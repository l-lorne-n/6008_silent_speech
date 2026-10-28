"""Experimental activity-first recognition controls, attached to the I/O owner."""
from pathlib import Path
from PySide6 import QtWidgets as W
from .lda_ui import Job, apply_lda_theme
from .streaming import load_model, train_activity, replay_activity


class StreamingWindow(W.QWidget):
    def __init__(self,root,main):
        super().__init__()
        self.root=Path(root);self.main=main;self.busy=False;self.live=False;self.job=None
        self.setWindowTitle('活动检测与在线识别 · 实验版');self.resize(850,650)
        layout=W.QVBoxLayout(self)
        intro=W.QLabel('先静息校准5秒 → 检测一次活动 → 识别或拒识 → 放松后重新等待。\n张嘴自然做口型，不出声。当前只显示结果，尚未连接网页控制。')
        intro.setWordWrap(True);layout.addWidget(intro)
        group=W.QGroupBox('01  用自动切段重新训练');form=W.QVBoxLayout(group)
        hint=W.QLabel('选择一份含至少3轮、开头至少5秒静息的完整录制。前80%整轮训练，最后20%留出。\n使用该录制启用的通道；双通道各自校准，任一路活动触发、两路放松后结束。请核对动作方式。')
        hint.setWordWrap(True);form.addWidget(hint)
        self.source=W.QLineEdit();self.source.setPlaceholderText('包含metadata.json与samples.csv的录制文件夹')
        row=W.QHBoxLayout();row.addWidget(self.source,1);self.choose=W.QPushButton('选择录制');row.addWidget(self.choose);form.addLayout(row)
        self.train=W.QPushButton('自动检测、训练并评估');form.addWidget(self.train);layout.addWidget(group)
        group=W.QGroupBox('02  加载模型与运行');form=W.QVBoxLayout(group)
        self.model=W.QLineEdit();self.model.setPlaceholderText('请选择活动切段模型；旧LDA模型不兼容')
        row=W.QHBoxLayout();row.addWidget(self.model,1);self.load=W.QPushButton('加载模型');row.addWidget(self.load);form.addLayout(row)
        row=W.QHBoxLayout();self.replay=W.QPushButton('回放所选录制');self.start=W.QPushButton('开始在线识别');self.stop=W.QPushButton('停止识别')
        for b in (self.replay,self.start,self.stop):row.addWidget(b)
        form.addLayout(row)
        self.status=W.QLabel('未运行');self.status.setWordWrap(True);form.addWidget(self.status)
        self.output=W.QLabel('等待指令');self.output.setStyleSheet('font-size:28px;font-weight:bold;padding:12px;color:#186647;');form.addWidget(self.output)
        note=W.QLabel('实验性拒识阈值尚未用独立非指令数据校准。显示词语不代表已验证正确；\n换人、移动电极或背景变化后，请停止再开始，重新静息校准。')
        note.setWordWrap(True);form.addWidget(note);layout.addWidget(group)
        self.log=W.QPlainTextEdit();self.log.setReadOnly(True);self.log.document().setMaximumBlockCount(500);layout.addWidget(self.log,1)
        self.choose.clicked.connect(self.choose_source);self.load.clicked.connect(self.choose_model)
        self.train.clicked.connect(self.train_model);self.replay.clicked.connect(self.replay_model)
        self.start.clicked.connect(self.start_live);self.stop.clicked.connect(self.stop_live)
        apply_lda_theme(self);self.controls()

    def controls(self):
        for widget in (self.choose,self.load,self.train,self.replay,self.source,self.model,self.start):widget.setEnabled(not self.busy and not self.live)
        self.stop.setEnabled(self.live)

    def choose_source(self):
        p=W.QFileDialog.getExistingDirectory(self,'选择完整录制',str(self.root/'recordings'))
        if p:self.source.setText(p)

    def choose_model(self):
        p,_=W.QFileDialog.getOpenFileName(self,'选择活动切段模型',str(self.root/'models'),'JSON (*.json)')
        if p:
            try:
                m=load_model(p);self.model.setText(p);self.log.appendPlainText('已加载词表：'+', '.join(m['classes'])+f"；通道{m.get('channels',[0])}")
            except Exception as exc:self.log.appendPlainText(str(exc))

    def launch(self,function,callback):
        if self.busy or self.live:return
        if self.main.lda_window and self.main.lda_window.busy:
            self.log.appendPlainText('请等待离线LDA任务完成。');return
        if self.main.recording or self.main.stopping:
            self.log.appendPlainText('请先结束并保存采集，再离线训练或回放。');return
        self.busy=True;self.controls();self.job=Job(function,self)
        self.job.progress.connect(self.log.appendPlainText);self.job.succeeded.connect(callback)
        self.job.failed.connect(lambda text:self.log.appendPlainText('失败：'+text))
        self.job.finished.connect(self.finished);self.job.start()

    def finished(self):
        self.busy=False;self.controls()

    def train_model(self):
        path=self.source.text().strip()
        def done(result):
            self.model.setText(result['model_path']);r=result['report']
            self.log.appendPlainText(f"已生成独立模型：{result['model_path']}\n留出{r['test_trials']}次：唯一匹配活动{r['test_detected']}次，接受且认对{r['test_correct']}次。未匹配候选中接受{r['unmatched_test_accepted']}次。\n这是同次录制内评估，尚非跨批或真人在线效果。")
        self.launch(lambda progress:train_activity(path,self.root/'models',progress),done)

    def replay_model(self):
        path=self.source.text().strip();model=self.model.text().strip()
        def done(result):
            r=result['report'];p=[e for e in r['events'] if e['kind']=='prediction']
            self.log.appendPlainText(f"回放完成：候选{len(p)}，接受{sum(e['accepted'] for e in p)}。{r['note']}\n报告：{result['folder']}")
        self.launch(lambda progress:replay_activity(model,path,self.root/'models',progress),done)

    def start_live(self):
        worker=self.main.worker
        if not worker or not worker.is_alive() or not worker.ready or worker.mode!='binary':
            self.log.appendPlainText('请先在主窗口连接真实1 kHz完整采样设备。');return
        if self.main.lda_window and self.main.lda_window.busy:
            self.log.appendPlainText('请等待离线LDA任务完成。');return
        try:load_model(self.model.text().strip())
        except Exception as exc:self.log.appendPlainText(str(exc));return
        self.live=True;self.controls();self.output.setText('请保持静息');self.status.setText('正在启动…')
        self.main.send('recognition_start',model_path=self.model.text().strip())

    def stop_live(self):
        self.main.send('recognition_stop')
        self.live=False;self.controls();self.status.setText('已请求停止');self.output.setText('等待指令')

    def message(self,m):
        kind=m['kind']
        if kind in ('disconnected','recognition_stopped','recognition_error'):
            self.live=False;self.controls();self.status.setText('已停止'+('：'+m['error'] if 'error' in m else ''));self.output.setText('等待指令')
        elif kind=='recognition_started':
            self.live=True;self.controls();self.status.setText('静息校准中，请放松5秒')
        elif kind=='recognition_status':
            s=m['status'];names={'calibrating':'静息校准','idle':'等待活动','active':'正在采集动作','refractory':'等待放松'}
            self.status.setText(f"{names[s['state']]} · 当前幅度 {s['level']:.0f}"+(f" · 剩余{s['calibration_remaining_s']:.1f}秒" if s['state']=='calibrating' else f" · 启动阈值{s['on']:.0f}"))
            if len(s.get('channel_levels',[]))>1:
                levels='/'.join(f'{v:.0f}' for v in s['channel_levels'])
                thresholds='/'.join(f'{v:.0f}' for v in (s.get('channel_on') or []))
                self.status.setText(f"{names[s['state']]} · 各路幅度 {levels} · 各路启动阈值 {thresholds or '校准中'}")
        elif kind=='recognition_event':
            e=m['event']
            if e['kind']=='prediction':
                self.output.setText(e['word'] if e['accepted'] else '未接受此动作')
                self.log.appendPlainText(f"样本{e['start']}–{e['end']}："+(e['word'] if e['accepted'] else '拒识')+f"；候选{e['candidate']}，模型分数{e['probability']:.2f}（不是校准置信度）")
            elif e['kind']=='reset':
                self.output.setText('请重新保持静息');self.log.appendPlainText('数据中断或削顶，活动已取消并重新校准：'+e['reason'])
            elif e['kind']=='rejected':
                self.output.setText('未接受此动作');self.log.appendPlainText('活动长度不合适：'+e['reason'])

    def closeEvent(self,event):
        if self.busy:event.ignore();return
        if self.live:self.stop_live()
        event.accept()
