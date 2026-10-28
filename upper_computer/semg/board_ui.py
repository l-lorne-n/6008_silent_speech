"""Serial display/cue UI. All signal processing and classification is on MCU."""
from datetime import datetime
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time
import asyncio

import numpy as np
import pyqtgraph as pg
import serial
from serial.tools import list_ports
from PySide6 import QtCore, QtGui, QtWidgets as W
from .board_protocol import BoardDecoder, CLASSES, model_info, result, score_trials, envelope

from .board_models import NAMES, NEW_MODEL, model_source
from .serial_settings import BAUDRATES, DEFAULT_BAUDRATE
from .ble_transport import BleByteStream, scan_bt24


class BleScanWorker(QtCore.QThread):
    found = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def run(self):
        try:self.found.emit(asyncio.run(scan_bt24()))
        except Exception as exc:self.failed.emit(f'{type(exc).__name__}: {exc}')


class BoardWindow(W.QMainWindow):
    def __init__(self,root,parent=None):
        super().__init__(parent)
        self.root=Path(root);self.port=None;self.busy=False;self.info=None;self.file=None
        self.ble_scan=None;self.ble_closing=None;self.link_initialized=False
        self.rx_bytes=0;self.text_probe=bytearray();self.csv_detected=False
        self.rows=[];self.outputs=[];self.session=None;self.markers={};self.token=100000
        self.last_rx=0;self.last_info_request=0;self.last_flush=0;self.last_device_end=None
        self.last_sequence=None;self.sequence_gaps=0
        self.envelope_points=deque(maxlen=501);self.last_envelope_rx=0;self.envelope_dirty=False
        self.setWindowTitle('sEMG · 板端实时监测与固件选择');self.resize(1200,960)
        self.apply_readable_theme()
        scroll=W.QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)
        widget=W.QWidget();scroll.setWidget(widget);layout=W.QVBoxLayout(widget)
        layout.setSizeConstraint(W.QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(22,18,22,18)
        title=W.QLabel('板端实时监测');title.setStyleSheet('font-size:25px;font-weight:bold');layout.addWidget(title)
        hint=W.QLabel('板端完成：20–500 Hz 滤波＋50 Hz 陷波 → 1400 ms 滑窗 → INT8 CNN → 静息解锁与连续确认。\n电脑只提词、接收结果和统计耗时；提示词不会发送给模型。')
        hint.setWordWrap(True);layout.addWidget(hint)
        controls=W.QHBoxLayout();layout.addLayout(controls)
        self.mode=W.QComboBox();self.mode.addItems(['实时监测模式','信号采集模式'])
        self.model=W.QComboBox();self.model.addItems(NAMES)
        self.model.setCurrentText(NEW_MODEL)
        self.flash_button=W.QPushButton('编译并烧录所选固件');self.flash_button.clicked.connect(self.flash)
        self.capture_button=W.QPushButton('返回信号采集界面');self.capture_button.clicked.connect(self.return_capture)
        for w in (self.mode,self.model,self.flash_button,self.capture_button):controls.addWidget(w)
        self.mode.currentIndexChanged.connect(self.change_mode)
        serial_row=W.QHBoxLayout();layout.addLayout(serial_row)
        self.transport=W.QComboBox();self.transport.addItems(['USB / COM 串口','BT24 / BLE 蓝牙'])
        self.ports=W.QComboBox();self.ports.setEditable(True)
        self.baud=W.QComboBox()
        for rate in BAUDRATES:self.baud.addItem(str(rate),rate)
        self.baud.setCurrentIndex(self.baud.findData(DEFAULT_BAUDRATE))
        self.refresh_button=W.QPushButton('刷新串口');self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button=W.QPushButton('连接监测');self.connect_button.clicked.connect(self.toggle_connection)
        self.state=W.QLabel('未连接 · 请按板端固件选择波特率');self.state.setWordWrap(True)
        for w in (self.transport,self.ports,self.refresh_button,W.QLabel('板端 UART'),self.baud,self.connect_button):serial_row.addWidget(w)
        layout.addWidget(self.state)
        baud_hint=W.QLabel('连接和监测固件构建使用所选波特率；须与板端及蓝牙模块一致。改变选项不会自动修改已烧录固件。\n采样仍为每路 2 kHz，模型不变；原始数据采集固件固定为 230400。')
        baud_hint.setWordWrap(True);layout.addWidget(baud_hint)
        ble_hint=W.QLabel('BT24：选择 BLE → 扫描 → 连接，无需在 Windows 设置配对。BLE 无波特率设置；上方 UART 仅记录/构建板端速率，不会配置模块。\n监测需要板端 CNN 二进制固件；队友的六列 CSV 测试固件只能验证蓝牙链路。')
        ble_hint.setWordWrap(True);layout.addWidget(ble_hint)
        self.identity=W.QLabel('模型身份：等待板端回报');self.identity.setWordWrap(True);layout.addWidget(self.identity)
        status=W.QLabel('mid64_temporal3 为旧版；mid64_temporal3_20261027 为新数据训练版，已于 2026-10-28 烧录并通过 BLE 身份及连续收包检查。真人识别效果仍需提词测试。')
        status.setWordWrap(True);layout.addWidget(status)
        cards=W.QHBoxLayout();layout.addLayout(cards)
        self.cue=W.QLabel('自由监测\n先放松，等待静息解锁')
        self.candidate=W.QLabel('当前滑窗候选\n—')
        self.command=W.QLabel('板端确认命令\n—')
        for w in (self.cue,self.candidate,self.command):
            w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);w.setMinimumHeight(95)
            w.setStyleSheet('background:#e9f1ed;border-radius:8px;padding:12px;font-size:24px;color:#163b31;')
            cards.addWidget(w,1)
        self.speed=W.QLabel('推理时间：等待板端 TIM2 / 1 MHz 测量');self.speed.setWordWrap(True);layout.addWidget(self.speed)
        self.health=W.QLabel('连续性：等待数据');self.health.setWordWrap(True);layout.addWidget(self.health)
        self.envelope_label=W.QLabel('双通道包络 · CH0 绿色 / CH1 橙色 · 固定 0–500 counts · 等待板端数据')
        self.envelope_label.setWordWrap(True);layout.addWidget(self.envelope_label)
        self.envelope_plot=pg.PlotWidget(background='#ffffff')
        self.envelope_plot.setMinimumHeight(190);self.envelope_plot.setMaximumHeight(230)
        self.envelope_plot.setLabel('left','包络 / ADC counts',color='#263d35')
        self.envelope_plot.setLabel('bottom','相对时间 / s',color='#263d35')
        for axis in ('left','bottom'):
            self.envelope_plot.getAxis(axis).setPen('#64786e')
            self.envelope_plot.getAxis(axis).setTextPen('#263d35')
        self.envelope_plot.setXRange(-5,0,padding=0)
        self.envelope_plot.setYRange(0,500,padding=0)
        self.envelope_plot.setLimits(xMin=-5,xMax=0,yMin=0,yMax=500)
        self.envelope_plot.disableAutoRange();self.envelope_plot.setMouseEnabled(x=False,y=False)
        self.envelope_plot.setMenuEnabled(False);self.envelope_plot.hideButtons()
        self.envelope_plot.showGrid(x=True,y=True,alpha=.18)
        self.envelope_curves=[self.envelope_plot.plot(pen=pg.mkPen(color,width=2.2)) for color in ('#14834b','#d66b12')]
        layout.addWidget(self.envelope_plot)
        cue_controls=W.QHBoxLayout();layout.addLayout(cue_controls)
        self.rounds=W.QSpinBox();self.rounds.setRange(1,20);self.rounds.setValue(1)
        self.start=W.QPushButton('开始十词提词测试');self.start.clicked.connect(self.start_cues)
        self.stop=W.QPushButton('停止测试');self.stop.clicked.connect(lambda:self.finish_cues(aborted=True))
        cue_controls.addWidget(W.QLabel('每词次数'));cue_controls.addWidget(self.rounds)
        cue_controls.addWidget(self.start);cue_controls.addWidget(self.stop)
        self.summary=W.QLabel('测试流程：静息5秒；每词准备1.5秒、动作2秒、休息2秒。静息类 rest 自动参与解锁。')
        self.summary.setWordWrap(True);layout.addWidget(self.summary)
        self.table=W.QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['板端时刻 / s','确认命令','置信度','纯推理 / ms'])
        self.table.setMinimumHeight(140);self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table,1)
        self.log=W.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMinimumHeight(65);self.log.setMaximumHeight(90);layout.addWidget(self.log)
        self.log.document().setMaximumBlockCount(1500)
        self.transport.currentIndexChanged.connect(self.change_transport)
        self.refresh_ports();self.update_controls()
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(25)

    def apply_readable_theme(self):
        # Fix both palette and explicit item-view surfaces: Windows dark mode
        # otherwise supplies a dark Base behind the parent's dark text CSS.
        palette=QtGui.QPalette()
        roles={'Window':'#f3f5f4','WindowText':'#193c31','Base':'#ffffff','AlternateBase':'#edf3ef',
               'Text':'#193c31','Button':'#ffffff','ButtonText':'#193c31','Highlight':'#216b50',
               'HighlightedText':'#ffffff','ToolTipBase':'#ffffff','ToolTipText':'#193c31','PlaceholderText':'#53675d'}
        for group in (QtGui.QPalette.ColorGroup.Active,QtGui.QPalette.ColorGroup.Inactive,QtGui.QPalette.ColorGroup.Disabled):
            for role,color in roles.items():palette.setColor(group,getattr(QtGui.QPalette.ColorRole,role),QtGui.QColor(color))
        for role in ('Text','ButtonText','WindowText'):
            palette.setColor(QtGui.QPalette.ColorGroup.Disabled,getattr(QtGui.QPalette.ColorRole,role),QtGui.QColor('#66776e'))
        self.setPalette(palette)
        self.setStyleSheet('''
            QMainWindow, QWidget {background:#f3f5f4;color:#193c31;font-family:'Microsoft YaHei';font-size:13px;}
            QTableWidget,QTableView,QPlainTextEdit,QTextEdit {
                background:#ffffff;color:#193c31;alternate-background-color:#edf3ef;
                selection-background-color:#216b50;selection-color:#ffffff;
                border:1px solid #b9cbbf;gridline-color:#d8e2db;border-radius:4px;
            }
            QTableWidget::item {padding:4px;color:#193c31;}
            QTableWidget::item:selected {background:#216b50;color:#ffffff;}
            QHeaderView::section,QTableCornerButton::section {
                background:#dfebe3;color:#193c31;padding:7px;border:0;border-bottom:1px solid #b9cbbf;font-weight:bold;
            }
            QLineEdit,QComboBox,QSpinBox {background:#ffffff;color:#193c31;border:1px solid #b9cbbf;padding:5px;border-radius:4px;}
            QComboBox QAbstractItemView {background:#ffffff;color:#193c31;selection-background-color:#216b50;selection-color:#ffffff;}
            QPushButton {background:#ffffff;color:#193c31;border:1px solid #b9cbbf;border-radius:5px;padding:8px 10px;}
            QPushButton:hover {background:#dfebe3;}
            QPushButton:disabled,QComboBox:disabled,QSpinBox:disabled {background:#e9eeeb;color:#66776e;}
            QScrollArea {border:0;}
            QScrollBar:vertical {background:#e5ece7;width:12px;}
            QScrollBar::handle:vertical {background:#9eafa4;min-height:25px;border-radius:4px;}
        ''')

    def clear_envelope(self,message):
        self.envelope_points.clear();self.last_envelope_rx=0;self.envelope_dirty=False
        for curve in self.envelope_curves:curve.setData([],[])
        self.envelope_label.setText(message)

    def refresh_envelope(self):
        if not self.envelope_dirty:return
        self.envelope_dirty=False
        points=np.asarray(self.envelope_points,dtype=float)
        if not len(points):return
        times=(points[:,0]-points[-1,0])/2000
        for ch,curve in enumerate(self.envelope_curves):curve.setData(times,points[:,ch+1])
        a,b=points[-1,1:];maximum=points[:,1:].max()
        overflow=f' · 最近5秒超出量程，峰值 {maximum:.0f}' if maximum>500 else ''
        self.envelope_label.setText(f'CH0 绿色：{a:.1f}  /  CH1 橙色：{b:.1f} counts · 200 ms包络 · 固定0–500{overflow}')

    @property
    def live(self):return self.port is not None

    def refresh_ports(self):
        if self.transport.currentIndex()==1:
            if self.port or self.ble_scan or self.ble_closing:return
            self.ports.clear();self.state.setText('正在扫描 BT24，请等待约 6 秒…')
            self.ble_scan=BleScanWorker(self)
            self.ble_scan.found.connect(self.ble_found)
            self.ble_scan.failed.connect(self.ble_scan_failed)
            self.ble_scan.finished.connect(self.ble_scan_finished)
            self.ble_scan.start();self.update_controls();return
        old=self.ports.currentText();self.ports.clear()
        found=list(list_ports.comports())
        for p in found:self.ports.addItem(p.device)
        chosen=old or next((p.device for p in found if 'CH340' in p.description),'')
        if chosen:self.ports.setCurrentText(chosen)

    def change_transport(self):
        self.ports.clear();self.ports.setEditable(self.transport.currentIndex()==0)
        self.refresh_button.setText('扫描 BT24' if self.transport.currentIndex()==1 else '刷新串口')
        self.state.setText('点击“扫描 BT24”，再选择设备连接。' if self.transport.currentIndex()==1 else '请选择串口。')
        if self.transport.currentIndex()==0:self.refresh_ports()
        self.update_controls()

    def ble_found(self,devices):
        for d in devices:self.ports.addItem(f'{d["name"]} · {d["address"]} · {d["rssi"]} dBm',d['address'])
        self.state.setText(f'找到 {len(devices)} 个 BLE 候选，请选择自己的模块。' if devices else '未发现 BT24：检查供电，并断开手机/其他程序的连接后重新扫描。')

    def ble_scan_failed(self,message):
        self.log.appendPlainText('BLE 扫描失败：'+message);self.state.setText('BLE 扫描失败，详见下方日志。')

    def ble_scan_finished(self):
        self.ble_scan.deleteLater();self.ble_scan=None;self.update_controls()

    def update_controls(self):
        pending=bool(self.ble_scan or self.ble_closing)
        self.flash_button.setEnabled(not self.busy and not self.session and not pending)
        self.connect_button.setEnabled(not self.busy and not pending)
        self.connect_button.setText('断开监测' if self.port else '连接监测')
        for w in (self.mode,self.model,self.ports,self.transport,self.refresh_button):w.setEnabled(not self.busy and not self.port and not pending)
        self.model.setEnabled(not self.busy and not self.port and self.mode.currentIndex()==0)
        self.baud.setEnabled(not self.busy and not self.port and not pending and self.mode.currentIndex()==0)
        self.start.setEnabled(bool(self.port and self.info and self.matches() and not self.session and self.last_rx and time.monotonic()-self.last_rx<1 and self.rows and not self.rows[-1]['flags']))
        self.stop.setEnabled(bool(self.session));self.capture_button.setEnabled(not self.busy and not self.session and not pending)

    def change_mode(self,index):
        if index==1:self.baud.setCurrentIndex(self.baud.findData(DEFAULT_BAUDRATE))
        self.update_controls()

    def matches(self):
        if not self.info:return False
        name=self.model.currentText()
        folder=model_source(name,self.root)
        path=folder/'model_int8_qdq.onnx'
        return (self.info.get('model')==name and path.exists() and
                hashlib.sha256(path.read_bytes()).hexdigest()[:16]==self.info.get('sha') and self.info.get('mode')==1)

    def write_log(self,kind,**data):
        if self.file:self.file.write(json.dumps(dict(kind=kind,host_monotonic=time.monotonic(),**data),ensure_ascii=False)+'\n')

    def send(self,cmd):
        if self.port:self.port.write((cmd+'\n').encode('ascii'))

    def toggle_connection(self):
        if self.port:self.disconnect();return
        if self.mode.currentIndex()!=0:
            self.log.appendPlainText('采集模式请烧录采集固件后返回主采集界面连接。');return
        parent=self.parent()
        if parent and parent.worker and parent.worker.is_alive():
            self.log.appendPlainText('请先断开主采集界面的串口。');return
        try:
            if self.transport.currentIndex()==1:
                address=self.ports.currentData()
                if not address:raise ValueError('请先扫描并选择 BT24。')
                self.port=BleByteStream(address)
            else:
                self.port=serial.Serial(port=None,baudrate=self.baud.currentData(),timeout=0,write_timeout=.3)
                self.port.dtr=False;self.port.rts=False;self.port.port=self.ports.currentText();self.port.open()
            self.link_initialized=False;self.rx_bytes=0;self.text_probe=bytearray();self.csv_detected=False
            self.decoder=BoardDecoder();self.info=None;self.rows=[];self.outputs=[];self.last_device_end=None
            self.last_sequence=None;self.sequence_gaps=0
            self.clear_envelope('双通道包络 · 固定0–500 counts · 等待板端数据')
            self.table.setRowCount(0);self.markers={};self.last_rx=0
            self.output_dir=self.root/'recordings_live'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.output_dir.mkdir(parents=True);self.file=(self.output_dir/'board_trace.jsonl').open('w',encoding='utf-8')
            self.write_log('connection',port=self.ports.currentData() if self.transport.currentIndex()==1 else self.ports.currentText(),
                           transport='ble' if self.transport.currentIndex()==1 else 'serial',baudrate=self.baud.currentData())
            if isinstance(self.port,BleByteStream):self.state.setText('正在连接 BT24 并订阅通知…')
            else:self.initialize_link()
            self.log.appendPlainText(f'保存到 {self.output_dir}')
        except Exception as exc:
            self.log.appendPlainText(str(exc));self.disconnect()
        self.update_controls()

    def initialize_link(self):
        if isinstance(self.port,BleByteStream):self.port.write(b'N\nI\n')
        else:self.send('N');self.send('I')
        self.last_info_request=time.monotonic();self.link_initialized=True
        self.state.setText(f'链路已连接 · 板端 UART {self.baud.currentData()}，等待模型身份及推理结果…')

    def disconnect(self):
        if self.session:self.finish_cues(aborted=True)
        if self.port:
            self.port.close()
            if isinstance(self.port,BleByteStream):self.ble_closing=self.port
            self.port=None
        if self.file:self.file.flush();self.file.close();self.file=None
        self.info=None;self.state.setText('已断开');self.candidate.setText('当前滑窗候选\n—')
        self.clear_envelope('双通道包络 · 已断开 · 固定0–500 counts')
        self.command.setText('板端确认命令\n—');self.update_controls()

    def flash(self):
        parent=self.parent()
        if parent and (parent.recording or parent.stopping or (parent.worker and parent.worker.is_alive())):
            self.log.appendPlainText('请先保存录制并断开主采集界面的串口。');return
        self.disconnect();self.busy=True;self.update_controls()
        self.process=QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(lambda:self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')))
        self.process.finished.connect(self.flash_finished)
        self.process.errorOccurred.connect(self.flash_error)
        py=self.root/'.runtime/Scripts/python.exe'
        args=self.firmware_arguments()
        self.log.appendPlainText('开始备份 / 编译 / 烧录 / 校验，请保持调试器和板子连接。')
        self.process.start(str(py),args)

    def firmware_arguments(self):
        args=['-u',str(self.root/'board/manage.py')]
        if self.mode.currentIndex()==1:return args+['flash','--capture','--baud',str(DEFAULT_BAUDRATE)]
        return args+['build-flash','--model',self.model.currentText(),'--baud',str(self.baud.currentData())]

    def flash_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.busy=False;self.log.appendPlainText('烧录进程无法启动：'+self.process.errorString());self.update_controls()

    def flash_finished(self,code,status):
        self.busy=False;self.update_controls()
        self.log.appendPlainText('烧录并校验完成。可连接监测。' if code==0 and self.mode.currentIndex()==0 else
                                 ('采集固件已恢复，可返回主界面连接采集。' if code==0 else '烧录未成功，请查看上方错误和 board/logs。'))

    def return_capture(self):
        self.disconnect()
        if self.parent() and self.mode.currentIndex()==1:
            self.parent().baud.setCurrentIndex(self.parent().baud.findData(DEFAULT_BAUDRATE))
        if self.parent():self.parent().show();self.parent().raise_()
        self.close()

    def marker(self,trial,key):
        self.token+=1;self.markers[self.token]=(trial,key,time.monotonic());self.send(f'M,{self.token}')

    def start_cues(self):
        if not self.port or not self.matches() or self.session or not self.rows or self.rows[-1]['flags']:return
        trials=[]
        for _ in range(self.rounds.value()):
            words=CLASSES[:10].copy();random.shuffle(words)
            trials.extend(dict(word=w) for w in words)
        self.session=dict(trials=trials,index=-1,phase='baseline',deadline=time.monotonic()+5,outputs_start=len(self.outputs),bad=False)
        self.cue.setText('静息 5 秒\n保持放松');self.write_log('test_start',words=[t['word'] for t in trials])
        self.summary.setText('测试进行中；只按“现在动作”提示发音，准备与休息时放松。')
        self.update_controls()

    def advance_cues(self):
        s=self.session
        if not s or time.monotonic()<s['deadline']:return
        trials=s['trials'];i=s['index'];phase=s['phase']
        if phase in ('baseline','rest'):
            if i>=0:self.marker(trials[i],'end_idx')
            i+=1;s['index']=i
            if i==len(trials):s.update(phase='drain',deadline=time.monotonic()+.7);self.cue.setText('本组结束\n正在收尾');return
            t=trials[i];s.update(phase='prepare',deadline=time.monotonic()+1.5)
            self.cue.setText(f'准备 · {i+1}/{len(trials)}\n{t["word"]}');self.marker(t,'prepare_idx')
        elif phase=='prepare':
            t=trials[i];s.update(phase='action',deadline=time.monotonic()+2)
            self.cue.setText(f'现在动作 · 无声口型一次\n{t["word"]}');self.marker(t,'action_idx')
        elif phase=='action':
            t=trials[i];s.update(phase='rest',deadline=time.monotonic()+2)
            self.cue.setText('休息 · 放松\n等待下一词');self.marker(t,'rest_idx')
        elif phase=='drain':self.finish_cues()

    def finish_cues(self,aborted=False):
        if not self.session:return
        s=self.session;self.session=None
        rows=score_trials(s['trials'],self.outputs[s['outputs_start']:])
        valid=[r for r in rows if r['valid']]
        report=dict(aborted=aborted,transport_invalid=s['bad'],baudrate=self.baud.currentData(),model=self.info,trials=s['trials'],scores=rows,
                    transport='ble' if self.transport.currentIndex()==1 else 'serial',
                    timing_note='Cue ACKs use 10 ms ADC block boundaries; not physical muscle-onset latency.')
        self.write_log('test_end',report=report)
        name='test_'+datetime.now().strftime('%H%M%S_%f')+'.json'
        (self.output_dir/name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if aborted or s['bad']:
            self.summary.setText('测试中止或通信异常，本组不作为有效正确率。已有输出已保存。')
        else:
            n=len(valid);correct=sum(r['correct'] for r in valid);strict=sum(r['strict'] for r in valid)
            wrong=sum(r['wrong'] for r in valid);miss=sum(r['miss'] for r in valid)
            self.summary.setText(f'有效试次 {n}/{len(rows)} · 正确 {correct}/{n} · 严格成功 {strict}/{n} · 错词输出 {wrong} · 漏检 {miss}。报告：{name}')
        self.cue.setText('自由监测\n本组已结束');self.update_controls()

    def poll(self):
        now=time.monotonic()
        if self.ble_closing and self.ble_closing.closed.is_set():self.ble_closing=None
        if self.port:
            try:
                if isinstance(self.port,BleByteStream):
                    self.port._check()
                    if not self.port.ready.is_set():self.update_controls();return
                    if not self.link_initialized:self.initialize_link()
                data=self.port.read(min(max(self.port.in_waiting,1),32768))
                self.rx_bytes+=len(data)
                if not self.info and not self.csv_detected:
                    self.text_probe.extend(data)
                    while b'\n' in self.text_probe:
                        line,_,rest=self.text_probe.partition(b'\n');self.text_probe=bytearray(rest)
                        try:
                            values=[int(x) for x in line.strip().split(b',')]
                            if len(values)==6 and all(0<=v<=65535 for v in values[:2]):self.csv_detected=True
                        except ValueError:pass
                    if len(self.text_probe)>4096:self.text_probe.clear()
                for packet in self.decoder.feed(data):self.handle_packet(packet)
                self.refresh_envelope()
                if self.last_envelope_rx and now-self.last_envelope_rx>1:
                    self.clear_envelope('双通道包络 · 数据已过期，请检查连接 · 固定0–500 counts')
                if not self.info and not self.csv_detected and now-self.last_info_request>2:self.send('I');self.last_info_request=now
                if self.session and (self.decoder.crc_errors or self.sequence_gaps or now-self.last_rx>1):
                    self.session['bad']=True;self.finish_cues(aborted=True)
                if now-self.last_rx>1:
                    self.state.setText(f'已接收 {self.rx_bytes} 字节：检测到六列 CSV 测试固件。蓝牙链路可用，但不能用于 CNN 实时监测；请使用匹配 UART 速率的监测固件。' if self.csv_detected and not self.info else
                                       f'等待有效监测结果 · 接收 {self.rx_bytes} 字节 · UART {self.baud.currentData()}：检查板端/蓝牙波特率、固件及模型身份')
                    self.candidate.setText('当前滑窗候选\n数据未就绪 / 已过期')
                if self.file and now-self.last_flush>1:self.file.flush();self.last_flush=now
                self.advance_cues()
            except Exception as exc:
                self.log.appendPlainText('通信错误：'+str(exc));self.disconnect()
        self.update_controls()

    def handle_packet(self,p):
        if self.last_sequence is not None and p.seq!=(self.last_sequence+1)&0xffffffff:
            self.sequence_gaps+=1
        self.last_sequence=p.seq
        if p.kind==5:
            self.info=model_info(p);self.write_log('model',info=self.info)
            if self.info.get('env_hz')!=100:
                self.clear_envelope('当前板端固件未提供包络，请点击“编译并烧录所选固件”更新。')
            self.identity.setText(f'板端模型：{self.info["model"]} / SHA256 {self.info["sha"]} · '+('与所选模型一致' if self.matches() else '与选择不一致，禁止开始提词测试'))
        elif p.kind==8:
            values=envelope(p)
            if self.envelope_points and p.end-self.envelope_points[-1][0]!=20:
                self.envelope_points.clear()
            self.envelope_points.append((p.end,*values));self.last_envelope_rx=time.monotonic();self.envelope_dirty=True
            self.write_log('envelope',end=p.end,ch0=values[0],ch1=values[1])
        elif p.kind==3:
            token,tick=struct.unpack('<II',p.payload)
            item=self.markers.pop(token,None)
            if item:item[0][item[1]]=p.end;self.write_log('cue_ack',token=token,end=p.end,tick=tick,key=item[1])
        elif p.kind==4:
            r=result(p);self.last_rx=time.monotonic();self.write_log('result',**r)
            if self.last_device_end is not None and p.end<self.last_device_end:
                if self.session:self.session['bad']=True;self.finish_cues(aborted=True)
                self.info=None;self.rows=[];self.outputs=[];self.log.appendPlainText('检测到设备复位，重新验证模型身份。')
            self.last_device_end=p.end
            if self.session and (p.flags or r['skipped_windows'] or r['tx_dropped']):self.session['bad']=True
            self.rows.append(r)
            if len(self.rows)>1000:self.rows=self.rows[-1000:]
            top=r['top'];self.candidate.setText(f'当前滑窗候选\n{CLASSES[top]} · {r["probabilities"][top]:.1%}')
            self.state.setText(f'连续监测中 · {self.baud.currentData()} baud · '+('已解锁，等待命令' if r['armed'] else '等待连续静息解锁'))
            if p.flags:self.state.setText('板端 UART 故障 4：请复位 STM32 后重新验证。' if p.flags&4 else f'板端故障 {p.flags}，禁止开始提词测试。')
            if r['emitted']!=255:
                label=r['emitted'];word=CLASSES[label]
                self.outputs.append(dict(end=p.end,word=word,probability=r['probabilities'][label]))
                self.command.setText(f'板端确认命令\n{word}')
                row=self.table.rowCount();self.table.insertRow(row)
                for col,value in enumerate([f'{p.end/2000:.2f}',word,f'{r["probabilities"][label]:.1%}',f'{r["inference_us"]/1000:.2f}']):self.table.setItem(row,col,W.QTableWidgetItem(value))
                self.table.scrollToBottom()
            infer=np.array([x['inference_us'] for x in self.rows])/1000
            total=np.array([x['pipeline_us'] for x in self.rows])/1000
            self.speed.setText(f'板端纯推理：当前 {infer[-1]:.2f} ms，近{len(infer)}次 P50 {np.median(infer):.2f} / P95 {np.percentile(infer,95):.2f} / 最大 {infer.max():.2f} ms\n'
                               f'每步滤波＋复制＋推理＋解码：当前 {total[-1]:.2f} ms，累计最大 {r["max_pipeline_us"]/1000:.2f} ms；≥100 ms {r["over_100ms"]} 次，≥200 ms {r["over_200ms"]} 次')
            self.health.setText(f'跳过滑窗 {r["skipped_windows"]} · 板端丢包 {r["tx_dropped"]} · 接收序号异常 {self.sequence_gaps} · CRC错误 {self.decoder.crc_errors} · 设备故障 {p.flags} · ADC削顶计数 {r["clipped_samples"]}\n'
                                '100 ms步进需持续跟上；上述计算耗时不等于从开始发音到确认命令的总响应时间。')

    def closeEvent(self,event):
        if self.busy:self.log.appendPlainText('烧录进行中，完成后再关闭。');event.ignore();return
        if self.ble_scan or self.ble_closing or isinstance(self.port,BleByteStream):
            if self.port:self.disconnect()
            self.log.appendPlainText('正在释放蓝牙，请稍候，窗口将自动关闭。')
            QtCore.QTimer.singleShot(200,self.close);event.ignore();return
        self.disconnect();event.accept()
