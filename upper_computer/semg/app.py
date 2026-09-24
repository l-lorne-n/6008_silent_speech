from collections import deque
import json
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets as W
import pyqtgraph as pg
from scipy.signal import sosfilt
from serial.tools import list_ports

from .acquisition import Acquisition
from .filtering import display_sos, display_filter_metadata
from .session import WORDS, TrialEngine, schedule
from . import __version__
from .rhythm import QuietRhythmMonitor, indicator_metadata
from .vocabulary import load_vocabulary, save_vocabulary

ROOT = Path(__file__).resolve().parents[1]
PHASES = {"idle": "等待开始", "baseline": "静息基线", "prepare": "准备 · 看词",
          "action": "现在 · 内部发音一次", "rest": "休息 · 放松", "paused": "已暂停",
          "finished": "本组完成", "stopped": "已终止"}


def configure_fonts(app):
    # Also works in headless Qt, where Windows font discovery is unavailable.
    for name in ("msyh.ttc", "msyhbd.ttc"):
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
        if path.exists():
            QtGui.QFontDatabase.addApplicationFont(str(path))
    app.setFont(QtGui.QFont("Microsoft YaHei", 10))


class CueWindow(W.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sEMG · 参与者提词")
        self.setStyleSheet("background:#102b2b; color:#f0f6f4;")
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        self.phase = W.QLabel("等待开始")
        self.phase.setStyleSheet("font-size:30px;color:#91d4b4")
        self.word = W.QLabel("准备好了吗")
        self.word.setStyleSheet("font-size:100px;font-weight:700;")
        self.time = W.QLabel("")
        self.time.setStyleSheet("font-size:36px")
        self.instruction = W.QLabel("保持闭唇，内部尝试发音一次，不出声\n完成后放松 · Esc 返回操作界面")
        self.instruction.setStyleSheet("font-size:22px;color:#b6cdca")
        for widget in (self.phase, self.word, self.time, self.instruction):
            widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(widget, 1)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"闭唇语音 · sEMG 提词与采集 v{__version__}")
        self.resize(1280, 900)
        self.worker = None
        self.engine = None
        self.event_id = 0
        self.recording = False
        self.stopping = False
        self.current_path = None
        self.raw = deque(maxlen=5000)
        self.filtered = deque(maxlen=5000)
        self.envelope = deque(maxlen=5000)
        self.env_ring = deque(maxlen=200)
        self.sos = display_sos()
        self.zi = np.zeros((len(self.sos), 2))
        self.rhythm = QuietRhythmMonitor()
        self.rhythm_context = None
        self.rhythm_after_ns = 0
        self.cue_window = CueWindow()
        self.lda_window = None
        self.vocabulary_error = None
        try:
            self.vocabulary, self.selected_words = load_vocabulary(ROOT / "vocabulary.json")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.vocabulary, self.selected_words = list(WORDS), list(WORDS)
            self.vocabulary_error = str(exc)
        self.build_ui()
        if self.vocabulary_error:
            self.banner.setText("词表配置读取失败，暂显示默认四词；请在“管理词表”中确认并保存后再采集。" + self.vocabulary_error)
        self.refresh_ports()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)
        self.plot_tick = 0

    def build_ui(self):
        self.setStyleSheet("""
        QMainWindow, QWidget#root {background:#f3f5f4;color:#153330;}
        QWidget#sidebar, QWidget#sidebarViewport, QScrollArea#sidebarScroll {background:#f3f5f4;color:#153330;}
        QScrollArea#sidebarScroll {border:0;}
        QScrollArea#sidebarScroll QScrollBar:vertical {background:#e4ebe7;width:12px;margin:0;}
        QScrollArea#sidebarScroll QScrollBar::handle:vertical {background:#9eb5a8;border-radius:5px;min-height:24px;}
        QScrollArea#sidebarScroll QScrollBar::add-line:vertical, QScrollArea#sidebarScroll QScrollBar::sub-line:vertical {height:0;}
        QScrollArea#sidebarScroll QScrollBar::add-page:vertical, QScrollArea#sidebarScroll QScrollBar::sub-page:vertical {background:#e4ebe7;}
        QWidget {font-family:'Microsoft YaHei';font-size:13px;color:#153330;}
        QWidget:disabled {color:#73847b;}
        QGroupBox {border:1px solid #d7e1dc;border-radius:8px;margin-top:14px;padding:14px 10px 10px;background:white;}
        QGroupBox::title {subcontrol-origin:margin;left:12px;font-weight:bold;}
        QPushButton {padding:9px 12px;border:1px solid #c9d8d0;border-radius:6px;background:white;color:#173f34;}
        QPushButton:hover {background:#e5f2e9;}
        QPushButton:disabled {color:#9daaa3;background:#eef1ef;}
        QPushButton#primary {background:#186647;color:white;border:0;font-weight:bold;}
        QPushButton#primary:disabled {background:#a7beb1;}
        QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox {background:white;color:#153330;border:1px solid #ccd8d0;border-radius:4px;padding:5px;selection-background-color:#186647;selection-color:white;}
        QComboBox QAbstractItemView {background:white;color:#153330;selection-background-color:#186647;selection-color:white;}
        QProgressBar {border:0;background:#dce7e1;border-radius:4px;height:10px;text-align:center;}
        QProgressBar::chunk {background:#2e8861;border-radius:4px;}
        """)
        root = W.QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = W.QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 18)
        title = W.QLabel("闭唇语音  /  sEMG 采集台")
        title.setStyleSheet("font-size:25px;font-weight:700")
        title_row = W.QHBoxLayout()
        title_row.addWidget(title, 1)
        self.lda_button = W.QPushButton("LDA 训练与模型测试")
        self.lda_button.clicked.connect(self.open_lda)
        title_row.addWidget(self.lda_button)
        outer.addLayout(title_row)
        self.banner = W.QLabel("连接后检查波形，再开始提词。保持闭唇，内部尝试发音一次，不出声。")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("padding:8px;background:#e4eee8;border-radius:6px;color:#24513e")
        outer.addWidget(self.banner)
        body = W.QHBoxLayout()
        outer.addLayout(body, 1)
        side = W.QWidget(objectName="sidebar")
        side.setFixedWidth(295)
        sl = W.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 10, 0)
        side_scroll = W.QScrollArea(objectName="sidebarScroll")
        side_scroll.viewport().setObjectName("sidebarViewport")
        palette = QtGui.QPalette(side_scroll.palette())
        for group in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive,
                      QtGui.QPalette.ColorGroup.Disabled):
            palette.setColor(group, QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f3f5f4"))
            palette.setColor(group, QtGui.QPalette.ColorRole.Base, QtGui.QColor("#f3f5f4"))
            palette.setColor(group, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#153330"))
        for widget in (side_scroll, side_scroll.viewport(), side):
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        side_scroll.setWidget(side)
        side_scroll.setFixedWidth(320)
        body.addWidget(side_scroll)

        connection = W.QGroupBox("01  连接设备")
        form = W.QFormLayout(connection)
        self.mode = W.QComboBox()
        self.mode.addItem("当前固件 · 三列文本联调", "legacy")
        self.mode.addItem("完整采样 · 二进制 v1", "binary")
        self.mode.addItem("模拟演练 · 非真实信号", "simulation")
        self.mode.setCurrentIndex(1)
        self.port = W.QComboBox()
        self.port.setEditable(True)
        form.addRow(self.mode)
        form.addRow("串口", self.port)
        self.refresh_button = W.QPushButton("刷新端口")
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button = W.QPushButton("连接 / 检查数据")
        self.connect_button.clicked.connect(self.toggle_connection)
        form.addRow(self.refresh_button, self.connect_button)
        self.connection_status = W.QLabel("未连接 · 115200 / 8N1")
        self.connection_status.setWordWrap(True)
        form.addRow(self.connection_status)
        sl.addWidget(connection)

        self.config_box = W.QGroupBox("02  本次采集")
        form = W.QFormLayout(self.config_box)
        self.subject = W.QLineEdit("S01")
        self.session_id = W.QSpinBox()
        self.session_id.setRange(1, 999)
        self.reps = W.QSpinBox()
        self.reps.setRange(1, 100)
        self.reps.setValue(10)
        self.seed = W.QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(20260923)
        self.site = W.QLineEdit()
        self.site.setPlaceholderText("例如：下颌左侧；参考电极…")
        self.notes = W.QLineEdit()
        self.notes.setPlaceholderText("贴片、姿势或异常备注")
        for label, widget in [("受试者", self.subject), ("Session", self.session_id), ("每词次数", self.reps),
                              ("随机种子", self.seed), ("电极位置", self.site), ("备注", self.notes)]:
            form.addRow(label, widget)
        self.durations = {}
        for key, label, default in [("baseline", "静息基线 / 秒", 10), ("prepare", "准备 / 秒", 1.5),
                                    ("action", "内部发音 / 秒", 2), ("rest", "休息 / 秒", 2)]:
            spin = W.QDoubleSpinBox()
            spin.setRange(.1, 120)
            spin.setValue(default)
            spin.setSingleStep(.5)
            self.durations[key] = spin
            form.addRow(label, spin)
        sl.addWidget(self.config_box)
        self.words_label = W.QLabel()
        self.words_label.setWordWrap(True)
        self.words_button = W.QPushButton("管理词表 / 添加词")
        self.words_button.clicked.connect(self.edit_vocabulary)
        form.addRow(self.words_label)
        form.addRow(self.words_button)
        self.reps.valueChanged.connect(self.update_words_summary)
        self.update_words_summary()
        self.fullscreen = W.QPushButton("打开参与者全屏提词")
        self.fullscreen.clicked.connect(self.cue_window.showFullScreen)
        sl.addWidget(self.fullscreen)
        sl.addStretch()

        right = W.QVBoxLayout()
        body.addLayout(right, 1)
        cue = W.QFrame()
        cue.setMinimumHeight(190)
        cue.setStyleSheet("QFrame{background:#113b31;border-radius:10px;color:#f6faf7;}")
        cl = W.QVBoxLayout(cue)
        self.phase_label = W.QLabel("等待开始")
        self.phase_label.setStyleSheet("font-size:16px;color:#a9dbc4")
        self.word_label = W.QLabel(" · ".join(self.selected_words))
        self.word_label.setWordWrap(True)
        self.word_label.setStyleSheet("font-size:38px;font-weight:700;color:#f6faf7")
        self.countdown = W.QLabel("先连接设备，或选择模拟演练")
        self.countdown.setStyleSheet("font-size:16px;color:#d0e6dc")
        for widget in (self.phase_label, self.word_label, self.countdown):
            widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(widget)
        right.addWidget(cue)
        controls = W.QHBoxLayout()
        self.start_button = W.QPushButton("开始采集")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.pause_button = W.QPushButton("暂停")
        self.reject_button = W.QPushButton("当前作废 + 补录")
        self.stop_button = W.QPushButton("结束并保存")
        self.start_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.pause)
        self.reject_button.clicked.connect(self.reject)
        self.stop_button.clicked.connect(self.stop_recording)
        for button in (self.start_button, self.pause_button, self.reject_button, self.stop_button):
            controls.addWidget(button)
        right.addLayout(controls)
        self.progress = W.QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)
        self.stats_label = W.QLabel("有效接收率 —    保存样本 —    缺失 —    标记往返 —")
        self.stats_label.setWordWrap(True)
        right.addWidget(self.stats_label)
        self.quality_label = W.QLabel("信号检查：等待数据")
        self.quality_label.setStyleSheet("color:#64776c")
        right.addWidget(self.quality_label)
        self.rhythm_label = W.QLabel("静息节律：连接完整采样设备后，保持放松约 10 秒")
        self.rhythm_label.setWordWrap(True)
        self.rhythm_label.setStyleSheet("padding:6px;background:#edf1f0;color:#52645d;border-radius:4px")
        self.rhythm_label.setToolTip(
            "独立分析最近连续10秒原始信号中的5–30 Hz脉冲；幅度为该频带内脉冲峰峰值。\n"
            "这是疑似心搏节律，不是经ECG验证的心率，也不证明接线或肌电质量合格。\n"
            "未检出不等于接线异常；发音流程中暂停，不拼接各次短休息段。")
        right.addWidget(self.rhythm_label)
        ranges = W.QHBoxLayout()
        self.range_status = W.QLabel("纵轴固定 · 不随波形自动缩放")
        ranges.addWidget(self.range_status, 1)
        for label, callback in [("适配当前波形并锁定", self.fit_ranges_once),
                                ("手动量程", self.edit_ranges),
                                ("恢复默认量程", self.reset_ranges)]:
            button = W.QPushButton(label)
            button.clicked.connect(callback)
            ranges.addWidget(button)
        right.addLayout(ranges)
        pg.setConfigOptions(antialias=True)
        self.curves = []
        self.plots = []
        for name, color in [("原始 ADC · counts", "#477db5"), ("带通 · counts", "#25885e"),
                            ("活动包络 · counts", "#b88525")]:
            plot = pg.PlotWidget(background="w", title=name)
            plot.setMinimumHeight(100)
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.setLabel("bottom", "相对最近样本", units="s")
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            self.curves.append(plot.plot(pen=pg.mkPen(color, width=1.3)))
            self.plots.append(plot)
            right.addWidget(plot, 1)
        self.reset_ranges()
        self.path_label = W.QLabel("保存位置：upper_computer / recordings（模拟数据另存 simulated）")
        self.path_label.setWordWrap(True)
        right.addWidget(self.path_label)
        self.open_folder_button = W.QPushButton("打开本次数据文件夹")
        self.open_folder_button.clicked.connect(self.open_folder)
        right.addWidget(self.open_folder_button)
        self.enable_controls()

    def update_words_summary(self):
        self.words_label.setText("本次词表：" + " · ".join(self.selected_words)
            + f"\n共 {len(self.selected_words)} 词 × {self.reps.value()} 次 = {len(self.selected_words)*self.reps.value()} 试次；轮内随机")

    def edit_vocabulary(self):
        if self.recording or self.stopping:
            return
        from .vocabulary_ui import VocabularyDialog
        dialog = VocabularyDialog(self.vocabulary, self.selected_words, self)
        if dialog.exec() != W.QDialog.DialogCode.Accepted:
            return
        words, selected = dialog.values()
        try:
            save_vocabulary(ROOT / "vocabulary.json", words, selected)
        except (OSError, ValueError) as exc:
            self.banner.setText("词表保存失败：" + str(exc)); return
        self.vocabulary, self.selected_words = words, selected
        self.vocabulary_error = None
        self.update_words_summary()
        if not self.engine:
            self.word_label.setText(" · ".join(selected))
        self.banner.setText("词表已保存，下次启动会保留；新录制使用本次勾选的词。")

    def apply_ranges(self, limits):
        self.y_limits = limits
        for plot, (low, high) in zip(self.plots, limits):
            plot.enableAutoRange(x=False, y=False)
            plot.setXRange(-5, 0, padding=0)
            plot.setYRange(low, high, padding=0)
        self.range_status.setText("纵轴固定 · 不随波形自动缩放")

    def reset_ranges(self):
        self.apply_ranges([(0, 65535), (-10000, 10000), (0, 5000)])

    def fit_ranges_once(self):
        fs = 100 if self.worker and self.worker.mode == "legacy" else 1000
        limits = []
        for i, data in enumerate((self.raw, self.filtered, self.envelope)):
            values = np.asarray(list(data)[-5*fs:])
            if not len(values):
                limits.append(self.y_limits[i])
                continue
            if i == 0:
                pad = max(float(np.ptp(values))*.15, 100)
                limits.append((max(0, float(values.min())-pad), min(65535, float(values.max())+pad)))
            elif i == 1:
                bound = max(float(np.max(np.abs(values)))*1.15, 100)
                limits.append((-bound, bound))
            else:
                limits.append((0, max(float(values.max())*1.15, 100)))
        self.apply_ranges(limits)

    def edit_ranges(self):
        dialog = W.QDialog(self)
        dialog.setWindowTitle("固定纵轴量程 · ADC counts")
        layout = W.QFormLayout(dialog)
        fields = []
        for name, pair in zip(("原始 ADC", "滤波波形", "活动包络"), self.y_limits):
            row = W.QHBoxLayout()
            spins = []
            for label, value in zip(("下限", "上限"), pair):
                row.addWidget(W.QLabel(label))
                spin = W.QDoubleSpinBox()
                spin.setRange(-1000000, 1000000)
                spin.setDecimals(1)
                spin.setValue(value)
                row.addWidget(spin)
                spins.append(spin)
            layout.addRow(name, row)
            fields.append(spins)
        error = W.QLabel("")
        layout.addRow(error)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        def accept():
            limits = [(low.value(), high.value()) for low, high in fields]
            if any(low >= high for low, high in limits):
                error.setText("每条图的下限必须小于上限。")
                return
            self.apply_ranges(limits)
            dialog.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        dialog.exec()

    def refresh_ports(self):
        old = self.port.currentText().split(" ")[0]
        self.port.clear()
        for p in list_ports.comports():
            self.port.addItem(f"{p.device}  {p.description}", p.device)
        for i in range(self.port.count()):
            if self.port.itemData(i) == (old or "COM4"):
                self.port.setCurrentIndex(i)
                break
        self.port.lineEdit().setCursorPosition(0)

    def send(self, kind, **kwargs):
        if self.worker and self.worker.is_alive():
            try:
                self.worker.command(kind, **kwargs)
            except queue.Full:
                self.worker.abort_reason = "控制队列已满，不能保证事件完整性，停止当前批次"
                if self.engine:
                    self.engine.phase = "stopped"
                self.banner.setText("控制队列已满；立即停止操作并检查数据文件。")
                self.banner.setStyleSheet("background:#fce4df;color:#962e1e;padding:8px")

    def toggle_connection(self):
        if self.worker and self.worker.is_alive():
            if not self.recording and not self.stopping:
                self.send("disconnect")
            return
        mode = self.mode.currentData()
        port = self.port.currentText().split(" ")[0]
        self.raw.clear(); self.filtered.clear(); self.envelope.clear(); self.env_ring.clear()
        self.zi.fill(0)
        self.rhythm.reset()
        self.rhythm_context = None
        self.plots[1].setTitle("板端带通 · counts" if mode == "legacy" else "20–400 Hz 带通 + 50 Hz 陷波 · counts")
        self.plots[2].setTitle("板端活动包络 · counts" if mode == "legacy" else "陷波后活动包络 · counts")
        self.worker = Acquisition(mode, port, ROOT)
        self.mode.setEnabled(False)
        self.port.setEnabled(False)
        self.connect_button.setText("断开连接")
        self.connection_status.setText("正在连接，等待有效数据…")
        self.banner.setStyleSheet("padding:8px;background:#e4eee8;border-radius:6px;color:#24513e")
        self.banner.setText({"legacy": "旧固件联调：约 100 行/秒；无设备序号，不能按接收时刻精确贴标签。",
                             "binary": "完整采样模式：设备块边界标记约 20 ms 量化；屏幕实际呈现时延尚未硬件验证。",
                             "simulation": "模拟演练：所有波形为合成信号，文件单独保存，不能用于真实实验结论。"}[mode])
        self.worker.start()

    def config(self):
        seed = self.seed.value()
        return dict(mode=self.worker.mode, port=self.worker.port, subject_id=self.subject.text().strip(),
                    session_id=self.session_id.value(), repetitions=self.reps.value(), random_seed=seed,
                    electrode_sites=self.site.text().strip(), notes=self.notes.text().strip(),
                    words=list(self.selected_words), trial_order=schedule(self.reps.value(), seed, self.selected_words),
                    durations={k: v.value() for k, v in self.durations.items()},
                    sampling_rate_hz=100 if self.worker.mode == "legacy" else 1000,
                    display_filter=display_filter_metadata(self.worker.mode),
                    quiet_rhythm_indicator=indicator_metadata())

    def start_recording(self):
        if self.vocabulary_error:
            self.banner.setText("请先在“管理词表”中确认并保存词表，再开始采集。")
            return
        if self.lda_window and self.lda_window.busy:
            self.banner.setText("LDA 正在处理数据，请等待完成后开始采集。")
            return
        if not self.worker or not self.worker.ready or self.recording or self.stopping:
            return
        config = self.config()
        if not config["subject_id"] or (self.worker.mode != "simulation" and not config["electrode_sites"]):
            self.banner.setText("请填写受试者编号和电极位置后开始；模拟模式可省略电极位置。")
            return
        self.event_id = 0
        self.engine = TrialEngine(config["trial_order"], config["durations"], self.record_event)
        self.recording = True
        self.config_box.setEnabled(False)
        self.start_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.send("start", config=config)
        self.enable_controls()

    def record_event(self, event):
        self.event_id += 1
        event["event_id"] = f"event_{self.event_id}"
        if event["kind"] == "phase":
            self.update_cue()
            # Submission time is explicit; it is not a photon/display measurement.
            event["gui_submit_ns"] = time.perf_counter_ns()
        self.send("event", event=event)

    def update_cue(self):
        if not self.engine:
            return
        phase = self.engine.phase
        current = self.engine.current
        word = current["word"] if current and phase in ("prepare", "action") else {
            "baseline": "放松", "rest": "放松", "paused": "已暂停", "finished": "完成", "stopped": "已结束"}.get(phase, "准备")
        self.phase_label.setText(PHASES[phase])
        self.word_label.setText(word)
        self.cue_window.phase.setText(PHASES[phase])
        self.cue_window.word.setText(word)
        self.progress.setMaximum(len(self.engine.trials))
        self.progress.setValue(self.engine.completed)
        self.progress.setFormat(f"%v / %m 试次（包含作废和补录）")

    def pause(self):
        if not self.engine or self.stopping:
            return
        now = time.perf_counter_ns()
        if self.engine.phase == "paused":
            self.engine.resume(now)
            self.pause_button.setText("暂停")
        else:
            self.engine.pause(now)
            self.pause_button.setText("继续")

    def reject(self):
        if self.engine and not self.stopping:
            self.engine.reject(time.perf_counter_ns())
            self.update_cue()

    def stop_recording(self):
        if self.recording and not self.stopping:
            self.engine.stop(time.perf_counter_ns())
            self.stopping = True
            self.send("stop", status="stopped")
            self.enable_controls()

    def enable_controls(self):
        for button in (self.pause_button, self.reject_button, self.stop_button):
            button.setEnabled(self.recording and not self.stopping)
        self.connect_button.setEnabled(not self.recording and not self.stopping)

    def poll(self):
        self.sync_rhythm_context()
        if self.worker:
            while True:
                try:
                    message = self.worker.messages.get_nowait()
                except queue.Empty:
                    break
                kind = message["kind"]
                if kind == "ready":
                    self.start_button.setEnabled(not self.recording and not self.stopping)
                    self.connection_status.setText("数据就绪 · 115200 / 8N1")
                    if not self.recording:
                        self.countdown.setText("填写受试者和电极位置，检查信号后开始采集")
                elif kind == "recording":
                    self.current_path = message["path"]
                    self.path_label.setText("正在保存：" + self.current_path)
                    self.engine.start(time.perf_counter_ns())
                elif kind == "saved":
                    self.current_path = message["path"]
                    self.path_label.setText(f"已保存 [{message['status']}]：{self.current_path}")
                    self.recording = self.stopping = False
                    self.config_box.setEnabled(True)
                    self.start_button.setEnabled(self.worker.ready and self.worker.is_alive())
                    self.pause_button.setText("暂停")
                    self.enable_controls()
                elif kind == "error":
                    self.banner.setText("采集已停止：" + message["error"])
                    self.banner.setStyleSheet("background:#fce4df;color:#962e1e;padding:8px")
                    if self.engine:
                        self.engine.phase = "stopped"
                        self.update_cue()
                    self.recording = self.stopping = False
                    self.config_box.setEnabled(True)
                    self.enable_controls()
                elif kind == "disconnected":
                    self.mode.setEnabled(True); self.port.setEnabled(True)
                    self.connect_button.setText("连接 / 检查数据")
                    self.start_button.setEnabled(False)
                    self.connection_status.setText("未连接 · 115200 / 8N1")
                elif kind == "stats":
                    s = message["stats"]
                    self.stats_label.setText(f"有效接收 {s['rate_hz']:.0f} 点/s    总接收 {s['samples']:,}    缺失 {s['missing']}    "
                        f"CRC {s['crc_errors']}    标记 RTT {s['marker_rtt_ms'] if s['marker_rtt_ms'] is not None else '—'} ms")
            while True:
                try:
                    preview = self.worker.preview.get_nowait()
                except queue.Empty:
                    break
                rows = preview["rows"]
                if not rows:
                    continue
                raw = np.array([v[0] for v in rows], dtype=float)
                if self.rhythm_context in ("idle", "baseline", "paused") and preview["rx_ns"] >= self.rhythm_after_ns:
                    self.rhythm.feed(raw, preview["first"], preview["rx_ns"], time.perf_counter_ns())
                if self.worker.mode == "legacy":
                    filt = [v[1] for v in rows]
                    env = [v[2] for v in rows]
                else:
                    filt, self.zi = sosfilt(self.sos, raw - 32768, zi=self.zi)
                    env = []
                    for value in filt:
                        self.env_ring.append(abs(value))
                        env.append(sum(self.env_ring) / len(self.env_ring))
                self.raw.extend(raw); self.filtered.extend(filt); self.envelope.extend(env)
        if self.recording and not self.stopping and self.engine:
            now = time.perf_counter_ns()
            self.engine.tick(now)
            self.update_cue()
            remaining = max(0, (self.engine.deadline - now) / 1e9)
            text = "保持闭唇，内部尝试发音一次，不出声"
            if self.engine.phase not in ("paused", "finished", "stopped"):
                text = f"{remaining:.1f} s  ·  " + text
            self.countdown.setText(text)
            self.cue_window.time.setText(f"{remaining:.1f} s" if self.engine.phase not in ("paused", "finished", "stopped") else "")
            if self.engine.phase == "finished":
                self.stopping = True
                self.send("stop", status="completed")
                self.enable_controls()
        self.update_rhythm_indicator()
        self.plot_tick += 1
        if self.plot_tick % 2 == 0 and self.raw:
            fs = 100 if self.worker and self.worker.mode == "legacy" else 1000
            raw = np.asarray(list(self.raw)[-5 * fs:])
            saturated = np.mean((raw <= 1) | (raw >= 65534)) * 100
            self.quality_label.setText(f"近窗检查：峰峰值 {np.ptp(raw):.0f} counts · 接近满量程 {saturated:.1f}%"
                                       + (" · 检查接触/增益" if saturated > 1 else ""))
            recent = np.asarray(list(self.filtered)[-fs:])
            if len(recent):
                self.quality_label.setText(self.quality_label.text()
                    + f" · 近1秒滤波 RMS {np.sqrt(np.mean(recent**2)):.0f} counts（非合格判定）")
            outside = []
            for i, (curve, data) in enumerate(zip(self.curves, (self.raw, self.filtered, self.envelope))):
                y = np.asarray(list(data)[-5 * fs:])
                curve.setData((np.arange(len(y)) - len(y) + 1) / fs, y)
                low, high = self.y_limits[i]
                if len(y) and (y.min() < low or y.max() > high):
                    outside.append(("原始", "滤波", "包络")[i])
            self.range_status.setText("纵轴固定" + (" · " + "、".join(outside) + "超出显示范围，请调整量程" if outside else " · 不随波形自动缩放"))
            self.range_status.setStyleSheet("color:#a34c14" if outside else "color:#153330")

    def sync_rhythm_context(self):
        if not self.worker or not self.worker.is_alive() or not self.worker.ready:
            context = "disconnected"
        elif self.worker.mode != "binary":
            context = self.worker.mode
        elif self.recording or self.stopping:
            context = self.engine.phase if self.engine else "starting"
        else:
            context = "idle"
        if context != self.rhythm_context:
            self.rhythm.reset()
            self.rhythm_context = context
            # Drain previews captured before this quiet phase (including an
            # extra half second after movement); never concatenate short rests.
            self.rhythm_after_ns = time.perf_counter_ns()+500_000_000

    def update_rhythm_indicator(self):
        self.sync_rhythm_context()
        context = self.rhythm_context
        color, background = "#52645d", "#edf1f0"
        if context == "disconnected":
            text = "静息节律：等待完整采样设备连接"
        elif context in ("legacy", "simulation"):
            text = "静息节律：仅真实 1 kHz 完整采样模式启用（当前不判断）"
        elif context not in ("idle", "baseline", "paused"):
            text = "静息节律：提词流程中暂停判断；可在开始前或暂停后静息检查"
        else:
            result = self.rhythm.evaluate(time.perf_counter_ns())
            text = "静息节律："+result.detail
            if result.status == "detected":
                color, background = "#20583f", "#e3efe7"
                text += f" · 约 {result.rate:.0f} 次/分 · 脉冲峰峰值 {result.amplitude:.0f} counts（5–30 Hz）"
            elif result.status == "artifact":
                color, background = "#934714", "#fff0dc"
        self.rhythm_label.setText(text+"\n辅助观察，不代表接线或肌电质量合格。")
        self.rhythm_label.setStyleSheet(f"padding:6px;background:{background};color:{color};border-radius:4px")

    def open_folder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.current_path or ROOT)))

    def open_lda(self):
        if self.recording or self.stopping:
            self.banner.setText("请先结束并保存当前采集，再打开 LDA 训练与测试。")
            return
        if self.lda_window is None:
            from .lda_ui import LDAWindow
            self.lda_window = LDAWindow(ROOT, lambda: self.recording or self.stopping, self)
        self.lda_window.show()
        self.lda_window.raise_()

    def closeEvent(self, event):
        if self.lda_window and self.lda_window.busy:
            self.banner.setText("LDA 正在处理，请等待完成后关闭程序。")
            event.ignore()
            return
        if self.recording or self.stopping:
            self.stop_recording()
            self.banner.setText("正在结束并保存本批次；保存完成后可关闭窗口。")
            event.ignore()
            return
        if self.worker and self.worker.is_alive():
            self.send("disconnect")
            self.worker.join(timeout=1)
            if self.worker.is_alive():
                event.ignore()
                return
        self.cue_window.close()
        event.accept()


def main():
    app = W.QApplication(sys.argv)
    configure_fonts(app)
    app.setApplicationName("Closed Lip sEMG")
    window = MainWindow()
    if "--demo" in sys.argv:
        window.mode.setCurrentIndex(2)
    window.show()
    return app.exec()
