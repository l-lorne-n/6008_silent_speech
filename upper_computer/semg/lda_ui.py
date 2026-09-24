"""Offline LDA workspace; all expensive I/O and fitting run off the GUI thread."""
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets as W

from . import lda


def apply_lda_theme(window):
    """Keep this dialog readable even when Windows supplies a dark palette."""
    palette = QtGui.QPalette()
    roles = {'Window':'#f3f5f4', 'WindowText':'#153330', 'Base':'#ffffff',
             'AlternateBase':'#eef3f0', 'Text':'#153330', 'Button':'#ffffff',
             'ButtonText':'#153330', 'Highlight':'#216647', 'HighlightedText':'#ffffff',
             'ToolTipBase':'#ffffff', 'ToolTipText':'#153330', 'PlaceholderText':'#52645d'}
    for group in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive,
                  QtGui.QPalette.ColorGroup.Disabled):
        for role, color in roles.items():
            palette.setColor(group, getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(color))
    for role in ('Text','WindowText','ButtonText'):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled,
                         getattr(QtGui.QPalette.ColorRole,role), QtGui.QColor('#59665f'))
    window.setPalette(palette)
    window.setAutoFillBackground(True)
    window.setStyleSheet("""
        QWidget {background:#f3f5f4;color:#153330;font-family:'Microsoft YaHei';font-size:14px;}
        QWidget:disabled {color:#59665f;}
        QLabel {background:transparent;}
        QTabWidget::pane {background:#f3f5f4;border:1px solid #bccdc3;}
        QTabBar::tab {background:#e4ebe7;color:#314d42;padding:9px 16px;border:1px solid #bccdc3;}
        QTabBar::tab:selected {background:#ffffff;color:#123e2a;border-bottom:3px solid #216647;}
        QTreeWidget,QTableWidget,QPlainTextEdit,QListWidget {background:white;alternate-background-color:#eef3f0;
            color:#153330;border:1px solid #bccdc3;selection-background-color:#216647;selection-color:white;}
        QTreeWidget::item,QTableWidget::item,QListWidget::item {padding:4px;}
        QTreeWidget::item:disabled,QTableWidget::item:disabled {color:#59665f;background:#edf1ee;}
        QTreeWidget::item:selected,QTableWidget::item:selected {background:#216647;color:white;}
        QHeaderView::section {background:#e4ebe7;color:#153330;border:1px solid #c6d3cb;padding:7px;font-weight:600;}
        QTableCornerButton::section {background:#e4ebe7;border:1px solid #c6d3cb;}
        QLineEdit,QComboBox,QSpinBox {background:white;color:#153330;border:1px solid #9eb5a8;
            border-radius:4px;padding:6px;selection-background-color:#216647;selection-color:white;}
        QLineEdit:disabled,QComboBox:disabled,QSpinBox:disabled {background:#edf1ee;color:#59665f;}
        QComboBox QAbstractItemView {background:white;color:#153330;selection-background-color:#216647;selection-color:white;}
        QPushButton {background:white;color:#173f34;border:1px solid #9eb5a8;border-radius:5px;padding:8px 12px;}
        QPushButton:hover {background:#e3efe7;}
        QPushButton:disabled {background:#edf1ee;color:#59665f;}
        QPushButton#primary {background:#186647;color:white;border:0;font-weight:bold;}
        QPushButton#primary:disabled {background:#d3dfd7;color:#43584a;}
        QToolTip {background:white;color:#153330;border:1px solid #9eb5a8;}
    """)
    # Native Windows dark controls can ignore some palette roles. Use Fusion
    # only inside this dialog, leaving the acquisition window/application alone.
    window._lda_style = W.QStyleFactory.create('Fusion')
    for widget in [window, *window.findChildren(W.QWidget)]:
        widget.setStyle(window._lda_style)
        widget.setPalette(palette)


class Job(QtCore.QThread):
    progress = QtCore.Signal(str)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.succeeded.emit(self.function(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class DataPicker(W.QWidget):
    changed = QtCore.Signal()

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.catalog = []
        self.record_items = []
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = W.QHBoxLayout()
        self.path = W.QLineEdit(str(root))
        self.browse = W.QPushButton("选择数据目录")
        self.scan = W.QPushButton("扫描 / 刷新")
        row.addWidget(self.path, 1);row.addWidget(self.browse);row.addWidget(self.scan)
        layout.addLayout(row)
        self.tree = W.QTreeWidget()
        self.tree.setHeaderLabels(["勾选受试者 / 录制 / 试次", "词语 / 动作条件", "次数 / 状态", "手动用途"])
        self.tree.setColumnWidth(0, 360); self.tree.setColumnWidth(1, 200)
        self.tree.setMinimumHeight(160)
        layout.addWidget(self.tree, 1)
        row = W.QHBoxLayout()
        self.select_all = W.QPushButton("全选可用")
        self.clear = W.QPushButton("清空选择")
        row.addWidget(self.select_all);row.addWidget(self.clear)
        self.summary = W.QLabel("未选择数据")
        row.addWidget(self.summary, 1)
        layout.addLayout(row)
        self.warning = W.QLabel("只列出完整试次；勾选不等于质量合格，训练时还会检查边界、连续性和削顶。")
        self.warning.setWordWrap(True);layout.addWidget(self.warning)
        self.browse.clicked.connect(self.choose)
        self.select_all.clicked.connect(lambda: self.set_all(True))
        self.clear.clicked.connect(lambda: self.set_all(False))
        self.tree.itemChanged.connect(self.update_summary)

    def choose(self):
        folder = W.QFileDialog.getExistingDirectory(self, "选择 recordings 根目录或单份录制目录", self.path.text())
        if folder:
            self.path.setText(folder)
            self.scan.click()

    def populate(self, result):
        self.catalog, errors = result
        self.tree.blockSignals(True);self.tree.clear();self.record_items=[]
        subjects = {}
        for item in self.catalog:
            subject = item['subject']
            if subject not in subjects:
                parent = W.QTreeWidgetItem(self.tree,[subject])
                parent.setFlags(parent.flags() | QtCore.Qt.ItemFlag.ItemIsAutoTristate | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                parent.setCheckState(0,QtCore.Qt.CheckState.Unchecked)
                subjects[subject]=parent
            record = W.QTreeWidgetItem(subjects[subject], [Path(item['path']).name, item['condition'], f"{len(item['trials'])} 次 · {item['status']}"])
            record.setToolTip(0,item['path']+'\n电极：'+item['site']+'\n备注：'+item['notes'])
            record.setFlags(record.flags() | QtCore.Qt.ItemFlag.ItemIsAutoTristate | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            record.setCheckState(0,QtCore.Qt.CheckState.Unchecked)
            role=W.QComboBox();role.addItem("训练",'train');role.addItem("测试",'test')
            role.currentIndexChanged.connect(self.changed)
            self.tree.setItemWidget(record,3,role)
            leaves=[]
            for trial in item['trials']:
                leaf=W.QTreeWidgetItem(record,[f"试次 {trial['trial_id']} · 第 {trial.get('block','?')} 轮",trial['word'],"完整"])
                leaf.setFlags(leaf.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0,QtCore.Qt.CheckState.Unchecked)
                leaves.append((leaf,trial))
            if item['disabled_reason'] or not leaves:
                record.setDisabled(True);record.setText(2,item['disabled_reason'] or '无完整试次')
                role.setEnabled(False)
            self.record_items.append((record,item,leaves,role))
        for parent in subjects.values(): parent.setExpanded(True)
        self.tree.blockSignals(False)
        self.warning.setText('扫描失败：'+'；'.join(errors[:3]) if errors else "勾选具体试次可调整每人的样本量。动作条件可能是历史默认值，请避免混入出声对照。")
        self.update_summary()

    def set_all(self, checked):
        self.tree.blockSignals(True)
        for record, _, leaves, _ in self.record_items:
            if not record.isDisabled():
                for leaf, _ in leaves:
                    leaf.setCheckState(0,QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        self.tree.blockSignals(False);self.update_summary()

    def selections(self):
        selected=[]
        for record,item,leaves,role in self.record_items:
            ids=[t['trial_id'] for leaf,t in leaves if leaf.checkState(0)==QtCore.Qt.CheckState.Checked]
            if ids and not record.isDisabled(): selected.append(dict(path=item['path'],trial_ids=ids,role=role.currentData()))
        return selected

    def update_summary(self, *_):
        counts={};subjects=set();recordings=0
        for record,item,leaves,_ in self.record_items:
            if record.isDisabled():continue
            rows=[t for leaf,t in leaves if leaf.checkState(0)==QtCore.Qt.CheckState.Checked]
            if rows: subjects.add(item['subject']);recordings+=1
            for t in rows: counts[t['word']]=counts.get(t['word'],0)+1
        self.summary.setText(f"已选 {len(subjects)} 人 · {recordings} 组 · {sum(counts.values())} 次\n"+' / '.join(f"{w} {n}" for w,n in counts.items()))
        self.changed.emit()


class LDAWindow(W.QDialog):
    def __init__(self, root, is_acquiring=lambda:False, parent=None):
        super().__init__(parent)
        self.root=Path(root);self.is_acquiring=is_acquiring
        self.busy=False;self.job=None;self.last_result=None
        self.setWindowTitle("LDA · 训练、评估与模型测试")
        area = self.screen().availableGeometry()
        self.resize(min(1250, area.width()-40), min(960, area.height()-60))
        layout=W.QVBoxLayout(self)
        intro=W.QLabel("选择完整试次 → 按组划分 → 一键训练 → 保存模型 → 加载模型测试其他数据")
        intro.setWordWrap(True)
        intro.setStyleSheet('font-size:18px;font-weight:700');layout.addWidget(intro)
        self.tabs=W.QTabWidget();layout.addWidget(self.tabs,1)
        training=W.QWidget();tl=W.QVBoxLayout(training)
        self.train_picker=DataPicker(self.root/'recordings');tl.addWidget(self.train_picker,1)
        self.strategy=W.QComboBox()
        for text,data in [("按轮次（同批次内部评估）","block"),("按整组录制（跨批次）","recording"),("按受试者（跨人）","subject"),("手动指定每组训练/测试用途","manual")]:self.strategy.addItem(text,data)
        self.percent=W.QSpinBox();self.percent.setRange(50,90);self.percent.setValue(80);self.percent.setSuffix('%')
        self.seed=W.QSpinBox();self.seed.setRange(0,2147483647);self.seed.setValue(20260924)
        self.variant=W.QComboBox();self.variant.addItem('20–400 Hz + 50 Hz 陷波','notch50');self.variant.addItem('20–400 Hz + 50/100/200 Hz 陷波','notch50_100_200')
        form=W.QGridLayout()
        for col,(label,widget) in enumerate([('划分单位',self.strategy),('训练占比',self.percent),('随机种子',self.seed)]):
            form.addWidget(W.QLabel(label),0,col);form.addWidget(widget,1,col)
        form.addWidget(W.QLabel('滤波方案（固定8个特征 + 自动收缩LDA）'),2,0);form.addWidget(self.variant,2,1,1,2)
        tl.addLayout(form)
        self.split_note=W.QLabel();self.split_note.setWordWrap(True);tl.addWidget(self.split_note)
        self.train_button=W.QPushButton('一键训练并评估');self.train_button.setObjectName('primary');tl.addWidget(self.train_button)
        self.tabs.addTab(training,'1  训练与评估')
        testing=W.QWidget();el=W.QVBoxLayout(testing)
        row=W.QHBoxLayout();self.model_path=W.QLineEdit();self.model_path.setPlaceholderText('选择本版导出的 model.json')
        self.load_button=W.QPushButton('加载模型');row.addWidget(self.model_path,1);row.addWidget(self.load_button);el.addLayout(row)
        self.model_info=W.QLabel('未加载模型；测试会使用模型保存的滤波与特征，不重新拟合。');self.model_info.setWordWrap(True);el.addWidget(self.model_info)
        self.test_picker=DataPicker(self.root/'recordings');self.test_picker.tree.setColumnHidden(3,True);el.addWidget(self.test_picker,1)
        self.test_button=W.QPushButton('用模型测试所选数据');self.test_button.setObjectName('primary');el.addWidget(self.test_button)
        self.tabs.addTab(testing,'2  加载模型与测试')
        results=W.QWidget();rl=W.QVBoxLayout(results)
        self.result_summary=W.QLabel('训练或测试完成后显示结果');self.result_summary.setWordWrap(True);rl.addWidget(self.result_summary)
        self.matrix=W.QTableWidget(0,0)
        self.matrix.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
        self.matrix.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.matrix.setMinimumHeight(180);rl.addWidget(W.QLabel('混淆矩阵：行是真实词语，列是预测词语'));rl.addWidget(self.matrix)
        self.details=W.QPlainTextEdit();self.details.setReadOnly(True);rl.addWidget(self.details,1)
        self.saved_model=W.QLineEdit();self.saved_model.setReadOnly(True);self.saved_model.setPlaceholderText('模型保存位置')
        rl.addWidget(self.saved_model)
        row=W.QHBoxLayout();self.open_result=W.QPushButton('打开结果文件夹');self.use_model=W.QPushButton('将此模型载入测试页')
        row.addWidget(self.open_result);row.addWidget(self.use_model);rl.addLayout(row)
        self.tabs.addTab(results,'3  结果与模型位置')
        self.status=W.QLabel('准备就绪');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.train_button.clicked.connect(self.train_selected);self.test_button.clicked.connect(self.test_selected)
        self.load_button.clicked.connect(self.choose_model)
        self.model_path.editingFinished.connect(self.inspect_model)
        self.strategy.currentIndexChanged.connect(self.update_split_note);self.percent.valueChanged.connect(self.update_split_note)
        self.open_result.clicked.connect(self.open_output);self.use_model.clicked.connect(self.use_saved_model)
        self.train_picker.scan.clicked.connect(lambda:self.scan_picker(self.train_picker))
        self.test_picker.scan.clicked.connect(lambda:self.scan_picker(self.test_picker))
        self.update_split_note()
        apply_lda_theme(self)
        self.scan_picker(self.train_picker)

    def update_split_note(self):
        manual=self.strategy.currentData()=='manual'
        self.percent.setEnabled(not manual);self.train_picker.tree.setColumnHidden(3,not manual)
        self.split_note.setText('在数据树每份录制的“手动用途”中指定训练或测试。' if manual else
            f"目标训练 {self.percent.value()}% / 测试 {100-self.percent.value()}%；完整组不拆开，实际比例以结果为准。")

    def launch(self,function,on_success):
        if self.busy:return
        if self.is_acquiring():
            self.status.setText('请先结束并保存当前采集，再运行离线分析。');return
        self.busy=True;self.tabs.setEnabled(False);self.status.setText('处理中…')
        job=Job(function,self);self.job=job
        job.progress.connect(self.status.setText)
        job.succeeded.connect(on_success)
        job.failed.connect(lambda message:self.status.setText('未完成：'+message))
        job.finished.connect(self.job_finished)
        job.start()

    def job_finished(self):
        self.busy=False;self.tabs.setEnabled(True)
        self.job.deleteLater();self.job=None

    def scan_picker(self,picker):
        path=picker.path.text()
        def done(result):
            picker.populate(result);self.status.setText(f'扫描完成：{len(result[0])} 份录制。默认不勾选数据。')
        def scan(progress):
            if Path(path).resolve() == (self.root/'recordings').resolve() and not Path(path).exists():
                return [], []  # A fresh clone has no recordings until the first capture.
            return lda.scan_recordings(path)
        self.launch(scan,done)

    def train_selected(self):
        selections=self.train_picker.selections()
        if not selections:self.status.setText('请先勾选训练/测试数据。');return
        config=dict(strategy=self.strategy.currentData(),train_percent=self.percent.value(),seed=self.seed.value(),variant=self.variant.currentData())
        self.launch(lambda progress:lda.train(selections,self.root/'models',progress=progress,**config),self.show_result)

    def test_selected(self):
        selections=self.test_picker.selections();path=self.model_path.text().strip()
        if not selections or not path:self.status.setText('请加载模型，并扫描、勾选需要测试的数据。');return
        self.launch(lambda progress:lda.evaluate(path,selections,self.root/'models',progress),self.show_result)

    def choose_model(self):
        file,_=W.QFileDialog.getOpenFileName(self,'选择 LDA 模型',str(self.root/'models'),'LDA 模型 (*.json)')
        if file:self.model_path.setText(file);self.inspect_model()

    def inspect_model(self):
        try:
            model=lda.load_model(self.model_path.text())
            self.model_info.setText(f"模型有效 · {len(model['classes'])}词：{', '.join(model['classes'])}\n{model['variant']} · 训练 {model['training_trials']} 次 · 受试者 {', '.join(model['training_subjects'])}\n可测试不同词表的录制；未训练词会被判成模型内某个词，报告单独标注。")
        except Exception as exc:self.model_info.setText('模型未就绪：'+str(exc))

    def show_result(self,result):
        self.last_result=result;report=result['report'];m=report['metrics']
        classes = report['classes']
        known = f"{m['known_accuracy']:.1%}" if m['known_accuracy'] is not None else '不适用'
        self.result_summary.setText(f"{report['scope']} · 模型 {len(classes)} 词\n全部准确率 {m['accuracy']:.1%}（{m['correct']}/{m['total']}） · 宏平均 F1 {m['macro_f1']:.3f}\n已知词 {m['known_total']} 次，准确率 {known}；未训练词 {m['unknown_total']} 次（无自动拒识）")
        self.matrix.setRowCount(len(m['row_labels']));self.matrix.setColumnCount(len(classes))
        self.matrix.horizontalHeader().setSectionResizeMode(
            W.QHeaderView.ResizeMode.Stretch if len(classes) <= 6 else W.QHeaderView.ResizeMode.ResizeToContents)
        self.matrix.setHorizontalHeaderLabels(classes)
        self.matrix.setVerticalHeaderLabels([w if w in classes else w+'（未训练）' for w in m['row_labels']])
        for i,row in enumerate(m['confusion_matrix']):
            for j,value in enumerate(row):
                item=W.QTableWidgetItem(str(value));item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if i==j:item.setBackground(QtGui.QColor('#e3efe7'))
                self.matrix.setItem(i,j,item)
        text=[]
        if 'train' in report:
            text += [f"实际训练 {len(report['train'])} 次 / 测试 {len(report['test'])} 次；训练占比 {report['actual_train_percent']:.1f}%"]
            for split in ('train','test'):
                for person in sorted({r['subject'] for r in report[split]}):
                    rows=[r for r in report[split] if r['subject']==person]
                    text.append(('训练' if split=='train' else '测试')+f" · {person} · "+' / '.join(f"{w} {sum(r['word']==w for r in rows)}" for w in m['row_labels']))
        else:
            for person in sorted({r['subject'] for r in report['test']}):
                rows=[r for r in report['test'] if r['subject']==person]
                text.append(f"测试 · {person} · "+' / '.join(f"{w} {sum(r['word']==w for r in rows)}" for w in m['row_labels']))
        text += ['','各词识别：']+[f"{w}: {s['correct']}/{s['count']}，召回率 {s['recall']:.1%}" for w,s in m['per_word'].items()]
        text += ['','注意：']+report['warnings']
        text += ['','排除试次：']+[f"{r['recording']} / {r['trial_id']}: {r['reason']}" for r in report['excluded']]
        text += ['','结果目录：'+result['folder'],'report.json 含完整划分与检查；predictions.csv 含逐试次预测。']
        self.details.setPlainText('\n'.join(text))
        self.saved_model.setText(result['model_path'] or report.get('model_path',''))
        self.use_model.setEnabled(bool(self.saved_model.text()))
        self.tabs.setCurrentIndex(2);self.status.setText('完成，结果已保存。')

    def use_saved_model(self):
        self.model_path.setText(self.saved_model.text());self.inspect_model();self.tabs.setCurrentIndex(1)
        if not self.test_picker.catalog:self.scan_picker(self.test_picker)

    def open_output(self):
        if self.last_result:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.last_result['folder']))

    def closeEvent(self,event):
        if self.busy:
            self.status.setText('正在处理，请等待完成后关闭；原始采集文件不会改动。');event.ignore()
        else:event.accept()
