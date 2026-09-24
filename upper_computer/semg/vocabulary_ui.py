"""Edit local vocabulary and choose the words for the next recording."""
from PySide6 import QtCore, QtWidgets as W
from .vocabulary import validate_words
from .lda_ui import apply_lda_theme


class VocabularyDialog(W.QDialog):
    def __init__(self, words, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle("管理采集词表")
        self.resize(520, 560)
        layout = W.QVBoxLayout(self)
        note = W.QLabel("添加词语后勾选本次要采集的词；每轮每个勾选词各一次。\n修改仅影响之后的新录制，已有数据和模型保留原词表。")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = W.QHBoxLayout()
        self.entry = W.QLineEdit()
        self.entry.setPlaceholderText("输入新词，例如 stop")
        self.add_button = W.QPushButton("添加词")
        row.addWidget(self.entry, 1); row.addWidget(self.add_button)
        layout.addLayout(row)
        self.items = W.QListWidget()
        layout.addWidget(self.items, 1)
        for word in words:
            self.add_item(word, word in selected)
        row = W.QHBoxLayout()
        remove = W.QPushButton("移除选中行")
        select = W.QPushButton("全部勾选")
        row.addWidget(remove); row.addWidget(select)
        layout.addLayout(row)
        self.error = W.QLabel("")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(W.QDialogButtonBox.StandardButton.Save).setText("保存词表")
        buttons.button(W.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        layout.addWidget(buttons)
        self.add_button.clicked.connect(self.add_word)
        remove.clicked.connect(lambda: self.items.takeItem(self.items.currentRow()) if self.items.currentRow() >= 0 else None)
        select.clicked.connect(self.select_all)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.add_button.setDefault(True)
        apply_lda_theme(self)

    def add_item(self, word, checked):
        item = W.QListWidgetItem(word, self.items)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def values(self):
        items = [self.items.item(i) for i in range(self.items.count())]
        return ([i.text() for i in items],
                [i.text() for i in items if i.checkState() == QtCore.Qt.CheckState.Checked])

    def add_word(self):
        word = self.entry.text().strip().casefold()
        try:
            validate_words(self.values()[0] + [word])
        except ValueError as exc:
            self.error.setText(str(exc)); return
        self.add_item(word, True)
        self.entry.clear(); self.error.clear()

    def select_all(self):
        for i in range(self.items.count()):
            self.items.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def accept(self):
        try:
            words, selected = self.values()
            validate_words(words); validate_words(selected)
            if self.entry.text().strip():
                raise ValueError("输入框还有未添加的词，请先点击“添加词”或清空输入框")
        except ValueError as exc:
            self.error.setText(str(exc)); return
        super().accept()
