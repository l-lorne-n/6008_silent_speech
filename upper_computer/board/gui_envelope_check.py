"""UI regression under an explicitly dark system palette, without a serial port."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys
import struct
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtWidgets as W,QtGui
from semg.app import MainWindow,configure_fonts
from semg.protocol import encode
from semg.board_protocol import BoardDecoder

app=W.QApplication([]);configure_fonts(app)
dark=QtGui.QPalette()
for name,color in {'Window':'#292929','Base':'#252525','AlternateBase':'#333333',
                   'Text':'#eeeeee','WindowText':'#eeeeee','Button':'#333333','ButtonText':'#eeeeee'}.items():
    dark.setColor(getattr(QtGui.QPalette.ColorRole,name),QtGui.QColor(color))
app.setPalette(dark)
window=MainWindow();window.open_board();b=window.board_window
b.log.appendPlainText('界面示例：在深色系统主题下，日志仍使用白底深色文字。')
b.table.setRowCount(2)
for i,row in enumerate([['760.00','previous','84.8%','68.00'],['765.40','start','80.3%','68.11']]):
    for j,text in enumerate(row):b.table.setItem(i,j,W.QTableWidgetItem(text))
b.table.selectRow(1)
for i in range(501):
    a=60+340*max(0,1-abs(i-300)/70);c=85+230*max(0,1-abs(i-330)/90)
    p=BoardDecoder().feed(encode(8,i,20*(i+1),struct.pack('<2f',a,c),count=1,version=2))[0]
    b.handle_packet(p)
b.refresh_envelope();app.processEvents()
assert b.envelope_plot.viewRange()[1]==[0,500]
assert len(b.envelope_curves[0].getData()[0])==501
assert b.table.palette().color(QtGui.QPalette.ColorRole.Base).lightness()>200
assert b.log.palette().color(QtGui.QPalette.ColorRole.Text).lightness()<100
b.grab().save(str(ROOT/'board/ui_envelope_dark_theme.png'))
b.resize(1000,700);app.processEvents()
assert b.centralWidget().verticalScrollBar().maximum()>0
b.clear_envelope('断流检查');assert not b.envelope_points
window.close();print('Dark-theme contrast, fixed 0-500 plot, two traces, scrolling and stale clearing verified.')
