"""Offscreen UI check with real serial data; does not impersonate a human trial."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtWidgets as W
from semg.app import MainWindow,configure_fonts
app=W.QApplication([]);configure_fonts(app);window=MainWindow();window.open_board();b=window.board_window
b.ports.setCurrentText('COM7');b.toggle_connection()
start=time.monotonic()
while time.monotonic()-start<6:app.processEvents();time.sleep(.01)
assert b.matches() and len(b.rows)>20 and b.start.isEnabled()
assert b.info.get('env_hz')==100 and len(b.envelope_points)>400
assert b.envelope_plot.viewRange()[1]==[0,500]
trial={};b.marker(trial,'action_idx')
start=time.monotonic()
while time.monotonic()-start<.5:app.processEvents();time.sleep(.01)
assert 'action_idx' in trial and b.sequence_gaps==0 and b.decoder.crc_errors==0
b.grab().save(str(ROOT/'board/ui_live_verified.png'))
print('UI live packets:',len(b.rows),'cue ACK:',trial,'max inference us:',max(r['inference_us'] for r in b.rows))
print('Board envelope points:',len(b.envelope_points),'last:',b.envelope_points[-1])
b.disconnect();window.close();app.processEvents()
