"""Offscreen UI smoke test; no serial port, original models or recordings written."""
import json,os,sys,time,uuid
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtWidgets as W
from semg.streaming import load_model
import semg.app as ui

OUT=ROOT/'tests/output'/('activity_gui_'+uuid.uuid4().hex);OUT.mkdir(parents=True)
app=W.QApplication([]);ui.configure_fonts(app)
ui.ROOT=OUT
main=ui.MainWindow();main.show();main.open_streaming();window=main.streaming_window
assert '自然做口型' in main.cue_window.instruction.text()
window.source.setText(str(ROOT/'recordings/S05/session_006/20261010_114855_a28346'))
window.train.click()
deadline=time.monotonic()+60
while window.busy:
    app.processEvents();time.sleep(.01)
    if time.monotonic()>deadline:raise AssertionError('Training timeout')
assert window.model.text(),window.log.toPlainText()
m=load_model(window.model.text());assert len(m['classes'])==10
window.replay.click()
while window.busy:
    app.processEvents();time.sleep(.01)
    if time.monotonic()>deadline:raise AssertionError('Replay timeout')
window.start.click();assert not window.live  # requires a real connected device
window.message(dict(kind='recognition_started'));assert window.live
window.message(dict(kind='recognition_event',event=dict(kind='prediction',start=100,end=1100,accepted=True,word='open',candidate='open',probability=.8)))
assert window.output.text()=='open'
window.message(dict(kind='disconnected'));assert not window.live
main.grab().save(str(OUT/'main.png'));window.grab().save(str(OUT/'activity.png'))
main.close();app.processEvents()
print('OUT',OUT)
print('MODEL',window.model.text())
print(window.log.toPlainText().encode('ascii','backslashreplace').decode())
