"""Real Qt end-to-end offline training, model load, holdout test and overlap error."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtCore,QtWidgets as W
from semg.app import MainWindow,configure_fonts
from semg.lda_ui import LDAWindow

app=W.QApplication([]);configure_fonts(app)
window=LDAWindow(ROOT);window.show()

def until(predicate,seconds=40):
    end=time.monotonic()+seconds
    while not predicate():
        app.processEvents();time.sleep(.01)
        if time.monotonic()>end:raise AssertionError(window.status.text())

until(lambda:not window.busy)
source=ROOT/'recordings/S05/session_001/20260924_123453_5ddcee'
window.train_picker.path.setText(str(source));window.train_picker.scan.click();until(lambda:not window.busy)
window.train_picker.set_all(True)
assert '1 人' in window.train_picker.summary.text() and '40 次' in window.train_picker.summary.text()
# Keep smoke artifacts away from the user's production model directory.
window.root=ROOT/'tests/output/lda_gui'
window.train_button.click();until(lambda:not window.busy)
assert window.last_result,window.status.text()
training=window.last_result
assert len(training['report']['train'])==32 and len(training['report']['test'])==8
model=Path(training['model_path']);assert model.exists()
out=ROOT/'tests/output';window.grab().save(str(out/'lda_training_result.png'))
window.use_model.click();until(lambda:not window.busy)
window.test_picker.path.setText(str(source));window.test_picker.scan.click();until(lambda:not window.busy)
test_ids={r['trial_id'] for r in training['report']['test']}
for _,_,leaves,_ in window.test_picker.record_items:
    for leaf,trial in leaves:
        leaf.setCheckState(0,QtCore.Qt.CheckState.Checked if trial['trial_id'] in test_ids else QtCore.Qt.CheckState.Unchecked)
window.test_button.click();until(lambda:not window.busy)
assert window.last_result['report']['metrics']==training['report']['metrics']
window.test_picker.set_all(True);window.test_button.click();until(lambda:not window.busy)
assert '已用于训练' in window.status.text(),window.status.text()
window.tabs.setCurrentIndex(0);app.processEvents();window.grab().save(str(out/'lda_data_picker.png'))
window.close()
main=MainWindow();main.timer.stop();main.open_lda();until(lambda:not main.lda_window.busy)
assert main.lda_window.isVisible();main.lda_window.close();main.close()
(out/'lda_gui_result.json').write_text(json.dumps(dict(status='PASS',model=str(model),metrics=training['report']['metrics'],checks=['subject/trial selection','background train','JSON reload','holdout prediction parity','overlap error','main app entry']),ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: Qt train / reload / test / overlap / app entry')
