"""Exercise requested time-domain features and frozen settings in Qt."""
import os,sys,time,uuid
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtCore,QtWidgets as W
from semg.app import configure_fonts
from semg.lda_ui import LDAWindow
from semg import lda
from test_vocabulary import make_recording,TEN
OUT=ROOT/'tests/output'/('lda_features_gui_'+uuid.uuid4().hex);OUT.mkdir(parents=True)
(OUT/'recordings').mkdir();source=make_recording(OUT/'recordings/synthetic',TEN)
app=W.QApplication([]);configure_fonts(app);window=LDAWindow(OUT);window.resize(1250,950);window.show()
def wait():
    deadline=time.monotonic()+30
    while window.busy:
        app.processEvents();time.sleep(.01)
        if time.monotonic()>deadline:raise AssertionError(window.status.text())
wait();window.train_picker.set_all(True);window.percent.setValue(50)
window.strategy.setCurrentIndex(window.strategy.findData('within_recording'))
assert '每个有效批次' in window.split_note.text()
assert '8维' in window.feature_summary.text()
window.segments.setValue(6)
for key,check in window.feature_checks.items():check.setChecked(key in ['rms','mav','zcr_zero','waveform_length'])
assert '24维' in window.feature_summary.text()
app.processEvents();window.grab().save(str(OUT/'feature_controls.png'))
window.train_button.click();wait();assert window.last_result,window.status.text()
training=window.last_result;assert len(lda.load_model(training['model_path'])['scaler_mean'])==24
assert training['report']['strategy']=='within_recording'
assert '受试者平均准确率' in window.details.toPlainText()
test_ids={r['trial_id'] for r in training['report']['test']}
window.use_model.click();wait()
# Change the training tab; the loaded model must still use frozen 6x4 settings.
window.segments.setValue(1)
for key,check in window.feature_checks.items():check.setChecked(key=='log_rms')
assert '24维' in window.model_info.text()
for _,_,leaves,_ in window.test_picker.record_items:
    for leaf,row in leaves:leaf.setCheckState(0,QtCore.Qt.CheckState.Checked if row['trial_id'] in test_ids else QtCore.Qt.CheckState.Unchecked)
window.test_button.click();wait()
assert window.last_result['report']['metrics']==training['report']['metrics']
assert window.last_result['report']['feature_config']['segments']==6
window.tabs.setCurrentIndex(1);app.processEvents();window.grab().save(str(OUT/'model_features.png'))
for check in window.feature_checks.values():check.setChecked(False)
assert not window.train_button.isEnabled()
# Real old v1 model remains loadable; do not evaluate/rewrite its training source.
old=ROOT/'models/train_20260924_150000_94777c/model.json'
if old.exists():
    window.model_path.setText(str(old));window.inspect_model();assert '8维' in window.model_info.text()
window.close();app.processEvents();print('PASS',OUT)
