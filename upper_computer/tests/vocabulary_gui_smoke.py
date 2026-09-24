"""Offscreen Qt checks; all new settings, recordings and models use a unique test directory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6 import QtCore, QtTest, QtWidgets as W
import semg.app as ui
from semg import lda
from semg.vocabulary import load_vocabulary
from semg.vocabulary_ui import VocabularyDialog
from export_trials import export
from test_vocabulary import TEN, make_recording


def input_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ('recordings', 'models') for p in (ROOT/name).rglob('*') if p.is_file()}


before = input_hashes()
(ROOT/'tests/output').mkdir(parents=True, exist_ok=True)
OUT = Path(tempfile.mkdtemp(prefix='vocabulary_', dir=ROOT/'tests/output'))
ui.ROOT = OUT
app = W.QApplication([]); ui.configure_fonts(app)
main = ui.MainWindow(); main.show()


def until(predicate, seconds=40):
    end = time.monotonic()+seconds
    while not predicate():
        app.processEvents(); time.sleep(.01)
        if time.monotonic() > end:
            raise AssertionError(main.banner.text())


dialog_errors = []
def edit_dialog():
    try:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, VocabularyDialog)
        for word in TEN[4:]:
            dialog.entry.setText(word)
            if word == 'stop':
                QtTest.QTest.keyClick(dialog.entry, QtCore.Qt.Key.Key_Return)
            else:
                dialog.add_button.click()
        assert dialog.values()[0] == TEN
        dialog.entry.setText('STOP'); dialog.add_button.click()
        assert '重复' in dialog.error.text()
        dialog.entry.clear()
        dialog.items.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert 'open' not in dialog.values()[1]
        dialog.select_all()
        dialog.grab().save(str(OUT/'vocabulary_editor.png'))
        dialog.accept()
    except Exception as exc:
        dialog_errors.append(str(exc))
        if app.activeModalWidget(): app.activeModalWidget().reject()


QtCore.QTimer.singleShot(50, edit_dialog)
main.words_button.click()
assert not dialog_errors, dialog_errors
assert main.selected_words == TEN
assert load_vocabulary(OUT/'vocabulary.json') == (TEN, TEN)
main.close()
# A fresh application window restores the persisted vocabulary.
main = ui.MainWindow(); main.show()
assert main.selected_words == TEN
main.grab().save(str(OUT/'main_ten_words.png'))
main.mode.setCurrentIndex(2); main.reps.setValue(1)
for spin in main.durations.values(): spin.setValue(.2)
main.toggle_connection(); until(lambda: main.worker.ready)
main.start_recording(); until(lambda: main.engine.phase == 'action')
assert not main.words_button.isEnabled()
main.reject(); main.pause()
assert main.engine.phase == 'paused'
main.pause()
until(lambda: not main.recording and main.current_path is not None)
folder = Path(main.current_path)
meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
trials = lda.lines((folder/'trials.jsonl').read_bytes())
assert meta['words'] == TEN and len(meta['label_map']) == 11
assert meta['status'] == 'completed' and len(trials) == 11
assert {t['word'] for t in trials if t['status'] == 'completed'} == set(TEN)
assert sum(t['exported'] for t in export(folder, allow_simulated=True)) == 10
assert main.words_button.isEnabled()
main.toggle_connection(); until(lambda: not main.worker.is_alive())

source = make_recording(OUT/'synthetic_ten', TEN)
main.open_lda(); window = main.lda_window
until(lambda: not window.busy)
window.train_picker.path.setText(str(source)); window.train_picker.scan.click()
until(lambda: not window.busy)
window.train_picker.set_all(True); window.percent.setValue(50)
assert '40 次' in window.train_picker.summary.text()
window.train_button.click(); until(lambda: not window.busy)
assert window.last_result, window.status.text()
assert len(window.last_result['report']['classes']) == 10
ten_model = window.last_result['model_path']
assert window.matrix.rowCount() == 10 and window.matrix.columnCount() == 10
window.grab().save(str(OUT/'ten_word_result.png'))

# Load the user's existing v1 model without modifying it or retraining it.
old_model = ROOT/'models/train_20260924_150000_94777c/model.json'
has_real_model = old_model.exists()
if not has_real_model:
    four = make_recording(OUT/'synthetic_four', list(TEN[:4]), seed=456)
    baseline = lda.train([dict(path=str(four))], OUT/'models', train_percent=50)
    legacy = lda.load_model(baseline['model_path']); legacy['schema'] = 'semg_lda_v1'
    old_model = OUT/'synthetic_v1.json'; lda.write_json(old_model, legacy)
window.model_path.setText(str(old_model)); window.inspect_model()
assert '模型有效' in window.model_info.text()
window.test_picker.path.setText(str(source)); window.test_picker.scan.click()
until(lambda: not window.busy)
window.test_picker.set_all(True); window.test_button.click(); until(lambda: not window.busy)
m = window.last_result['report']['metrics']
assert m['unknown_total'] == 24 and m['known_total'] == 16
assert window.matrix.rowCount() == 10 and window.matrix.columnCount() == 4
window.grab().save(str(OUT/'four_word_model_ten_word_test.png'))

real = ROOT/'recordings/S05/session_001/20260924_123453_5ddcee'
real_metrics = None
if has_real_model and real.exists():
    window.test_picker.path.setText(str(real)); window.test_picker.scan.click(); until(lambda: not window.busy)
    held_out = {5,6,7,8,17,18,19,20}
    for _,_,leaves,_ in window.test_picker.record_items:
        for leaf, trial in leaves:
            leaf.setCheckState(0, QtCore.Qt.CheckState.Checked if trial['trial_id'] in held_out else QtCore.Qt.CheckState.Unchecked)
    window.test_button.click(); until(lambda: not window.busy)
    real_metrics = window.last_result['report']['metrics']
    assert real_metrics['correct'] == 7 and real_metrics['total'] == 8 and real_metrics['unknown_total'] == 0
else:
    print('SKIP: original 7/8 check needs the separately shared S05 recording and model')
# Overlap protection is also exercised when no private recordings are present.
window.model_path.setText(ten_model); window.inspect_model()
window.test_picker.path.setText(str(source)); window.test_picker.scan.click(); until(lambda: not window.busy)
window.test_picker.set_all(True); window.test_button.click(); until(lambda: not window.busy)
assert '已用于训练' in window.status.text()
window.close(); main.close()
assert input_hashes() == before, 'Production recording/model files changed'
(OUT/'result.json').write_text(json.dumps(dict(status='PASS', old_model_metrics=real_metrics,
    inputs_unchanged=True, checks=['edit/persist/reload vocabulary','ten-word simulated capture',
    'pause/repeat/export','ten-class Qt training','v1 model with untrained words',
    'training overlap guard'], original_holdout_check='PASS' if real_metrics else 'SKIPPED: private data absent'), ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS: ' + str(OUT))
