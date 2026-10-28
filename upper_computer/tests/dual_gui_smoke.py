"""Real Qt + dual simulator + raw recording/export + two-channel LDA dialog."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PySide6 import QtWidgets as W
import semg.app as ui
from semg.lda_ui import LDAWindow
from semg import lda
from semg.protocol import Decoder, DATA
from export_trials import export
from test_dual_channel import dual_recording

OUT=ROOT/'tests/output'/('dual_gui_'+uuid.uuid4().hex);OUT.mkdir(parents=True)
ui.ROOT=OUT;app=W.QApplication([]);ui.configure_fonts(app)
window=ui.MainWindow();window.resize(1450,1000);window.show()
window.channel_checks[1].setChecked(True);window.mode.setCurrentIndex(2)
window.selected_words=['open','close'];window.update_words_summary();window.reps.setValue(1)
for k,s in window.durations.items():s.setValue(1.2 if k=='prepare' else .3)

def until(check,timeout=20):
    deadline=time.monotonic()+timeout
    while not check():
        app.processEvents();time.sleep(.01)
        if time.monotonic()>deadline:raise AssertionError(window.banner.text())

window.toggle_connection();until(lambda:window.worker.ready)
until(lambda:len(window.raw1)>250)
assert window.worker.info['channels']==2
assert window.worker.info['sampling_rate_hz']==2000
until(lambda:window.display_fs==2000)
assert window.raw.maxlen==10000 and window.env_ring.maxlen==400
assert not window.channel_checks[0].isEnabled()
window.start_recording();until(lambda:window.recording and window.engine.phase=='action')
assert window.config()['selected_channels']==[0,1]
assert 'CH1' in window.quality_label.text()
assert window.site.text() == '喉咙脸'
assert len(window.curves1[1].getData()[1])>0
assert not window.curves1[0].isVisible() and not window.curves[1].isVisible()
np.testing.assert_array_equal(window.curves[0].getData()[1], window.curves[2].getData()[1])
np.testing.assert_array_equal(window.curves1[1].getData()[1], window.curves1[2].getData()[1])
window.fit_ranges_once()
assert window.y_limits[2][1] == max(window.y_limits[0][1], window.y_limits[1][1])
window.grab().save(str(OUT/'dual_main.png'))
until(lambda:not window.recording and window.current_path is not None)
folder=Path(window.current_path);meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
assert meta['channel_count']==2 and meta['selected_channels']==[0,1]
assert meta['status']=='completed'
assert meta['sampling_rate_hz']==2000 and meta['display_filter']['bandpass_hz']==[20,500]
assert meta['sensor_supply_v']==5.0 and meta['sensor_supply_source']=='operator_entered'
rows=list(csv.DictReader((folder/'samples.csv').open()))
raw=np.array([[int(r['raw_ch0']),int(r['raw_ch1'])] for r in rows])
assert not np.array_equal(raw[:,0],raw[:,1])
frames=Decoder().feed((folder/'transport.bin').read_bytes())
decoded={f.first+i:pair for f in frames if f.kind==DATA for i,pair in enumerate(np.frombuffer(f.payload,dtype='<u2').reshape(-1,2))}
np.testing.assert_array_equal(raw,[decoded[int(r['device_sample_idx'])] for r in rows])
assert all(m['exported'] for m in export(folder,allow_simulated=True))
window.close();app.processEvents()

(OUT/'recordings').mkdir();source=dual_recording(OUT/'recordings/dual')
dialog=LDAWindow(OUT);dialog.resize(1300,960);dialog.show()
until(lambda:not dialog.busy);dialog.train_picker.set_all(True)
dialog.channel_checks[1].setChecked(True)
assert '16维' in dialog.feature_summary.text()
dialog.strategy.setCurrentIndex(dialog.strategy.findData('within_recording'))
dialog.grab().save(str(OUT/'dual_lda.png'))
dialog.train_button.click();until(lambda:not dialog.busy)
assert dialog.last_result,dialog.status.text()
run=dialog.last_result;model=lda.load_model(run['model_path'])
assert model['channels']==[0,1] and len(model['feature_names'])==16
dialog.use_model.click();until(lambda:not dialog.busy)
assert '总计16维' in dialog.model_info.text()
ids={r['trial_id'] for r in run['report']['test']}
for _,_,leaves,_ in dialog.test_picker.record_items:
    for leaf,row in leaves:
        from PySide6 import QtCore
        leaf.setCheckState(0,QtCore.Qt.CheckState.Checked if row['trial_id'] in ids else QtCore.Qt.CheckState.Unchecked)
dialog.channel_checks[1].setChecked(False)  # testing stays frozen at two channels
dialog.test_button.click();until(lambda:not dialog.busy)
assert dialog.last_result['report']['metrics']==run['report']['metrics']
dialog.grab().save(str(OUT/'dual_results.png'));dialog.close();app.processEvents()
from test_2khz import make_2k_recording
new_source=make_2k_recording(OUT/'recordings/new_2k')
dialog=LDAWindow(OUT);dialog.show();until(lambda:not dialog.busy)
dialog.train_picker.populate(lda.scan_recordings(new_source));dialog.train_picker.set_all(True)
assert dialog.variant.currentData()=='notch50_bp500'
dialog.channel_checks[1].setChecked(True)
dialog.train_button.click();until(lambda:not dialog.busy)
assert dialog.last_result,dialog.status.text()
assert lda.load_model(dialog.last_result['model_path'])['preprocessing']['fs']==2000
dialog.tabs.setCurrentIndex(0);dialog.grab().save(str(OUT/'new_2k_lda.png'))
dialog.train_picker.populate(lda.scan_recordings(source));dialog.train_picker.set_all(True)
assert dialog.variant.currentData()=='notch50'
dialog.close();app.processEvents()
print('PASS',OUT)
