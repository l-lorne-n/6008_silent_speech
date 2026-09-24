"""Replay a recorded baseline into real Qt without opening a serial port."""
import csv
import json
import os
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6 import QtWidgets as W
from semg.app import MainWindow, configure_fonts

app=W.QApplication([]);configure_fonts(app)
window=MainWindow();window.timer.stop()
window.resize(1400,1050);window.show();app.processEvents()
source=ROOT/'recordings/S05/session_001/20260924_123453_5ddcee'
with (source/'samples.csv').open() as f:
    rows=list(csv.DictReader(f))[140:10140]
alive=[True]
worker=SimpleNamespace(mode='binary',ready=True,port='REPLAY',preview=queue.Queue(),messages=queue.Queue(),is_alive=lambda:alive[0])
window.worker=worker;window.sync_rhythm_context();window.rhythm_after_ns=0
now=time.perf_counter_ns()
for start in range(0,len(rows),20):
    chunk=rows[start:start+20]
    worker.preview.put(dict(first=int(chunk[0]['device_sample_idx']),rx_ns=now,
                            rows=[(float(r['raw_ch0']),'','') for r in chunk]))
window.poll();window.poll();app.processEvents()
assert window.rhythm.result.status=='detected',window.rhythm_label.text()
assert '81' in window.rhythm_label.text()
assert window.y_limits==[(0,65535),(-10000,10000),(0,5000)]
out=ROOT/'tests/output';out.mkdir(exist_ok=True)
window.grab().save(str(out/'rhythm_gui.png'))
detected_text=window.rhythm_label.text()
window.recording=True;window.engine=SimpleNamespace(phase='action')
window.update_rhythm_indicator()
assert '暂停判断' in window.rhythm_label.text() and not window.rhythm.raw
window.engine.phase='paused';window.update_rhythm_indicator()
assert '检测到稳定' not in window.rhythm_label.text()
window.recording=False;worker.mode='simulation';window.update_rhythm_indicator()
assert '当前不判断' in window.rhythm_label.text()
worker.mode='legacy';window.update_rhythm_indicator()
assert '当前不判断' in window.rhythm_label.text()
alive[0]=False;window.update_rhythm_indicator()
assert '等待完整采样设备连接' in window.rhythm_label.text()
window.worker=None;window.close()
report=dict(status='PASS',source=str(source),display=detected_text,serial_port_opened=False,
            checks=['real baseline replay','action clears result','pause resets','simulation disabled','legacy disabled','disconnect clears','fixed axes preserved'])
(out/'rhythm_gui_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
