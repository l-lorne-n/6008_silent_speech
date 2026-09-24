"""Read-only legacy firmware integration. Automated cues are NOT subject trials."""
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtWidgets as W
import semg.app as ui

OUT = Path(__file__).resolve().parent / "output"
ui.ROOT = OUT
app = W.QApplication([])
ui.configure_fonts(app)
window = ui.MainWindow()
window.port.setEditText("COM4")
window.subject.setText("HARDWARE_LINK_TEST")
window.site.setText("自动链路测试，非受试者实验")
window.notes.setText("无人按提示执行动作；仅测试串口/GUI/文件保存；不得用于训练")
window.reps.setValue(1)
for spin in window.durations.values(): spin.setValue(.3)
window.show()
window.toggle_connection()


def until(predicate, timeout=15):
    end = time.monotonic() + timeout
    while not predicate():
        app.processEvents(); time.sleep(.01)
        if time.monotonic() > end:
            raise AssertionError(window.banner.text())


until(lambda: window.worker.ready)
app.processEvents()
window.start_recording()
until(lambda: window.engine.phase == "action")
window.pause()
assert window.engine.phase == "paused"
window.pause()
until(lambda: window.engine.completed >= 2)
window.grab().save(str(OUT / "hardware_preview.png"))
until(lambda: not window.recording and window.current_path is not None)
path = Path(window.current_path)
meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
assert meta["status"] == "completed" and meta["mode"] == "legacy"
assert meta["samples_saved"] > 250
with (path / "samples.csv").open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert all(row["device_sample_idx"] == "" for row in rows)
results = [json.loads(s) for s in (path / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(results) == 5 and results[0]["status"] == "interrupted"
window.close()
report = dict(status="PASS", port="COM4", mode="legacy", recording=str(path),
              samples_saved=len(rows), trial_result_count=len(results), commands_sent=False,
              purpose="Automated link test only. No subject compliance or word classification validation.")
(OUT / "hardware_test_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
