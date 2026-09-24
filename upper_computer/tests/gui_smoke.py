"""Headless real Qt + simulated wire + recorder + export integration test."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtWidgets as W
import semg.app as ui
from export_trials import export

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
ui.ROOT = OUT
app = W.QApplication([])
ui.configure_fonts(app)
window = ui.MainWindow()
assert window.reps.value() == 10
assert window.mode.currentData() == "binary"
assert window.y_limits == [(0, 65535), (-10000, 10000), (0, 5000)]
assert all(p.getViewBox().autoRangeEnabled() == [False, False] for p in window.plots)
window.mode.setCurrentIndex(2)
window.reps.setValue(1)
for spin in window.durations.values(): spin.setValue(.2)
window.show()
window.toggle_connection()


def until(predicate, timeout=12):
    end = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        time.sleep(.01)
        if time.monotonic() > end:
            raise AssertionError("Timeout: " + window.banner.text())


until(lambda: window.worker.ready)
until(lambda: len(window.raw) >= 200)
window.fit_ranges_once()
locked_ranges = [tuple(p.getViewBox().viewRange()[1]) for p in window.plots]
window.start_recording()
until(lambda: window.engine.phase == "action")
assert locked_ranges == [tuple(p.getViewBox().viewRange()[1]) for p in window.plots]
assert "滤波 RMS" in window.quality_label.text()
window.reject()
window.grab().save(str(OUT / "gui_preview.png"))
window.cue_window.show()
window.cue_window.resize(1200, 750)
app.processEvents()
window.cue_window.grab().save(str(OUT / "cue_preview.png"))
until(lambda: not window.recording and window.current_path is not None)
path = Path(window.current_path)
meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
assert meta["status"] == "completed", meta
assert meta["samples_saved"] > 2000
assert meta["display_filter"]["notch_hz"] == 50
assert meta["display_filter"]["raw_recording_modified"] is False
results = [json.loads(s) for s in (path / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(results) == 5 and results[0]["status"] == "rejected", results
manifest = export(path, allow_simulated=True)
assert sum(t["exported"] for t in manifest) == 4, manifest
window.close()
report = dict(recording=str(path), samples_saved=meta["samples_saved"], trial_results=results,
              exported_trials=sum(t["exported"] for t in manifest), status="PASS")
(OUT / "gui_test_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
