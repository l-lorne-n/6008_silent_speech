"""Explicit hardware smoke: connect the monitor to --address and save evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from semg.board_ui import BoardWindow

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--address', required=True)
parser.add_argument('--model', default='mid64_temporal3_20261027')
args = parser.parse_args()
output = ROOT/'history/ble_bt24_20261028'
app = QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
window = BoardWindow(ROOT)
window.model.setCurrentText(args.model)
window.resize(1280, 1050)
window.show()
window.transport.setCurrentIndex(1)
window.ports.addItem('BT24 · '+args.address, args.address)
window.baud.setCurrentIndex(window.baud.findData(115200))
window.toggle_connection()
deadline = time.monotonic()+35
while time.monotonic()<deadline:
    app.processEvents()
    if window.csv_detected or (window.info and len(window.rows)>=30) or not window.port:break
    time.sleep(.02)
report = dict(state=window.state.text(), bytes=window.rx_bytes, csv_detected=window.csv_detected,
              info=window.info, identity_verified=window.matches(), results=len(window.rows),
              last=window.rows[-1] if window.rows else None, crc_errors=window.decoder.crc_errors,
              sequence_gaps=window.sequence_gaps, test_enabled=window.start.isEnabled(), log=window.log.toPlainText())
window.grab().save(str(output/'monitor_ble.png'))
window.disconnect()
deadline = time.monotonic()+12
while window.ble_closing and time.monotonic()<deadline:
    app.processEvents();time.sleep(.02)
report['disconnected_cleanly'] = window.ble_closing is None
(output/'monitor_smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
window.close()
app.processEvents()
sys.exit(0 if report['identity_verified'] and report['results']>=30 and report['test_enabled'] and report['disconnected_cleanly'] else 1)
