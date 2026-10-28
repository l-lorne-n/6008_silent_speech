"""Baud selection must leave the frozen sampling/model path unchanged."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from semg.board_models import ROOT, NEW_MODEL
from semg.serial_settings import validate_baudrate
from board.manage import build_dir, built_image, uart_source, main


@pytest.mark.parametrize('value', [9600, 0, None, True, 115200.0, '115200'])
def test_invalid_baud_rejected(value):
    with pytest.raises(ValueError):
        validate_baudrate(value)


def test_capture_cannot_flash_115200():
    with patch('sys.argv', ['manage.py', 'flash', '--capture', '--baud', '115200']), patch('board.manage.flash') as flash:
        with pytest.raises(SystemExit):
            main()
        flash.assert_not_called()


@pytest.mark.parametrize('name', ['mid64_temporal3', NEW_MODEL])
def test_both_firmware_variants_preserve_model_and_sampling(name):
    legacy = ROOT / 'board/build' / name
    a, b = (build_dir(name, rate) for rate in (115200, 230400))
    assert a != b != legacy
    # This checks the actual compiled exports, not a newly trained substitute.
    assert (a/'model_data.h').read_bytes() == (b/'model_data.h').read_bytes() == (legacy/'model_data.h').read_bytes()
    assert uart_source((a/'main.c').read_text(), 230400) == (b/'main.c').read_text()
    assert (b/'main.c').read_bytes() == (legacy/'main.c').read_bytes()
    for folder, rate in ((a, 115200), (b, 230400)):
        manifest = json.loads((folder/'build_manifest.json').read_text())
        assert manifest['baudrate'] == rate and manifest['model'] == name
        assert built_image(name, rate) == folder/'Objects/realtime.hex'
        assert '0 Error(s), 0 Warning(s)' in (folder/'build.log').read_text(encoding='utf-8-sig')
        for path, digest in manifest['sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('rate', [115200, 230400])
def test_monitor_selects_same_baud_for_connection_and_firmware(rate, tmp_path):
    from PySide6.QtWidgets import QApplication
    from semg.board_ui import BoardWindow
    app = QApplication.instance() or QApplication([])
    with patch('semg.board_ui.list_ports.comports', return_value=[]):
        window = BoardWindow(tmp_path)
    window.timer.stop()
    try:
        assert window.baud.currentData() == 230400
        window.baud.setCurrentIndex(window.baud.findData(rate))
        assert window.firmware_arguments()[-2:] == ['--baud', str(rate)]
        fake = Mock()
        with patch('semg.board_ui.serial.Serial', return_value=fake) as serial:
            window.toggle_connection()
            assert serial.call_args.kwargs['baudrate'] == rate
            assert window.port is fake and not window.baud.isEnabled()
            fake.write.assert_any_call(b'N\n')
        window.disconnect()
        assert window.baud.isEnabled()
        record = json.loads((window.output_dir/'board_trace.jsonl').read_text().splitlines()[0])
        assert record['baudrate'] == rate
        window.mode.setCurrentIndex(1)
        assert window.baud.currentData() == 230400 and not window.baud.isEnabled()
        assert window.firmware_arguments()[-4:] == ['flash', '--capture', '--baud', '230400']
    finally:
        window.close()
        app.processEvents()
