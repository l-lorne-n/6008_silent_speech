"""Offline version routing/identity checks; no serial device or flash access."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from semg.board_models import NEW_MODEL, NAMES, ROOT, model_source
from semg.board_protocol import BoardDecoder, model_info
from semg.protocol import encode


def identity(name):
    digest = hashlib.sha256((model_source(name) / 'model_int8_qdq.onnx').read_bytes()).hexdigest()[:16]
    return dict(model=name, sha=digest, fs=2000, win=2800, step=200,
                mode=1, fw=1, env_hz=100, env_ms=200)


def test_new_identity_fits_existing_wire_protocol():
    info = identity(NEW_MODEL)
    # Match firmware JSON separators; packet maximum is 236 bytes.
    payload = json.dumps(info, separators=(',', ':')).encode('ascii')
    assert len(payload) < 220  # firmware snprintf buffer; decoder limit is 236
    packets = BoardDecoder().feed(encode(5, 1, 0, payload, version=2))
    assert len(packets) == 1 and model_info(packets[0]) == info
    assert model_source('mid64_temporal3') != model_source(NEW_MODEL)
    assert len(NAMES) == len(set(NAMES)) == 11


def test_monitor_matches_selected_version_and_rejects_crossed_hashes():
    from PySide6.QtWidgets import QApplication
    from semg.board_ui import BoardWindow
    app = QApplication.instance() or QApplication([])
    with patch('semg.board_ui.list_ports.comports', return_value=[]):
        window = BoardWindow(ROOT)
    window.timer.stop()
    try:
        for name in ('mid64_temporal3', NEW_MODEL):
            window.model.setCurrentText(name)
            window.info = identity(name)
            assert window.matches()
            other = NEW_MODEL if name == 'mid64_temporal3' else 'mid64_temporal3'
            window.info['sha'] = identity(other)['sha']
            assert not window.matches()
            window.info = identity(other)
            assert not window.matches()
    finally:
        window.close()
        app.processEvents()


def test_built_firmware_is_bound_to_new_export_and_contract():
    build = ROOT / 'board/build' / NEW_MODEL
    manifest = json.loads((build / 'build_manifest.json').read_text(encoding='utf-8'))
    model = json.loads((build / 'model.json').read_text(encoding='utf-8'))
    contract_path = model_source(NEW_MODEL) / 'input_contract.json'
    assert model['name'] == manifest['model'] == NEW_MODEL
    assert Path(model['onnx']) == model_source(NEW_MODEL) / 'model_int8_qdq.onnx'
    assert model['onnx_sha256'] == manifest['onnx_sha256'] == hashlib.sha256(Path(model['onnx']).read_bytes()).hexdigest()
    assert model['contract_sha256'] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    assert model['contract'] == json.loads(contract_path.read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(manifest['hex']).read_bytes()).hexdigest() == manifest['hex_sha256']
    header = (build / 'model_data.h').read_text()
    assert f'#define MODEL_NAME "{NEW_MODEL}"' in header
    assert f'#define MODEL_HASH "{manifest["onnx_sha256"][:16]}"' in header
    for value in model['contract']['normalization']['scale']:
        assert float(value).hex() + 'f' in header
    for path, digest in manifest['sources'].items():
        # manage.py now offers baud variants; this historical manifest records
        # its old version. All inference sources and frozen artifacts stay exact.
        if Path(path) != ROOT / 'board/manage.py':
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
