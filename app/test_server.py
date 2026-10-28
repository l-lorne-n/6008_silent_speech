"""Run with python -m unittest -v test_server. No third-party dependencies."""
import http.client
import json
from pathlib import Path
import base64
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch
from urllib.parse import unquote
from server import LocalServer, WORDS, media_manifest, validate_command, validate_signal


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep HTTP/media coverage runnable on a fresh clone without private media.
        temporary = tempfile.TemporaryDirectory(prefix='quietly-http-test-')
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name).resolve()
        (cls.root / 'photo').mkdir()
        (cls.root / 'music').mkdir()
        (cls.root / 'photo/测试 图片.png').write_bytes(base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII='))
        with wave.open(str(cls.root / 'music/测试 音频.wav'), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(bytes(512))
        (cls.root / 'server.py').write_text('private test fixture', encoding='utf-8')
        for replacement in (patch('server.ROOT', cls.root),
                            patch('server.media_manifest', side_effect=lambda: media_manifest(cls.root))):
            replacement.start()
            cls.addClassCleanup(replacement.stop)
        cls.server = LocalServer(('127.0.0.1', 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        body = json.dumps(data) if data is not None else None
        conn.request(method, path, body=body, headers=headers or ({'Content-Type': 'application/json'} if body else {}))
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    def test_media_files_and_unicode_urls(self):
        status, _, body = self.request('GET', '/api/media')
        self.assertEqual(status, 200)
        manifest = json.loads(body)
        self.assertGreater(len(manifest['photos']), 0)
        self.assertGreater(len(manifest['music']), 0)
        for item in manifest['photos'] + manifest['music']:
            status, headers, _ = self.request('HEAD', '/' + item['url'])
            self.assertEqual(status, 200)
            self.assertEqual(int(headers['Content-Length']), (self.root / unquote(item['url'])).stat().st_size)

    def test_audio_range_and_suffix(self):
        manifest = json.loads(self.request('GET', '/api/media')[2])
        item = manifest['music'][0]
        original = (self.root / unquote(item['url'])).read_bytes()
        for requested, expected in [('bytes=10-29', original[10:30]), ('bytes=-20', original[-20:])]:
            status, headers, body = self.request('GET', '/' + item['url'], headers={'Range': requested})
            self.assertEqual(status, 206)
            self.assertEqual(body, expected)
            self.assertEqual(int(headers['Content-Length']), len(expected))
        self.assertEqual(self.request('GET', '/' + item['url'], headers={'Range': 'bytes=999999999-'})[0], 416)

    def test_private_files_and_traversal(self):
        for path in ['/server.py', '/start.ps1', '/%2e%2e/EE6008_sEMG_Silent_Speech.docx', '/photo/../../EE6008_sEMG_Silent_Speech.docx']:
            self.assertEqual(self.request('GET', path)[0], 404)

    def test_reject_other_origins(self):
        self.assertEqual(self.request('POST', '/api/command', {'word': 'music'}, {'Content-Type': 'application/json', 'Origin': 'https://example.org'})[0], 403)
        self.assertEqual(self.request('GET', '/api/media', headers={'Host': 'example.org'})[0], 403)

    def test_validation_and_all_words(self):
        for word in WORDS:
            self.assertEqual(validate_command({'word': word.upper(), 'confidence': .9})['word'], word)
        for payload in [{'word': 'delete'}, {'word': 'music', 'confidence': 90}, {'word': 'music', 'confidence': True}, {'word': 'music', 'confidence': float('nan')}, {'word': 1}, {'word': 'photo', 'id': ''}]:
            with self.assertRaises(ValueError):
                validate_command(payload)
        self.assertEqual(self.request('POST', '/api/command', {'word': 'unknown'})[0], 400)

    def test_signal_validation(self):
        self.assertEqual(validate_signal({'channels': [[1, 2], [3, 4]], 'sample_rate': 2000})['sample_rate'], 2000)
        for payload in [{'channels': []}, {'channels': [[1]]}, {'channels': [[1, float('inf')]]}, {'channels': [[1, 2]], 'sample_rate': 0}]:
            with self.assertRaises(ValueError):
                validate_signal(payload)

    def test_board_status_and_unconnected_control(self):
        status, _, body = self.request('GET', '/api/board/status')
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['connected'])
        self.assertTrue(json.loads(body)['stale'])
        self.assertEqual(self.request('POST', '/api/board/control', {'enabled': True})[0], 400)
        self.assertEqual(self.request('POST', '/api/board/control', {'enabled': 'yes'})[0], 400)
        self.assertEqual(self.request('POST', '/api/board/disconnect', {})[0], 200)

    def test_board_endpoints_reject_external_origin(self):
        self.assertEqual(self.request('POST', '/api/board/connect', {'port': 'COM7'},
            {'Content-Type': 'application/json', 'Origin': 'https://example.org'})[0], 403)

    def test_board_connect_baud_selection_and_legacy_default(self):
        for extra, expected in (({}, 230400), ({'baudrate': 115200}, 115200), ({'baudrate': 230400}, 230400)):
            with patch.object(self.server.board, 'connect', return_value={'baudrate': expected}) as connect:
                self.assertEqual(self.request('POST', '/api/board/connect', {'port': 'COM7', **extra})[0], 200)
                connect.assert_called_once_with('COM7', expected)
        for bad in (9600, None, True, 115200.0, '115200'):
            with patch.object(self.server.board, '_disconnect') as disconnect:
                self.assertEqual(self.request('POST', '/api/board/connect', {'port': 'COM7', 'baudrate': bad})[0], 400)
                disconnect.assert_not_called()

    def test_ble_scan_connect_and_transport_validation(self):
        with patch.object(self.server.board, 'scan_ble', return_value=[{'device': 'BT24', 'description': 'test'}]):
            status, _, body = self.request('POST', '/api/board/scan-ble', {})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['ports'][0]['device'], 'BT24')
        with patch.object(self.server.board, 'connect', return_value={}) as connect:
            self.assertEqual(self.request('POST', '/api/board/connect', {'transport': 'ble', 'port': 'BT24', 'baudrate': 115200})[0], 200)
            connect.assert_called_once_with('BT24', 115200, 'ble')
        for value in ('unknown', None, 123):
            self.assertEqual(self.request('POST', '/api/board/connect', {'transport': value, 'port': 'BT24'})[0], 400)

    def test_sse_delivery_and_deduplication(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        conn.request('GET', '/api/events')
        stream = conn.getresponse()
        self.assertEqual(stream.status, 200)
        self.assertEqual(stream.readline(), b'retry: 1500\n')
        stream.readline(); stream.readline()
        result = json.loads(self.request('POST', '/api/command', {'word': 'photo', 'id': 'one-action', 'confidence': .91})[2])
        self.assertEqual(result['subscribers'], 1)
        self.assertEqual(stream.readline(), b'event: command\n')
        received = json.loads(stream.readline().decode()[6:])
        self.assertEqual(received['word'], 'photo')
        self.assertEqual(received['confidence'], .91)
        retry = json.loads(self.request('POST', '/api/command', {'word': 'photo', 'id': 'one-action'})[2])
        self.assertTrue(retry['duplicate'])
        stream.close(); conn.close()


if __name__ == '__main__':
    unittest.main()
