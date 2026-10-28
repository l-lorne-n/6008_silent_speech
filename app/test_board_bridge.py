"""Transport and command-gating regressions, no physical serial port needed."""
import queue
import hashlib
import json
import time
import threading
import unittest
from unittest.mock import patch, Mock
from server import Hub
from board_bridge import BoardBridge
from semg.board_protocol import Packet, RESULT
from semg.board_models import NEW_MODEL, model_source
from semg.protocol import encode


def frame(end=200, top=5, emitted=255, flags=0, mode=1):
    probabilities = [0.] * 11
    probabilities[top] = 1.
    payload = RESULT.pack(1000, 68000, 68500, 20, 0, 68500, 0, 0, 0,
                          mode, top, emitted, 1, *probabilities, 0, 0)
    return Packet(4, end // 200, end, 0, flags, payload)


class BoardTest(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()
        self.events = queue.Queue()
        self.hub.clients.add(self.events)
        self.board = BoardBridge(self.hub)
        self.board.state.update(connected=True, verified=True, enabled=True)

    def test_candidates_do_not_operate_media(self):
        self.board.accept_result(frame(), 'test')
        self.assertTrue(self.events.empty())

    def test_confirmed_uses_emitted_class_not_top(self):
        self.board.accept_result(frame(top=5, emitted=3), 'test')
        message = self.events.get_nowait().decode()
        data = json.loads(message.split('data: ')[1])
        self.assertEqual(data['word'], 'music')
        self.assertTrue(data['board'])

    def test_pause_drops_commands_without_replay(self):
        self.board.accept_result(frame(), 'test')
        self.board.enable(False)
        self.events.get_nowait()  # status
        self.board.accept_result(frame(end=400, emitted=5), 'test')
        self.assertTrue(self.events.empty())
        self.board.enable(True)
        self.events.get_nowait()
        self.board.accept_result(frame(end=600), 'test')
        self.assertTrue(self.events.empty())
        self.board.accept_result(frame(end=800, emitted=5), 'test')
        self.assertIn(b'event: command', self.events.get_nowait())

    def test_faults_unverified_and_wrong_mode_never_dispatch(self):
        self.board.accept_result(frame(flags=1, emitted=5), 'test')
        self.board.accept_result(frame(end=400, mode=0, emitted=5), 'test')
        self.board.state['verified'] = False
        self.board.accept_result(frame(end=600, emitted=5), 'test')
        self.assertTrue(self.events.empty())

    def test_reboot_and_duplicate_disable_execution(self):
        self.board.accept_result(frame(end=1000), 'test')
        self.board.accept_result(frame(end=1000, emitted=5), 'test')
        self.assertFalse(self.board.state['verified'])
        self.assertFalse(self.board.state['enabled'])
        self.assertTrue(self.events.empty())

    def test_stale_cannot_enable(self):
        self.board.last_rx = time.monotonic() - 2
        with self.assertRaises(ValueError):
            self.board.enable(True)

    def test_uart_fault_cannot_enable_control(self):
        self.board.accept_result(frame(flags=4), 'test')
        with self.assertRaisesRegex(ValueError, '故障'):
            self.board.enable(True)

    def test_model_identity_rejects_mismatch(self):
        with self.assertRaises(ValueError):
            self.board.verify({'model': 'mid64_temporal3', 'mode': 1, 'sha': 'wrong'})

    def test_new_and_old_frozen_models_are_accepted_but_not_interchangeable(self):
        identities = {}
        for name in ('mid64_temporal3', NEW_MODEL):
            digest = hashlib.sha256((model_source(name) / 'model_int8_qdq.onnx').read_bytes()).hexdigest()[:16]
            identities[name] = digest
            self.board.verify({'model': name, 'mode': 1, 'sha': digest})
        self.assertNotEqual(*identities.values())
        for name, other in (('mid64_temporal3', NEW_MODEL), (NEW_MODEL, 'mid64_temporal3')):
            with self.assertRaises(ValueError):
                self.board.verify({'model': name, 'mode': 1, 'sha': identities[other]})
        with self.assertRaises(ValueError):
            self.board.verify({'model': '../' + NEW_MODEL, 'mode': 1, 'sha': identities[NEW_MODEL]})
        with self.assertRaises(ValueError):
            self.board.verify({'model': NEW_MODEL, 'mode': 0, 'sha': identities[NEW_MODEL]})

    def test_unknown_port_not_opened(self):
        with patch.object(self.board, 'ports', return_value=[{'device': 'COM7'}]):
            with self.assertRaises(ValueError):
                self.board.connect('COM99')

    def test_selected_baud_passes_to_worker_and_invalid_choice_keeps_connection(self):
        for rate in (115200, 230400):
            with patch.object(self.board, 'ports', return_value=[{'device': 'COM7'}]), patch('board_bridge.threading.Thread') as thread:
                thread.return_value.is_alive.return_value = False
                self.board.connect('COM7', rate)
                self.assertEqual(thread.call_args.kwargs['args'], ('COM7', rate))
                self.assertEqual(self.board.snapshot()['baudrate'], rate)
        for bad in (9600, None, True, 115200.0, '115200'):
            with patch.object(self.board, '_disconnect') as disconnect, self.assertRaises(ValueError):
                self.board.connect('COM7', bad)
            disconnect.assert_not_called()

    def test_fragmented_serial_commands_identical_for_both_bauds_and_models(self):
        for name in ('mid64_temporal3', NEW_MODEL):
            digest = hashlib.sha256((model_source(name)/'model_int8_qdq.onnx').read_bytes()).hexdigest()[:16]
            identity = dict(model=name, sha=digest, fs=2000, win=2800, step=200, mode=1, fw=1, env_hz=100, env_ms=200)
            result = frame(end=2800, top=5, emitted=5)
            stream = encode(5, 0, 0, json.dumps(identity).encode(), version=2) + encode(4, 1, 2800, result.payload, version=2)
            for rate in (115200, 230400):
                with self.subTest(model=name, baud=rate):
                    self.setUp()
                    self.board.state.update(verified=False, baudrate=rate)
                    chunks = [stream[i:i+7] for i in range(0, len(stream), 7)]
                    def read(_):
                        if chunks:
                            return chunks.pop(0)
                        self.board.stop.set()
                        return b''
                    with patch('serial.Serial') as serial:
                        serial.return_value.in_waiting = 7
                        serial.return_value.read.side_effect = read
                        self.board._run('COM_TEST', rate)
                        self.assertEqual(serial.call_args.kwargs['baudrate'], rate)
                        serial.return_value.close.assert_called_once()
                    events = []
                    while not self.events.empty():
                        message = self.events.get_nowait().decode()
                        if 'event: command' in message:
                            events.append(json.loads(message.split('data: ')[1]))
                    self.assertEqual([event['word'] for event in events], ['photo'])
                    self.assertFalse(self.board.state['error'])
                    self.assertEqual(self.board.state['crc_errors'], 0)
                    self.assertEqual(self.board.state['sequence_gaps'], 0)

    def test_ble_scan_and_connect_validate_transport_and_selection(self):
        async def scan():return [dict(name='BT24', address='48:87:2D:D0:B5:43', rssi=-50)]
        with patch('board_bridge.scan_bt24', side_effect=scan):
            devices = self.board.scan_ble()
        self.assertEqual(devices[0]['device'], '48:87:2D:D0:B5:43')
        with patch('board_bridge.threading.Thread') as thread:
            thread.return_value.is_alive.return_value = False
            self.board.connect(devices[0]['device'], 115200, 'ble')
            self.assertEqual(thread.call_args.kwargs['args'], (devices[0]['device'], 115200, 'ble'))
        for transport, address in [('BLE', 'x'), ('ble', 'unknown'), (None, 'x')]:
            with self.assertRaises(ValueError), patch.object(self.board, '_disconnect') as disconnect:
                self.board.connect(address, 115200, transport)
            disconnect.assert_not_called()

    def test_ble_new_model_identity_then_fragmented_confirmation_reaches_dispatcher(self):
        digest = hashlib.sha256((model_source(NEW_MODEL)/'model_int8_qdq.onnx').read_bytes()).hexdigest()[:16]
        identity = dict(model=NEW_MODEL, sha=digest, fs=2000, win=2800, step=200, mode=1, fw=1, env_hz=100)
        r = frame(end=2800, top=5, emitted=5)
        stream = encode(5, 0, 0, json.dumps(identity).encode(), version=2)+encode(4, 1, 2800, r.payload, version=2)
        chunks = [stream[i:i+20] for i in range(0, len(stream), 20)]
        port = Mock();port.ready = threading.Event();port.ready.set();port.in_waiting = 20
        port.closed = threading.Event();port.closed.set()
        def read(_):
            if chunks:return chunks.pop(0)
            self.board.stop.set();return b''
        port.read.side_effect = read
        self.board.state.update(verified=False, transport='ble')
        with patch('board_bridge.BleByteStream', return_value=port):self.board._run('BT24', 115200, 'ble')
        messages = []
        while not self.events.empty():messages.append(self.events.get_nowait())
        commands = [json.loads(m.decode().split('data: ')[1]) for m in messages if b'event: command' in m]
        self.assertEqual([c['word'] for c in commands], ['photo'])
        self.assertEqual(self.board.state['info']['model'], NEW_MODEL)
        self.assertEqual(self.board.state['rx_bytes'], len(stream))
        self.assertEqual(self.board.state['error'], '')
        port.write.assert_any_call(b'N\nI\n');port.close.assert_called_once()

    def test_ble_csv_firmware_never_dispatches_and_explains_mismatch(self):
        port = Mock();port.ready = threading.Event();port.ready.set();port.in_waiting = 50
        port.closed = threading.Event();port.closed.set()
        port.read.return_value = b'123,456,-1,2,3,4\r\n'
        self.board.state['verified'] = False
        with patch('board_bridge.BleByteStream', return_value=port):self.board._run('BT24', 115200, 'ble')
        self.assertIn('CSV', self.board.state['error'])
        while not self.events.empty():self.assertNotIn(b'event: command', self.events.get_nowait())
        self.assertFalse(self.board.state['connected'])
        port.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
