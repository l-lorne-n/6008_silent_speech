import asyncio
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from semg.ble_transport import BleByteStream, SERVICE_UUID, NOTIFY_UUID, WRITE_UUID, uart_characteristics
from semg.board_protocol import BoardDecoder
from semg.protocol import encode

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class Services:
    def __init__(self, notify_props=('notify', 'write')):
        self.notify = SimpleNamespace(uuid=NOTIFY_UUID, properties=notify_props, max_write_without_response_size=20)
        self.write = SimpleNamespace(uuid=WRITE_UUID, properties=['write-without-response'], max_write_without_response_size=20)
        self.service = SimpleNamespace(characteristics=[self.notify, self.write])

    def get_service(self, uuid):
        return self.service if uuid == SERVICE_UUID else None


class Scanner:
    @staticmethod
    async def find_device_by_address(address, timeout):
        return SimpleNamespace(address=address)


class Client:
    def __init__(self):
        self.services = Services()
        self.is_connected = False
        self.writes = []
        self.written = threading.Event()
        self.disconnected = threading.Event()

    def factory(self, device, **kwargs):
        self.disconnected_callback = kwargs['disconnected_callback']
        return self

    async def connect(self):
        self.is_connected = True

    async def start_notify(self, char, callback):
        self.received = callback

    async def write_gatt_char(self, char, data, response):
        self.writes.append((char.uuid, data, response))
        self.written.set()

    async def disconnect(self):
        self.is_connected = False
        self.disconnected_callback(self)
        self.disconnected.set()


def port_for(client, **kwargs):
    port = BleByteStream('test', scanner=Scanner, client_factory=client.factory, **kwargs)
    assert port.ready.wait(2), port.error
    return port


def test_fragmented_notifications_and_ordered_command_writes():
    client = Client();port = port_for(client)
    try:
        payload = encode(3, 1, 200, bytes(8), version=2)
        for i in range(0, len(payload), 7):client.received(None, payload[i:i+7])
        decoder = BoardDecoder()
        packets = []
        while port.in_waiting:packets.extend(decoder.feed(port.read(3)))
        assert len(packets) == 1 and packets[0].seq == 1 and decoder.crc_errors == 0
        port.write(b'I\n')
        assert client.written.wait(2)
        assert client.writes == [(NOTIFY_UUID, b'I\n', True)]
    finally:
        port.close();assert port.closed.wait(2)
    assert client.disconnected.is_set() and port.error is None


def test_ffe2_fallback_and_chunking():
    client = Client();client.services = Services(('notify',))
    port = port_for(client)
    try:
        port.write(b'x'*45)
        deadline = time.monotonic()+2
        while len(client.writes)<3 and time.monotonic()<deadline:time.sleep(.01)
        assert client.writes == [(WRITE_UUID, b'x'*20, False), (WRITE_UUID, b'x'*20, False), (WRITE_UUID, b'x'*5, False)]
    finally:
        port.close();assert port.closed.wait(2)


def test_missing_notify_is_rejected():
    with pytest.raises(ValueError, match='FFE1'):uart_characteristics(Services(('read',)))


def test_overflow_fails_instead_of_silently_dropping_data():
    client = Client();port = port_for(client, max_buffer=10)
    client.received(None, b'x'*11)
    with pytest.raises(OSError, match='溢出'):port.read(1)
    assert port.closed.wait(2) and client.disconnected.is_set()


def test_unexpected_disconnect_is_reported():
    client = Client();port = port_for(client)
    client.is_connected = False;client.disconnected_callback(client)
    with pytest.raises(OSError, match='断开'):port.read(1)
    assert port.closed.wait(2)


def test_cancel_during_scan_finishes_without_connection():
    class SlowScanner:
        @staticmethod
        async def find_device_by_address(*args, **kwargs):await asyncio.sleep(60)
    client = Client()
    port = BleByteStream('test', scanner=SlowScanner, client_factory=client.factory)
    port.close()
    assert port.closed.wait(2) and not port.ready.is_set() and not client.is_connected


def test_transient_advertising_miss_retries_once():
    class TransientScanner:
        calls = 0
        @classmethod
        async def find_device_by_address(cls, address, timeout):
            cls.calls += 1
            return None if cls.calls == 1 else SimpleNamespace(address=address)
    client = Client()
    port = BleByteStream('test', scanner=TransientScanner, client_factory=client.factory)
    try:
        assert port.ready.wait(2)
        assert TransientScanner.calls == 2
    finally:
        port.close();assert port.closed.wait(2)


def test_gatt_write_error_is_not_hidden_by_cleanup_disconnect():
    class FailingClient(Client):
        async def write_gatt_char(self, *args, **kwargs):raise OSError('specific GATT write failure')
    client = FailingClient();port = port_for(client)
    port.write(b'I\n')
    assert port.closed.wait(2)
    assert 'specific GATT write failure' in port.error
    assert client.disconnected.is_set()


def test_close_waits_for_native_disconnect_even_when_session_is_cancelled():
    class SlowCloseClient(Client):
        async def disconnect(self):
            await asyncio.sleep(.12)
            await super().disconnect()
    client = SlowCloseClient();port = port_for(client)
    port.close()
    assert port.closed.wait(2)
    assert client.disconnected.is_set(), 'native disconnect was cancelled before releasing the device'


def test_ui_csv_is_not_accepted_as_model_data(tmp_path):
    from PySide6.QtWidgets import QApplication
    from semg.board_ui import BoardWindow
    app = QApplication.instance() or QApplication([])
    with patch('semg.board_ui.list_ports.comports', return_value=[]):window = BoardWindow(tmp_path)
    window.timer.stop()
    client = Client();port = port_for(client)
    try:
        window.transport.setCurrentIndex(1)
        window.ble_found([dict(name='BT24', address='test', rssi=-50)])
        with patch('semg.board_ui.BleByteStream.__new__', return_value=port), patch('semg.board_ui.BleByteStream.__init__', return_value=None):
            window.toggle_connection()
        window.poll()
        client.received(None, b'123,456,-1,2,3,4\r\n')
        window.poll()
        assert window.csv_detected and 'CSV' in window.state.text()
        assert window.info is None and not window.start.isEnabled()
        assert not window.transport.isEnabled() and not window.baud.isEnabled()
        window.disconnect()
        assert port.closed.wait(2)
        window.poll()
        assert window.transport.isEnabled()
    finally:
        port.close();port.closed.wait(2);window.poll();window.close();app.processEvents()
