"""BT24 GATT byte stream. One asyncio loop per connection, off the Qt thread."""
import asyncio
from queue import Empty, Full, Queue
import threading

SERVICE_UUID = '0000ffe0-0000-1000-8000-00805f9b34fb'
NOTIFY_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'
WRITE_UUID = '0000ffe2-0000-1000-8000-00805f9b34fb'


async def scan_bt24(timeout=6):
    from bleak import BleakScanner
    found = await asyncio.wait_for(BleakScanner.discover(timeout=timeout, return_adv=True), timeout + 8)
    return sorted([
        dict(name=a.local_name or d.name or 'BLE UART', address=d.address, rssi=a.rssi)
        for d, a in found.values()
        if 'BT24' in (a.local_name or d.name or '').upper() or SERVICE_UUID in a.service_uuids
    ], key=lambda x: -x['rssi'])


def uart_characteristics(services):
    service = services.get_service(SERVICE_UUID)
    if service is None:
        raise ValueError('设备没有 BT24 FFE0 服务，请确认模块型号及 UUID 配置。')
    chars = {c.uuid: c for c in service.characteristics}
    notify = chars.get(NOTIFY_UUID)
    if notify is None or not {'notify', 'indicate'}.intersection(notify.properties):
        raise ValueError('BT24 FFE1 不支持通知。')
    for uuid in (NOTIFY_UUID, WRITE_UUID):
        write = chars.get(uuid)
        if write and {'write', 'write-without-response'}.intersection(write.properties):
            return notify, write
    raise ValueError('BT24 没有可写入的 FFE1 / FFE2。')


class BleByteStream:
    """Nonblocking serial-like adapter. Errors propagate through read/write."""
    def __init__(self, address, *, scanner=None, client_factory=None, max_buffer=1024*1024):
        self.address = address
        self._scanner = scanner
        self._client_factory = client_factory
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._writes = Queue(maxsize=64)
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.error = None
        self._closing = False
        self.thread = threading.Thread(target=self._run, name='BT24-BLE', daemon=True)
        self.thread.start()

    def _fail(self, message):
        with self._lock:
            if self.error is None:
                self.error = str(message)

    def _check(self):
        if self.error:
            raise OSError(self.error)
        if self._stop.is_set():
            raise OSError('BLE 已关闭。')

    @property
    def in_waiting(self):
        self._check()
        with self._lock:
            return len(self._buffer)

    def read(self, size):
        self._check()
        with self._lock:
            data = bytes(self._buffer[:size])
            del self._buffer[:size]
        return data

    def write(self, data):
        self._check()
        if not self.ready.is_set():
            raise OSError('BLE 正在连接，请等待。')
        try:
            self._writes.put_nowait(bytes(data))
        except Full:
            self._fail('BLE 发送队列已满，请断开后重试。')
            self._check()
        return len(data)

    def close(self):
        # The UI observes closed before permitting a new session. Never blocks Qt.
        self._stop.set()

    def _received(self, sender, data):
        with self._lock:
            if len(self._buffer) + len(data) > self._max_buffer:
                self.error = 'BLE 接收缓存溢出，本次连接数据无效，请重新连接。'
            else:
                self._buffer.extend(data)

    def _disconnected(self, client):
        if not self._stop.is_set() and not self._closing:
            self._fail('BT24 蓝牙连接已断开，请检查供电、距离及其他设备占用。')

    async def _session(self):
        scanner, factory = self._scanner, self._client_factory
        if scanner is None or factory is None:
            from bleak import BleakClient, BleakScanner
            scanner, factory = BleakScanner, BleakClient
        # BT24 / Windows can take a moment to resume advertising after disconnect.
        # A bounded second scan avoids requiring another user click in that case.
        device = None
        for timeout in (6, 10):
            device = await asyncio.wait_for(scanner.find_device_by_address(self.address, timeout=timeout), timeout+4)
            if device is not None:break
        if device is None:
            raise OSError('没有发现所选 BT24，请检查供电，并断开手机/其他软件的连接。')
        client = factory(device, disconnected_callback=self._disconnected, timeout=15)
        try:
            await asyncio.wait_for(client.connect(), 20)
            notify, write = uart_characteristics(client.services)
            await asyncio.wait_for(client.start_notify(notify, self._received), 8)
            self.ready.set()
            while not self._stop.is_set():
                self._check()
                if not client.is_connected:
                    raise OSError('BT24 蓝牙连接已断开。')
                try:
                    data = self._writes.get_nowait()
                except Empty:
                    await asyncio.sleep(.01)
                    continue
                response = 'write' in write.properties
                size = 512 if response else write.max_write_without_response_size
                for start in range(0, len(data), size):
                    await asyncio.wait_for(client.write_gatt_char(write, data[start:start+size], response=response), 5)
        except Exception as exc:
            self._fail(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            self._closing = True
            self.ready.clear()
            # Stop may cancel _session while it is ALREADY releasing WinRT GATT.
            # Always finish that release before closing the loop/thread, otherwise
            # Windows can retain the peripheral and subsequent scans miss BT24.
            cleanup = asyncio.create_task(asyncio.wait_for(client.disconnect(), 8))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def _main(self):
        task = asyncio.create_task(self._session())
        try:
            while not task.done() and not self._stop.is_set():
                await asyncio.sleep(.02)
            if not task.done():
                task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as exc:
            self._fail(f'{type(exc).__name__}: {exc}')
        finally:
            self.ready.clear()
            self.closed.set()
