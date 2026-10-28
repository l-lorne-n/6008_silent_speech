"""Serial/BLE transport: filtering, inference and confirmation stay on STM32."""
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import uuid

UPPER = Path(__file__).resolve().parents[1] / 'upper_computer'
sys.path.insert(0, str(UPPER))
from semg.board_protocol import BoardDecoder, CLASSES, result, model_info, envelope
from semg.board_models import NAMES, model_source
from semg.serial_settings import DEFAULT_BAUDRATE, validate_baudrate
from semg.ble_transport import BleByteStream, scan_bt24


class BoardBridge:
    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.RLock()
        self.lifecycle = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.ble_devices = {}
        self.state = dict(connected=False, verified=False, enabled=False, port='', error='', info=None,
                          connecting=False, transport='serial', rx_bytes=0,
                          baudrate=DEFAULT_BAUDRATE, latest=None, last_command=None, crc_errors=0, sequence_gaps=0)
        self.last_rx = 0

    def snapshot(self):
        with self.lock:
            return {**self.state, 'stale': not self.last_rx or time.monotonic() - self.last_rx > 1}

    @staticmethod
    def ports():
        try:
            from serial.tools import list_ports
            return [dict(device=p.device, description=p.description) for p in list_ports.comports()]
        except ImportError:
            raise ValueError('缺少 pyserial，请用 app/start.cmd 启动（复用上位机 Python）。')

    def publish(self):
        self.hub.publish('board', self.snapshot())

    def scan_ble(self):
        with self.lifecycle:
            if self.thread and self.thread.is_alive():
                raise ValueError('请先断开当前设备再扫描 BLE。')
            try:
                devices = asyncio.run(scan_bt24())
            except Exception as exc:
                raise ValueError(f'BLE 扫描失败：{type(exc).__name__}: {exc}') from exc
            self.ble_devices = {d['address']: d for d in devices}
            return [dict(device=d['address'], description=f'{d["name"]} · {d["rssi"]} dBm') for d in devices]

    def connect(self, port, baudrate=DEFAULT_BAUDRATE, transport='serial'):
        validate_baudrate(baudrate)
        if transport not in ('serial', 'ble'):
            raise ValueError('transport 必须为 serial 或 ble。')
        available = self.ble_devices if transport == 'ble' else [p['device'] for p in self.ports()]
        if not isinstance(port, str) or port not in available:
            raise ValueError('请先扫描并选择 BT24。' if transport == 'ble' else '请选择当前可用的串口。')
        with self.lifecycle:
            self._disconnect()
            self.stop.clear()
            with self.lock:
                self.state.update(connected=False, verified=False, enabled=True, port=port, baudrate=baudrate, error='',
                                  connecting=True, transport=transport, rx_bytes=0,
                                  info=None, latest=None, last_command=None, crc_errors=0, sequence_gaps=0)
                self.last_rx = 0
            args = (port, baudrate, transport) if transport == 'ble' else (port, baudrate)
            self.thread = threading.Thread(target=self._run, args=args, daemon=True)
            self.thread.start()
        return self.snapshot()

    def _disconnect(self):
        self.stop.set()
        with self.lock:
            self.state['enabled'] = False
        if self.thread:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise ValueError('设备仍在关闭，请稍后重试。')
            self.thread = None
        with self.lock:
            self.state.update(connected=False, connecting=False, verified=False, enabled=False, latest=None)
            self.last_rx = 0

    def disconnect(self):
        with self.lifecycle:
            self._disconnect()
        self.publish()
        return self.snapshot()

    def enable(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be boolean')
        with self.lock:
            if not self.state['verified'] or not self.state['connected'] or self.snapshot()['stale']:
                raise ValueError('请先连接并等待板端模型验证和有效推理数据。')
            if enabled and self.state['latest'] and self.state['latest']['flags']:
                raise ValueError('板端存在故障，禁止启用控制；UART 故障 4 需要复位 STM32 后重新验证。')
            self.state['enabled'] = enabled
        self.publish()
        return self.snapshot()

    @staticmethod
    def verify(info):
        # Bind to the actual frozen graph, never accept an arbitrary serial model.
        name = info.get('model')
        if name not in NAMES or info.get('mode') != 1:
            raise ValueError('板端不是受支持的实时监测模型，请先在上位机烧录监测固件。')
        folder = model_source(name, UPPER)
        model = folder / 'model_int8_qdq.onnx'
        if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest()[:16] != info.get('sha'):
            raise ValueError('板端模型校验不一致，已禁止执行命令。')

    def accept_result(self, packet, session):
        r = result(packet)
        with self.lock:
            previous = self.state['latest']
            if previous and r['end'] <= previous['end']:
                # Reboot / replay needs a new INFO handshake before any command.
                self.state.update(verified=False, enabled=False, error='板端时间回退或复位，正在重新验证；通过后请手动启用控制。')
            self.last_rx = time.monotonic()
            self.state['latest'] = r
            if (self.state['verified'] and self.state['enabled'] and r['mode'] == 1
                    and not r['flags'] and r['emitted'] != 255):
                index = r['emitted']
                event = dict(word=CLASSES[index], confidence=r['probabilities'][index], source='semg',
                             id=f'board-{session}-{r["end"]}', board=True, sent_at=time.time() * 1000)
                self.state['last_command'] = event
                self.hub.publish('command', event)

    def _run(self, port, baudrate=DEFAULT_BAUDRATE, transport='serial'):
        serial_port = None
        session = uuid.uuid4().hex
        decoder = BoardDecoder()
        points = deque(maxlen=501)
        last_seq = None
        started = time.monotonic()
        last_publish = started
        last_info_request = started
        text_probe = bytearray()
        try:
            if transport == 'ble':
                serial_port = BleByteStream(port)
                deadline = time.monotonic()+55
                while not serial_port.ready.is_set():
                    serial_port._check()
                    if self.stop.wait(.05):return
                    if time.monotonic()>deadline:raise ValueError('BT24 连接超时，请重新扫描。')
            else:
                import serial
                serial_port = serial.Serial(port=None, baudrate=baudrate, timeout=.05, write_timeout=1)
                serial_port.dtr = False
                serial_port.rts = False
                serial_port.port = port
                serial_port.open()
                serial_port.reset_input_buffer()
            started = last_info_request = time.monotonic()
            serial_port.write(b'N\nI\n')
            with self.lock:
                self.state.update(connected=True, connecting=False)
            self.publish()
            while not self.stop.is_set():
                # Windows drivers can wait for a full large read despite short
                # timeouts. Drain available bytes; block only for one new byte.
                data = serial_port.read(max(1, min(4096, serial_port.in_waiting)))
                with self.lock:self.state['rx_bytes'] += len(data)
                if transport == 'ble' and not self.state['verified']:
                    text_probe.extend(data)
                    while b'\n' in text_probe:
                        line,_,rest = text_probe.partition(b'\n');text_probe = bytearray(rest)
                        try:values = [int(v) for v in line.strip().split(b',')]
                        except ValueError:continue
                        if len(values)==6 and all(0<=v<=65535 for v in values[:2]):
                            raise ValueError('BT24 已收到六列 CSV：板端仍是测试固件，请烧录 CNN 实时监测固件后连接。')
                    if len(text_probe)>4096:text_probe.clear()
                for packet in decoder.feed(data):
                    if last_seq is not None and packet.seq != (last_seq + 1) & 0xffffffff:
                        with self.lock:
                            self.state['sequence_gaps'] += 1
                    last_seq = packet.seq
                    if packet.kind == 5:
                        info = model_info(packet)
                        self.verify(info)
                        with self.lock:
                            self.state.update(info=info, verified=True, error='')
                    elif packet.kind == 4:
                        self.accept_result(packet, session)
                    elif packet.kind == 8:
                        values = envelope(packet)
                        if points and packet.end - points[-1][0] != 20:
                            points.clear()
                        points.append([packet.end, *values])
                now = time.monotonic()
                with self.lock:
                    self.state['crc_errors'] = decoder.crc_errors
                if now - last_publish >= .1:
                    self.publish()
                    if points:
                        self.hub.publish('envelope', {'points': list(points), 'sent_at': time.time()*1000})
                        # The browser retains the five-second history, transport sends batches.
                        points.clear()
                    last_publish = now
                if not self.state['verified'] and now - last_info_request > 1:
                    serial_port.write(b'I\n')
                    last_info_request = now
                if now - (self.last_rx or started) > 4:
                    raise ValueError(f'未收到有效推理数据。请检查连接，并确认板端监测固件及串口链路均为 {baudrate} 波特率，然后重新连接。')
                if transport == 'ble' and not data:self.stop.wait(.005)
        except Exception as exc:
            with self.lock:
                self.state['error'] = str(exc)
        finally:
            if serial_port:
                serial_port.close()
                if transport == 'ble':serial_port.closed.wait(10)
            with self.lock:
                self.state.update(connected=False, connecting=False, verified=False, enabled=False, latest=None)
                self.last_rx = 0
            self.publish()
