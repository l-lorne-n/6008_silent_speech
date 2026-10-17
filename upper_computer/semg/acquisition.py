"""Single I/O owner: serial, decoder and disk; GUI receives disposable previews."""
import math
import queue
import random
import struct
import threading
import time

import serial

from .protocol import ACK, DATA, INFO, Continuity, Decoder, LegacyDecoder, encode
from .session import Recorder


class Simulator:
    def __init__(self, channels=1, fs=1000):
        self.channels = channels
        self.fs = fs
        self.start = time.perf_counter()
        self.first = 0
        self.seq = 0
        self.pending = bytearray()
        self.phase = "rest"
        self.word = "open"
        self.rng = random.Random(7)
        self.write(b"I\n")

    def packet(self, kind, payload, count=0):
        self.pending.extend(encode(kind, self.seq, self.first, payload, count, version=self.channels))
        self.seq += 1

    def write(self, data):
        for line in data.splitlines():
            if line == b"I":
                self.packet(INFO, struct.pack("<IHHI", self.fs, self.channels, 16, 0))
            elif line.startswith(b"M,"):
                self.packet(ACK, struct.pack("<II", int(line[2:]), int((time.perf_counter() - self.start) * 1000)))

    def read(self, _size):
        due = int((time.perf_counter() - self.start) * self.fs)
        while self.first + 20 <= due:
            # Clearly synthetic; this is not evidence of word separability.
            amp = 420 if self.phase == "action" else 12
            freq = {"open": 83, "close": 117, "next": 163, "music": 211}.get(
                self.word, 70 + sum(self.word.encode("utf-8")) % 230)
            values = [int(32768 + amp * math.sin(2 * math.pi * freq * (self.first + j) / self.fs)
                          + self.rng.gauss(0, 8)) for j in range(20)]
            if self.channels == 2:
                values = [v for j,a in enumerate(values) for v in (a, int(32768 + .6*amp*math.sin(2*math.pi*(freq+31)*(self.first+j)/self.fs)+self.rng.gauss(0,8)))]
            self.packet(DATA, struct.pack(f"<{len(values)}H", *values), 20)
            self.first += 20
        data = bytes(self.pending)
        self.pending.clear()
        if not data:
            time.sleep(.005)
        return data

    def close(self):
        pass


class Acquisition(threading.Thread):
    def __init__(self, mode, port, root, channels=None, baudrate=230400, simulation_fs=2000):
        super().__init__(daemon=True)
        self.mode, self.port, self.root = mode, port, root
        self.baudrate = baudrate
        self.simulation_fs = simulation_fs
        from .channels import validate_channels
        self.selected_channels = validate_channels(channels)
        self.wire_version = None
        self.commands = queue.Queue(maxsize=1024)
        self.messages = queue.Queue()
        self.preview = queue.Queue(maxsize=8)
        self.decoder = LegacyDecoder() if mode == "legacy" else Decoder()
        self.continuity = Continuity()
        self.rec = None
        self.source = None
        self.last_trial = None
        self.info = None
        self.token = 0
        self.pending = {}
        self.stats = dict(samples=0, missing=0, crc_errors=0, bad_lines=0,
                          device_dropped=0, marker_timeouts=0, marker_rtt_ms=None,
                          frame_gaps=0, rate_hz=0)
        self.ready = False
        self.ack_seen = False
        self.abort_reason = None
        self.previous_seq = None
        self.batch = 0
        self.recognizer = None

    def command(self, kind, **kwargs):
        self.commands.put_nowait(dict(kind=kind, **kwargs))

    def notify(self, kind, **kwargs):
        self.messages.put(dict(kind=kind, **kwargs))

    def marker(self, event_id):
        if self.mode == "legacy":
            return
        self.token += 1
        sent = time.perf_counter_ns()
        self.pending[self.token] = (event_id, sent)
        self.source.write(f"M,{self.token}\n".encode("ascii"))
        if self.rec:
            self.rec.json("events.jsonl", dict(kind="marker_sent", event_id=event_id,
                                              token=self.token, host_ns=sent))

    def close_recording(self, status, error=None):
        if self.rec:
            path = str(self.rec.path)
            rec, self.rec = self.rec, None
            try:
                if error:
                    rec.json("events.jsonl", dict(kind="acquisition_error", host_ns=time.perf_counter_ns(), error=error))
                    if self.last_trial:
                        rec.json("trials.jsonl", dict(self.last_trial, kind="trial_result",
                                 status="interrupted", reason=error, host_ns=time.perf_counter_ns()))
            finally:
                rec.close(status, dict(self.stats), error)
            self.notify("saved", path=path, status=status)
            self.last_trial = None

    def run(self):
        try:
            if self.mode == "simulation":
                self.source = Simulator(2 if 1 in self.selected_channels else 1, self.simulation_fs)
            else:
                self.source = serial.Serial()
                self.source.port = self.port
                self.source.baudrate = self.baudrate
                self.source.timeout = .01
                self.source.write_timeout = .2
                self.source.dtr = False
                self.source.rts = False
                self.source.open()
            if self.mode != "legacy":
                self.source.write(b"I\n")
            self.notify("connected")
            last_data = time.perf_counter()
            connected_at = last_data
            last_status = last_data
            rate_time, rate_count = last_data, 0
            last_sync = last_data - 3
            stop_at = None
            stop_status = None
            while True:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                for _ in range(100):
                    try:
                        cmd = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    kind = cmd["kind"]
                    if kind == "recognition_start":
                        try:
                            if self.mode != "binary" or not self.ready or self.info['sampling_rate_hz'] != 1000:
                                raise ValueError("在线识别需要已就绪的真实1 kHz完整采样连接")
                            from .streaming import Recognizer, load_model
                            self.recognizer = Recognizer(load_model(cmd['model_path']))
                            if not set(self.recognizer.channels).issubset(self.selected_channels):
                                raise ValueError('在线模型要求的通道未在主界面启用')
                            self.notify('recognition_started')
                        except Exception as exc:
                            self.recognizer = None
                            self.notify('recognition_error', error=str(exc))
                        continue
                    if kind == "recognition_stop":
                        self.recognizer = None
                        self.notify('recognition_stopped')
                        continue
                    if kind == "disconnect":
                        if self.rec:
                            self.close_recording("interrupted", "connection_closed")
                        return
                    if kind == "start":
                        if not self.ready or self.rec:
                            raise RuntimeError("采集尚未就绪或已有记录")
                        config = dict(cmd['config'])
                        if self.mode != 'legacy':
                            from .filtering import display_filter_metadata
                            config['sampling_rate_hz'] = self.info['sampling_rate_hz']
                            config['display_filter'] = display_filter_metadata(self.mode, config['sampling_rate_hz'])
                        config['serial_baudrate'] = self.baudrate
                        self.rec = Recorder(self.root, dict(config, device_info=self.info, selected_channels=self.selected_channels,
                                                           connection_stats_at_start=dict(self.stats)))
                        self.rec.files["transport.bin"].write(bytes(self.decoder.buffer))
                        self.notify("recording", path=str(self.rec.path))
                    elif kind == "event":
                        e = cmd["event"]
                        if self.rec:
                            self.rec.json("events.jsonl", e)
                            if e["kind"] == "trial_result":
                                self.rec.json("trials.jsonl", e)
                                self.last_trial = None
                            elif e.get("trial_id") and e["kind"] == "phase" and e["phase"] in ("prepare", "action", "rest"):
                                self.last_trial = {k: e[k] for k in ("trial_id", "word", "label_id", "block", "repeat_of")}
                        if e["kind"] == "phase":
                            if self.mode == "simulation":
                                self.source.phase = e["phase"]
                                self.source.word = e.get("word", "open")
                            self.marker(e["event_id"])
                    elif kind == "stop":
                        # Drain final marker ACKs and the short device/USB backlog.
                        stop_at = time.perf_counter() + .75
                        stop_status = cmd.get("status", "stopped")
                now = time.perf_counter()
                if self.mode != "legacy" and now - last_sync >= 2:
                    self.source.write(b"I\n")
                    self.marker(f"sync_{self.token + 1}")
                    last_sync = now
                for token, (event_id, sent) in list(self.pending.items()):
                    if time.perf_counter_ns() - sent > 1_000_000_000:
                        self.stats["marker_timeouts"] += 1
                        del self.pending[token]
                        if self.rec:
                            self.rec.json("events.jsonl", dict(kind="marker_timeout", event_id=event_id, token=token,
                                                              host_ns=time.perf_counter_ns()))
                            if not event_id.startswith("sync_"):
                                raise RuntimeError("提示事件未收到设备应答，当前批次中断；检查 UART RX 接线")
                size = min(max(getattr(self.source, "in_waiting", 1), 1), 8192)
                data = self.source.read(size)
                rx_ns = time.perf_counter_ns()
                if data:
                    self.batch += 1
                    if self.rec:
                        self.rec.files["transport.bin"].write(data)
                        self.rec.json("frames.jsonl", dict(kind="rx", batch_id=self.batch, host_ns=rx_ns, bytes=len(data)))
                    valid_samples = self.consume(data, rx_ns)
                    if valid_samples:
                        last_data = now
                        self.stats["samples"] += valid_samples
                        if not self.ready and (self.mode == "legacy" or (self.info and self.ack_seen)):
                            self.ready = True
                            self.notify("ready")
                if now - last_data > 3:
                    raise RuntimeError("连续 3 秒未收到有效采样。检查端口、固件模式和连接；当前试次已中断。")
                if not self.ready and now - connected_at > 5:
                    raise RuntimeError("收到采样但双向握手未完成。检查完整采样固件及 USB 串口 TX → PB15(RX) 接线。")
                if stop_at and now >= stop_at:
                    self.close_recording(stop_status)
                    stop_at = None
                if now - rate_time >= 1:
                    self.stats["rate_hz"] = round((self.stats["samples"] - rate_count) / (now - rate_time), 1)
                    rate_time, rate_count = now, self.stats["samples"]
                if now - last_status >= .25:
                    self.stats["missing"] = self.continuity.missing
                    self.stats["crc_errors"] = getattr(self.decoder, "crc_errors", 0)
                    self.stats["bad_lines"] = getattr(self.decoder, "bad_lines", 0)
                    self.notify("stats", stats=dict(self.stats))
                    if self.recognizer:
                        self.notify('recognition_status', status=self.recognizer.detector.status())
                    last_status = now
        except Exception as exc:
            error = str(exc)
            try:
                self.close_recording("error", error)
            except Exception as save_exc:
                error += f"；保存失败：{save_exc}（请检查已有文件）"
            self.notify("error", error=error)
        finally:
            self.recognizer = None
            if self.source:
                self.source.close()
            self.notify("disconnected")

    def samples(self, values, first, rx_ns):
        if self.rec:
            self.rec.samples(values, first, self.batch, rx_ns)
        # Consume every decoded sample before the intentionally lossy preview queue.
        # Recognition failures must not abort or corrupt an ongoing raw recording.
        if self.recognizer and values and first is not None:
            try:
                physical = [[v[0], v[3]] if len(v) > 3 else [v[0]] for v in values]
                selected = [[v[c] for c in self.recognizer.channels] for v in physical]
                for event in self.recognizer.feed(selected, first):
                    self.notify('recognition_event', event=event, rx_ns=rx_ns)
            except Exception as exc:
                self.recognizer = None
                self.notify('recognition_error', error=str(exc))
        try:
            self.preview.put_nowait(dict(rows=values, first=first, rx_ns=rx_ns))
        except queue.Full:
            # Preview only; recording has already received every decoded sample.
            pass

    def consume(self, data, rx_ns):
        if self.mode == "legacy":
            rows = self.decoder.feed(data)
            self.stats["bad_lines"] = self.decoder.bad_lines
            self.samples(rows, None, rx_ns)
            return len(rows)
        n = 0
        frames = self.decoder.feed(data)
        self.stats["crc_errors"] = self.decoder.crc_errors
        for frame in frames:
            if self.wire_version is not None and self.wire_version != frame.version:
                raise RuntimeError('连接中协议版本发生变化，请重新连接')
            self.wire_version = frame.version
            if self.previous_seq is not None:
                delta = (frame.seq - self.previous_seq) & 0xFFFFFFFF
                if not 0 < delta < 0x80000000:
                    raise RuntimeError("设备帧号回退或重复，可能已重启。请重新连接并新建批次。")
                self.stats["frame_gaps"] += delta - 1
            self.previous_seq = frame.seq
            if self.rec:
                self.rec.json("frames.jsonl", dict(kind="frame", type=frame.kind, seq=frame.seq, first=frame.first,
                         count=frame.count, flags=frame.flags, rx_batch_id=self.batch, host_ns=rx_ns))
            if frame.flags:
                raise RuntimeError(f"设备报告采集/队列错误，flags={frame.flags}；停止本批次")
            if frame.kind == DATA:
                gap = self.continuity.accept(frame.first, frame.count)
                self.stats["missing"] = self.continuity.missing
                if gap and self.rec:
                    raise RuntimeError(f"采样序号缺失 {gap} 点；停止本批次以免混入不完整试次")
                channels = frame.version
                raw = struct.unpack(f"<{frame.count*channels}H", frame.payload)
                rows = [(v, '', '') for v in raw] if channels == 1 else [(raw[i], '', '', raw[i+1]) for i in range(0,len(raw),2)]
                self.samples(rows, frame.first, rx_ns)
                n += frame.count
            elif frame.kind == INFO:
                fs, channels, bits, dropped = struct.unpack("<IHHI", frame.payload)
                if fs not in (1000, 2000) or bits != 16 or channels != frame.version:
                    raise RuntimeError(f"不支持的设备参数：{fs} Hz / {channels} 通道 / {bits} 位")
                if max(self.selected_channels) >= channels:
                    raise RuntimeError('已勾选CH1，但设备只提供CH0；请烧录双通道v2固件或只选CH0')
                info = dict(sampling_rate_hz=fs, channels=channels, adc_bits=bits, protocol_version=frame.version)
                if self.info is not None and self.info != info:
                    raise RuntimeError('连接中采样率或设备参数发生变化，请重新连接并新建批次')
                self.info = info
                self.stats["device_dropped"] = dropped
            elif frame.kind == ACK:
                token, tick_ms = struct.unpack("<II", frame.payload)
                event_id, sent = self.pending.pop(token, (None, None))
                if sent is not None:
                    self.ack_seen = True
                    rtt = (rx_ns - sent) / 1e6
                    self.stats["marker_rtt_ms"] = round(rtt, 2)
                    if self.rec:
                        self.rec.json("events.jsonl", dict(kind="marker_ack", event_id=event_id, token=token,
                            host_ns=rx_ns, sent_host_ns=sent, device_sample_idx=frame.first,
                            device_tick_ms=tick_ms, rtt_ms=rtt, marker_quantization_samples=20,
                            marker_quantization_ms=20000/self.info['sampling_rate_hz'] if self.info else None))
        return n
