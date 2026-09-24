"""Wire framing shared with firmware/PROTOCOL.md; no GUI dependencies."""
from dataclasses import dataclass
import binascii
import struct

MAGIC = b"\xa5\x5a"
HEADER = struct.Struct("<2sBBHIIHH")
DATA, INFO, ACK = 1, 2, 3
MAX_PAYLOAD = 512


@dataclass
class Frame:
    kind: int
    seq: int
    first: int
    count: int
    flags: int
    payload: bytes


def encode(kind, seq, first, payload=b"", count=0, flags=0):
    body = HEADER.pack(MAGIC, 1, kind, len(payload), seq, first, count, flags) + payload
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


class Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.discarded = 0

    def feed(self, data):
        self.buffer.extend(data)
        out = []
        while len(self.buffer) >= 2:
            pos = self.buffer.find(MAGIC)
            if pos < 0:
                keep = int(self.buffer[-1] == MAGIC[0])
                self.discarded += len(self.buffer) - keep
                self.buffer[:] = self.buffer[-1:] if keep else b""
                break
            if pos:
                self.discarded += pos
                del self.buffer[:pos]
            if len(self.buffer) < HEADER.size:
                break
            _, version, kind, size, seq, first, count, flags = HEADER.unpack_from(self.buffer)
            valid = version == 1 and kind in (DATA, INFO, ACK) and size <= MAX_PAYLOAD
            valid &= (kind != DATA or (0 < count <= 256 and size == count * 2))
            valid &= (kind != INFO or (size == 12 and count == 0))
            valid &= (kind != ACK or (size == 8 and count == 0))
            if not valid:
                self.discarded += 1
                del self.buffer[0]
                continue
            total = HEADER.size + size + 2
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<H", self.buffer, total - 2)[0]
            if binascii.crc_hqx(self.buffer[:total - 2], 0xFFFF) != expected:
                self.crc_errors += 1
                del self.buffer[0]
                continue
            out.append(Frame(kind, seq, first, count, flags, bytes(self.buffer[HEADER.size:total - 2])))
            del self.buffer[:total]
        return out


class LegacyDecoder:
    def __init__(self):
        self.buffer = bytearray()
        self.bad_lines = 0

    def feed(self, data):
        self.buffer.extend(data)
        out = []
        while b"\n" in self.buffer:
            line, _, self.buffer = self.buffer.partition(b"\n")
            try:
                values = [int(v) for v in line.strip().split(b",")]
                if len(values) != 3 or not 0 <= values[0] <= 65535:
                    raise ValueError("invalid legacy line")
                out.append(tuple(values))
            except ValueError:
                self.bad_lines += 1
        if len(self.buffer) > 4096:
            self.bad_lines += 1
            self.buffer.clear()
        return out


class Continuity:
    """Detect restarts/out-of-order data. Do not renumber missing samples."""
    def __init__(self):
        self.expected = None
        self.missing = 0

    def accept(self, first, count):
        gap = 0
        if self.expected is not None:
            gap = (first - self.expected) & 0xFFFFFFFF
            if gap >= 0x80000000:
                raise ValueError("设备采样序号回退：可能重启或乱序，请新建采集批次")
            self.missing += gap
        self.expected = (first + count) & 0xFFFFFFFF
        return gap
