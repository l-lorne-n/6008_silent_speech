import binascii
import json
import struct

import pytest

from semg.protocol import ACK, DATA, INFO, Continuity, Decoder, LegacyDecoder, encode
from semg.session import Recorder, TrialEngine, schedule
from semg.acquisition import Acquisition


def test_fragmentation_crc_and_recovery():
    good = encode(DATA, 3, 80, struct.pack("<3H", 1, 32768, 65535), 3)
    broken = bytearray(good); broken[20] ^= 0x80
    stream = b"noise" + broken + b"junk" + good
    decoder = Decoder(); frames = []
    for byte in stream:
        frames += decoder.feed(bytes([byte]))
    assert len(frames) == 1
    assert frames[0].first == 80
    assert decoder.crc_errors == 1
    assert struct.unpack("<3H", frames[0].payload) == (1, 32768, 65535)
    assert binascii.crc_hqx(b"123456789", 0xFFFF) == 0x29B1


def test_reject_bad_frame_shape():
    decoder = Decoder()
    assert decoder.feed(encode(DATA, 0, 0, b"\x00\x00", 2)) == []
    assert len(decoder.feed(encode(INFO, 1, 0, struct.pack("<IHHI", 1000, 1, 16, 0)))) == 1


def test_continuity_gap_reset_and_wrap():
    c = Continuity()
    assert c.accept(100, 20) == 0
    assert c.accept(140, 20) == 20
    assert c.missing == 20
    with pytest.raises(ValueError): c.accept(0, 20)
    c = Continuity(); c.accept(0xFFFFFFF0, 16)
    assert c.accept(0, 20) == 0


def test_legacy_split_lines_and_range():
    d = LegacyDecoder()
    assert d.feed(b"33900,-") == []
    assert d.feed(b"2,3\r\n65536,2,3\nno\n") == [(33900, -2, 3)]
    assert d.bad_lines == 2


def test_balanced_schedule_reproducible():
    s = schedule(10, 42)
    assert s == schedule(10, 42)
    assert len({t["trial_id"] for t in s}) == 40
    for i in range(0, 40, 4):
        assert {t["word"] for t in s[i:i + 4]} == {"open", "close", "next", "music"}


def engine():
    events = []
    e = TrialEngine(schedule(1, 4), dict(baseline=1, prepare=1, action=1, rest=1), events.append)
    return e, events


def test_delayed_gui_never_skips_action():
    e, events = engine(); e.start(0)
    e.tick(5_000_000_000)
    assert e.phase == "prepare"
    assert e.deadline == 6_000_000_000
    e.tick(9_000_000_000)
    assert e.phase == "action" and e.deadline == 10_000_000_000


def test_pause_interrupts_requeues_once_and_reject_keeps_original():
    e, events = engine(); e.start(0); e.tick(1_000_000_000); e.tick(2_000_000_000)
    e.reject(2_100_000_000); e.reject(2_200_000_000)
    assert len(e.trials) == 5
    e.pause(2_300_000_000)
    assert len(e.trials) == 5 and e.phase == "paused"
    e.resume(3_000_000_000)
    assert e.current["trial_id"] == 2
    assert [x for x in events if x["kind"] == "trial_result"][0]["status"] == "interrupted"
    assert e.trials[-1]["repeat_of"] == 1


def test_recorder_legacy_index_blank_and_metadata(tmp_path):
    config = dict(subject_id="S01", session_id=1, mode="legacy")
    r = Recorder(tmp_path, config)
    r.samples([(33000, -5, 3)], None, 1, 123)
    r.close("completed", {})
    meta = json.loads((r.path / "metadata.json").read_text())
    assert meta["samples_saved"] == 1 and meta["timing"] == "legacy_unaligned"
    assert (r.path / "samples.csv").read_text().splitlines()[1].split(",")[1] == ""
    assert meta["articulation_mode"] == "closed_lip"
    with pytest.raises(ValueError): Recorder(tmp_path, dict(config, subject_id="../../oops"))


def test_acquisition_refuses_reset_and_device_fault(tmp_path):
    w = Acquisition("binary", "", tmp_path)
    w.consume(encode(DATA, 10, 100, struct.pack("<2H", 33000, 33001), 2), 123)
    with pytest.raises(RuntimeError, match="帧号"):
        w.consume(encode(DATA, 0, 0, struct.pack("<2H", 1, 2), 2), 124)
    w = Acquisition("binary", "", tmp_path)
    with pytest.raises(RuntimeError, match="flags"):
        w.consume(encode(DATA, 0, 0, struct.pack("<2H", 1, 2), 2, flags=1), 123)


def test_gap_during_recording_is_error_not_compressed_time(tmp_path):
    w = Acquisition("binary", "", tmp_path)
    w.rec = Recorder(tmp_path, dict(subject_id="S01", session_id=1, mode="binary"))
    path = w.rec.path
    w.consume(encode(DATA, 0, 100, struct.pack("<2H", 1, 2), 2), 123)
    with pytest.raises(RuntimeError, match="缺失 2"):
        w.consume(encode(DATA, 1, 104, struct.pack("<2H", 3, 4), 2), 125)
    w.close_recording("error", "missing_samples")
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["stats"]["missing"] == 2 and meta["samples_saved"] == 2


def test_marker_ack_has_device_index_and_separate_host_clock(tmp_path):
    w = Acquisition("binary", "", tmp_path)
    w.rec = Recorder(tmp_path, dict(subject_id="S01", session_id=1, mode="binary"))
    path = w.rec.path
    w.pending[7] = ("event_3", 1_000_000)
    w.consume(encode(ACK, 0, 800, struct.pack("<II", 7, 801)), 4_000_000)
    w.close_recording("completed")
    e = json.loads((path / "events.jsonl").read_text())
    assert e["device_sample_idx"] == 800 and e["host_ns"] == 4_000_000
    assert e["rtt_ms"] == 3 and e["marker_quantization_samples"] == 20
    assert w.ack_seen
