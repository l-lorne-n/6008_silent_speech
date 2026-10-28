import numpy as np
import pytest

from semg.rhythm import inspect_quiet, QuietRhythmMonitor
from semg.acquisition import Acquisition


def pulses(times=None):
    t = np.arange(10000)/1000
    x = 32768 + np.random.default_rng(28).normal(0, 15, len(t))
    x += 120*np.sin(2*np.pi*50*t) + 100*np.sin(2*np.pi*100*t)
    for beat in np.arange(.6, 9.8, .75) if times is None else times:
        x += 1000*np.exp(-((t-beat)/.012)**2)-800*np.exp(-((t-beat-.028)/.013)**2)
    return x


def test_regular_pulses_and_inverted_electrodes():
    x = pulses(); original = x.copy()
    for signal in (x, 65536-x):
        result = inspect_quiet(signal)
        assert result.status == "detected"
        assert abs(result.rate-80) < 2
        assert result.amplitude > 100
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("kind", ["flat", "mains", "noise", "step", "clipped", "irregular", "single"])
def test_does_not_approve_obvious_non_pulses(kind):
    t=np.arange(10000)/1000
    rng=np.random.default_rng(31)
    signals = {
        "flat": np.full(10000, 32768.),
        "mains": 32768+2000*np.sin(2*np.pi*50*t)+2000*np.sin(2*np.pi*100*t),
        "noise": 32768+rng.normal(0, 500, 10000),
        "step": pulses()+np.where(t>5, 4000, 0),
        "clipped": np.where(np.arange(10000)==7000, 65535, pulses()),
        "irregular": pulses([.6, 1.1, 2.3, 2.85, 4.0, 4.5, 5.7, 6.2, 7.4, 8., 9.3]),
        "single": pulses([5.0]),
    }
    assert inspect_quiet(signals[kind]).status != "detected"


def test_preview_gap_and_stale_data_clear_detection():
    m=QuietRhythmMonitor(); x=pulses()
    m.feed(x, 200, 100_000_000_000, 100_000_000_000)
    assert m.evaluate(100_000_000_000).status == "detected"
    m.feed(x[:20], 10240, 100_100_000_000, 100_100_000_000)
    assert m.evaluate(100_100_000_000).status == "waiting"
    assert len(m.raw)==20
    m.feed(x, 200, 101_000_000_000, 101_000_000_000)
    assert m.evaluate(101_000_000_000).status == "detected"
    assert m.evaluate(103_000_000_000).status == "waiting"
    assert not m.raw


def test_preview_reports_sample_index_and_drop_does_not_join_windows(tmp_path):
    w=Acquisition("binary", "", tmp_path)
    for first in range(0, 200, 20):
        w.samples([(32768, "", "")]*20, first, first+1)
    previews=[]
    while not w.preview.empty(): previews.append(w.preview.get_nowait())
    assert len(previews)==8 and previews[0]['first']==0
    w.samples([(32768, "", "")]*20, 200, 201)
    assert w.preview.get_nowait()['first']==200  # missing preview 160..199 is detectable
