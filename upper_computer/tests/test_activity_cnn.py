import numpy as np
import pytest
from semg.activity_cnn import ActivityGate, CausalRMSFrontend, GateConfig
from semg.lda import filter_sos


def replay(x, size):
    f = CausalRMSFrontend(filter_sos('notch50_bp500'))
    g = ActivityGate()
    events, samples = [], []
    for i in range(0, len(x), size):
        for frame in f.feed(x[i:i+size], i+7000):
            samples.append(frame['filtered'])
            event = g.feed(frame['rms'], frame['end'], frame['clipped'])
            if event:
                events.append(event)
    return events, np.concatenate(samples), g


def signal(seconds=10):
    rng = np.random.default_rng(44)
    x = 32768+rng.normal(0, 5, (seconds*2000, 2))
    # One word with a short internal gap must remain one event.
    for a,b in [(12000,12800),(13000,14000),(16600,18200)]:
        if a < len(x):
            b = min(b,len(x))
            x[a:b,0] += 500*np.sin(2*np.pi*90*np.arange(b-a)/2000)
    return x


def test_causal_chunk_invariance_and_word_gap():
    a,ya,_ = replay(signal(),40)
    b,yb,_ = replay(signal(),137)
    assert a == b and len(a) == 2
    np.testing.assert_allclose(ya,yb,rtol=0,atol=1e-9)
    assert all(e['kind']=='segment' and e['end']<=e['emitted'] for e in a)


def test_quiet_and_no_eof_flush():
    rng = np.random.default_rng(3)
    assert replay(32768+rng.normal(0,5,(20000,2)),1000)[0] == []
    events,_,gate = replay(signal(7),73)
    assert events == [] and gate.state == 'active'


def test_gap_and_clipping_are_not_commands():
    f = CausalRMSFrontend(filter_sos('notch50_bp500'))
    f.feed(signal()[:100],0)
    with pytest.raises(ValueError,match='gap'):
        f.feed(signal()[120:160],120)
    x = signal(8)
    x[13500,0] = 65535
    events,_,_ = replay(x,73)
    assert any(e.get('reason')=='clipping' for e in events)
    assert not any(e['kind']=='segment' for e in events)


def test_long_event_is_rejected():
    g = ActivityGate(GateConfig())
    events=[]
    for end in range(40,24001,40):
        e=g.feed([1000,5] if end>12000 else [5,5],end)
        if e:events.append(e)
    assert len(events)==1 and events[0]['reason']=='too_long'


def test_calibration_clip_guard_and_causal_prefix():
    g=ActivityGate()
    for end in range(40,10001,40):g.feed([5,5],end,clipped=end==4000)
    assert g.state=='idle' and g.calibration_excluded_frames==11
    x=signal()
    _,full,_=replay(x,2000)
    _,prefix,_=replay(x[:13600],137)
    np.testing.assert_allclose(prefix,full[:len(prefix)],rtol=0,atol=1e-9)
