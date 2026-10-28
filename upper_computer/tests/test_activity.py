import json
import queue
import numpy as np
import pytest
from semg.activity import ActivityConfig, ActivityDetector
from semg.acquisition import Acquisition
from semg.streaming import Recognizer, classify, validate_model, SCHEMA, FEATURE_VERSION
from dataclasses import asdict


def signal(duration=10000, intervals=((6000,7000),)):
    rng=np.random.default_rng(44);x=32768+rng.normal(0,5,duration)
    for a,b in intervals:
        t=np.arange(b-a)/1000;x[a:b]+=500*np.sin(2*np.pi*90*t)
    return x


def feed_chunks(x,size=20):
    detector=ActivityDetector();events=[]
    for pos in range(0,len(x),size):events+=detector.feed(x[pos:pos+size],pos)
    return events,detector


def model():
    return dict(schema=SCHEMA,features=FEATURE_VERSION,detector=asdict(ActivityConfig()),channels=[0],
                classes=['open','close'],mean=[0.]*8,scale=[1.]*8,coef=[[0.]*8],intercept=[3.],min_probability=.6,min_margin=.2)


def test_quiet_does_not_emit_and_model_can_reject():
    events,d=feed_chunks(signal(intervals=()))
    assert [e['kind'] for e in events]==['calibrated'] and d.state=='idle'
    m=model();m['intercept']=[0.]
    assert classify(m,[1]*8)['word'] is None


def test_chunk_invariant_and_one_event_per_action():
    x=signal(intervals=((6000,6500),(6680,7050),(8200,8900)))
    a,_=feed_chunks(x,20);b,_=feed_chunks(x,73)
    seg=[e for e in a if e['kind']=='segment']
    assert len(seg)==2  # brief within-word pause must not cause another command
    assert len(a)==len(b)
    for aa,bb in zip(a,b):
        assert aa['kind']==bb['kind']
        assert aa['emitted']==bb['emitted']
        if aa['kind']=='segment':
            assert aa['start']==bb['start'] and aa['end']==bb['end']
            assert np.allclose(aa['features'],bb['features'])
            assert aa['emitted']>=aa['end']


def test_gap_cancels_inflight_and_recalibrates():
    x=signal();d=ActivityDetector();d.feed(x[:6500],0)
    assert d.state=='active'
    events=d.feed(x[6600:],6600)
    assert events[0]['reason']=='sample_gap'
    assert not any(e['kind']=='segment' for e in events)
    assert d.state=='calibrating'


def test_clipping_cancels_activity():
    x=signal();x[6500]=65535
    events,_=feed_chunks(x,71)
    assert any(e.get('reason')=='clipping' for e in events)
    assert not any(e['kind']=='segment' for e in events)


def test_long_activity_emits_no_command_and_waits_for_release():
    events,d=feed_chunks(signal(12000,((6000,11000),)))
    assert sum(e.get('reason')=='activity_too_long' for e in events)==1
    assert not any(e['kind']=='segment' for e in events)
    assert d.state=='idle'


def test_partial_activity_at_eof_not_forced_to_word():
    events,d=feed_chunks(signal(6600,((6000,6600),)))
    assert d.state=='active'
    assert not any(e['kind']=='segment' for e in events)


def test_release_wait_not_counted_as_active_duration():
    events,_=feed_chunks(signal(12000,((6000,8840),)))
    assert sum(e['kind']=='segment' for e in events)==1
    assert not any(e.get('reason')=='activity_too_long' for e in events)


def test_recorder_persists_explicit_mouthing_mode(tmp_path):
    from semg.session import Recorder
    r=Recorder(tmp_path,dict(subject_id='T01',session_id=1,mode='simulation',articulation_mode='silent_mouthing'))
    r.close('completed',{})
    meta=json.loads((r.path/'metadata.json').read_text(encoding='utf-8'))
    assert meta['articulation_mode']=='silent_mouthing'


def test_reject_legacy_model_and_invalid_settings():
    with pytest.raises(ValueError):validate_model({'schema':'semg_lda_v2'})
    with pytest.raises(ValueError):ActivityConfig(off_ratio=4,on_ratio=3)
    with pytest.raises(ValueError):ActivityConfig(release_ms=20,post_ms=150)
    m=model();m['scale']=[0.]*8
    with pytest.raises(ValueError):validate_model(m)


def test_acquisition_recognition_not_using_lossy_preview(tmp_path):
    a=Acquisition('binary','unused',tmp_path);a.recognizer=Recognizer(model())
    for _ in range(a.preview.maxsize):a.preview.put({})
    x=signal().astype(int)
    for pos in range(0,len(x),20):a.samples([(v,'','') for v in x[pos:pos+20]],pos,0)
    events=[]
    while not a.messages.empty():events.append(a.messages.get_nowait())
    words=[e for e in events if e.get('event',{}).get('kind')=='prediction']
    assert len(words)==1 and words[0]['event']['word']=='close'
    assert a.preview.qsize()==a.preview.maxsize


def test_classifier_failure_does_not_stop_raw_recording(tmp_path):
    class Broken:
        def feed(self,*args):raise ValueError('bad model')
    class Recorder:
        def __init__(self):self.count=0
        def samples(self,values,*args):self.count+=len(values)
    a=Acquisition('binary','unused',tmp_path);a.recognizer=Broken();a.rec=Recorder()
    a.samples([(32768,'','')]*20,0,0)
    assert a.rec.count==20 and a.recognizer is None
    assert a.messages.get_nowait()['kind']=='recognition_error'
