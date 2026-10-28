import copy
import csv
import json
import struct

import numpy as np
import pytest

from semg import lda, streaming
from semg.acquisition import Acquisition
from semg.activity import ActivityDetector
from semg.protocol import Decoder, encode, DATA, INFO
from semg.session import Recorder, schedule
from export_trials import export


def dual_recording(folder, selected=None):
    folder.mkdir()
    channels=[0,1] if selected is None else selected
    meta=dict(mode='binary',sampling_rate_hz=1000,status='completed',subject_id='DUAL',
              channel_count=2,selected_channels=channels,words=['open','close'])
    (folder/'metadata.json').write_text(json.dumps(meta))
    rng=np.random.default_rng(194);raw=32768+rng.normal(0,4,(40000,2))
    events=[];trials=[]
    for i,trial in enumerate(schedule(5,91,['open','close'])):
        start=5000+i*3500;k=['open','close'].index(trial['word'])
        t=np.arange(900)/1000
        raw[start+1200:start+2100,0]+=(250+300*k)*np.sin(2*np.pi*(85+50*k)*t)
        raw[start+1200:start+2100,1]+=(600-250*k)*np.sin(2*np.pi*(140+50*k)*t)
        trials.append(dict(trial,status='completed'))
        for phase,offset in [('prepare',0),('action',1200),('rest',2100)]:
            eid=f"{trial['trial_id']}_{phase}"
            events += [dict(kind='phase',phase=phase,trial_id=trial['trial_id'],event_id=eid),
                       dict(kind='marker_ack',event_id=eid,device_sample_idx=start+offset,rtt_ms=2)]
    raw=np.rint(raw).astype(int)
    with (folder/'samples.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['device_sample_idx','raw_ch0','raw_ch1'])
        writer.writerows((i,*r) for i,r in enumerate(raw))
    for name,rows in [('events.jsonl',events),('trials.jsonl',trials)]:
        (folder/name).write_text('\n'.join(json.dumps(r) for r in rows))
    return folder


def test_dual_wire_fragmentation_channel_order_and_recorder(tmp_path):
    info=encode(INFO,0,0,struct.pack('<IHHI',1000,2,16,0),version=2)
    payload=struct.pack('<6H',100,40000,200,41000,300,42000)
    packet=encode(DATA,1,0,payload,3,version=2)
    decoder=Decoder();frames=[]
    for b in info+packet:frames+=decoder.feed(bytes([b]))
    assert frames[-1].version==2 and frames[-1].count==3
    assert frames[-1].payload==payload
    assert not Decoder().feed(encode(DATA,0,0,payload,6,version=2))
    a=Acquisition('binary','',tmp_path,[0,1]);a.consume(info,0)
    assert a.info['channels']==2
    a.rec=Recorder(tmp_path,dict(mode='binary',subject_id='T',session_id=1,selected_channels=[0,1],device_info=a.info))
    folder=a.rec.path;a.consume(packet,1);a.close_recording('completed')
    rows=list(csv.DictReader((folder/'samples.csv').open()))
    assert [int(r['raw_ch1']) for r in rows]==[40000,41000,42000]
    assert [int(r['raw_ch0']) for r in rows]==[100,200,300]
    assert [int(r['device_sample_idx']) for r in rows]==[0,1,2]
    assert json.loads((folder/'metadata.json').read_text())['selected_channels']==[0,1]
    with pytest.raises(RuntimeError,match='CH1'):
        Acquisition('binary','',tmp_path,[1]).consume(encode(INFO,0,0,struct.pack('<IHHI',1000,1,16,0)),0)
    with pytest.raises(RuntimeError,match='版本'):
        a.consume(encode(INFO,2,3,struct.pack('<IHHI',1000,1,16,0)),2)


@pytest.mark.parametrize('channels',[[0],[1],[0,1]])
def test_dual_lda_order_roundtrip_missing_and_overlap(tmp_path,channels):
    folder=dual_recording(tmp_path/'dual')
    run=lda.train([dict(path=str(folder))],tmp_path/'models',channels=channels,strategy='within_recording')
    model=lda.load_model(run['model_path']);report=run['report']
    assert len(model['scaler_mean'])==8*len(channels)
    assert model['channels']==channels
    two,_,_=lda.extract_recording(dict(path=str(folder)),'notch50',channels=[0,1])
    a,_,_=lda.extract_recording(dict(path=str(folder)),'notch50',channels=[0])
    b,_,_=lda.extract_recording(dict(path=str(folder)),'notch50',channels=[1])
    np.testing.assert_array_equal(two[0]['features'],a[0]['features']+b[0]['features'])
    assert a[0]['fingerprint']==b[0]['fingerprint']==two[0]['fingerprint']
    testing=[dict(path=str(folder),trial_ids=[r['trial_id'] for r in report['test']])]
    assert lda.evaluate(run['model_path'],testing,tmp_path/'models')['report']['metrics']==report['metrics']
    damaged=copy.deepcopy(model);damaged['channels']=[1] if channels==[0] else [0]
    with pytest.raises(ValueError):lda.validate_model(damaged)
    # A copied single-channel CSV cannot bypass overlap detection in a dual model.
    selected=dict(path=str(folder))
    with pytest.raises(ValueError,match='已用于训练'):lda.evaluate(run['model_path'],[selected],tmp_path/'models')
    meta=json.loads((folder/'metadata.json').read_text());meta['selected_channels']=[0]
    (folder/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='缺失通道'):
        lda.extract_recording(selected,'notch50',channels=[0,1])


def test_v3_four_feature_model_still_loads(tmp_path):
    folder=dual_recording(tmp_path/'dual')
    settings=dict(segments=6,metrics=['log_rms','log_mean_abs_diff','log_mav','mean_frequency_hz'])
    run=lda.train([dict(path=str(folder))],tmp_path/'models',feature_settings=settings)
    m=lda.load_model(run['model_path']);m['schema']='semg_lda_v3';m.pop('channels');m.pop('training_signal_fingerprints')
    assert len(lda.validate_model(m)['feature_names'])==24


def test_export_preserves_two_columns_and_selected_channel_identity(tmp_path):
    folder=dual_recording(tmp_path/'dual',selected=[1]);manifest=export(folder)
    assert all(r['exported'] for r in manifest)
    data=np.load(next((folder/'cue_windows').glob('*.npz')))
    assert data['raw'].shape==(900,1) and data['channels'].tolist()==[1]
    assert 'raw_ch0' not in data.files and 'raw_ch1' in data.files


def test_dual_activity_train_replay_and_chunk_invariance(tmp_path):
    folder=dual_recording(tmp_path/'dual');data=streaming.recording_stream(folder)
    events=[];detector=ActivityDetector(channels=2)
    for i in range(0,len(data),73):events+=detector.feed(data[i:i+73,1:],i)
    expected,_=streaming.detect_recording(data,detector.config)
    assert events==expected
    segments=[e for e in events if e['kind']=='segment']
    assert len(segments)==10 and all(len(e['features'])==16 for e in segments)
    run=streaming.train_activity(folder,tmp_path/'models');model=streaming.load_model(run['model_path'])
    assert model['channels']==[0,1] and len(model['mean'])==16
    replay=streaming.replay_activity(run['model_path'],folder,tmp_path/'models')
    assert replay['report']['includes_training_recording']
    assert len([e for e in replay['report']['events'] if e['kind']=='prediction'])==10
    a=Acquisition('binary','',tmp_path,[0,1]);a.recognizer=streaming.Recognizer(model)
    for _ in range(a.preview.maxsize):a.preview.put({})
    for i in range(0,len(data),20):a.samples([(r[1],'','',r[2]) for r in data[i:i+20]],i,0)
    messages=[]
    while not a.messages.empty():messages.append(a.messages.get_nowait())
    assert len([m for m in messages if m.get('event',{}).get('kind')=='prediction'])==10


def test_dual_activity_uses_either_channel_and_rejects_clipping():
    t=np.arange(10000)/1000;rng=np.random.default_rng(54)
    raw=32768+rng.normal(0,3,(10000,2))
    raw[6000:7000,1]+=500*np.sin(2*np.pi*90*t[6000:7000])
    events=ActivityDetector(channels=2).feed(raw,0)
    assert sum(e['kind']=='segment' for e in events)==1
    raw[6500,0]=65535
    events=ActivityDetector(channels=2).feed(raw,0)
    assert any(e.get('reason')=='clipping' for e in events)
    assert not any(e['kind']=='segment' for e in events)


def test_inactive_flat_channel_does_not_make_valid_trials_duplicates(tmp_path):
    folder=dual_recording(tmp_path/'dual',selected=[1])
    path=folder/'samples.csv'
    rows=list(csv.DictReader(path.open()))
    for row in rows:row['raw_ch0']='0'  # unused/unconnected input, saved but not trained
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['device_sample_idx','raw_ch0','raw_ch1'])
        writer.writeheader();writer.writerows(rows)
    run=lda.train([dict(path=str(folder))],tmp_path/'models',channels=[1])
    assert len(run['report']['train'])+len(run['report']['test'])==10


def test_original_activity_model_without_channels_is_single_channel():
    from test_activity import model
    old=model();old['schema']='semg_activity_lda_v1';old.pop('channels')
    assert streaming.Recognizer(old).channels==[0]


def test_legacy_recording_accepts_absent_device_handshake(tmp_path):
    rec=Recorder(tmp_path,dict(mode='legacy',subject_id='T',session_id=1,device_info=None))
    rec.samples([(30000,1,2)],None,1,0);rec.close('completed',{})
    assert rec.meta['channel_count']==1 and rec.meta['selected_channels']==[0]
