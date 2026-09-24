import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from semg import lda
from semg.session import WORDS


@pytest.fixture
def recording(tmp_path):
    folder=tmp_path/'recording';folder.mkdir()
    meta=dict(mode='binary',sampling_rate_hz=1000,status='completed',subject_id='A',session_id=1,condition='silent',articulation_mode='closed_lip')
    (folder/'metadata.json').write_text(json.dumps(meta))
    rng=np.random.default_rng(99);events=[];trials=[];samples=[]
    for n in range(16):
        tid=n+1;word=WORDS[n%4];start=n*2500
        t=np.arange(2500)/1000
        amplitude=np.where(t>=1.2,100+150*(n%4),20)
        raw=np.rint(32768+amplitude*np.sin(2*np.pi*(70+30*(n%4))*t)+rng.normal(0,7,len(t))).astype(int)
        samples.extend(zip(start+np.arange(2500),raw))
        for phase,offset in [('prepare',0),('action',1200),('rest',2400)]:
            event_id=f'{tid}_{phase}'
            events.extend([dict(kind='phase',phase=phase,trial_id=tid,event_id=event_id),dict(kind='marker_ack',event_id=event_id,device_sample_idx=start+offset,rtt_ms=5)])
        trials.append(dict(status='completed',trial_id=tid,word=word,block=n//4+1))
    with (folder/'samples.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['device_sample_idx','raw_ch0']);writer.writerows(samples)
    for name,rows in [('events.jsonl',events),('trials.jsonl',trials)]:
        (folder/name).write_text('\n'.join(json.dumps(r) for r in rows))
    return folder


def test_extraction_train_only_scaler_roundtrip_and_overlap(recording,tmp_path):
    selected=[dict(path=str(recording))]
    result=lda.train(selected,tmp_path/'output',train_percent=50,seed=4)
    report=result['report'];model=lda.load_model(result['model_path'])
    assert len(report['train'])==8 and len(report['test'])==8
    assert set(t['block'] for t in report['train']).isdisjoint(t['block'] for t in report['test'])
    np.testing.assert_allclose(model['scaler_mean'],np.array([t['features'] for t in report['train']]).mean(axis=0))
    tested=lda.evaluate(result['model_path'],[dict(path=str(recording),trial_ids=[r['trial_id'] for r in report['test']])],tmp_path/'output')
    assert tested['report']['metrics']==report['metrics']
    with pytest.raises(ValueError,match='已用于训练'):
        lda.evaluate(result['model_path'],selected,tmp_path/'output')
    # Folder copies/renames cannot bypass overlap checks.
    import shutil
    copied=tmp_path/'copy';shutil.copytree(recording,copied)
    with pytest.raises(ValueError,match='已用于训练'):
        lda.evaluate(result['model_path'],[dict(path=str(copied))],tmp_path/'output')
    with pytest.raises(ValueError,match='相同的原始试次'):
        lda.extract(selected+[dict(path=str(copied))],'notch50')


def test_catalog_excludes_legacy_simulated_and_live(recording):
    catalog,errors=lda.scan_recordings(recording)
    assert not errors and len(catalog[0]['trials'])==16 and not catalog[0]['disabled_reason']
    meta_path=recording/'metadata.json';meta=json.loads(meta_path.read_text())
    for mode,status in [('legacy','completed'),('simulation','completed'),('binary','recording')]:
        meta_path.write_text(json.dumps(dict(meta,mode=mode,status=status)))
        catalog,_=lda.scan_recordings(recording)
        assert catalog[0]['disabled_reason']
        with pytest.raises(ValueError):lda.extract_recording(dict(path=str(recording)),'notch50')


def test_reject_gap_and_report_clipped_trial(recording):
    path=recording/'samples.csv';rows=list(csv.reader(path.open()))
    original=[r[:] for r in rows]
    rows[1500][1]='65535'
    def write(values):
        with path.open('w',newline='') as f:csv.writer(f).writerows(values)
    write(rows)
    accepted,bad,_=lda.extract_recording(dict(path=str(recording)),'notch50')
    assert len(accepted)==15 and bad[0]['trial_id']==1 and '削顶' in bad[0]['reason']
    del original[1500];write(original)
    with pytest.raises(ValueError,match='不连续'):
        lda.extract_recording(dict(path=str(recording)),'notch50')


def test_reject_and_missing_ack_not_silently_used(recording):
    path=recording/'trials.jsonl';rows=lda.lines(path.read_bytes());rows[0]['status']='rejected'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    events_path=recording/'events.jsonl'
    events=[e for e in lda.lines(events_path.read_bytes()) if not (e['kind']=='marker_ack' and e['event_id']=='2_action')]
    events_path.write_text('\n'.join(json.dumps(e) for e in events))
    accepted,bad,_=lda.extract_recording(dict(path=str(recording)),'notch50')
    assert len(accepted)==14 and {r['trial_id'] for r in bad}=={1,2}


def test_group_splits_ratios_and_class_coverage():
    records=[]
    for person in ['A','B','C','D']:
        for batch in range(2):
            for block in range(5):
                for word in WORDS:records.append(dict(subject=person,source_hash=person+str(batch),block=block,word=word,role='train' if batch==0 else 'test'))
    for strategy,key in [('subject',lambda r:r['subject']),('recording',lambda r:r['source_hash']),('block',lambda r:(r['source_hash'],r['block']))]:
        tr,te=lda.split_records(records,strategy,70,1)
        assert {key(records[i]) for i in tr}.isdisjoint({key(records[i]) for i in te})
        a,b=lda.split_records(records,strategy,70,1)
        np.testing.assert_array_equal(tr,a);np.testing.assert_array_equal(te,b)
    # Exactly 10 groups at 70% must produce 3 test groups, not 4 from float roundoff.
    tr,te=lda.split_records(records[:40],'block',70,1)
    assert len(te)==12
    tr,te=lda.split_records(records,'manual',70,1)
    assert len(tr)==len(te)==80
    with pytest.raises(ValueError,match='两个不同'):
        lda.split_records(records[:20],'recording',80,1)
    # The selected words now define the task; a three-word dataset is valid.
    three=[r for r in records if r['word']!='music']
    tr,te=lda.split_records(three,'block',80,1)
    assert {three[i]['word'] for i in tr}=={'open','close','next'}
    with pytest.raises(ValueError,match='每词至少'):
        lda.split_records(three+[dict(records[0],word='rare')],'block',80,1)


def test_invalid_json_model_is_rejected(recording,tmp_path):
    result=lda.train([dict(path=str(recording))],tmp_path/'output',train_percent=50)
    model=lda.load_model(result['model_path'])
    for mutate in [lambda m:m.update(schema='other'),lambda m:m['scaler_scale'].__setitem__(0,0),lambda m:m['coef'][0].__setitem__(0,float('nan')),lambda m:m['sos'][0].__setitem__(0,123),lambda m:m.update(training_fingerprints=[])]:
        damaged=copy.deepcopy(model);mutate(damaged)
        with pytest.raises(ValueError):lda.validate_model(damaged)


def test_overlapping_trial_intervals_are_rejected(recording):
    path=recording/'events.jsonl';events=lda.lines(path.read_bytes())
    for event in events:
        if event['kind']=='marker_ack' and event['event_id'].startswith('2_'):
            event['device_sample_idx']-=1500
    path.write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError,match='采样区间重叠'):
        lda.extract_recording(dict(path=str(recording)),'notch50')
