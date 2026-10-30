from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_cnn_individual_80_20 as run


def trials():
    rows=[]
    for subject in ['S011','S012']:
        for rec in range(2):
            for block in range(5):
                for word in ['open','close','next','music','start','stop','quieter','louder','previous','photo']:
                    fp=f'{subject}-{rec}-{block}-{word}'
                    rows.append(dict(subject=subject,recording=f'{subject}-{rec}',source_hash=f'{subject}-{rec}',
                        trial_id=len(rows),block=block,word=word,fingerprint=fp,signal_fingerprints=[fp+'-0',fp+'-1']))
    return rows


def test_personal_80_20_is_grouped_deterministic_and_does_not_mutate_input():
    valid=trials();before=deepcopy(valid)
    rows,classes=run.subject_split(valid,'S011')
    assert valid==before
    assert {r['subject'] for r in rows}=={'S011'}
    assert len(classes)==11 and classes[-1]=='rest'
    assert sum(r['split']=='test' for r in rows)==20
    assert sum(r['split'] in ['train','validation'] for r in rows)==80
    assert sum(r['split']=='validation' for r in rows)==20
    assignment={(r['recording'],r['trial_id']):r['split'] for r in rows}
    repeat,_=run.subject_split(list(reversed(valid)),'S011')
    assert assignment=={(r['recording'],r['trial_id']):r['split'] for r in repeat}
    for recording,block in {(r['recording'],r['block']) for r in rows}:
        assert len({r['split'] for r in rows if (r['recording'],r['block'])==(recording,block)})==1


def test_duplicated_raw_trial_is_rejected():
    valid=trials();valid.append(deepcopy(valid[0]))
    with pytest.raises(AssertionError):run.subject_split(valid,'S011')


def data():
    record=dict(subject='S011',recording='r',annotations=[])
    rows=[]
    for i,split in enumerate(['train','validation','test']):
        a=i*10000
        record['annotations'].append(dict(trial_id=i,split=split,prepare=a,context_end=a+10000))
        rows.append(dict(subject='S011',recording='r',trial_id=i,split=split,start=a+100,end=a+2900))
    indices=dict(train=[0],validation=[1],test=[2],fit=[0,1],dev_calibration=[0],final_calibration=[1])
    return dict(records=[record],rows=rows,indices=indices)


def test_audit_rejects_other_person_or_test_normalization_pool():
    original=data();run.audit_person(original,'S011')
    wrong=deepcopy(original);wrong['rows'][0]['subject']='S012'
    with pytest.raises(AssertionError):run.audit_person(wrong,'S011')
    wrong=deepcopy(original);wrong['indices']['fit'].append(2)
    with pytest.raises(AssertionError):run.audit_person(wrong,'S011')
    wrong=deepcopy(original);wrong['indices']['final_calibration']=[2]
    with pytest.raises(AssertionError):run.audit_person(wrong,'S011')


def test_audit_rejects_cross_partition_signal_overlap():
    wrong=data();wrong['rows'][0]['end']=10100
    with pytest.raises(AssertionError):run.audit_person(wrong,'S011')
    wrong=data();wrong['records'][0]['annotations'][0]['context_end']=10100
    with pytest.raises(AssertionError):run.audit_person(wrong,'S011')
