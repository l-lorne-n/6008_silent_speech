from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
import run_personal_sliding_finetune as run


def fixture():
    classes = [f'w{i}' for i in range(10)] + ['rest']
    records, rows = [], []
    for subject in ['a','b']:
        rec = dict(recording=subject, subject=subject, first=0, annotations=[])
        for i in range(30):
            start = i*12000
            rec['annotations'].append(dict(trial_id=i, block=i//10, word=classes[i%10], split='test',
                prepare=start, action=start+2000, rest=start+6000, context_end=start+12000))
            for label, offset in [(i%10,2200),(i%10,2400),(10,7400),(10,7600)]:
                rows.append(dict(recording=subject, subject=subject, trial_id=i, block=i//10,
                    label=label, start=start+offset, end=start+offset+2800,
                    offset=start+offset, split='test', fingerprint=f'{subject}:{start+offset}'))
        records.append(rec)
    return records, rows, classes


def test_two_episodes_each_class_and_no_signal_overlap():
    original, rows, classes = fixture()
    plans, heldout = run.select_calibration(original, rows, classes)
    assert all(t['split']=='test' for r in original for t in r['annotations'])
    for record in heldout:
        subject = record['subject']
        assert sum(t['split']=='test' for t in record['annotations']) == 10
        assert len(plans[subject]['episodes']) == 22
        selected = [rows[i] for e in plans[subject]['episodes'] for i in e['indices']]
        for a,b in run.base.split_runs(record,'test'):
            for end in run.base.window_ends(record,a,b,2800):
                assert all(w['end'] <= end-2800 or w['start'] >= end for w in selected)
    other,_ = run.select_calibration(list(reversed(original)),rows,classes)
    assert other == plans
    invalid = deepcopy(rows)
    invalid[plans['a']['episodes'][0]['indices'][0]]['end'] = 240001
    with pytest.raises(AssertionError):
        run.audit_partition(heldout,invalid,plans,classes)


def test_incomplete_round_skipped_and_insufficient_rest_fails():
    records,rows,classes = fixture()
    records[0]['annotations'][0]['split'] = 'excluded'
    rows = [r for r in rows if not (r['subject']=='a' and r['trial_id']==0)]
    plans,heldout = run.select_calibration(records,rows,classes)
    assert [g['block'] for g in plans['a']['rounds']] == [1,2]
    assert sum(t['split']=='test' for t in heldout[0]['annotations']) == 9
    rows = [r for r in rows if not (r['subject']=='a' and r['label']==10)]
    with pytest.raises(ValueError,match='fewer than two'):
        run.select_calibration(records,rows,classes)


def synthetic():
    run.torch.set_num_threads(2);run.base.cnn.set_seed(43)
    length=2800
    waves={'r':run.np.random.default_rng(44).normal(size=(22*length,2)).astype(run.np.float32)}
    rows=[dict(recording='r', subject='a', trial_id=i, label=i%11, offset=i*length,
        start=i*length, end=(i+1)*length, fingerprint=f'r:{i}') for i in range(22)]
    cp=dict(state_dict=deepcopy(run.base.SlidingNet(1400).state_dict()), window_ms=1400,
        classes=[f'w{i}' for i in range(10)]+['rest'], normalization=dict(mean=[0.,0.],scale=[1.,1.]),
        training_fingerprints=['original-training'])
    plan=dict(episodes=[dict(indices=[i]) for i in range(22)])
    config=dict(seed=19, learning_rate=1e-4,epochs=2,base_checkpoint_sha256='synthetic')
    return cp,rows,waves,plan,config


def test_independent_initialization_frozen_bn_and_onnx_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, 'tempdir', str(tmp_path))
    cp,rows,waves,plan,config = synthetic()
    original=deepcopy(cp)
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    first=run.train_person(cp,rows,waves,plan,'a',config,a)
    second=run.train_person(cp,rows,waves,plan,'b',config,b)
    changed=[]
    for name,value in cp['state_dict'].items():
        assert run.torch.equal(value,original['state_dict'][name])
        assert run.torch.equal(first['state_dict'][name],second['state_dict'][name])
        if 'running_' in name or 'num_batches_tracked' in name:
            assert run.torch.equal(value,first['state_dict'][name])
        else:
            changed.append(not run.torch.equal(value,first['state_dict'][name]))
    assert any(changed) and cp['training_fingerprints']==['original-training']
    assert first['normalization']==cp['normalization']
    assert len(first['personal_adaptation']['calibration_fingerprints'])==22
    x=run.base.batch(rows,list(range(22)),waves,cp['normalization'])
    pre=run.base.export_fp(run.base.restore(first),a/'model_fp32.onnx',x[:3])
    dest=a/'model_int8_qdq.onnx'
    run.base.auto.quantize(pre,dest,x,run.np.arange(len(x)),99.99)
    run.base.previous.check_quantized_graph(dest)
    logits=run.base.auto.infer(dest,x,run.np.arange(3))
    assert logits.shape==(3,11) and run.np.isfinite(logits).all()


def test_scoring_wrong_then_correct_and_matching_trial_sets():
    records,rows,classes=fixture()
    _,heldout=run.select_calibration(records,rows,classes)
    events=[dict(recording='a',emitted=3000,candidate='w0'),
            dict(recording='a',emitted=243000,candidate='w1'),
            dict(recording='a',emitted=243200,candidate='w0')]
    metrics,trials,confusion=run.score(events,heldout,classes)
    assert metrics['trials']==20 and metrics['correct_unique_trials']==1
    assert metrics['clean_correct_trials']==0
    assert metrics['wrong_label_outputs']==1 and metrics['wrong_response_trials']==1
    assert metrics['wrong_fraction_of_response_outputs']==.5
    assert sum(c['count'] for c in confusion)==2
    pairs=run.paired_trials(trials,list(reversed(trials)))
    assert len(pairs)==20 and sum(t['base_wrong'] for t in pairs)==1
    with pytest.raises(AssertionError):
        run.paired_trials(trials,trials[:-1])
