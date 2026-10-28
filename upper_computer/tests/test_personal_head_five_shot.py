from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_personal_head_five_shot as run
from test_personal_sliding_finetune import fixture


def six_rounds():
    records,rows,classes=fixture()
    for r in records:
        extra=deepcopy(r['annotations'])
        for t in extra:
            t['trial_id']+=30;t['block']+=3
            for k in ['prepare','action','rest','context_end']:t[k]+=360000
        r['annotations'].extend(extra)
    extra=deepcopy(rows)
    for w in extra:
        w['trial_id']+=30;w['block']+=3
        for k in ['start','end','offset']:w[k]+=360000
        w['fingerprint']+=':later'
    rows+=extra
    return records,rows,classes


def test_expansion_keeps_two_shots_and_excludes_entire_five_rounds():
    records,rows,classes=six_rounds()
    old,_=run.previous.select_calibration(records,rows,classes)
    saved=deepcopy(old)
    plans,heldout=run.expand_calibration(records,rows,classes,old)
    assert old==saved and all(t['split']=='test' for r in records for t in r['annotations'])
    for s,p in plans.items():
        assert p['episodes'][:22]==old[s]['episodes'] and p['rounds'][:2]==old[s]['rounds']
        assert len(p['episodes'])==55
        own=next(r for r in heldout if r['subject']==s)
        assert sum(t['split']=='test' for t in own['annotations'])==10
        # All replay windows are disjoint, including unused calibration windows.
        selected=[w for w in rows if w['subject']==s and w['trial_id']<50]
        for a,b in run.base.split_runs(own,'test'):
            for end in run.base.window_ends(own,a,b,2800):
                assert all(w['end']<=end-2800 or w['start']>=end for w in selected)
    bad=deepcopy(plans);bad['a']['episodes'][-1]=deepcopy(bad['a']['episodes'][10])
    with pytest.raises(AssertionError):run.audit_partition(heldout,rows,classes,bad)
    short=deepcopy(records)
    for r in short:r['annotations']=r['annotations'][:40]
    with pytest.raises(ValueError,match='insufficient'):
        run.expand_calibration(short,rows,classes,old)


def test_55_window_batch_only_changes_classifier(tmp_path):
    run.torch.set_num_threads(2);run.base.cnn.set_seed(123)
    records,rows,classes=six_rounds();old,_=run.previous.select_calibration(records,rows,classes)
    plans,_=run.expand_calibration(records,rows,classes,old)
    schedule=run.make_schedule(plans['a'],rows,2,123)
    assert schedule==run.make_schedule(plans['a'],rows,2,123)
    assert all(len(ix)==55 for ix in schedule)
    cp=dict(state_dict=deepcopy(run.base.SlidingNet(1400).state_dict()),window_ms=1400,
        classes=classes,normalization=dict(mean=[0.,0.],scale=[1.,1.]),training_fingerprints=[])
    original=deepcopy(cp['state_dict'])
    waves={'a':run.np.random.default_rng(12).normal(size=(720000,2)).astype(run.np.float32)}
    config=dict(seed=123,learning_rate=.0001,base_checkpoint_sha256='synthetic')
    personal=run.head.train_person(cp,rows,waves,plans['a'],'a',config,schedule,tmp_path)
    changed={n for n,v in personal['state_dict'].items() if not run.torch.equal(v,cp['state_dict'][n])}
    assert changed=={'layers.23.weight','layers.23.bias'}
    assert all(run.torch.equal(v,cp['state_dict'][n]) for n,v in original.items())
    assert len(personal['personal_adaptation']['source_episodes'])==55


def test_baseline_and_head_both_replayed_on_new_partition(tmp_path,monkeypatch):
    records,rows,classes=six_rounds();old,_=run.previous.select_calibration(records,rows,classes)
    _,heldout=run.expand_calibration(records,rows,classes,old)
    cp=dict(classes=classes,window_ms=1400)
    for s in ['a','b']:
        (tmp_path/s).mkdir();run.torch.save(cp,tmp_path/s/'model.pt')
    calls=[]
    def infer(path,checkpoint,own,waves,split):
        calls.append((str(path),own[0]['subject'],split))
        keys=[(r['recording'],n,end) for r in own for n,(a,b) in enumerate(run.base.split_runs(r,'test'))
              for end in run.base.window_ends(r,a,b,2800)]
        trace=dict(recording=run.np.array([k[0] for k in keys]),run=run.np.array([k[1] for k in keys]),
                   end=run.np.array([k[2] for k in keys]),probabilities=run.np.zeros((len(keys),11),run.np.float32))
        trace['probabilities'][:,10]=1
        if path.parent.name!='export':
            t=next(t for t in own[0]['annotations'] if t['split']=='test')
            mask=(trace['end']>=t['action']+2000)&(trace['end']<t['action']+2600)
            trace['probabilities'][mask]=0;trace['probabilities'][mask,classes.index(t['word'])]=1
        return trace
    monkeypatch.setattr(run,'SUBJECTS',['a','b'])
    monkeypatch.setattr(run.base,'load_waves',lambda records:{})
    monkeypatch.setattr(run.base,'infer_runs',infer)
    config=dict(base=str(tmp_path/'base'),remaining_test={'a':10,'b':10},
        decoder=dict(probability=.5,confirm=2,rest_frames=2,margin=0.))
    run.compare(tmp_path,cp,heldout,config)
    result=run.io.read(tmp_path/'comparison.json')['overall']
    assert len(calls)==8 and sum(Path(c[0]).parent.name=='export' for c in calls)==4
    assert result['base_int8']['trials']==result['head5_int8']['trials']==20
    assert result['base_int8']['correct_unique_trials']==0 and result['head5_int8']['correct_unique_trials']==2
    assert (tmp_path/'a'/'paired_base_vs_head5_int8.csv').exists()
