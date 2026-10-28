from copy import deepcopy
import csv
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_personal_head_finetune as run
from test_personal_sliding_finetune import synthetic, fixture


def test_only_classifier_changes_and_independent_initialization(tmp_path,monkeypatch):
    monkeypatch.setattr(run.tempfile,'tempdir',str(tmp_path))
    cp,rows,waves,plan,config=synthetic()
    before=deepcopy(cp)
    schedule=[list(range(22)),list(reversed(range(22)))]
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    first=run.train_person(cp,rows,waves,plan,'a',config,schedule,a)
    second=run.train_person(cp,rows,waves,plan,'b',config,schedule,b)
    changed=[]
    for name,value in first['state_dict'].items():
        assert run.torch.equal(cp['state_dict'][name],before['state_dict'][name])
        assert run.torch.equal(value,second['state_dict'][name])
        if not run.torch.equal(value,cp['state_dict'][name]):changed.append(name)
    assert set(changed)=={'layers.23.weight','layers.23.bias'}
    assert cp['training_fingerprints']==['original-training']
    history=run.io.read(a/'training_history.json')
    assert run.schedule_indices(history,rows,plan,2)==schedule
    assert run.io.read(a/'freeze_audit.json')['all_frozen_bitwise_identical']
    x=run.base.batch(rows,list(range(22)),waves,cp['normalization'])
    pre=run.base.export_fp(run.base.restore(first),a/'model_fp32.onnx',x[:3])
    path=a/'model_int8_qdq.onnx'
    run.base.auto.quantize(pre,path,x,run.np.arange(len(x)),99.99)
    run.base.previous.check_quantized_graph(path)
    logits=run.base.auto.infer(path,x,run.np.arange(3))
    assert logits.shape==(3,11) and run.np.isfinite(logits).all()


def test_schedule_rejects_duplicates_and_non_calibration_sources():
    _,rows,_,plan,_=synthetic()
    history=[dict(epoch=1,window_fingerprints=[r['fingerprint'] for r in rows])]
    assert run.schedule_indices(history,rows,plan,1)==[list(range(22))]
    bad=deepcopy(history);bad[0]['window_fingerprints'][0]=rows[1]['fingerprint']
    with pytest.raises(AssertionError):run.schedule_indices(bad,rows,plan,1)
    bad_rows=rows+[dict(rows[0],fingerprint='heldout')]
    bad=deepcopy(history);bad[0]['window_fingerprints'][0]='heldout'
    with pytest.raises(AssertionError):run.schedule_indices(bad,bad_rows,plan,1)


def test_three_way_comparison_uses_exact_grid_and_correct_deltas(tmp_path,monkeypatch):
    records,rows,classes=fixture()
    _,heldout=run.previous.select_calibration(records,rows,classes)
    heldout=heldout[:1]
    keys=[(r['recording'],n,end) for r in heldout for n,(a,b) in enumerate(run.base.split_runs(r,'test'))
          for end in run.base.window_ends(r,a,b,2800)]
    trace=dict(recording=run.np.array([k[0] for k in keys]),run=run.np.array([k[1] for k in keys]),
               end=run.np.array([k[2] for k in keys]),probabilities=run.np.zeros((len(keys),11),run.np.float32))
    trace['probabilities'][:,10]=1
    def prediction(count):
        out=deepcopy(trace)
        trials=[t for t in heldout[0]['annotations'] if t['split']=='test'][:count]
        for t in trials:
            mask=(out['end']>=t['action']+2000)&(out['end']<t['action']+2600)
            out['probabilities'][mask]=0
            out['probabilities'][mask,classes.index(t['word'])]=1
        return out
    run.verify_trace_grid(trace,heldout,1400)
    invalid=deepcopy(trace);invalid['end'][0]+=1
    with pytest.raises(AssertionError):run.verify_trace_grid(invalid,heldout,1400)
    reference=tmp_path/'reference';(reference/'a').mkdir(parents=True)
    target=tmp_path/'output';(target/'a').mkdir(parents=True)
    for kind in ['fp32','int8']:
        run.np.savez(reference/'a'/f'base_{kind}_trace.npz',**prediction(0))
        run.np.savez(reference/'a'/f'personal_{kind}_trace.npz',**prediction(1))
    cp=dict(classes=classes,window_ms=1400)
    run.torch.save(cp,target/'a'/'model.pt')
    config=dict(reference=str(reference),remaining_test={'a':10},
                decoder=dict(probability=.5,confirm=2,rest_frames=2,margin=0.))
    monkeypatch.setattr(run,'SUBJECTS',['a'])
    monkeypatch.setattr(run.base,'load_waves',lambda records: {})
    monkeypatch.setattr(run.base,'infer_runs',lambda *args: prediction(2))
    run.compare(target,cp,heldout,config)
    result=run.io.read(target/'comparison.json')['overall']
    assert result['base_int8']['correct_unique_trials']==0
    assert result['full_int8']['correct_unique_trials']==1
    assert result['head_int8']['correct_unique_trials']==2
    with (target/'head_deltas.csv').open(encoding='utf-8-sig',newline='') as f:
        changes=list(csv.DictReader(f))
    assert next(r for r in changes if r['subject']=='ALL' and r['format']=='int8' and r['reference']=='full')['correct_change']=='1'
    with (target/'a'/'paired_full_vs_head_int8.csv').open(encoding='utf-8-sig',newline='') as f:
        pairs=list(csv.DictReader(f))
    assert len(pairs)==10 and sum(t['head_correct']=='True' for t in pairs)==2
