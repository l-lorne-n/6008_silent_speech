from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_sliding_architecture_sweep as run


def test_baseline_exact_initialization_and_shapes():
    run.torch.set_num_threads(2)
    run.base.cnn.set_seed(123);old=run.base.SlidingNet(1400)
    run.base.cnn.set_seed(123);new=run.SweepNet(run.specs()[0])
    assert all(run.torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    expected=[19235,23875,46475,75323,21283,19587,54667,47179,55371]
    for spec,count in zip(run.specs(),expected):
        net=run.SweepNet(spec).eval()
        assert sum(p.numel() for p in net.parameters())==count
        with run.torch.no_grad():assert net(run.torch.zeros(2,2,2800)).shape==(2,11)
        restored=run.restore(dict(state_dict=net.state_dict(),architecture=spec,window_ms=1400))
        assert all(run.torch.equal(v,restored.state_dict()[k]) for k,v in net.state_dict().items())


def test_validation_rank_prioritizes_strict_success_then_errors():
    metrics=dict(clean_success_rate=.7,end_to_end_recall=.72,wrong_label_outputs=10,
                 no_response_trials=5,rest_proxy_outputs_per_min=.2)
    ref=dict(metrics=metrics,resources=dict(macs_per_window=100))
    better=deepcopy(ref);better['metrics']['clean_success_rate']+=.01
    assert run.validation_rank(better)>run.validation_rank(ref)
    better=deepcopy(ref);better['metrics']['wrong_label_outputs']-=1
    assert run.validation_rank(better)>run.validation_rank(ref)
    # Unrelated test scores never enter the ranking.
    ref['test_accuracy']=1.;better['test_accuracy']=0.
    assert run.validation_rank(better)>run.validation_rank(ref)


def synthetic_data():
    rng=run.np.random.default_rng(17);rows=[];records=[];waves={}
    classes=[f'w{i}' for i in range(10)]+['rest']
    for split in ['train','validation','test']:
        rec=dict(recording=split,subject='S011',first=0,annotations=[])
        waves[split]=rng.normal(size=(22*5000,2)).astype(run.np.float32)
        for i in range(22):
            start=i*5000;label=i%11
            rec['annotations'].append(dict(trial_id=i,word=classes[label%10],block=i//11,split=split,
                prepare=start,action=start+1200,rest=start+4200,context_end=start+5000))
            rows.append(dict(recording=split,subject='S011',trial_id=i,label=label,split=split,
                start=start+1800,end=start+4600,offset=start+1800,fingerprint=f'{split}:{i}'))
        records.append(rec)
    indices={k:[i for i,r in enumerate(rows) if r['split']==k] for k in ['train','validation','test']}
    indices['fit']=indices['train']+indices['validation']
    indices['dev_calibration']=indices['train'];indices['final_calibration']=indices['fit']
    cfg=dict(seeds=[123],max_epochs=2,early_stop=2,classes=classes,
        optimizer=dict(lr=.001,weight_decay=.0001),scheduler=dict(factor=.1,patience=2),
        dev_normalization=run.base.normalization(records,waves,{'train'}),
        final_normalization=run.base.normalization(records,waves,{'train','validation'}),
        decoder=dict(probability=.5,confirm=2,rest_frames=2,margin=0.),
        counts=dict(train=22,validation=22,test=22),architecture_selection='synthetic validation rule')
    return dict(rows=rows,records=records,waves=waves,indices=indices),cfg


def test_small_synthetic_end_to_end_resume_and_test_isolation(tmp_path,monkeypatch):
    run.torch.set_num_threads(2)
    monkeypatch.setattr(run.tempfile,'tempdir',str(tmp_path))
    monkeypatch.setattr(run.base,'PER_CLASS',2)
    data,cfg=synthetic_data()
    spec=run.specs()[-1];cfg['architectures']=[spec]
    folder=tmp_path/spec['name']
    # A realistic full stage chain, but only synthetic data and two epochs.
    run.develop(folder,spec,data,cfg)
    cp=run.torch.load(folder/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    assert cp['training_fingerprints']==[data['rows'][i]['fingerprint'] for i in data['indices']['train']]
    digest=run.io.digest(folder/'seeds/123/model.pt')
    with monkeypatch.context() as m:
        m.setattr(run.base,'train_epoch',lambda *a:pytest.fail('Completed seed was retrained'))
        run.develop(folder,spec,data,cfg)
    assert run.io.digest(folder/'seeds/123/model.pt')==digest
    run.freeze_recommendation(tmp_path,cfg)
    assert run.io.read(tmp_path/'validation_selection.json')['test_used'] is False
    source=tmp_path/'source';(source/'export').mkdir(parents=True)
    run.io.write(source/'export/input_contract.json',dict(classes=cfg['classes']))
    cfg['base']=str(source)
    run.finalize(folder,spec,data,cfg)
    final=run.torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert final['training_fingerprints']==[data['rows'][i]['fingerprint'] for i in data['indices']['fit']]
    assert not any(f.startswith('test:') for f in final['training_fingerprints'])
    run.test_model(folder/'test',folder,data,cfg)
    (tmp_path/'current_base').mkdir()
    run.test_model(tmp_path/'current_base/test',folder,data,cfg)
    with monkeypatch.context() as m:
        m.setattr(run,'infer_runs',lambda *a:pytest.fail('Completed test was replayed'))
        run.test_model(folder/'test',folder,data,cfg)
    run.summarize(tmp_path,data,cfg)
    report=run.io.read(tmp_path/'comparison.json')
    assert report['test_targets']==22 and report['recommended_by_validation']==spec['name']
    assert report['models']['current_base']==report['models'][spec['name']]
    assert (folder/'test/paired_vs_current_int8.csv').exists()
    # Tampering with a completed checkpoint must stop resume, not silently accept.
    with (folder/'seeds/123/model.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='Input changed'):
        run.done(folder/'seeds/123','seed_sha256.json')


def test_trace_grid_rejects_other_split():
    data,_=synthetic_data()
    keys=[(r['recording'],n,e) for r in data['records'] for n,(a,b) in enumerate(run.base.split_runs(r,'test'))
          for e in run.base.window_ends(r,a,b,2800)]
    trace=dict(recording=run.np.array([x[0] for x in keys]),run=run.np.array([x[1] for x in keys]),
        end=run.np.array([x[2] for x in keys]),probabilities=run.np.full((len(keys),11),1/11))
    run.validate_grid(trace,data['records'],'test')
    with pytest.raises(AssertionError):run.validate_grid(trace,data['records'],'validation')
