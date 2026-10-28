from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import run_cnn_goodone_plus_current as run
from test_sliding_architecture_sweep import synthetic_data


def test_fresh_split_uses_whole_rounds_and_rejects_duplicate_signal():
    records=[]
    for subject in ['S011','S013']:
        for block in range(1,6):
            for n,word in enumerate(['open','close','next','music','louder','photo','previous','quieter','start','stop']):
                key=f'{subject}:{block}:{n}'
                records.append(dict(subject=subject,recording=subject,source_hash=subject,block=block,
                    trial_id=block*10+n,word=word,fingerprint=key,signal_fingerprints=[key+':0',key+':1']))
    rows,classes=run.make_split(records)
    assert len(classes)==11 and classes[-1]=='rest'
    assert run.Counter(r['split'] for r in rows)==dict(train=60,validation=20,test=20)
    assert run.make_split(records)==(rows,classes)
    for recording in ['S011','S013']:
        for block in range(1,6):
            assert len({r['split'] for r in rows if r['recording']==recording and r['block']==block})==1
    for record in records:record['signal_fingerprints']=['same']
    with pytest.raises(AssertionError):run.make_split(records)


def test_synthetic_single_architecture_pipeline_and_resume(tmp_path,monkeypatch):
    run.torch.set_num_threads(2)
    monkeypatch.setattr(run.tempfile,'tempdir',str(tmp_path))
    monkeypatch.setattr(run.base,'PER_CLASS',2)
    data,cfg=synthetic_data()
    spec=run.architecture()
    assert spec['widths']==[8,24,32,64,64] and spec['tail_kernel']==3 and spec['pool_bins']==1
    cfg.update(schema=run.SCHEMA,architectures=[spec],versions={},base=str(tmp_path/'template'))
    template=Path(cfg['base'])/'export';template.mkdir(parents=True)
    run.io.write(template/'input_contract.json',run.input_contract(cfg['classes'],cfg['final_normalization']))
    run.io.write(tmp_path/'prepared_sha256.json',{})
    run.io.write(tmp_path/'inputs_sha256.json',{})
    monkeypatch.setattr(run,'verify_prepared',lambda out:(data,cfg))
    run.run_all(tmp_path)
    folder=tmp_path/spec['name']
    cp=run.torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert cp['training_fingerprints']==[data['rows'][i]['fingerprint'] for i in data['indices']['fit']]
    assert not any(f.startswith('test:') for f in cp['training_fingerprints'])
    contract=run.io.read(folder/'export/input_contract.json')
    assert contract['normalization']==cfg['final_normalization']
    assert contract['architecture']==spec and contract['decoder']==run.DECODER
    assert run.io.read(tmp_path/'completed.json')['no_deployment']
    report=run.io.read(tmp_path/'comparison.json')
    assert report['test_targets']==22 and set(report['results'])=={'fp32','int8'}
    assert report['results']['int8']['overall']['trials']==22
    digest=run.io.digest(folder/'model.pt')
    monkeypatch.setattr(run.sweep,'develop',lambda *args:pytest.fail('Completed run must not train again'))
    run.run_all(tmp_path)
    assert run.io.digest(folder/'model.pt')==digest
    with (folder/'model.pt').open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(RuntimeError,match='Input changed'):run.run_all(tmp_path)


def test_output_lock_prevents_concurrent_run_and_releases(tmp_path):
    with run.output_lock(tmp_path):
        with pytest.raises(OSError):
            with run.output_lock(tmp_path):
                pytest.fail('Second writer entered')
    with run.output_lock(tmp_path):pass
