"""Verify release bytes; optionally verify raw data and saved models/trace scores."""
from pathlib import Path
import argparse,csv,hashlib,json,sys

HERE=Path(__file__).resolve().parent
UPPER=HERE.parents[1]
REPO=UPPER.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',action='store_true')
    parser.add_argument('--models',action='store_true')
    args=parser.parse_args();manifest=read(HERE/'manifest.json')
    for name,sha in manifest['files'].items():
        p=(HERE/name).resolve();assert p.is_relative_to(HERE)
        assert digest(p)==sha,f'Changed or missing release file: {name}'
    result=dict(files_verified=len(manifest['files']),final_models=len(read(HERE/'models.json')))
    for name,sha in read(HERE/'training_sources_sha256.json').items():
        assert digest(REPO/name)==sha,f'Training source mismatch: {name}'
    if args.data:
        hashes=read(HERE/'cross_subject/raw_inputs_sha256.json')
        assert hashes==read(HERE/'individual_80_20/raw_inputs_sha256.json')
        for name,sha in hashes.items():assert digest(REPO/name)==sha,f'Dataset mismatch: {name}'
        result['raw_files_verified']=len(hashes)
    if args.models:
        sys.path.insert(0,str(UPPER/'experiments'))
        import run_cnn_goodone_plus_current as current
        sweep,base,io,np,torch=current.sweep,current.base,current.io,current.np,current.torch
        torch.set_num_threads(2);checks=[]
        for entry in read(HERE/'models.json'):
            folder=(HERE/entry['model']).parent;fold=folder.parent
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            contract=read(folder/'export/input_contract.json')
            assert cp['architecture']==current.architecture()
            assert cp['normalization']==contract['normalization'] and cp['classes']==contract['classes']
            records=read(fold/'recordings.json');lookup={r['recording']:r for r in records}
            for fp in cp['training_fingerprints']:
                rec,a,b=fp.rsplit(':',2);r=lookup[rec];a,b=int(a),int(b)
                assert any(t['split'] in ['train','validation'] and t['prepare']<=a<b<=t['context_end'] for t in r['annotations'])
                assert (r['subject']!=entry['subject']) if entry['experiment']=='cross_subject' else (r['subject']==entry['subject'])
            x=np.random.default_rng(42).normal(size=(3,2,2800)).astype(np.float32)
            net=sweep.restore(cp).eval()
            with torch.inference_mode():expected=net(torch.from_numpy(x)).numpy()
            got=base.auto.infer(folder/'export/model_fp32.onnx',x,np.arange(len(x)))
            np.testing.assert_allclose(got,expected,rtol=1e-4,atol=1e-4)
            base.previous.check_quantized_graph(folder/'export/model_int8_qdq.onnx')
            int8=base.auto.infer(folder/'export/model_int8_qdq.onnx',x,np.arange(len(x)))
            assert int8.shape==(3,11) and np.isfinite(int8).all()
            for kind in ['fp32','int8']:
                with np.load(folder/'test'/f'{kind}_trace.npz') as z:trace={k:z[k] for k in z.files}
                sweep.validate_grid(trace,records,'test')
                events=base.decode(trace,contract['decoder'],cp['classes'])
                assert events==read(folder/'test'/f'{kind}_events.json')
                assert sweep.score(events,records,cp['classes'])[0]==read(folder/'test/comparison.json')[kind]['overall']
                with (HERE/entry['experiment']/f'{kind}_ten_word_trials.csv').open(encoding='utf-8-sig',newline='') as f:
                    predictions={(r['recording'],int(r['trial_id'])):r for r in csv.DictReader(f) if r['subject']==entry['subject']}
                checked_trials=0
                for record in records:
                    ix=np.flatnonzero(trace['recording']==record['recording'])
                    for trial in record['annotations']:
                        if trial['split']!='test':continue
                        chosen=[i for i in ix if (label:=base.training_label(trial,int(trace['end'][i])-2800,
                            int(trace['end'][i]),cp['classes'])) is not None and label<10]
                        assert chosen
                        mean=trace['probabilities'][chosen].astype(np.float64).mean(axis=0)
                        pred=cp['classes'][int(mean[:10].argmax())]
                        saved=predictions[(record['recording'],trial['trial_id'])]
                        assert pred==saved['prediction'] and len(chosen)==int(saved['windows'])
                        checked_trials+=1
                assert checked_trials==len(predictions)
            checks.append(dict(experiment=entry['experiment'],subject=entry['subject'],
                fp32_max_absolute_error=float(np.abs(got-expected).max()),training_partition_verified=True,
                trace_rescoring=True,ten_word_predictions_recomputed=True))
        result['model_checks']=checks
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
