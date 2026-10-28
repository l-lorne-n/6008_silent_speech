"""Choose INT8 calibration on development validation, then test frozen choice."""
import argparse
from collections import Counter
import gc
import hashlib
import shutil
import tempfile

from export_h7_cnn import *


def representative(rows,indices):
    rng=np.random.default_rng(20261021)
    selected=[]
    for subject,word in sorted({(rows[i]['subject'],rows[i]['word']) for i in indices}):
        group=[int(i) for i in indices if (rows[i]['subject'],rows[i]['word'])==(subject,word)]
        selected.extend(rng.choice(group,min(10,len(group)),replace=False).tolist())
    return np.array(sorted(selected))


def quantize(source,dest,x,indices,percentile):
    extra={'ActivationSymmetric':False,'WeightSymmetric':True}
    method=CalibrationMethod.MinMax if percentile is None else CalibrationMethod.Percentile
    if percentile is not None:extra['CalibPercentile']=percentile
    quantize_static(str(source),str(dest),Reader(x,indices),quant_format=QuantFormat.QDQ,
                    per_channel=True,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,
                    calibrate_method=method,op_types_to_quantize=['Conv','Gemm','AveragePool','Flatten'],extra_options=extra)
    onnx.checker.check_model(onnx.load(dest))


def infer(path,x,indices):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    return np.concatenate([session.run(None,{'input':x[i:i+1,:,None,:]})[0] for i in indices])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('cnn',type=Path)
    args=parser.parse_args();folder=args.cnn.resolve()
    target=folder/'deployment_calibrated';target.mkdir(exist_ok=False)
    tmp=target/'temporary';tmp.mkdir();tempfile.tempdir=str(tmp)
    shutil.copy2(__file__,target/Path(__file__).name)
    torch.set_num_threads(2)
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    baseline=json.loads((Path(report['protocol']['baseline'])/'report.json').read_text(encoding='utf-8'))
    rows,split=split_data(baseline)
    protocol=dict(candidates=[None,99.99,99.9,99.5],selection='minimum development validation cross entropy, using saved selection checkpoint trained only on development train split',
                  calibration='Up to 10 randomly selected development-training trials per subject/word, seed 20261021; final chosen method recalibrated on an independently sampled final-training subset by the same rule.',
                  test_used_for_candidate_selection=False,
                  caveat='These recordings and the earlier smaller architecture holdout scores were previously inspected. Candidates below are chosen only on development validation; results remain exploratory, not a fresh blind test.')
    write_json(target/'protocol.json',protocol)
    raw,lengths=extract_waveforms(rows,baseline,input_length=6144)
    np.savez(folder/'waveform_cache.npz',raw=raw,lengths=lengths)
    write_json(folder/'waveform_cache_manifest.json',dict(fingerprints=[r['fingerprint'] for r in rows],
               sha256=hashlib.sha256((folder/'waveform_cache.npz').read_bytes()).hexdigest()))
    x,scaler=normalize(raw,lengths,split['train']);x=x.numpy()
    chosen=torch.load(folder/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    assert scaler==chosen['normalization']
    net=h7_from_checkpoint(chosen).eval()
    source=target/'development_fp32.onnx'
    torch.onnx.export(fused_layers(net),torch.zeros(1,2,1,6144),str(source),opset_version=17,dynamo=False,
                      input_names=['input'],output_names=['logits'],do_constant_folding=True)
    pre=target/'development_preprocessed.onnx'
    quant_pre_process(str(source),str(pre),skip_optimization=True)
    representatives=representative(rows,split['train'])
    validation_rows=[rows[i] for i in split['validation']]
    labels=torch.tensor([report['protocol']['classes'].index(r['word']) for r in validation_rows])
    classes=report['protocol']['classes']
    fp=infer(source,x,split['validation'])
    outcomes=[]
    for percentile in protocol['candidates']:
        name='minmax' if percentile is None else 'percentile_'+str(percentile)
        path=target/(name+'_development.onnx')
        quantize(pre,path,x,representatives,percentile)
        logits=infer(path,x,split['validation'])
        metric=lda.metrics(validation_rows,logits.argmax(1),classes)
        loss=float(nn.functional.cross_entropy(torch.from_numpy(logits),labels))
        outcomes.append(dict(name=name,percentile=percentile,validation_loss=loss,validation_metrics=metric))
        write_json(target/'validation_candidates.json',dict(fp32_metrics=lda.metrics(validation_rows,fp.argmax(1),classes),
                   calibration_fingerprints=[rows[i]['fingerprint'] for i in representatives],candidates=outcomes))
        print('VALIDATION',name,metric['accuracy'],loss,flush=True);gc.collect()
    selected=min(outcomes,key=lambda r:r['validation_loss'])
    write_json(target/'selected_calibration.json',selected)
    print('CALIBRATION FROZEN BEFORE FINAL TEST',selected['name'],flush=True)
    final=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    train=np.sort(np.r_[split['train'],split['validation']])
    x,scaler=normalize(raw,lengths,train);x=x.numpy()
    assert scaler==final['normalization']
    representatives=representative(rows,train)
    path=target/'model_int8_qdq.onnx'
    quantize(folder/'deployment/model_preprocessed.onnx',path,x,representatives,selected['percentile'])
    logits=infer(path,x,split['test'])
    test_rows=[rows[i] for i in split['test']];pred=logits.argmax(1)
    metrics=lda.metrics(test_rows,pred,classes)
    result=dict(fp32=report['metrics'],int8=metrics,
                group_metrics=lda.group_metrics([rows[i] for i in train],test_rows,pred,classes),
                selected=selected,calibration_trials=len(representatives),test_used_for_calibration=False,
                calibration_fingerprints=[rows[i]['fingerprint'] for i in representatives],
                calibration_selection=protocol,
                onnx_bytes=path.stat().st_size,
                fp32_int8_prediction_agreement=float(np.mean(pred==np.array([classes.index(r['prediction']) for r in csv.DictReader((folder/'predictions.csv').open(encoding='utf-8-sig'))]))))
    write_json(target/'report.json',result)
    with (target/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject','recording','trial_id','word','int8','correct'])
        for r,p in zip(test_rows,pred):writer.writerow([r['subject'],r['recording'],r['trial_id'],r['word'],classes[p],int(r['word']==classes[p])])
    for filename in ['input_contract.json','resource_estimate.json']:
        shutil.copy2(folder/'deployment'/filename,target/filename)
    print('FINAL_INT8',metrics['correct'],metrics['total'],metrics['accuracy'],flush=True)


if __name__=='__main__':main()
