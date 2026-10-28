"""Train the fixed H7 CNN on label-blind gate segments; compare full replay.

No devices, no deployment, no changes to the frozen detector or source models.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import time

import train_small_cnn as cnn
from train_small_cnn import np, torch, nn
import replay_activity_cnn as replay
from export_h7_cnn import fused_layers, resource_report, onnx, quant_pre_process
from calibrate_h7_cnn import quantize, representative, infer
from semg.activity_cnn import GateConfig

ROOT=cnn.ROOT
REPLAY=ROOT/'analysis/activity_cnn_replay_20261021'
OLD=replay.MODEL
SEEDS=[20261010,20261011,20261012]
LENGTH=6144
JITTER=200  # independently jitter each boundary by up to 100 ms


class Placeholder:
    """Only exercises the shared scoring/eligibility path, never a model."""
    def run(self,*args):return [np.zeros((1,10),np.float32)]


def boundaries(start,end,lo,hi,ds,de):
    """Bound augmentation to same-trial context and to the CNN input capacity."""
    a=max(lo,min(start+int(ds),end-400))
    b=min(hi,end+int(de),a+LENGTH)
    if b-a<400:
        a,b=start,end
    assert lo<=a<b<=hi and 400<=b-a<=LENGTH
    return a,b


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    records=replay.read(REPLAY/'recordings.json')
    selected=replay.read(REPLAY/'selected.json');cfg=GateConfig(**selected['config'])
    contract=replay.read(OLD/'deployment_calibrated/input_contract.json')
    provenance=replay.read(REPLAY/'source_sha256.json')
    for path,h in provenance.items():assert replay.digest(path)==h, path
    for p in [REPLAY/'recordings.json',REPLAY/'selected.json',OLD/'protocol.json']:
        provenance[str(p)]=replay.digest(p)
    rows=[];raw=[];contexts=[];lengths=[];offsets=[];context_lengths=[];all_events=[];excluded=[]
    for r in records:
        path=REPLAY/'cache'/(r['recording']+'.npz')
        provenance[str(path)]=replay.digest(path)
        with np.load(path) as z:cache={k:z[k] for k in z.files}
        detected,state=replay.detect(r,cache,cfg)
        annotations={t['trial_id']:t for t in r['annotations']}
        for split in ['train','validation','test']:
            ev=replay.attach_scores(detected,r,cache,split,Placeholder(),contract['normalization'],contract['classes'])
            # Placeholder outputs are explicitly erased to avoid confusing them with predictions.
            for e in ev:
                e['eligible']=e['candidate'] is not None
                for k in ['candidate','probability','margin']:e[k]=None
            all_events+=ev
            counts=Counter(e['matched_trial'] for e in ev if e['eligible'] and e['matched_trial'] is not None)
            for e in ev:
                reason=None
                if not e['eligible']:reason=e.get('reason','gate_rejected')
                elif e['matched_trial'] is None:reason='no_unambiguous_word_label'
                elif counts[e['matched_trial']]!=1:reason='multiple_candidates_for_word'
                else:
                    t=annotations[e['matched_trial']]
                    if not (t['prepare']<=e['start']<e['end']<=t['context_end']):reason='cross_trial_context'
                    elif np.any(cache['clipped'][(cache['ends']>e['start'])&(cache['ends']-40<e['end'])]):reason='clipping_in_padding'
                if reason:
                    excluded.append(dict(recording=r['recording'],subject=r['subject'],split=split,event_id=e['event_id'],reason=reason))
                    continue
                start,end=e['start'],e['end'];length=end-start
                # Training augmentation may use only already available waveform within this trial.
                left=max(t['prepare'],start-JITTER)
                right=min(t['context_end'],end+JITTER,e['emitted'])
                waveform=cache['filtered'][start-r['first']:end-r['first']].T
                context=cache['filtered'][left-r['first']:right-r['first']].T
                x=np.zeros((2,LENGTH),np.float32);x[:,:length]=waveform
                cx=np.zeros((2,LENGTH+2*JITTER),np.float32);cx[:,:context.shape[1]]=context
                row=dict(index=len(rows),split=split,subject=r['subject'],recording=r['recording'],
                         trial_id=t['trial_id'],block=t['block'],word=t['word'],event_id=e['event_id'],
                         start=start,end=end,emitted=e['emitted'],context_start=left,context_end=right,
                         fingerprint=__import__('hashlib').sha256(waveform.tobytes()+f'{start}:{end}'.encode()).hexdigest())
                rows.append(row);raw.append(x);contexts.append(cx);lengths.append(length)
                offsets.append(start-left);context_lengths.append(context.shape[1])
        print('SEGMENTED',r['subject'],r['recording'],flush=True)
    raw=np.stack(raw);contexts=np.stack(contexts);lengths=np.array(lengths);offsets=np.array(offsets);context_lengths=np.array(context_lengths)
    np.savez(out/'segments.npz',raw=raw,contexts=contexts,lengths=lengths,offsets=offsets,context_lengths=context_lengths)
    replay.write(out/'segments.json',rows);replay.write_csv(out/'segment_manifest.csv',rows)
    replay.write(out/'all_gate_events.json',all_events);replay.write_csv(out/'excluded_segment_candidates.csv',excluded)
    split={k:np.array([i for i,r in enumerate(rows) if r['split']==k]) for k in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not {(r['recording'],r['block']) for r in rows if r['split']==a}&{(r['recording'],r['block']) for r in rows if r['split']==b}
    for k,ix in split.items():assert {rows[i]['word'] for i in ix}==set(contract['classes'])
    summary=dict(usable_by_split={k:len(v) for k,v in split.items()},
                 original_targets=dict(train=1507,validation=490,test=499),
                 usable_by_subject={s:dict(Counter(r['split'] for r in rows if r['subject']==s)) for s in sorted({r['subject'] for r in rows})},
                 exclusions_by_split={k:dict(Counter(r['reason'] for r in excluded if r['split']==k)) for k in split})
    protocol=dict(architecture='H7SmallCNN 32/64/128, kernels15/9/7, stride2, pool2/2/24, dropout0.35, 87434 parameters',
                  original_model=str(OLD),replay_reference=str(REPLAY),detector=selected['config'],
                  min_probability=0.,min_margin=0.,classes=contract['classes'],
                  selection='Minimum cross entropy on uniquely labelled validation auto-segments; same 3 seeds,120-epoch cap,patience20 as original CNN.',
                  training=dict(seeds=SEEDS,epochs=120,patience=20,batch=32,optimizer='AdamW',lr=.001,weight_decay=.001),
                  augmentation='Independently jitter start/end by +/-100ms from continuous same-trial context, constrained to <=6144 samples and <=event emission. Same original +/-40ms zero-filled shift and per-channel gain0.9-1.1.',
                  normalization='Fit channel mean/std on unpadded eligible automatic TRAIN segments; refit using train+validation only for final model.',
                  labels='Gate receives no cues. Posthoc cue-proxy match plus unique segment, no cross-trial padding, no clipping. Unmatched/ambiguous segments not assigned words.',
                  final_evaluation='All original499 test targets and all detector outputs, including unmatched background and misses. No confidence tuning. Same detector and event list for both CNNs.',
                  quantization='PTQ candidates MinMax/99.99/99.9/99.5; training-only balanced calibration; minimum development validation CE selects method, then final-training recalibration.',
                  caveats=['Previously viewed test scores and detector chosen on prior validation; exploratory seen-person comparison, not blind or LOSO.',
                           'Cue tags are proxy truth, not manually annotated true muscle boundaries.',
                           'Training uses fewer automatically segmentable trials; preprocessing, normalization and boundary augmentation change with the training data.',
                           '10-word model has no rest/unknown output; fixed zero confidence threshold cannot reduce number of background triggers.',
                           'No deployment, no STM32 timing measurement. Detector release remains320ms.'])
    replay.write(out/'protocol.json',protocol);replay.write(out/'dataset_summary.json',summary)
    replay.write(out/'source_sha256.json',provenance)
    shutil.copy2(__file__,out/Path(__file__).name)
    for path in ['semg/activity_cnn.py','experiments/replay_activity_cnn.py','experiments/train_small_cnn.py','experiments/calibrate_h7_cnn.py','experiments/export_h7_cnn.py']:
        shutil.copy2(ROOT/path,out/Path(path).name)
    print('DATASET',summary,flush=True)


def load_data(out):
    rows=replay.read(out/'segments.json')
    with np.load(out/'segments.npz') as z:data={k:z[k] for k in z.files}
    split={k:np.array([i for i,r in enumerate(rows) if r['split']==k]) for k in ['train','validation','test']}
    return rows,data,split


def augmented_batch(data,indices,norm):
    batch=np.zeros((len(indices),2,LENGTH),np.float32)
    mean=np.asarray(norm['mean'])[:,None];scale=np.asarray(norm['scale'])[:,None]
    for j,i in enumerate(indices):
        start=int(data['offsets'][i]);end=start+int(data['lengths'][i])
        ds,de=np.random.randint(-JITTER,JITTER+1,2)
        a,b=boundaries(start,end,0,int(data['context_lengths'][i]),ds,de)
        batch[j,:,:b-a]=(data['contexts'][i,:,a:b]-mean)/scale
    return cnn.augmentation(torch.from_numpy(batch),shift_samples=80)


def train_epoch(net,optim,data,y,ix,norm):
    net.train();loss_sum=correct=0
    order=np.random.permutation(ix)
    for pos in range(0,len(order),32):
        part=order[pos:pos+32];optim.zero_grad(set_to_none=True)
        logits=net(augmented_batch(data,part,norm));loss=nn.functional.cross_entropy(logits,y[part])
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optim.step()
        loss_sum+=float(loss.detach())*len(part);correct+=int((logits.argmax(1)==y[part]).sum())
    return loss_sum/len(ix),correct/len(ix)


def train(out):
    assert not (out/'model.pt').exists()
    rows,data,split=load_data(out);protocol=replay.read(out/'protocol.json');classes=protocol['classes']
    y=torch.tensor([classes.index(r['word']) for r in rows]);raw=data['raw'];lengths=data['lengths']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    x,norm=cnn.normalize(raw,lengths,split['train']);histories=[];best=float('inf');selected=None
    began=time.monotonic()
    for seed in SEEDS:
        cnn.set_seed(seed);net=cnn.H7SmallCNN(widths=(32,64,128));optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        history=[];local=float('inf');stale=0
        for epoch in range(1,121):
            tl,ta=train_epoch(net,optim,data,y,split['train'],norm)
            vl,va,_=cnn.evaluate(net,x,y,split['validation'])
            history.append(dict(epoch=epoch,train_loss=tl,train_accuracy_augmented=ta,validation_loss=vl,validation_accuracy=va))
            if vl<local-1e-4:
                local=vl;stale=0
                if vl<best:
                    best=vl;selected=dict(seed=seed,epoch=epoch,validation_loss=vl,validation_accuracy=va)
                    torch.save(dict(state_dict=net.state_dict(),normalization=norm,selected=selected,
                                    training_fingerprints=[rows[i]['fingerprint'] for i in split['train']]),out/'selection_checkpoint.pt')
            else:stale+=1
            if epoch==1 or epoch%10==0:print('SEED',seed,'EPOCH',epoch,'TRAIN',round(ta,4),'VAL',round(va,4),'LOSS',round(vl,4),flush=True)
            if stale>=20:break
        histories.append(dict(seed=seed,epochs=history));replay.write(out/'selection_history.json',histories)
    replay.write(out/'selected_training.json',selected)
    print('TRAINING SELECTION FROZEN',selected,flush=True)
    final_ix=np.sort(np.r_[split['train'],split['validation']]);x,norm=cnn.normalize(raw,lengths,final_ix)
    cnn.set_seed(selected['seed']);net=cnn.H7SmallCNN(widths=(32,64,128));optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    history=[]
    for epoch in range(1,selected['epoch']+1):
        tl,ta=train_epoch(net,optim,data,y,final_ix,norm);history.append(dict(epoch=epoch,loss=tl,accuracy_augmented=ta))
        if epoch==1 or epoch%10==0:print('REFIT',epoch,'/',selected['epoch'],'TRAIN',round(ta,4),flush=True)
    replay.write(out/'refit_history.json',history)
    checkpoint=dict(schema='semg_h7_cnn_auto_segment_v1',state_dict=net.state_dict(),normalization=norm,
                    classes=classes,channels=[0,1],input_length=LENGTH,selected=selected,protocol=protocol,
                    training_fingerprints=[rows[i]['fingerprint'] for i in final_ix])
    torch.save(checkpoint,out/'model.pt')
    restored=cnn.h7_from_checkpoint(torch.load(out/'model.pt',map_location='cpu',weights_only=True)).eval();net.eval()
    with torch.inference_mode():torch.testing.assert_close(net(x[final_ix[:8]]),restored(x[final_ix[:8]]),rtol=0,atol=0)
    replay.write(out/'training_report.json',dict(selected=selected,elapsed_seconds=time.monotonic()-began,
                 parameters=sum(p.numel() for p in net.parameters()),final_training_segments=len(final_ix),roundtrip_verified=True,
                 test_evaluated=False))
    print('TRAINING COMPLETE',flush=True)


def export(net,path,x,indices):
    net.eval()
    torch.onnx.export(fused_layers(net),torch.zeros(1,2,1,LENGTH),str(path),opset_version=17,dynamo=False,
                      input_names=['input'],output_names=['logits'],do_constant_folding=True)
    onnx.checker.check_model(onnx.load(path))
    got=infer(path,x,indices)
    with torch.inference_mode():
        want=np.concatenate([net(torch.from_numpy(x[ix])).numpy() for ix in np.array_split(indices,max(1,int(np.ceil(len(indices)/32))))])
    np.testing.assert_allclose(got,want,rtol=1e-4,atol=1e-4)
    pre=path.with_name(path.stem+'_preprocessed.onnx')
    quant_pre_process(str(path),str(pre),skip_optimization=True)
    return pre


def quantization(out):
    target=out/'export';target.mkdir(exist_ok=False)
    tmp=target/'temporary';tmp.mkdir();tempfile.tempdir=str(tmp.resolve())
    torch.set_num_threads(2)
    rows,data,split=load_data(out);classes=replay.read(out/'protocol.json')['classes']
    y=torch.tensor([classes.index(r['word']) for r in rows])
    cp=torch.load(out/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    x,norm=cnn.normalize(data['raw'],data['lengths'],split['train']);x=x.numpy();assert norm==cp['normalization']
    pre=export(cnn.h7_from_checkpoint(cp),target/'development_fp32.onnx',x,split['validation'])
    ix=representative(rows,split['train']);candidates=[]
    for pct in [None,99.99,99.9,99.5]:
        name='minmax' if pct is None else f'percentile_{pct}'
        path=target/(name+'_development.onnx');quantize(pre,path,x,ix,pct)
        logits=infer(path,x,split['validation'])
        loss=float(nn.functional.cross_entropy(torch.from_numpy(logits),y[split['validation']]))
        accuracy=float(np.mean(logits.argmax(1)==y[split['validation']].numpy()))
        candidates.append(dict(name=name,percentile=pct,validation_loss=loss,validation_accuracy=accuracy))
        print('QUANT VALIDATION',name,accuracy,loss,flush=True)
    chosen=min(candidates,key=lambda r:r['validation_loss'])
    replay.write(target/'quantization_selection.json',dict(selected=chosen,candidates=candidates,
                  calibration_fingerprints=[rows[i]['fingerprint'] for i in ix],test_used=False))
    print('QUANTIZATION FROZEN',chosen,flush=True)
    cp=torch.load(out/'model.pt',map_location='cpu',weights_only=True);net=cnn.h7_from_checkpoint(cp)
    final_ix=np.sort(np.r_[split['train'],split['validation']])
    x,norm=cnn.normalize(data['raw'],data['lengths'],final_ix);x=x.numpy();assert norm==cp['normalization']
    pre=export(net,target/'model_fp32.onnx',x,final_ix[:100])
    ix=representative(rows,final_ix)
    quantize(pre,target/'model_int8_qdq.onnx',x,ix,chosen['percentile'])
    replay.write(target/'final_calibration.json',dict(selected=chosen,calibration_fingerprints=[rows[i]['fingerprint'] for i in ix],test_used=False))
    replay.write(target/'resource_estimate.json',resource_report(net))
    original=replay.read(OLD/'deployment_calibrated/input_contract.json')
    original.update(normalization=norm,preprocessing=dict(fs=2000,center=32768,continuous_causal_filter=True,
                    detector=replay.read(out/'protocol.json')['detector'],extra_cue_trimming=False),
                    requires='Causal activity gate segments; same 5s initial calibration, no cue boundaries; not validated on STM32.')
    replay.write(target/'input_contract.json',original)
    print('EXPORT COMPLETE',flush=True)


def compare(out):
    assert not (out/'comparison.json').exists()
    records=replay.read(REPLAY/'recordings.json');protocol=replay.read(out/'protocol.json');cfg=GateConfig(**protocol['detector'])
    classes=protocol['classes'];results={};all_predictions={}
    norms={key:replay.read(path)['normalization'] for key,path in [('old',OLD/'deployment_calibrated/input_contract.json'),('new',out/'export/input_contract.json')]}
    for label,path in [('old_fp32',OLD/'deployment/model_fp32.onnx'),('old_int8',OLD/'deployment_calibrated/model_int8_qdq.onnx'),
                       ('new_fp32',out/'export/model_fp32.onnx'),('new_int8',out/'export/model_int8_qdq.onnx')]:
        events,states=replay.run(REPLAY,records,cfg,'test',replay.session(path),norms[label.split('_')[0]],classes)
        all_predictions[label]=events
        replay.write(out/(label+'_events.json'),events);replay.write_csv(out/(label+'_events.csv'),events)
        trials=replay.trial_diagnostics(events,records,0.)
        replay.write_csv(out/(label+'_trials.csv'),trials)
        results[label]=dict(overall=replay.metrics(events,records,'test'),
                           by_subject={s:replay.metrics(events,records,'test',subject=s) for s in sorted({r['subject'] for r in records})})
        print('REPLAY',label,results[label]['overall'],flush=True)
    reference=replay.read(REPLAY/'test_report.json')
    for kind in ['fp32','int8']:
        assert results['old_'+kind]['overall']==reference['results'][kind]['selected_rejection']
        old=all_predictions['old_'+kind];new=all_predictions['new_'+kind]
        fields=['recording','event_id','kind','start','end','emitted','matched_trial','matched_word','reason']
        assert [[e.get(k) for k in fields] for e in old]==[[e.get(k) for k in fields] for e in new]
        def correct_set(events):return {(e['recording'],e['matched_trial']) for e in events if e['candidate'] and e['matched_trial'] is not None and e['candidate']==e['matched_word']}
        a,b=correct_set(old),correct_set(new)
        results['paired_'+kind]=dict(new_only_correct=len(b-a),old_only_correct=len(a-b),both_correct=len(a&b),both_wrong_or_missed=499-len(a|b))
    # Check export numerics on all test candidates against restored PyTorch.
    cp=torch.load(out/'model.pt',map_location='cpu',weights_only=True);net=cnn.h7_from_checkpoint(cp).eval();torch.set_num_threads(2)
    norm=norms['new'];runtime=replay.session(out/'export/model_fp32.onnx');checked=0
    for r in records:
        with np.load(REPLAY/'cache'/(r['recording']+'.npz')) as z:y=z['filtered']
        for e in all_predictions['new_fp32']:
            if e['recording']!=r['recording'] or e['candidate'] is None:continue
            a,b=e['start']-r['first'],e['end']-r['first'];x=np.zeros((1,2,LENGTH),np.float32)
            x[0,:,:b-a]=((y[a:b]-np.array(norm['mean']))/np.array(norm['scale'])).T
            with torch.inference_mode():want=net(torch.from_numpy(x)).numpy()
            got=runtime.run(None,{'input':x[:,:,None,:]})[0]
            np.testing.assert_allclose(got,want,rtol=1e-4,atol=1e-4);checked+=1
    for path,h in replay.read(out/'source_sha256.json').items():assert replay.digest(path)==h,path
    replay.write(out/'comparison.json',dict(results=results,test_targets=499,all_test_targets_retained=True,
                  same_detector_events=True,original_replay_reproduced=True,onnx_pytorch_test_candidates_checked=checked,
                  sources_models_and_cache_unchanged=True,no_deployment=True))
    print('COMPARISON COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','train','quantize','compare','all'],default='all');args=p.parse_args()
    out=args.output.resolve()
    stages={'prepare':prepare,'train':train,'quantize':quantization,'compare':compare}
    for phase,fn in stages.items():
        if args.phase in [phase,'all']:
            print('STAGE',phase,'OUTPUT',out,flush=True);fn(out)


if __name__=='__main__':main()
