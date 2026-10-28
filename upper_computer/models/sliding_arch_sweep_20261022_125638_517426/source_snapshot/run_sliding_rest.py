"""User-run sliding SpeechNet + rest experiment. No hardware access.

Balanced train windows; validation-only model/INT8/decoder selection; frozen
499-command test with independent command scoring. No gate-derived labels.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import gc
import shutil
import sys
import tempfile
import time
import traceback

import run_speechnet_comparison as previous
from train_small_cnn import np, torch, nn
from speechnet_two_channel import SpeechNet2, ExportModel
from sliding_speech import (FS, STEP, REST, split_runs, window_ends, training_label,
                            balanced_indices, decode, event_metrics)

cnn, replay, auto = previous.cnn, previous.replay, previous.auto
ROOT = cnn.ROOT
REFERENCE = ROOT/'models/speechnet2_20261021_171418'
WINDOWS = [800, 1400]
SEEDS = [20261010, 20261011, 20261012]
PER_CLASS = 400


class SlidingNet(SpeechNet2):
    def __init__(self, window_ms):
        super().__init__('time_scaled', 11)
        self.window_ms = int(window_ms)
        self.length = self.window_ms*2
        width = self.length
        for kernel, pool in zip([16, 64, 32], [8, 4, 4]):
            width = (width+1)//pool
        self.layers[20] = nn.AvgPool2d((1, width))


def restore(cp):
    net = SlidingNet(cp['window_ms'])
    net.load_state_dict(cp['state_dict'])
    return net


def read_records():
    return replay.read(auto.REPLAY/'recordings.json')


def load_waves(records):
    waves = {}
    for record in records:
        with np.load(auto.REPLAY/'cache'/(record['recording']+'.npz')) as z:
            waves[record['recording']] = z['filtered']
    return waves


def verify(out):
    previous.verify_hashes(replay.read(out/'inputs_sha256.json'))


def seal(out, marker, paths):
    replay.write(out/marker, {str(p): replay.digest(p) for p in paths})


def check_seal(out, marker):
    previous.verify_hashes(replay.read(out/marker))


def prepare(out):
    # The frozen prior run's source hashes anchor the same raw files, split and cache.
    hashes = replay.read(REFERENCE/'inputs_sha256.json')
    hashes.update({str(REFERENCE/name): replay.digest(REFERENCE/name) for name in
                   ['comparison.json', 'fp32_events.json', 'int8_events.json']})
    previous.verify_hashes(hashes)
    files = previous.dependencies()+[Path(__file__), ROOT/'experiments/sliding_speech.py']
    hashes.update({str(p): replay.digest(p) for p in files})
    replay.write(out/'inputs_sha256.json', hashes)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for p in files:
        shutil.copy2(p, snapshot/p.name)
    for p in (ROOT/'experiments/third_party').glob('SilentWear-*'):
        shutil.copy2(p, snapshot/p.name)
    records = read_records()
    classes = replay.read(previous.SOURCE/'protocol.json')['classes']+['rest']
    assert len(classes) == 11 and classes[REST] == 'rest'
    totals = Counter(t['split'] for r in records for t in r['annotations'])
    assert totals['train'] == 1507 and totals['validation'] == 490 and totals['test'] == 499
    # The split is inherited at whole-round level BEFORE windows are generated.
    groups = {s: {(r['recording'],t['block']) for r in records for t in r['annotations'] if t['split']==s}
              for s in ['train','validation','test']}
    assert not groups['train'] & groups['validation']
    assert not (groups['train'] | groups['validation']) & groups['test']
    summaries = {}
    for ms in WINDOWS:
        length = 2*ms
        rows = []
        excluded = Counter()
        for r in records:
            with np.load(auto.REPLAY/'cache'/(r['recording']+'.npz')) as z:
                clip_ends, clips = z['ends'], z['clipped']
            for t in r['annotations']:
                if t['split'] == 'excluded':
                    continue
                for end in window_ends(r, t['prepare'], t['context_end'], length):
                    start = end-length
                    label = training_label(t, start, end, classes)
                    if label is None:
                        excluded['ambiguous_transition'] += 1
                        continue
                    if np.any(clips[(clip_ends > start) & (clip_ends-40 < end)]):
                        excluded['clipping'] += 1
                        continue
                    rows.append(dict(recording=r['recording'], subject=r['subject'], split=t['split'],
                                     trial_id=t['trial_id'], block=t['block'], label=label,
                                     start=start, end=end, offset=start-r['first'],
                                     fingerprint=f'{r["recording"]}:{start}:{end}'))
        for s in ['train','validation','test']:
            assert {x['label'] for x in rows if x['split']==s} == set(range(11)), (ms,s)
        validation = [i for i,x in enumerate(rows) if x['split']=='validation']
        val_ix = balanced_indices(rows, validation, 200, 20261022, False)
        replay.write(out/f'windows_{ms}.json', rows)
        replay.write(out/f'validation_{ms}.json', val_ix.tolist())
        summaries[str(ms)] = dict(candidates_by_split={s:dict(Counter(classes[x['label']] for x in rows if x['split']==s))
                                                   for s in ['train','validation','test']},
                                  validation_sample_counts=dict(Counter(classes[rows[i]['label']] for i in val_ix)),
                                  excluded=dict(excluded), candidate_windows=len(rows))
    protocol = dict(classes=classes, window_ms_candidates=WINDOWS, step_ms=100, seeds=SEEDS,
                    max_epochs=100, early_stop=10, batch=44, windows_per_class_per_epoch=PER_CLASS,
                    optimizer='Adam lr0.001 weight_decay0.0001',
                    scheduler='ReduceLROnPlateau factor0.1 patience2; selected trajectory reused for refit',
                    model='SpeechNet2 temporal16/64/32 spatial2x1,1x1; widths8/16/16/32/32; dropout0.5; 11 outputs',
                    train_balance='Exactly400 per class per epoch; equal subjects then random trial and window; rest1/11. Epoch seed changes.',
                    window_labels='Cue-proxy word if >=75% window overlaps action and end inside action after150ms; '
                                  'rest only fully inside guarded prep/rest. Ambiguous windows ignored in supervised loss, INCLUDED in replay.',
                    normalization='Per-channel statistics of unique continuous train contexts only; final train+validation, never test.',
                    validation='Fixed unique round-robin subject/trial window subset, up to200/class; macro class CE chooses seed/window/epoch.',
                    augmentation='Independent channel gain0.9-1.1; many clock-aligned window positions, no zero-padding or waveform time scaling.',
                    replay='Infer every100ms without gate/cue input on continuous held-out runs. Runs cut ONLY at original split boundaries; '
                           'decoder resets there, not at trial boundaries; initial window warmup and predicted-rest arming required.',
                    split_boundary_note='No window or decoder history crosses train/validation/test boundary. All499 test commands retained even during warmup. '
                                        'Causal filter cache retains preceding filter history, as in existing experiments.',
                    decoder_selection='Development checkpoint+selected development INT8; maximize validation clean success, then eventF1, then fewer rest outputs.',
                    score='Common emitted-time response [cue_action,cue_rest+300ms); strict success requires one output in entire trial context. '
                          'Prior models rescored by same rule; NOT directly identical to earlier gate-overlap68.14%.',
                    no_test_selection=True, no_deployment=True,
                    caveats=['Cue timings are proxy truth, not actual muscle boundaries.',
                             'Correlated sliding windows are NOT independent utterances; whole-round split preserved.',
                             'Previously viewed seen-person test; not fresh blind or cross-subject evaluation.',
                             'Balanced training does not guarantee no rest bias or no false alarms; continuous metrics remain essential.'])
    replay.write(out/'protocol.json', protocol)
    replay.write(out/'dataset_summary.json', summaries)
    seal(out,'dataset_sha256.json',[out/'protocol.json',out/'dataset_summary.json']+
         [out/f'{prefix}_{ms}.json' for ms in WINDOWS for prefix in ['windows','validation']])
    print('PREPARED',summaries,flush=True)


def normalization(records, waves, splits):
    total = np.zeros(2,np.float64)
    squares = np.zeros(2,np.float64)
    count = 0
    for r in records:
        for t in r['annotations']:
            if t['split'] in splits:
                x = waves[r['recording']][t['prepare']-r['first']:t['context_end']-r['first']].astype(np.float64)
                total += x.sum(0)
                squares += (x*x).sum(0)
                count += len(x)
    mean = total/count
    scale = np.sqrt(np.maximum(squares/count-mean*mean,1e-12))
    return dict(mean=mean.tolist(),scale=scale.tolist(),samples=count,fit_splits=sorted(splits))


def batch(rows, indices, waves, norm, augment=False):
    x = np.stack([waves[rows[int(i)]['recording']][rows[int(i)]['offset']:
                 rows[int(i)]['offset']+rows[int(i)]['end']-rows[int(i)]['start']].T for i in indices])
    x = ((x-np.asarray(norm['mean'])[None,:,None])/np.asarray(norm['scale'])[None,:,None]).astype(np.float32)
    if augment:
        x *= np.random.uniform(.9,1.1,(len(x),2,1)).astype(np.float32)
    return x


def evaluate(net, rows, indices, waves, norm):
    net.eval()
    losses, predictions, labels = [], [], []
    with torch.inference_mode():
        for start in range(0,len(indices),64):
            part = indices[start:start+64]
            logits = net(torch.from_numpy(batch(rows,part,waves,norm)))
            y = torch.tensor([rows[int(i)]['label'] for i in part])
            losses.extend(nn.functional.cross_entropy(logits,y,reduction='none').tolist())
            predictions.extend(logits.argmax(1).tolist())
            labels.extend(y.tolist())
    labels,predictions,losses = map(np.asarray,[labels,predictions,losses])
    macro_loss = float(np.mean([losses[labels==c].mean() for c in range(11)]))
    macro_accuracy = float(np.mean([(predictions[labels==c]==c).mean() for c in range(11)]))
    return macro_loss,macro_accuracy


def train_epoch(net, optimizer, rows, indices, waves, norm, seed):
    order = balanced_indices(rows,indices,PER_CLASS,seed,True)
    assert Counter(rows[int(i)]['label'] for i in order) == {c:PER_CLASS for c in range(11)}
    net.train()
    loss_total = correct = 0
    for start in range(0,len(order),44):
        part = order[start:start+44]
        y = torch.tensor([rows[int(i)]['label'] for i in part])
        optimizer.zero_grad(set_to_none=True)
        logits = net(torch.from_numpy(batch(rows,part,waves,norm,True)))
        loss = nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),5.)
        optimizer.step()
        loss_total += float(loss.detach())*len(part)
        correct += int((logits.argmax(1)==y).sum())
    return loss_total/len(order),correct/len(order)


def cp_for(net,norm,selected,classes,rows,indices):
    return dict(schema='sliding_rest_v1',state_dict=net.state_dict(),window_ms=net.window_ms,
                classes=classes,normalization=norm,selected=selected,
                training_fingerprints=[rows[int(i)]['fingerprint'] for i in indices])


def fit(out):
    check_seal(out,'dataset_sha256.json')
    records=read_records();waves=load_waves(records)
    norm=normalization(records,waves,{'train'})
    classes=replay.read(out/'protocol.json')['classes']
    best=float('inf');selected=None;histories=[];began=time.monotonic()
    for ms in WINDOWS:
        rows=replay.read(out/f'windows_{ms}.json')
        train_ix=np.asarray([i for i,r in enumerate(rows) if r['split']=='train'])
        val_ix=np.asarray(replay.read(out/f'validation_{ms}.json'))
        for seed in SEEDS:
            cnn.set_seed(seed);net=SlidingNet(ms)
            optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=2)
            local=float('inf');stale=0;history=[]
            for epoch in range(1,101):
                lr=float(optimizer.param_groups[0]['lr'])
                tl,ta=train_epoch(net,optimizer,rows,train_ix,waves,norm,seed+epoch)
                vl,va=evaluate(net,rows,val_ix,waves,norm)
                if not np.isfinite(vl): raise RuntimeError('Nonfinite validation loss')
                history.append(dict(epoch=epoch,lr=lr,train_loss=tl,train_accuracy=ta,
                                    validation_macro_loss=vl,validation_macro_accuracy=va))
                if vl<local:local=vl;stale=0
                else:stale+=1
                if vl<best:
                    best=vl;selected=dict(window_ms=ms,seed=seed,epoch=epoch,validation_macro_loss=vl,
                        validation_macro_accuracy=va,learning_rates=[h['lr'] for h in history])
                    torch.save(cp_for(net,norm,selected,classes,rows,train_ix),out/'selection_checkpoint.pt')
                scheduler.step(vl)
                print(f'SELECT window={ms} seed={seed} epoch={epoch}/100 train={ta:.3%} val_macro={va:.3%} CE={vl:.4f}',flush=True)
                if stale>=10:break
            histories.append(dict(window_ms=ms,seed=seed,epochs=history))
            replay.write(out/'selection_history.json',histories)
            del net,optimizer,scheduler;gc.collect()
    replay.write(out/'selected_training.json',selected)
    print('MODEL SELECTION FROZEN',selected,flush=True)
    rows=replay.read(out/f'windows_{selected["window_ms"]}.json')
    ix=np.asarray([i for i,r in enumerate(rows) if r['split'] in ['train','validation']])
    norm=normalization(records,waves,{'train','validation'})
    cnn.set_seed(selected['seed']);net=SlidingNet(selected['window_ms'])
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    history=[]
    for epoch,lr in enumerate(selected['learning_rates'],1):
        for group in optimizer.param_groups:group['lr']=lr
        loss,acc=train_epoch(net,optimizer,rows,ix,waves,norm,selected['seed']+epoch)
        history.append(dict(epoch=epoch,lr=lr,loss=loss,accuracy=acc))
        print(f'REFIT {epoch}/{selected["epoch"]} train={acc:.3%}',flush=True)
    torch.save(cp_for(net,norm,selected,classes,rows,ix),out/'model.pt')
    replay.write(out/'refit_history.json',history)
    reloaded=restore(torch.load(out/'model.pt',weights_only=True,map_location='cpu')).eval();net.eval()
    example=torch.from_numpy(batch(rows,ix[:3],waves,norm))
    with torch.inference_mode():torch.testing.assert_close(net(example),reloaded(example),rtol=0,atol=0)
    replay.write(out/'training_report.json',dict(selected=selected,elapsed_seconds=time.monotonic()-began,
        class_draws_per_epoch={c:PER_CLASS for c in classes},parameters=sum(p.numel() for p in net.parameters()),
        distinct_final_training_trials=len({(rows[int(i)]['recording'],rows[int(i)]['trial_id']) for i in ix}),
        test_used=False,checkpoint_roundtrip=True))
    seal(out,'training_sha256.json',[out/name for name in ['model.pt','selection_checkpoint.pt','selected_training.json',
         'training_report.json','selection_history.json']])


def export_fp(net,path,x):
    net.eval()
    torch.onnx.export(ExportModel(net).eval(),torch.zeros(1,2,1,net.length),str(path),opset_version=17,
                      dynamo=False,input_names=['input'],output_names=['logits'],do_constant_folding=True)
    auto.onnx.checker.check_model(auto.onnx.load(path))
    actual=auto.infer(path,x,np.arange(len(x)))
    with torch.inference_mode():expected=net(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
    pre=path.with_name(path.stem+'_preprocessed.onnx')
    auto.quant_pre_process(str(path),str(pre),skip_optimization=True)
    return pre


def resource_estimate(net):
    layers=ExportModel(net).layers
    x=torch.zeros(1,1,2,net.length);entries=[];macs=0
    with torch.inference_mode():
        for layer in layers:
            before=x.numel();x=layer(x)
            if isinstance(layer,nn.Conv2d):macs+=x.numel()*layer.in_channels*np.prod(layer.kernel_size).item()
            elif isinstance(layer,nn.Linear):macs+=layer.in_features*layer.out_features
            entries.append(dict(type=type(layer).__name__,elements=x.numel(),shape=list(x.shape),adjacent_bytes=before+x.numel()))
    tensors=2*net.length+sum(e['elements'] for e in entries if e['type'] not in ['Identity','Flatten','Dropout'])
    weights=sum(p.numel() for n,p in layers.named_parameters() if n.endswith('weight'))
    biases=sum(p.numel() for n,p in layers.named_parameters() if n.endswith('bias'))
    return dict(parameters=sum(p.numel() for p in net.parameters()),macs_per_window=int(macs),
                macs_per_second_at_10hz=int(macs*10),window_ms=net.window_ms,
                int8_weights_int32_bias_bytes=weights+4*biases,int8_tensor_inventory_bytes=tensors,
                planning_reserve_bytes=256*1024,int8_inventory_plus_reserve_bytes=tensors+256*1024,
                layers=entries,st_compiler_verified=False,board_latency_verified=False)


def infer_runs(path,cp,records,waves,split):
    runtime=replay.session(path)
    checked=previous.CheckedSession(path,restore(cp)) if path.name.endswith('fp32.onnx') else None
    mean=np.asarray(cp['normalization']['mean']);scale=np.asarray(cp['normalization']['scale'])
    length=2*cp['window_ms']
    recnames=[];runs=[];ends=[];probs=[]
    for r in records:
        for n,(start,end) in enumerate(split_runs(r,split)):
            for pos in window_ends(r,start,end,length):
                wave=waves[r['recording']][pos-length-r['first']:pos-r['first']]
                assert len(wave)==length
                x=((wave-mean)/scale).T[None,:,None,:].astype(np.float32)
                # Check a deterministic prefix for each held-out run; every window still inferred.
                engine=checked if checked is not None and len(ends)%100==0 else runtime
                logits=engine.run(None,{'input':x})[0][0]
                p=np.exp(logits-logits.max());p/=p.sum()
                recnames.append(r['recording']);runs.append(n);ends.append(pos);probs.append(p)
        print('SLIDING INFERENCE',split,path.name,r['recording'],flush=True)
    return dict(recording=np.asarray(recnames),run=np.asarray(runs),end=np.asarray(ends),
                probabilities=np.asarray(probs,np.float32),numeric_checks=np.asarray(0 if checked is None else checked.checked))


def choose_decoder(trace,records,classes):
    scores=[]
    for p in [.5,.65,.8]:
        for confirm in [2,3]:
            for rest in [2,3]:
                for margin in [0.,.15]:
                    config=dict(probability=p,confirm=confirm,rest_frames=rest,margin=margin)
                    metrics,_=event_metrics(decode(trace,config,classes),records,'validation')
                    scores.append(dict(config=config,metrics=metrics))
    best=max(scores,key=lambda s:(s['metrics']['clean_success_rate'],s['metrics']['event_f1'],
                                 -s['metrics']['rest_proxy_outputs_per_min']))
    return dict(selected=best,candidates=scores,selection_split='validation',test_used=False)


def export_and_select(out):
    check_seal(out,'dataset_sha256.json');check_seal(out,'training_sha256.json')
    records=read_records();waves=load_waves(records)
    dev=torch.load(out/'selection_checkpoint.pt',weights_only=True,map_location='cpu')
    rows=replay.read(out/f'windows_{dev["window_ms"]}.json')
    train_ix=np.asarray([i for i,r in enumerate(rows) if r['split']=='train'])
    val_ix=np.asarray(replay.read(out/f'validation_{dev["window_ms"]}.json'))
    assert dev['training_fingerprints']==[rows[int(i)]['fingerprint'] for i in train_ix]
    assert dev['normalization']==normalization(records,waves,{'train'})
    calibration=balanced_indices(rows,train_ix,50,20261022,False)
    target=out/'export';target.mkdir(exist_ok=True)
    xcal=batch(rows,calibration,waves,dev['normalization'])
    xval=batch(rows,val_ix,waves,dev['normalization'])
    labels=np.asarray([rows[int(i)]['label'] for i in val_ix])
    pre=export_fp(restore(dev),target/'development_fp32.onnx',xval[:16])
    candidates=[]
    for percentile in [None,99.99,99.9,99.5]:
        name='minmax' if percentile is None else f'percentile_{percentile}'
        dest=target/(name+'.onnx')
        auto.quantize(pre,dest,xcal,np.arange(len(xcal)),percentile)
        logits=auto.infer(dest,xval,np.arange(len(xval)))
        losses=nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(labels),reduction='none').numpy()
        candidates.append(dict(name=name,percentile=percentile,validation_macro_loss=float(np.mean([
            losses[labels==c].mean() for c in range(11)])),validation_macro_accuracy=float(np.mean([
            (logits.argmax(1)[labels==c]==c).mean() for c in range(11)]))))
        print('QUANT VALIDATION',candidates[-1],flush=True)
    chosen=min(candidates,key=lambda c:c['validation_macro_loss'])
    replay.write(target/'quantization_selection.json',dict(selected=chosen,candidates=candidates,
        calibration_fingerprints=[rows[int(i)]['fingerprint'] for i in calibration],test_used=False))
    trace=infer_runs(target/(chosen['name']+'.onnx'),dev,records,waves,'validation')
    np.savez_compressed(out/'validation_trace.npz',**trace)
    decoder=choose_decoder(trace,records,dev['classes'])
    replay.write(out/'decoder_selection.json',decoder)
    print('DECODER FROZEN BEFORE TEST',decoder['selected'],flush=True)
    final=torch.load(out/'model.pt',weights_only=True,map_location='cpu')
    fit_ix=np.asarray([i for i,r in enumerate(rows) if r['split'] in ['train','validation']])
    assert final['training_fingerprints']==[rows[int(i)]['fingerprint'] for i in fit_ix]
    assert final['normalization']==normalization(records,waves,{'train','validation'})
    assert not set(final['training_fingerprints']) & {r['fingerprint'] for r in rows if r['split']=='test'}
    calibration=balanced_indices(rows,fit_ix,50,20261022,False)
    xcal=batch(rows,calibration,waves,final['normalization'])
    net=restore(final)
    pre=export_fp(net,target/'model_fp32.onnx',xcal[:16])
    auto.quantize(pre,target/'model_int8_qdq.onnx',xcal,np.arange(len(xcal)),chosen['percentile'])
    previous.check_quantized_graph(target/'model_int8_qdq.onnx')
    replay.write(target/'calibration_manifest.json',dict(calibration_fingerprints=[rows[int(i)]['fingerprint'] for i in calibration],test_used=False))
    resources=resource_estimate(net)
    assert resources['int8_inventory_plus_reserve_bytes'] < 1024*1024
    replay.write(target/'resource_estimate.json',resources)
    old=replay.read(previous.SOURCE/'export/input_contract.json')
    replay.write(target/'input_contract.json',dict(classes=final['classes'],normalization=final['normalization'],
        fs=FS,step_samples=STEP,input_shape=[1,2,1,net.length],filter_sos=old['filter_sos'],
        preprocessing='continuous causal filter; ring buffer; no padding; no activity gate; standardize using saved training statistics',
        decoder=decoder['selected']['config'],initial_state='disarmed; requires predicted rest before command',
        note='FP32 I/O around internalINT8; no STM32 compiler/latency validation. Never use old gate-segment preprocessing.'))
    seal(out,'export_sha256.json',[p for p in target.iterdir() if p.is_file()]+[out/'decoder_selection.json'])


def compare(out):
    check_seal(out,'export_sha256.json');check_seal(out,'training_sha256.json')
    records=read_records();waves=load_waves(records)
    cp=torch.load(out/'model.pt',weights_only=True,map_location='cpu')
    config=replay.read(out/'decoder_selection.json')['selected']['config']
    results={}
    # Re-score prior command outputs, not their old overlap-matching labels.
    old_sources=[('original_int8',previous.SOURCE/'old_int8_events.json'),
                 ('auto32_int8',previous.SOURCE/'new_int8_events.json'),
                 ('gated_speechnet_int8',REFERENCE/'int8_events.json')]
    events_by_model={}
    for name,path in old_sources:
        events_by_model[name]=[e for e in replay.read(path) if e['candidate'] is not None]
    for kind in ['fp32','int8']:
        path=out/'export'/('model_fp32.onnx' if kind=='fp32' else 'model_int8_qdq.onnx')
        trace=infer_runs(path,cp,records,waves,'test')
        np.savez_compressed(out/f'{kind}_trace.npz',**trace)
        events=decode(trace,config,cp['classes'])
        replay.write(out/f'{kind}_events.json',events);replay.write_csv(out/f'{kind}_events.csv',events)
        events_by_model['sliding_'+kind]=events
    for name,events in events_by_model.items():
        overall,rows=event_metrics(events,records,'test')
        assert overall['trials']==499
        results[name]=dict(overall=overall,by_subject={s:event_metrics(events,records,'test',s)[0]
            for s in sorted({r['subject'] for r in records})})
        replay.write_csv(out/(name+'_trials.csv'),rows)
    verify(out)
    replay.write(out/'comparison.json',dict(results=results,test_targets=499,rest_windows_not_in_denominator=True,
        common_emission_time_scoring=True,previous_overlap_scores_not_directly_comparable=True,no_deployment=True,
        target_met_int8=results['sliding_int8']['overall']['end_to_end_recall']>=.85))
    table=[]
    for name,group in results.items():
        for s,m in [('ALL',group['overall'])]+list(group['by_subject'].items()):table.append(dict(model=name,subject=s,**m))
    replay.write_csv(out/'comparison.csv',table)
    lines=['滑动窗口+静息类：完整命令评估','',
        '分母为原499次测试命令，不包含静息窗口。新旧模型按统一输出时刻规则重算，不能直接与旧68.14%混用。',
        '响应区间为提示action至rest后300ms，严格成绩要求整个试次上下文只有一次输出且正确。',
        '只在原划分的连续留出区间运行，区间边界重置状态，不在每条命令边界重置；预热/未触发造成的漏检也计入499。',
        f'选中窗口{cp["window_ms"]}ms，步进100ms；冻结解码设置{config}。','',
        '模型                         正确命令         严格成功率       静息输出/分钟']
    for name,group in results.items():
        m=group['overall'];lines.append(f'{name:26} {m["correct_unique_trials"]}/499 ({m["end_to_end_recall"]:.2%})   {m["clean_success_rate"]:.2%}   {m["rest_proxy_outputs_per_min"]:.3f}')
    lines+=['','新INT8逐人：']
    for s,m in results['sliding_int8']['by_subject'].items():lines.append(f'{s}: {m["correct_unique_trials"]}/{m["trials"]}={m["end_to_end_recall"]:.2%}, 严格{m["clean_success_rate"]:.2%}')
    lines+=['','训练每轮11类各400个窗口，静息占1/11；静息来源按人、试次轮换。',
        '验证损失按11类等权，窗口长度/种子/轮数、量化和解码阈值均只由验证集选择。',
        '训练标签仍由提示代理生成，不等于人工确认的实际肌肉起止；过渡窗口未强行标静息，但全都参与回放推理。',
        '静息误触发是提示静息区输出计数，非真实生活吞咽/走动负例验证。',
        '原测试已查看，属于探索性已见人员比较。窗口很多不代表独立动作样本增加。',
        '尚未部署，资源估算不代表ST实际内存或10Hz实时推理性能保证。']
    (out/'结果汇总.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def preflight(out):
    cnn.set_seed(20261022)
    for ms in WINDOWS:
        net=SlidingNet(ms)
        x=np.random.default_rng(ms).normal(size=(6,2,ms*2)).astype(np.float32)
        optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        loss=nn.functional.cross_entropy(net(torch.from_numpy(x[:2])),torch.tensor([0,REST]))
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
        pre=export_fp(net,out/f'check_{ms}_fp32.onnx',x[:2])
        quant=out/f'check_{ms}_int8.onnx'
        auto.quantize(pre,quant,x,np.arange(len(x)),99.9)
        previous.check_quantized_graph(quant)
        assert np.isfinite(auto.infer(quant,x,np.arange(2))).all()
        replay.write(out/f'resources_{ms}.json',resource_estimate(net))
    replay.write(out/'preflight.json',dict(synthetic_only=True,full_training_started=False,checks_passed=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','prepare','train','export','replay','check'],default='all')
    p.add_argument('--output',type=Path)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    if args.phase in ['train','export','replay'] and args.output is None:p.error('--output required for continuing a prepared run')
    out=(args.output or ROOT/'models'/('sliding_rest_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if args.phase in ['all','prepare','check']:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():p.error('Run directory does not exist')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    temp=out/'temporary';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
    with (out/'run.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(previous.Tee(sys.stdout,log)),redirect_stderr(previous.Tee(sys.stderr,log)):
            try:
                print('OUTPUT',out,'PHASE',args.phase,flush=True)
                if args.phase=='check':preflight(out);print('SYNTHETIC CHECK PASSED',flush=True);return
                if args.phase in ['all','prepare']:prepare(out)
                else:verify(out)
                if args.phase in ['all','train']:
                    if (out/'model.pt').exists():raise RuntimeError('Model exists; refusing to retrain/overwrite')
                    fit(out)
                if args.phase in ['all','export']:
                    if (out/'export_sha256.json').exists():raise RuntimeError('Export completed; use --phase replay')
                    export_and_select(out)
                if args.phase in ['all','replay']:
                    if (out/'comparison.json').exists():raise RuntimeError('Test already completed; refusing to overwrite')
                    compare(out);replay.write(out/'completed.json',dict(complete=True,no_deployment=True))
                print('PHASE COMPLETE',args.phase,out,flush=True)
            except BaseException:
                traceback.print_exc();raise


if __name__=='__main__':main()
