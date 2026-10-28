"""Nine architectures x three seeds on the original sliding/rest split; user-run.

Same 1400ms windows, optimization rules, sampling and fixed command decoder.
Development validation chooses seeds/epochs/PTQ and the recommended architecture.
All final models are frozen before the original 499-command test is replayed.
Resume at completed seed/stage boundaries, never at a partially trained epoch.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import gc
import shutil
import sys
import tempfile
import time
import traceback

import run_sliding_rest as base
from run_personal_sliding_finetune import score, paired_trials

np, torch, nn, io = base.np, base.torch, base.nn, base.replay
SOURCE = base.ROOT/'models/sliding_rest_20261022_103406'
SEEDS = [20261010,20261011,20261012]
SUBJECTS = ['S011','S012','S013','S014','S015']
MAX_EPOCHS, EARLY_STOP = 100, 10


def specs():
    rows = [
        ('baseline',[8,16,16,32,32],1,1,'原结构重新训练，对照随机种子及训练流程'),
        ('tail64',[8,16,16,64,64],1,1,'只加宽后两层'),
        ('mid64',[8,24,32,64,64],1,1,'适度加宽中后层'),
        ('wide2x',[16,32,32,64,64],1,1,'五层通道全部翻倍'),
        ('temporal3',[8,16,16,32,32],3,1,'末层1x1改为1x3时间卷积'),
        ('pool2',[8,16,16,32,32],1,2,'前后两段时间池化'),
        ('mid64_temporal3',[8,24,32,64,64],3,1,'中后层加宽+末层时间卷积'),
        ('mid64_pool2',[8,24,32,64,64],1,2,'中后层加宽+两段时间池化'),
        ('mid64_temporal3_pool2',[8,24,32,64,64],3,2,'中后层加宽+末层时间卷积+两段池化'),
    ]
    return [dict(name=n,widths=w,tail_kernel=k,pool_bins=b,description=d) for n,w,k,b,d in rows]


class SweepNet(base.SlidingNet):
    def __init__(self, architecture):
        super().__init__(1400)
        self.architecture = deepcopy(architecture)
        widths = architecture['widths'];tail = architecture['tail_kernel'];bins = architecture['pool_bins']
        assert len(widths)==5 and tail in [1,3] and bins in [1,2]
        incoming=1
        for index,ch,kernel,padding in zip([0,4,8,12,16],widths,
                [(1,16),(1,64),(1,32),(2,1),(1,tail)],[(0,8),(0,32),(0,16),(0,0),(0,tail//2)]):
            old=self.layers[index]
            if (old.in_channels,old.out_channels,old.kernel_size)!=(incoming,ch,kernel):
                self.layers[index]=nn.Conv2d(incoming,ch,kernel,padding=padding)
                self.layers[index+1]=nn.BatchNorm2d(ch)
            incoming=ch
        self.layers[20]=nn.AvgPool2d((1,22//bins))
        if widths[-1]*bins!=32:self.layers[23]=nn.Linear(widths[-1]*bins,11)


def restore(cp):
    net=SweepNet(cp['architecture']) if 'architecture' in cp else base.SlidingNet(cp['window_ms'])
    net.load_state_dict(cp['state_dict'])
    return net


def checkpoint(net, norm, selected, classes, rows, indices):
    cp=base.cp_for(net,norm,selected,classes,rows,indices)
    cp.update(schema='sliding_architecture_sweep_v1',architecture=deepcopy(net.architecture))
    return cp


def seal(folder, name, files):
    # A completed stage becomes resumable only after the hash manifest is atomic.
    marker=folder/name;temporary=marker.with_suffix('.tmp')
    io.write(temporary,{str(p.resolve()):io.digest(p) for p in files})
    temporary.replace(marker)


def done(folder, name):
    if not (folder/name).exists():return False
    base.check_seal(folder,name)
    return True


def prepare(out):
    assert io.read(SOURCE/'completed.json')['complete']
    hashes={}
    for marker in ['inputs_sha256.json','dataset_sha256.json','training_sha256.json','export_sha256.json']:
        base.check_seal(SOURCE,marker)
        hashes.update(io.read(SOURCE/marker));hashes[str(SOURCE/marker)]=io.digest(SOURCE/marker)
    source_cp=torch.load(SOURCE/'model.pt',map_location='cpu',weights_only=True)
    dev_cp=torch.load(SOURCE/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    rows=io.read(SOURCE/'windows_1400.json');records=base.read_records()
    counts=Counter(t['split'] for r in records for t in r['annotations'])
    assert {k:counts[k] for k in ['train','validation','test']}==dict(train=1507,validation=490,test=499)
    groups={split:{(r['recording'],t['block']) for r in records for t in r['annotations'] if t['split']==split}
            for split in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not (groups['train']|groups['validation'])&groups['test']
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    fit=[i for i,r in enumerate(rows) if r['split'] in ['train','validation']]
    assert dev_cp['training_fingerprints']==[rows[i]['fingerprint'] for i in train]
    assert source_cp['training_fingerprints']==[rows[i]['fingerprint'] for i in fit]
    val=io.read(SOURCE/'validation_1400.json')
    assert len(set(val))==len(val) and all(rows[i]['split']=='validation' for i in val)
    dev_cal=base.balanced_indices(rows,np.array(train),50,20261022,False).tolist()
    lookup={r['fingerprint']:i for i,r in enumerate(rows)}
    final_cal=[lookup[f] for f in io.read(SOURCE/'export/calibration_manifest.json')['calibration_fingerprints']]
    assert all(i in set(fit) for i in final_cal)
    config=dict(schema='sliding_architecture_sweep_v1',base=str(SOURCE),architectures=specs(),seeds=SEEDS,
        window_ms=1400,step_ms=100,classes=source_cp['classes'],max_epochs=MAX_EPOCHS,early_stop=EARLY_STOP,
        batch=44,windows_per_class_per_epoch=400,dropout=.5,
        optimizer=dict(name='Adam',lr=.001,weight_decay=.0001),scheduler=dict(factor=.1,patience=2),
        dev_normalization=dev_cp['normalization'],final_normalization=source_cp['normalization'],
        decoder=io.read(SOURCE/'decoder_selection.json')['selected']['config'],
        counts={k:counts[k] for k in ['train','validation','test']},
        seed_epoch_selection='Minimum fixed validation-window macro cross entropy within each architecture.',
        quantization='Same four candidates: MinMax,99.99,99.9,99.5; development-validation macro CE selection.',
        architecture_selection='Development INT8 continuous validation: highest strict command success, then correct '
            'command recall, fewer wrong response outputs, fewer misses, fewer rest outputs, lower MACs; frozen before test.',
        refit='From scratch on original train+validation, same selected epoch count and LR trajectory; no test adaptation.',
        common_controls='Original split/cache/window labels/validation subset; per-class400 draws, batch44, '
            'gain0.9-1.1, dropout0.5; identical sampler seed for a given seed/epoch. No personal calibration.',
        decoder_policy='Fixed original decoder for every architecture; no new threshold sweep.',
        test_policy='All final models frozen first, then all original499 commands replayed with same runs/grid; '
            'current deployed-candidate base also freshly replayed. Do not promote test winner automatically.',
        caveats=['Previously inspected seen-person test, exploratory not blind/new-day/LOSO.',
                 'Overlapping windows are not independent actions; cue labels are proxies.',
                 'Memory figures are inventory estimates, not ST arena. No board latency/deployment claim.'],
        versions=dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__))
    io.write(out/'protocol.json',config);io.write(out/'recordings.json',records)
    shutil.copy2(SOURCE/'windows_1400.json',out/'windows_1400.json')
    io.write(out/'indices.json',dict(train=train,fit=fit,validation=val,dev_calibration=dev_cal,final_calibration=final_cal))
    resources={a['name']:base.resource_estimate(SweepNet(a)) for a in config['architectures']}
    io.write(out/'planned_resources.json',resources)
    io.write_csv(out/'architecture_plan.csv',[dict(**a,parameters=resources[a['name']]['parameters'],
        macs_per_window=resources[a['name']]['macs_per_window'],
        estimated_inventory_plus_reserve=resources[a['name']]['int8_inventory_plus_reserve_bytes']) for a in config['architectures']])
    files=[Path(__file__).resolve(),base.ROOT/'experiments/run_personal_sliding_finetune.py',SOURCE/'completed.json',SOURCE/'comparison.json']
    for p in files:hashes[str(p)]=io.digest(p)
    io.write(out/'inputs_sha256.json',hashes)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in base.previous.dependencies()+[Path(base.__file__),base.ROOT/'experiments/sliding_speech.py']+files[:2]:
        shutil.copy2(p,snapshot/p.name)
    seal(out,'prepared_sha256.json',[out/n for n in ['protocol.json','recordings.json','windows_1400.json','indices.json','planned_resources.json']])
    print('PREPARED: 9 architectures x 3 seeds; original 1507/490/499 split; fixed decoder',config['decoder'],flush=True)
    for a in config['architectures']:
        m=resources[a['name']];print(a['name'],m['parameters'],'parameters',m['macs_per_window'],'MAC/window',flush=True)


def train_seed(target, architecture, seed, data, cfg):
    target.mkdir(parents=True,exist_ok=True)
    if done(target,'seed_sha256.json'):
        print('RESUME completed seed',architecture['name'],seed,flush=True)
        return
    base.cnn.set_seed(seed);net=SweepNet(architecture)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**cfg['scheduler'])
    history=[];best=float('inf');stale=0;began=time.monotonic()
    for epoch in range(1,cfg['max_epochs']+1):
        lr=float(optimizer.param_groups[0]['lr'])
        tl,ta=base.train_epoch(net,optimizer,data['rows'],data['indices']['train'],data['waves'],cfg['dev_normalization'],seed+epoch)
        vl,va=base.evaluate(net,data['rows'],data['indices']['validation'],data['waves'],cfg['dev_normalization'])
        if not np.isfinite(vl):raise RuntimeError('Nonfinite validation loss')
        history.append(dict(epoch=epoch,lr=lr,train_loss=tl,train_accuracy=ta,validation_macro_loss=vl,validation_macro_accuracy=va))
        if vl<best:
            best=vl;stale=0
            selected=dict(seed=seed,epoch=epoch,window_ms=1400,validation_macro_loss=vl,validation_macro_accuracy=va,
                          learning_rates=[h['lr'] for h in history])
            torch.save(checkpoint(net,cfg['dev_normalization'],selected,cfg['classes'],data['rows'],data['indices']['train']),target/'model.pt')
        else:stale+=1
        scheduler.step(vl)
        io.write(target/'history.json',history)
        print(f'TRAIN {architecture["name"]} seed={seed} epoch={epoch}/{cfg["max_epochs"]} '
              f'train={ta:.3%} val={va:.3%} valCE={vl:.5f} lr={lr:.1g}',flush=True)
        if stale>=cfg['early_stop']:break
    io.write(target/'training_summary.json',dict(elapsed_seconds=time.monotonic()-began,epochs_run=len(history),best_loss=best))
    seal(target,'seed_sha256.json',[target/n for n in ['model.pt','history.json','training_summary.json']])
    del net,optimizer,scheduler;gc.collect()


def infer_runs(path, cp, records, waves, split):
    runtime=io.session(path)
    checked=base.previous.CheckedSession(path,restore(cp)) if path.name.endswith('fp32.onnx') else None
    mean=np.asarray(cp['normalization']['mean']);scale=np.asarray(cp['normalization']['scale'])
    recnames=[];runs=[];ends=[];probs=[];length=cp['window_ms']*2
    for r in records:
        for n,(a,b) in enumerate(base.split_runs(r,split)):
            for end in base.window_ends(r,a,b,length):
                wave=waves[r['recording']][end-length-r['first']:end-r['first']]
                assert len(wave)==length
                x=((wave-mean)/scale).T[None,:,None,:].astype(np.float32)
                engine=checked if checked is not None and len(ends)%100==0 else runtime
                logits=engine.run(None,{'input':x})[0][0]
                p=np.exp(logits-logits.max());p/=p.sum()
                recnames.append(r['recording']);runs.append(n);ends.append(end);probs.append(p)
        print('REPLAY',path.parent.parent.name,split,path.name,r['recording'],flush=True)
    return dict(recording=np.array(recnames),run=np.array(runs),end=np.array(ends),
                probabilities=np.asarray(probs,np.float32),numeric_checks=np.asarray(0 if checked is None else checked.checked))


def validate_grid(trace, records, split, length=2800):
    expected=[(r['recording'],n,e) for r in records for n,(a,b) in enumerate(base.split_runs(r,split))
              for e in base.window_ends(r,a,b,length)]
    assert list(zip(trace['recording'].tolist(),trace['run'].tolist(),trace['end'].tolist()))==expected
    assert trace['probabilities'].shape==(len(expected),11) and np.isfinite(trace['probabilities']).all()


def develop(folder, architecture, data, cfg):
    folder.mkdir(exist_ok=True)
    for seed in cfg['seeds']:train_seed(folder/'seeds'/str(seed),architecture,seed,data,cfg)
    if done(folder,'development_sha256.json'):return
    options=[]
    for seed in cfg['seeds']:
        cp=torch.load(folder/'seeds'/str(seed)/'model.pt',map_location='cpu',weights_only=True)
        options.append(dict(seed=seed,**{k:v for k,v in cp['selected'].items() if k!='seed'}))
    selected=min(options,key=lambda s:s['validation_macro_loss'])
    shutil.copy2(folder/'seeds'/str(selected['seed'])/'model.pt',folder/'selection_checkpoint.pt')
    cp=torch.load(folder/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    io.write(folder/'seed_selection.json',dict(selected=selected,seeds=options,test_used=False))
    target=folder/'development';target.mkdir(exist_ok=True)
    xcal=base.batch(data['rows'],data['indices']['dev_calibration'],data['waves'],cp['normalization'])
    xval=base.batch(data['rows'],data['indices']['validation'],data['waves'],cp['normalization'])
    labels=np.array([data['rows'][i]['label'] for i in data['indices']['validation']])
    pre=base.export_fp(restore(cp),target/'model_fp32.onnx',xval[:16])
    candidates=[]
    for percentile in [None,99.99,99.9,99.5]:
        name='minmax' if percentile is None else f'percentile_{percentile}'
        path=target/(name+'.onnx')
        base.auto.quantize(pre,path,xcal,np.arange(len(xcal)),percentile)
        base.previous.check_quantized_graph(path)
        logits=base.auto.infer(path,xval,np.arange(len(xval)))
        losses=nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(labels),reduction='none').numpy()
        candidates.append(dict(name=name,percentile=percentile,validation_macro_loss=float(np.mean([
            losses[labels==c].mean() for c in range(11)])),validation_macro_accuracy=float(np.mean([
            (logits.argmax(1)[labels==c]==c).mean() for c in range(11)]))))
        print('PTQ VALIDATION',architecture['name'],candidates[-1],flush=True)
    quant=min(candidates,key=lambda c:c['validation_macro_loss'])
    io.write(folder/'quantization_selection.json',dict(selected=quant,candidates=candidates,test_used=False))
    trace=infer_runs(target/(quant['name']+'.onnx'),cp,data['records'],data['waves'],'validation')
    validate_grid(trace,data['records'],'validation')
    np.savez_compressed(folder/'validation_int8_trace.npz',**trace)
    events=base.decode(trace,cfg['decoder'],cfg['classes'])
    metrics,trials=base.event_metrics(events,data['records'],'validation')
    assert metrics['trials']==cfg['counts']['validation']
    io.write(folder/'validation_report.json',dict(architecture=architecture,selected=selected,quantization=quant,metrics=metrics,
        seed_validation_accuracy_mean=float(np.mean([s['validation_macro_accuracy'] for s in options])),
        seed_validation_accuracy_std=float(np.std([s['validation_macro_accuracy'] for s in options])),
        resources=base.resource_estimate(restore(cp)),test_used=False))
    io.write_csv(folder/'validation_trials.csv',trials)
    seal(folder,'development_sha256.json',[folder/n for n in ['selection_checkpoint.pt','seed_selection.json',
        'quantization_selection.json','validation_report.json','validation_int8_trace.npz']]+list(target.glob('*.onnx')))


def validation_rank(report):
    m=report['metrics']
    return (m['clean_success_rate'],m['end_to_end_recall'],-m['wrong_label_outputs'],-m['no_response_trials'],
            -m['rest_proxy_outputs_per_min'],-report['resources']['macs_per_window'])


def freeze_recommendation(out, cfg):
    if done(out,'selection_sha256.json'):return
    reports=[io.read(out/a['name']/'validation_report.json') for a in cfg['architectures']]
    best=max(reports,key=validation_rank)
    io.write(out/'validation_selection.json',dict(recommended_architecture=best['architecture']['name'],
        rule=cfg['architecture_selection'],reports=reports,test_used=False))
    io.write_csv(out/'validation_comparison.csv',[dict(model=r['architecture']['name'],**r['metrics'],
        selected_seed=r['selected']['seed'],selected_epoch=r['selected']['epoch'],
        seed_window_accuracy_mean=r['seed_validation_accuracy_mean'],seed_window_accuracy_std=r['seed_validation_accuracy_std'],
        parameters=r['resources']['parameters'],macs_per_window=r['resources']['macs_per_window']) for r in reports])
    seal(out,'selection_sha256.json',[out/'validation_selection.json',out/'validation_comparison.csv'])
    print('RECOMMENDATION FROZEN BEFORE TEST:',best['architecture']['name'],flush=True)


def finalize(folder, architecture, data, cfg):
    cp=torch.load(folder/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    selected=cp['selected']
    if not done(folder,'refit_sha256.json'):
        base.cnn.set_seed(selected['seed']);net=SweepNet(architecture)
        optimizer=torch.optim.Adam(net.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
        history=[]
        for epoch,lr in enumerate(selected['learning_rates'],1):
            for g in optimizer.param_groups:g['lr']=lr
            loss,acc=base.train_epoch(net,optimizer,data['rows'],data['indices']['fit'],data['waves'],cfg['final_normalization'],selected['seed']+epoch)
            history.append(dict(epoch=epoch,lr=lr,loss=loss,accuracy=acc))
            io.write(folder/'refit_history.json',history)
            print('REFIT',architecture['name'],epoch,'/',selected['epoch'],f'accuracy={acc:.3%}',flush=True)
        final=checkpoint(net,cfg['final_normalization'],selected,cfg['classes'],data['rows'],data['indices']['fit'])
        torch.save(final,folder/'model.pt')
        loaded=restore(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
        assert all(torch.equal(v,loaded.state_dict()[n]) for n,v in net.state_dict().items())
        seal(folder,'refit_sha256.json',[folder/'model.pt',folder/'refit_history.json'])
        del net,optimizer;gc.collect()
    if done(folder,'final_export_sha256.json'):return
    final=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert final['training_fingerprints']==[data['rows'][i]['fingerprint'] for i in data['indices']['fit']]
    assert not set(final['training_fingerprints'])&{r['fingerprint'] for r in data['rows'] if r['split']=='test'}
    target=folder/'export';target.mkdir(exist_ok=True)
    xcal=base.batch(data['rows'],data['indices']['final_calibration'],data['waves'],final['normalization'])
    net=restore(final);pre=base.export_fp(net,target/'model_fp32.onnx',xcal[:16])
    quant=io.read(folder/'quantization_selection.json')['selected']
    base.auto.quantize(pre,target/'model_int8_qdq.onnx',xcal,np.arange(len(xcal)),quant['percentile'])
    base.previous.check_quantized_graph(target/'model_int8_qdq.onnx')
    contract=io.read(Path(cfg['base'])/'export/input_contract.json')
    contract.update(architecture=architecture,decoder=cfg['decoder'],normalization=final['normalization'])
    io.write(target/'input_contract.json',contract);io.write(target/'resource_estimate.json',base.resource_estimate(net))
    io.write(target/'calibration_manifest.json',dict(calibration_fingerprints=[data['rows'][i]['fingerprint'] for i in data['indices']['final_calibration']],test_used=False))
    seal(folder,'final_export_sha256.json',[p for p in target.iterdir() if p.is_file()])


def test_model(folder, model_folder, data, cfg):
    folder.mkdir(exist_ok=True)
    if done(folder,'test_sha256.json'):return
    cp=torch.load(model_folder/'model.pt',map_location='cpu',weights_only=True)
    assert cp['normalization']==cfg['final_normalization']
    report={};files=[]
    for kind in ['fp32','int8']:
        filename='model_fp32.onnx' if kind=='fp32' else 'model_int8_qdq.onnx'
        trace=infer_runs(model_folder/'export'/filename,cp,data['records'],data['waves'],'test')
        validate_grid(trace,data['records'],'test')
        np.savez_compressed(folder/f'{kind}_trace.npz',**trace)
        events=base.decode(trace,cfg['decoder'],cfg['classes'])
        metrics,trials,confusion=score(events,data['records'],cfg['classes'])
        assert metrics['trials']==cfg['counts']['test']
        report[kind]=dict(overall=metrics,by_subject={s:score(events,[r for r in data['records'] if r['subject']==s],cfg['classes'])[0] for s in SUBJECTS})
        io.write(folder/f'{kind}_events.json',events);io.write_csv(folder/f'{kind}_trials.csv',trials)
        io.write_csv(folder/f'{kind}_confusion.csv',confusion)
        files.extend(folder/f'{kind}_{n}' for n in ['trace.npz','events.json','trials.csv'])
    io.write(folder/'comparison.json',report)
    seal(folder,'test_sha256.json',files+[folder/'comparison.json'])


def summarize(out, data, cfg):
    names=['current_base']+[a['name'] for a in cfg['architectures']]
    all_results={n:io.read(out/n/'test/comparison.json') for n in names}
    reference=all_results['current_base'];table=[];deltas=[]
    base_events={kind:io.read(out/'current_base/test'/f'{kind}_events.json') for kind in ['fp32','int8']}
    for name in names:
        for kind,group in all_results[name].items():
            for s,m in [('ALL',group['overall'])]+list(group['by_subject'].items()):
                table.append(dict(model=name,format=kind,subject=s,**m))
                ref=reference[kind]['overall'] if s=='ALL' else reference[kind]['by_subject'][s]
                deltas.append(dict(model=name,format=kind,subject=s,
                    correct_change=m['correct_unique_trials']-ref['correct_unique_trials'],
                    accuracy_change_pp=100*(m['end_to_end_recall']-ref['end_to_end_recall']),
                    strict_change_pp=100*(m['clean_success_rate']-ref['clean_success_rate']),
                    wrong_output_change=m['wrong_label_outputs']-ref['wrong_label_outputs'],
                    no_response_change=m['no_response_trials']-ref['no_response_trials']))
            if name!='current_base':
                before=score(base_events[kind],data['records'],cfg['classes'])[1]
                after=score(io.read(out/name/'test'/f'{kind}_events.json'),data['records'],cfg['classes'])[1]
                io.write_csv(out/name/'test'/f'paired_vs_current_{kind}.csv',paired_trials(before,after))
    recommended=io.read(out/'validation_selection.json')['recommended_architecture']
    io.write(out/'comparison.json',dict(models=all_results,recommended_by_validation=recommended,test_targets=cfg['counts']['test']))
    io.write_csv(out/'comparison.csv',table);io.write_csv(out/'deltas_vs_current.csv',deltas)
    lines=['滑窗CNN结构对照：所有模型使用原499条测试，不包含个人校准；固定1400ms/100ms步进/解码。',
           f'测试前由验证集选定的候选：{recommended}（不是按下表测试成绩挑选）',
           'INT8：模型                           正确命令        严格成功率   错词输出/试次   无响应   静息输出/分钟']
    for n in names:
        m=all_results[n]['int8']['overall']
        lines.append(f'{n:31s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
            f'{m["clean_success_rate"]:.2%}   {m["wrong_label_outputs"]}/{m["wrong_response_trials"]} '
            f'  {m["no_response_trials"]}   {m["rest_proxy_outputs_per_min"]:.3f}')
    for s in SUBJECTS:
        lines.append('\nINT8 '+s)
        for n in names:
            m=all_results[n]['int8']['by_subject'][s]
            lines.append(f'{n:31s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
                f'错词{m["wrong_label_outputs"]} 无响应{m["no_response_trials"]} 严格{m["clean_success_rate"]:.2%}')
    lines+=['','FP32总体（量化前对照）：']
    for n in names:
        m=all_results[n]['fp32']['overall']
        lines.append(f'{n:31s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
                     f'错词{m["wrong_label_outputs"]} 无响应{m["no_response_trials"]}')
    lines+=['','每结构先做3种子开发训练，再按验证损失选择一个检查点并重训；不是3种子测试均值。',
             'validation_comparison.csv保留种子间窗口验证波动。解码冻结，PTQ使用同一验证选择流程。',
             '数据和测试此前已查看；这是探索性已见人员比较，不能替代新的跨日/跨人验证。',
             '窗口标签是提示代理；静息输出不是生活负例误触发。未部署，资源估算不保证STM32时延。']
    report='\n'.join(lines);(out/'results.txt').write_text(report,encoding='utf-8-sig');print(report,flush=True)
    seal(out,'results_sha256.json',[out/n for n in ['comparison.json','comparison.csv','deltas_vs_current.csv','results.txt']])


def run_all(out):
    base.verify(out);base.check_seal(out,'prepared_sha256.json')
    cfg=io.read(out/'protocol.json')
    assert cfg['versions']==dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__)
    data=dict(rows=io.read(out/'windows_1400.json'),records=io.read(out/'recordings.json'),indices=io.read(out/'indices.json'))
    data['waves']=base.load_waves(data['records'])
    assert base.normalization(data['records'],data['waves'],{'train'})==cfg['dev_normalization']
    assert base.normalization(data['records'],data['waves'],{'train','validation'})==cfg['final_normalization']
    for a in cfg['architectures']:develop(out/a['name'],a,data,cfg)
    freeze_recommendation(out,cfg)
    for a in cfg['architectures']:finalize(out/a['name'],a,data,cfg)
    seal(out,'all_models_frozen_sha256.json',[out/a['name']/n for a in cfg['architectures']
        for n in ['refit_sha256.json','final_export_sha256.json']]+[out/'selection_sha256.json'])
    base.verify(out)
    (out/'current_base').mkdir(exist_ok=True)
    test_model(out/'current_base/test',Path(cfg['base']),data,cfg)
    # Reproduce the current model's original command score, on exactly its split.
    old=io.read(Path(cfg['base'])/'comparison.json')['results']
    measured=io.read(out/'current_base/test/comparison.json')
    for kind in ['fp32','int8']:
        for k,v in old['sliding_'+kind]['overall'].items():assert measured[kind]['overall'][k]==v
    for a in cfg['architectures']:test_model(out/a['name']/'test',out/a['name'],data,cfg)
    summarize(out,data,cfg)
    base.check_seal(out,'all_models_frozen_sha256.json')
    io.write(out/'completed.json',dict(complete=True,no_deployment=True,architectures=len(cfg['architectures']),
        seeds_per_architecture=len(cfg['seeds']),test_targets=cfg['counts']['test']))


def preflight(out):
    checks=[]
    for a in specs():
        folder=out/a['name'];folder.mkdir()
        base.cnn.set_seed(20261022);net=SweepNet(a)
        x=np.random.default_rng(20261022).normal(size=(6,2,2800)).astype(np.float32)
        optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        loss=nn.functional.cross_entropy(net(torch.from_numpy(x[:2])),torch.tensor([0,10]))
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
        cp=dict(state_dict=net.state_dict(),architecture=a,window_ms=1400)
        other=restore(cp)
        assert all(torch.equal(v,other.state_dict()[n]) for n,v in net.state_dict().items())
        pre=base.export_fp(net,folder/'model_fp32.onnx',x[:2])
        path=folder/'model_int8_qdq.onnx';base.auto.quantize(pre,path,x,np.arange(len(x)),99.99)
        base.previous.check_quantized_graph(path)
        got=base.auto.infer(path,x,np.arange(2));assert got.shape==(2,11) and np.isfinite(got).all()
        checks.append(dict(architecture=a,resources=base.resource_estimate(net),synthetic_gradient_export_int8_passed=True))
        print('SYNTHETIC CHECK PASSED',a['name'],flush=True)
    io.write(out/'preflight.json',dict(checks=checks,real_training_started=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','prepare','check'],default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true',help='Resume completed seed/stage checkpoints in --output')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    if args.resume and (args.output is None or args.phase!='all'):parser.error('--resume requires --output and --phase all')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=(args.output or base.ROOT/'models'/('sliding_arch_sweep_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if args.resume:
        if not (out/'prepared_sha256.json').exists():parser.error('Not a prepared sweep directory')
    else:out.mkdir(parents=True,exist_ok=False)
    tmp=out/'temporary';tmp.mkdir(exist_ok=True);tempfile.tempdir=str(tmp)
    with (out/'run.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(base.previous.Tee(sys.stdout,log)),redirect_stderr(base.previous.Tee(sys.stderr,log)):
            try:
                print('OUTPUT',out,'PHASE',args.phase,'RESUME',args.resume,flush=True)
                if args.phase=='check':preflight(out)
                else:
                    if not args.resume:prepare(out)
                    if args.phase=='all':run_all(out)
                print('PHASE COMPLETE',args.phase,out,flush=True)
            except BaseException:
                traceback.print_exc();raise


if __name__=='__main__':main()
