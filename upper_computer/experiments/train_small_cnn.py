"""Small offline two-channel CNN with frozen LDA test trials and grouped validation.

Run using .runtime/Scripts/python.exe; optional CPU torch is isolated in .cnn-deps.
No GUI, device connection, original recording mutation, or test-set model selection.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.cnn-deps'))
import numpy as np
import scipy
from scipy.signal import sosfilt, sosfilt_zi
import torch
from torch import nn
from semg import lda

LENGTH = 2048
SEEDS = [20261010, 20261011, 20261012]
MAX_EPOCHS, PATIENCE, BATCH = 120, 20, 32

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class SmallCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(2, 16, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(16, 32, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(6), nn.Flatten(), nn.Dropout(0.35), nn.Linear(64*6, classes))

    def forward(self, x): return self.layers(x)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def split_data(report):
    rows = sorted(report['train']+report['test'], key=lambda r: (r['subject'],r['recording'],r['trial_id']))
    old_test = {r['fingerprint'] for r in report['test']}
    rng = np.random.default_rng(20261010)
    validation = set()
    for recording in sorted({r['recording'] for r in rows}):
        training = [r for r in rows if r['recording']==recording and r['fingerprint'] not in old_test]
        blocks = sorted({r['block'] for r in training})
        if not blocks:
            continue  # A recording held out entirely by the baseline stays test-only.
        if len(blocks)<2:
            raise ValueError(f'{recording}: need at least two training rounds for grouped validation')
        chosen = int(rng.choice(blocks))
        validation.update(r['fingerprint'] for r in training if r['block']==chosen)
    indices = {key:[] for key in ['train','validation','test']}
    for i,row in enumerate(rows):
        key = 'test' if row['fingerprint'] in old_test else 'validation' if row['fingerprint'] in validation else 'train'
        indices[key].append(i)
    assert all(indices.values()), 'Training, validation, and test must all be nonempty'
    group_sets=[{(rows[i]['recording'],rows[i]['block']) for i in indices[k]} for k in indices]
    assert all(not group_sets[a]&group_sets[b] for a,b in [(0,1),(0,2),(1,2)])
    assert len({r['fingerprint'] for r in rows})==len(rows), 'Duplicate trials across splits'
    for key in ['train','validation']:
        assert {rows[i]['word'] for i in indices[key]}==set(report['classes']), f'{key} misses a class'
    assert {rows[i]['fingerprint'] for i in indices['test']}==old_test
    return rows, {k:np.array(v) for k,v in indices.items()}

def extract_waveforms(rows, report):
    fs = lda.variant_fs(report['variant'])
    length = round(2.048*fs)
    trim = round(.15*fs)
    x=np.zeros((len(rows),2,length),dtype=np.float32)
    lengths=np.zeros(len(rows),dtype=np.int64)
    sos=lda.filter_sos(report['variant'])
    for source in report['sources']:
        folder=Path(source['path'])
        for name,expected in source['hashes'].items():
            assert digest(folder/name)==expected, f'Source changed: {folder/name}'
        with (folder/'samples.csv').open(encoding='utf-8-sig',newline='') as f:
            data=np.array([(int(r['device_sample_idx']),int(r['raw_ch0']),int(r['raw_ch1'])) for r in csv.DictReader(f)],dtype=np.int64)
        assert np.all(np.diff(data[:,0])==1)
        for i,row in enumerate(rows):
            if Path(row['path'])!=folder: continue
            raw=data[row['prepare']-data[0,0]:row['rest']-data[0,0],1:]
            offset=row['action']-row['prepare']
            fp=hashlib.sha256(raw.astype('<u2').tobytes()+str(offset).encode('ascii')).hexdigest()
            assert fp==row['fingerprint']
            # Match LDA's causal filtering, preparation warm-up, and 150 ms trims.
            feat=np.concatenate([lda.feature_vector(raw[:,c],offset,sos,report['feature_config'],fs=fs) for c in range(2)])
            np.testing.assert_allclose(feat,row['features'],rtol=0,atol=1e-10)
            centered=raw.astype(float)-32768
            filtered=sosfilt(sos,centered,axis=0,zi=sosfilt_zi(sos)[:,:,None]*centered[0])[0]
            action=filtered[offset+trim:len(filtered)-trim]
            assert round(.4*fs)<=len(action)<=length, 'Unexpected length; do not silently truncate'
            lengths[i]=len(action); x[i,:,:len(action)]=action.T
        print('Loaded',folder.name,flush=True)
    assert (lengths>0).all() and np.isfinite(x).all()
    return x,lengths

def normalize(x,lengths,fit_indices):
    valid=np.arange(x.shape[-1])[None,:]<lengths[:,None]
    count=lengths[fit_indices].sum()
    mean=x[fit_indices].astype(np.float64).sum(axis=(0,2))/count
    variance=(((x[fit_indices].astype(np.float64)-mean[None,:,None])**2)*valid[fit_indices,None,:]).sum(axis=(0,2))/count
    scale=np.sqrt(variance).clip(1e-6)
    z=((x-mean[None,:,None])/scale[None,:,None])*valid[:,None,:]
    assert np.isfinite(z).all()
    return torch.from_numpy(z.astype(np.float32)),dict(mean=mean.tolist(),scale=scale.tolist(),fit_trials=len(fit_indices))

def augmentation(batch, shift_samples=40):
    batch=batch.clone()
    # Same small time translation for both channels of each trial.
    for i,shift in enumerate(torch.randint(-shift_samples,shift_samples+1,(len(batch),)).tolist()):
        batch[i]=torch.roll(batch[i],shift,dims=-1)
        if shift>0: batch[i,:,:shift]=0
        elif shift<0: batch[i,:,shift:]=0
    batch*=torch.empty((len(batch),2,1)).uniform_(0.9,1.1)
    return batch

def evaluate(net,x,y,indices):
    net.eval()
    with torch.inference_mode():
        logits=torch.cat([net(x[part]) for part in np.array_split(indices,max(1,int(np.ceil(len(indices)/BATCH))))])
        loss=float(nn.functional.cross_entropy(logits,y[indices]))
        accuracy=float((logits.argmax(1)==y[indices]).float().mean())
    return loss,accuracy,logits

def train_epoch(net,optimizer,x,y,indices,shift_samples=40):
    net.train(); total=0.; correct=0
    order=np.random.permutation(indices)
    for start in range(0,len(order),BATCH):
        ix=order[start:start+BATCH]; optimizer.zero_grad(set_to_none=True)
        logits=net(augmentation(x[ix],shift_samples));loss=nn.functional.cross_entropy(logits,y[ix])
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
        total+=float(loss.detach())*len(ix);correct+=int((logits.argmax(1)==y[ix]).sum())
    return total/len(order),correct/len(order)

def main():
    global LENGTH
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,default=ROOT/'models/train_20261010_173540_db0bb3')
    parser.add_argument('--cross-subject',action='store_true',help='Require disjoint training and test subjects; validate only on source subjects.')
    parser.add_argument('--output',type=Path,help='New output directory; refuses to overwrite an existing directory.')
    parser.add_argument('--resume-refit',action='store_true',help='Resume only after completed validation selection; restart deterministic final refit.')
    args=parser.parse_args();baseline=args.baseline.resolve()
    report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    fs=lda.variant_fs(report['variant'])
    LENGTH=round(2.048*fs)
    shift_samples=round(.04*fs)
    assert report['channels']==[0,1] and len(report['classes'])==10
    rows,split=split_data(report)
    train_subjects=sorted({r['subject'] for r in report['train']})
    test_subjects=sorted({r['subject'] for r in report['test']})
    if args.cross_subject:
        assert not set(train_subjects)&set(test_subjects), 'Cross-subject mode requires disjoint subjects'
        assert {rows[i]['subject'] for i in split['validation']} <= set(train_subjects)
    scope='cross_subject' if args.cross_subject else 'baseline_holdout'
    evaluation_warning=('Directional cross-subject test; validation uses source subjects only. Not five-fold LOSO. '
                        'These recordings were inspected in previous experiments, so this is an exploratory transfer evaluation.'
                        if args.cross_subject else
                        'Exploratory evaluation on the baseline holdout of previously inspected data; no broader generalization claim.')
    prefix='cnn_cross_' if args.cross_subject else 'cnn_'
    output=args.output.resolve() if args.output else ROOT/'models'/(prefix+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if args.resume_refit:
        assert args.output and (output/'selected_hyperparameters.json').is_file()
        assert not (output/'model.pt').exists() and not (output/'report.json').exists()
        shutil.copy2(__file__,output/'train_small_cnn_resume.py')
    else:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,output/'train_small_cnn.py')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parameter_count=sum(p.numel() for p in SmallCNN().parameters())
    protocol=dict(baseline=str(baseline),baseline_report_sha256=digest(baseline/'report.json'),
        baseline_model_sha256=digest(baseline/'model.json'),evaluation_scope=scope,train_subjects=train_subjects,test_subjects=test_subjects,
        architecture=str(SmallCNN()),parameters=parameter_count,seeds=SEEDS,max_epochs=MAX_EPOCHS,patience=PATIENCE,
        batch_size=BATCH,optimizer='AdamW',learning_rate=0.001,weight_decay=0.001,selection='minimum validation cross entropy; ties keep earlier epoch',
        final_refit=f'Fresh model on all original {len(report["train"])} training trials, selected seed and selected epoch count; then one test evaluation.',
        validation_split='One randomly selected whole round per baseline training recording, seed 20261010; baseline test is unchanged.',
        augmentation=f'training only: shared +/-{shift_samples} sample zero-filled shift; independent channel gain 0.9-1.1',
        channels=[0,1],length=LENGTH,preprocessing='Same LDA causal filter and cue boundaries; trim 150 ms per edge; right pad with zeros; no time resampling.',
        normalization='Per-channel mean/std on unpadded training samples only; normalized padding remains zero.',
        split_counts={k:len(v) for k,v in split.items()},torch_version=str(torch.__version__),numpy_version=np.__version__,scipy_version=scipy.__version__,
        device='cpu',classes=report['classes'],source_hashes=report['sources'],test_used_for_selection=False)
    if fs == 2000:
        protocol['sampling_rate_hz']=fs
        protocol['bandpass_hz']=[20,500]
    if args.resume_refit:
        assert json.loads((output/'protocol.json').read_text(encoding='utf-8')) == protocol
    else:
        write_json(output/'protocol.json',protocol)
    with (output/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['split','subject','recording','path','trial_id','block','word','prepare','action','rest','fingerprint']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key,indices in split.items():
            for i in indices:writer.writerow(dict(split=key,**{k:rows[i][k] for k in fields if k!='split'}))
    print('OUTPUT',output,'parameters',parameter_count,flush=True)
    raw,lengths=extract_waveforms(rows,report)
    write_json(output/'input_lengths.json',dict(min=int(lengths.min()),max=int(lengths.max()),median=float(np.median(lengths))))
    classes=report['classes'];y=torch.tensor([classes.index(r['word']) for r in rows],dtype=torch.long)
    x,selection_scaler=normalize(raw,lengths,split['train'])
    histories=[];best_loss=float('inf');selected=None
    started=time.monotonic()
    if args.resume_refit:
        selected=json.loads((output/'selected_hyperparameters.json').read_text(encoding='utf-8'))
        histories=json.loads((output/'selection_history.json').read_text(encoding='utf-8'))
        checkpoint=torch.load(output/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
        assert [h['seed'] for h in histories]==SEEDS
        assert checkpoint['selected']==selected and checkpoint['normalization']==selection_scaler
        assert selected['seed'] in SEEDS and 1<=selected['epoch']<=MAX_EPOCHS
        print('RESUME: frozen selection; restarting final refit from the same seed.',flush=True)
    for seed in ([] if args.resume_refit else SEEDS):
        set_seed(seed);net=SmallCNN();optimizer=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.001)
        history=[];local_best=float('inf');stale=0;best_epoch=0
        for epoch in range(1,MAX_EPOCHS+1):
            tl,ta=train_epoch(net,optimizer,x,y,split['train'],shift_samples)
            vl,va,_=evaluate(net,x,y,split['validation'])
            history.append(dict(epoch=epoch,train_loss=tl,train_accuracy_augmented=ta,validation_loss=vl,validation_accuracy=va))
            if vl<local_best-1e-4:
                local_best=vl;stale=0;best_epoch=epoch
                if vl<best_loss:
                    best_loss=vl;selected=dict(seed=seed,epoch=epoch,validation_loss=vl,validation_accuracy=va)
                    torch.save(dict(state_dict=net.state_dict(),normalization=selection_scaler,selected=selected),output/'selection_checkpoint.pt')
            else: stale+=1
            if epoch==1 or epoch%10==0: print('seed',seed,'epoch',epoch,'train',round(ta,3),'val',round(va,3),'val_loss',round(vl,4),flush=True)
            if stale>=PATIENCE:break
        histories.append(dict(seed=seed,best_epoch=best_epoch,best_validation_loss=local_best,epochs=history))
        write_json(output/'selection_history.json',histories)
    assert selected is not None
    write_json(output/'selected_hyperparameters.json',selected)
    print('Selection frozen BEFORE test:',selected,flush=True)
    all_train=np.sort(np.concatenate((split['train'],split['validation'])))
    assert {rows[i]['fingerprint'] for i in all_train}=={r['fingerprint'] for r in report['train']}
    x,final_scaler=normalize(raw,lengths,all_train)
    set_seed(selected['seed']);net=SmallCNN();optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    final_history=[]
    for epoch in range(1,selected['epoch']+1):
        loss,acc=train_epoch(net,optimizer,x,y,all_train,shift_samples)
        final_history.append(dict(epoch=epoch,loss=loss,accuracy_augmented=acc))
        if epoch==1 or epoch%10==0:print('refit',epoch,'/',selected['epoch'],'train',round(acc,3),flush=True)
    write_json(output/'refit_history.json',final_history)
    checkpoint=dict(schema='semg_small_cnn_v1',state_dict=net.state_dict(),normalization=final_scaler,
        classes=classes,channels=[0,1],input_length=LENGTH,selected=selected,protocol=protocol,
        training_fingerprints=[rows[i]['fingerprint'] for i in all_train])
    torch.save(checkpoint,output/'model.pt')
    # Roundtrip numerical check before reporting a loadable model.
    reloaded=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
    copy=SmallCNN(len(reloaded['classes']));copy.load_state_dict(reloaded['state_dict']);copy.eval();net.eval()
    with torch.inference_mode():
        torch.testing.assert_close(net(x[all_train[:8]]),copy(x[all_train[:8]]),rtol=0,atol=0)
    test_loss,test_acc,logits=evaluate(copy,x,y,split['test'])
    probabilities=logits.softmax(1).numpy();pred=logits.argmax(1).numpy()
    test_rows=[rows[i] for i in split['test']]
    metrics=lda.metrics(test_rows,pred,classes)
    groups=lda.group_metrics([rows[i] for i in all_train],test_rows,pred,classes)
    lda_model=lda.load_model(baseline/'model.json')
    lda_pred=lda.predict(lda_model,[r['features'] for r in test_rows])
    lda_metrics=lda.metrics(test_rows,lda_pred,classes)
    assert lda_metrics==report['metrics']
    cnn_correct=pred==y[split['test']].numpy();lda_correct=lda_pred==y[split['test']].numpy()
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject','recording','trial_id','word','prediction','confidence','correct','lda_prediction','lda_correct'])
        for row,p,probs,lp in zip(test_rows,pred,probabilities,lda_pred):
            writer.writerow([row['subject'],row['recording'],row['trial_id'],row['word'],classes[p],float(probs[p]),int(classes[p]==row['word']),classes[lp],int(classes[lp]==row['word'])])
    # CPU model-only batch=1 timing, excludes acquisition, window wait and preprocessing.
    with torch.inference_mode():
        for _ in range(10):copy(x[split['test'][:1]])
        times=[]
        for _ in range(100):
            t=time.perf_counter();copy(x[split['test'][:1]]);times.append((time.perf_counter()-t)*1000)
    for source in report['sources']:
        for name,h in source['hashes'].items():assert digest(Path(source['path'])/name)==h
    assert digest(baseline/'report.json')==protocol['baseline_report_sha256']
    assert digest(baseline/'model.json')==protocol['baseline_model_sha256']
    result=dict(schema='semg_small_cnn_experiment_v1',folder=str(output),protocol=protocol,selected=selected,
        metrics=metrics,group_metrics=groups,baseline_metrics=lda_metrics,test_loss=test_loss,
        paired=dict(cnn_only_correct=int((cnn_correct&~lda_correct).sum()),lda_only_correct=int((lda_correct&~cnn_correct).sum()),both_correct=int((cnn_correct&lda_correct).sum())),
        model_roundtrip_verified=True,input_hashes_unchanged=True,final_train_trials=len(all_train),test_trials=len(split['test']),
        final_normalization=final_scaler,elapsed_seconds=time.monotonic()-started,
        inference_ms_median=float(np.median(times)),inference_ms_p95=float(np.percentile(times,95)),
        warnings=[evaluation_warning,
                  'Offline cue-delimited ten-word classifier; no rest/unknown rejection or full streaming pipeline evaluated.',
                  'Model checkpoint is separate from LDA JSON and cannot be loaded by the current LDA GUI.',
                  'Only one architecture, three predetermined seeds; model and duration selected using validation, not test accuracy.'])
    write_json(output/'report.json',result)
    text=['小型双通道一维CNN实验','',f'总体：{metrics["correct"]}/{len(split["test"])} = {metrics["accuracy"]:.2%}；宏平均F1 {metrics["macro_f1"]:.4f}',
        f'评估方式：{scope}；训练人员：{train_subjects}；测试人员：{test_subjects}',
        f'同一测试集LDA：{lda_metrics["correct"]}/{len(split["test"])} = {lda_metrics["accuracy"]:.2%}',
        f'模型参数：{parameter_count}；选定随机种子：{selected["seed"]}；重训轮数：{selected["epoch"]}',
        f'开发阶段{len(split["train"])}训练/{len(split["validation"])}验证/{len(split["test"])}测试；验证选定训练轮数后，从头使用原LDA的{len(all_train)}训练试次重训，再评估原{len(split["test"])}测试试次。',
        '每份录制的整轮不跨集合；测试集未参与标准化拟合、早停、随机种子选择。','']
    for g in groups['subject']:
        if g['test_total']>0:
            text.append(f'{g["subject"]}：{g["metrics"]["correct"]}/{g["test_total"]} = {g["metrics"]["accuracy"]:.2%}')
        else:
            text.append(f'{g["subject"]}：仅用于训练，无测试试次。')
    text+=['','当前是离线十词分类，尚未加入上位机CNN加载界面或验证连续在线识别。',
           '复现脚本：experiments/train_small_cnn.py；本目录保留脚本快照、协议、三份训练曲线、划分、逐次预测与model.pt。']
    (output/'结果说明.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)
    print('OUTPUT',output,flush=True)

if __name__=='__main__':main()
