"""Small offline two-channel framed LSTM with frozen LDA test trials and grouped validation.

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

class SmallLSTM(nn.Module):
    """Lossless 16-sample frames -> unidirectional LSTM -> six-bin pooling.

    Input remains normalized (batch, 2, 2048). Framing is a reshape, not
    decimation or envelope extraction. All samples, including the same padded
    tail used by the CNN, are retained. No convolutional frontend is used.
    """
    def __init__(self, classes=10):
        super().__init__()
        self.recurrent = nn.LSTM(input_size=32, hidden_size=64, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.pool = nn.AdaptiveAvgPool1d(6)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(64*6, classes)

    def forward(self, x):
        # (B, channel, frame, sample) -> (B, frame, channel*sample)
        framed = x.reshape(x.shape[0], 2, 128, 16).permute(0, 2, 1, 3).reshape(x.shape[0], 128, 32)
        sequence, _ = self.recurrent(framed)
        features = self.pool(sequence.transpose(1, 2)).flatten(1)
        return self.classifier(self.dropout(features))


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
    x=np.zeros((len(rows),2,LENGTH),dtype=np.float32)
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
            feat=np.concatenate([lda.feature_vector(raw[:,c],offset,sos,report['feature_config']) for c in range(2)])
            np.testing.assert_allclose(feat,row['features'],rtol=0,atol=1e-10)
            centered=raw.astype(float)-32768
            filtered=sosfilt(sos,centered,axis=0,zi=sosfilt_zi(sos)[:,:,None]*centered[0])[0]
            action=filtered[offset+150:len(filtered)-150]
            assert 400<=len(action)<=LENGTH, 'Unexpected length; do not silently truncate'
            lengths[i]=len(action); x[i,:,:len(action)]=action.T
        print('Loaded',folder.name,flush=True)
    assert (lengths>0).all() and np.isfinite(x).all()
    return x,lengths

def normalize(x,lengths,fit_indices):
    valid=np.arange(LENGTH)[None,:]<lengths[:,None]
    count=lengths[fit_indices].sum()
    mean=x[fit_indices].astype(np.float64).sum(axis=(0,2))/count
    variance=(((x[fit_indices].astype(np.float64)-mean[None,:,None])**2)*valid[fit_indices,None,:]).sum(axis=(0,2))/count
    scale=np.sqrt(variance).clip(1e-6)
    z=((x-mean[None,:,None])/scale[None,:,None])*valid[:,None,:]
    assert np.isfinite(z).all()
    return torch.from_numpy(z.astype(np.float32)),dict(mean=mean.tolist(),scale=scale.tolist(),fit_trials=len(fit_indices))

def augmentation(batch):
    batch=batch.clone()
    # Same small time translation for both channels of each trial.
    for i,shift in enumerate(torch.randint(-40,41,(len(batch),)).tolist()):
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

def train_epoch(net,optimizer,x,y,indices):
    net.train(); total=0.; correct=0
    order=np.random.permutation(indices)
    for start in range(0,len(order),BATCH):
        ix=order[start:start+BATCH]; optimizer.zero_grad(set_to_none=True)
        logits=net(augmentation(x[ix]));loss=nn.functional.cross_entropy(logits,y[ix])
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
        total+=float(loss.detach())*len(ix);correct+=int((logits.argmax(1)==y[ix]).sum())
    return total/len(order),correct/len(order)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,default=ROOT/'models/train_20261012_211031_2fa041')
    parser.add_argument('--cnn-reference',type=Path,default=ROOT/'models/cnn_20261013_101157_1f3069')
    args=parser.parse_args();baseline=args.baseline.resolve()
    reference=args.cnn_reference.resolve()
    reference_hashes={name:digest(reference/name) for name in ['report.json','split_manifest.csv','predictions.csv']}
    reference_report=json.loads((reference/'report.json').read_text(encoding='utf-8'))
    report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    assert report['channels']==[0,1] and len(report['classes'])==10
    rows,split=split_data(report)
    # Freeze the exact CNN development split, not just the final test set.
    with (reference/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        reference_split=list(csv.DictReader(f))
    expected={(key, rows[i]['fingerprint']) for key,indices in split.items() for i in indices}
    actual={(r['split'],r['fingerprint']) for r in reference_split}
    assert len(reference_split)==len(rows) and expected==actual, 'CNN split differs'
    assert reference_report['protocol']['baseline_report_sha256']==digest(baseline/'report.json')
    assert reference_report['protocol']['classes']==report['classes']
    output=ROOT/'models'/('lstm_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir()
    shutil.copy2(__file__,output/'train_small_lstm.py')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parameter_count=sum(p.numel() for p in SmallLSTM().parameters())
    protocol=dict(baseline=str(baseline),baseline_report_sha256=digest(baseline/'report.json'),
        architecture=str(SmallLSTM()),parameters=parameter_count,seeds=SEEDS,max_epochs=MAX_EPOCHS,patience=PATIENCE,
        batch_size=BATCH,optimizer='AdamW',learning_rate=0.001,weight_decay=0.001,selection='minimum validation cross entropy; ties keep earlier epoch',
        final_refit=f'Fresh model on all original {len(report["train"])} training trials, selected seed and selected epoch count; then one test evaluation.',
        validation_split='One randomly selected whole round per baseline training recording, seed 20261010; baseline test is unchanged.',
        augmentation='training only: shared +/-40 sample zero-filled shift; independent channel gain 0.9-1.1',
        channels=[0,1],length=LENGTH,preprocessing='Same LDA causal filter and cue boundaries; trim 150 ms per edge; right pad with zeros; no time resampling.',
        normalization='Per-channel mean/std on unpadded training samples only; normalized padding remains zero.',
        split_counts={k:len(v) for k,v in split.items()},torch_version=str(torch.__version__),numpy_version=np.__version__,scipy_version=scipy.__version__,
        framing='Lossless nonoverlapping 16-sample frames: (2,2048) -> (128,32); no decimation or convolution.',
        reference_cnn=str(reference),reference_cnn_hashes=reference_hashes,
        device='cpu',classes=report['classes'],source_hashes=report['sources'],test_used_for_selection=False)
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
    for seed in SEEDS:
        set_seed(seed);net=SmallLSTM();optimizer=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.001)
        history=[];local_best=float('inf');stale=0;best_epoch=0
        for epoch in range(1,MAX_EPOCHS+1):
            tl,ta=train_epoch(net,optimizer,x,y,split['train'])
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
    set_seed(selected['seed']);net=SmallLSTM();optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    final_history=[]
    for epoch in range(1,selected['epoch']+1):
        loss,acc=train_epoch(net,optimizer,x,y,all_train)
        final_history.append(dict(epoch=epoch,loss=loss,accuracy_augmented=acc))
        if epoch==1 or epoch%10==0:print('refit',epoch,'/',selected['epoch'],'train',round(acc,3),flush=True)
    write_json(output/'refit_history.json',final_history)
    checkpoint=dict(schema='semg_small_lstm_v1',state_dict=net.state_dict(),normalization=final_scaler,
        classes=classes,channels=[0,1],input_length=LENGTH,selected=selected,protocol=protocol,
        training_fingerprints=[rows[i]['fingerprint'] for i in all_train])
    torch.save(checkpoint,output/'model.pt')
    # Roundtrip numerical check before reporting a loadable model.
    reloaded=torch.load(output/'model.pt',map_location='cpu',weights_only=True)
    copy=SmallLSTM(len(reloaded['classes']));copy.load_state_dict(reloaded['state_dict']);copy.eval();net.eval()
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
    lstm_correct=pred==y[split['test']].numpy();lda_correct=lda_pred==y[split['test']].numpy()
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
    for name,h in reference_hashes.items():assert digest(reference/name)==h
    with (reference/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        reference_predictions={(r['recording'],int(r['trial_id'])):r for r in csv.DictReader(f)}
    cnn_correct=[]
    for row in test_rows:
        old=reference_predictions[(row['recording'],row['trial_id'])]
        assert old['subject']==row['subject'] and old['word']==row['word']
        cnn_correct.append(bool(int(old['correct'])))
    cnn_correct=np.array(cnn_correct)
    assert len(reference_predictions)==len(test_rows) and int(cnn_correct.sum())==reference_report['metrics']['correct']
    result=dict(schema='semg_small_lstm_experiment_v1',folder=str(output),protocol=protocol,selected=selected,
        metrics=metrics,group_metrics=groups,baseline_metrics=lda_metrics,test_loss=test_loss,
        cnn_metrics=reference_report['metrics'],cnn_group_metrics=reference_report['group_metrics'],
        paired_vs_cnn=dict(lstm_only_correct=int((lstm_correct&~cnn_correct).sum()),cnn_only_correct=int((cnn_correct&~lstm_correct).sum()),both_correct=int((lstm_correct&cnn_correct).sum())),
        paired=dict(lstm_only_correct=int((lstm_correct&~lda_correct).sum()),lda_only_correct=int((lda_correct&~lstm_correct).sum()),both_correct=int((lstm_correct&lda_correct).sum())),
        model_roundtrip_verified=True,input_hashes_unchanged=True,final_train_trials=len(all_train),test_trials=len(split['test']),
        final_normalization=final_scaler,elapsed_seconds=time.monotonic()-started,
        inference_ms_median=float(np.median(times)),inference_ms_p95=float(np.percentile(times,95)),
        warnings=['Exploratory within-recording evaluation on previously inspected data; no unseen-subject or unseen-session claim.',
                  'Offline cue-delimited ten-word classifier; no rest/unknown rejection or full streaming pipeline evaluated.',
                  'Model checkpoint is separate from LDA JSON and cannot be loaded by the current LDA GUI.',
                  'Only one architecture, three predetermined seeds; model and duration selected using validation, not test accuracy.'])
    write_json(output/'report.json',result)
    text=['小型双通道分帧LSTM实验','',f'总体：{metrics["correct"]}/{len(split["test"])} = {metrics["accuracy"]:.2%}；宏平均F1 {metrics["macro_f1"]:.4f}',
        f'同一测试集CNN：{reference_report["metrics"]["correct"]}/{len(split["test"])} = {reference_report["metrics"]["accuracy"]:.2%}',
        f'同一测试集LDA：{lda_metrics["correct"]}/{len(split["test"])} = {lda_metrics["accuracy"]:.2%}',
        f'模型参数：{parameter_count}；选定随机种子：{selected["seed"]}；重训轮数：{selected["epoch"]}',
        f'开发阶段{len(split["train"])}训练/{len(split["validation"])}验证/{len(split["test"])}测试；验证选定训练轮数后，从头使用原LDA的{len(all_train)}训练试次重训，再评估原{len(split["test"])}测试试次。',
        '每份录制的整轮不跨集合；测试集未参与标准化拟合、早停、随机种子选择。','']
    for g in groups['subject']:
        text.append(f'{g["subject"]}：{g["metrics"]["correct"]}/{g["test_total"]} = {g["metrics"]["accuracy"]:.2%}')
    text+=['','当前是离线十词分类，尚未加入上位机LSTM加载界面或验证连续在线识别。',
           '复现脚本：experiments/train_small_lstm.py；本目录保留脚本快照、协议、三份训练曲线、划分、逐次预测与model.pt。']
    (output/'结果说明.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)
    print('OUTPUT',output,flush=True)

if __name__=='__main__':main()
