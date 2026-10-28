"""User-run mid64_temporal3 retraining on the CURRENT goodone_plus recordings.

Fresh raw-data preparation; same sliding/rest architecture and training recipe.
No old cache, split, checkpoint or normalization is used. No device access.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import shutil
import sys
import tempfile
import traceback

import run_sliding_architecture_sweep as sweep
from semg import lda

base, io = sweep.base, sweep.io
np, torch, nn = sweep.np, sweep.torch, sweep.nn
ROOT = base.ROOT
DATA = ROOT/'recordings/goodone_plus'
SCHEMA = 'current_goodone_mid64_temporal3_v1'
DECODER = dict(probability=.5, confirm=2, rest_frames=2, margin=0.)


def architecture():
    return next(a for a in sweep.specs() if a['name'] == 'mid64_temporal3')


def inventory(root):
    selections = []
    for subject in sweep.SUBJECTS:
        paths = sorted((root/subject.lower()).rglob('metadata.json'))
        if not paths:
            raise ValueError(f'No recordings for {subject}')
        for path in paths:
            meta = io.read(path)
            expected = (subject, 2000, 'completed', 'silent', 'silent_mouthing')
            actual = tuple(meta.get(k) for k in ['subject_id','sampling_rate_hz','status','condition','articulation_mode'])
            if actual != expected:
                raise ValueError(f'Unexpected recording metadata: {path}: {actual}')
            selections.append(dict(path=str(path.parent.resolve()), role='train'))
    names = [Path(s['path']).name for s in selections]
    if len(names) != 49 or len(set(names)) != len(names):
        raise ValueError('Expected 49 distinct recordings in the current five-person dataset')
    return selections


def make_split(records):
    """Original outer and inner algorithms, applied afresh to current sources."""
    train, test = lda.split_records(records, 'within_recording', 80, 20260924)
    report = dict(train=[records[int(i)] for i in train], test=[records[int(i)] for i in test],
                  classes=lda.ordered_classes(records))
    rows, indices = base.cnn.split_data(report)
    for split, ix in indices.items():
        for i in ix:
            rows[int(i)] = dict(rows[int(i)], split=split)
    groups = [{(r['recording'],r['block']) for r in rows if r['split']==s}
              for s in ['train','validation','test']]
    signals = [{f for r in rows if r['split']==s for f in r['signal_fingerprints']}
               for s in ['train','validation','test']]
    for a,b in [(0,1),(0,2),(1,2)]:
        assert not groups[a]&groups[b] and not signals[a]&signals[b]
    return rows, report['classes']+['rest']


def build_cache(out, sources, rows):
    cache = out/'cache'; cache.mkdir()
    records = []
    for source in sources:
        folder = Path(source['path'])
        with (folder/'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        cols = [header.index(k) for k in ['device_sample_idx','raw_ch0','raw_ch1']]
        data = np.loadtxt(folder/'samples.csv', delimiter=',', skiprows=1, usecols=cols, dtype=np.int64)
        assert np.all(np.diff(data[:,0])==1)
        first, end = int(data[0,0]), int(data[-1,0])+1
        annotations = io.cues(folder, [r for r in rows if r['recording']==folder.name], end)
        assert min(t['prepare'] for t in annotations)-first >= 10000
        frontend = io.CausalRMSFrontend(lda.filter_sos('notch50_bp500'))
        wave, ends, clipped = [], [], []
        for start in range(0,len(data),20000):
            for frame in frontend.feed(data[start:start+20000,1:], first+start):
                wave.append(frame['filtered']); ends.append(frame['end']); clipped.append(frame['clipped'])
        filtered = np.concatenate(wave).astype(np.float32)
        np.savez(cache/(folder.name+'.npz'), filtered=filtered, ends=np.array(ends), clipped=np.array(clipped))
        records.append(dict(recording=folder.name, subject=source['subject'], path=str(folder),
                            first=first,end=end,samples=len(data),seconds=len(data)/2000,
                            annotations=annotations,clipped_frames=int(sum(clipped)),
                            tail_unprocessed_samples=len(data)-len(filtered)))
        print('CACHE',source['subject'],folder.name,flush=True)
    return records


def load_waves(out, records):
    waves = {}
    for r in records:
        with np.load(out/'cache'/(r['recording']+'.npz')) as z:
            waves[r['recording']] = z['filtered']
    return waves


def make_windows(out, records, classes):
    rows, excluded = [], Counter()
    for record in records:
        with np.load(out/'cache'/(record['recording']+'.npz')) as z:
            ends, clipped = z['ends'], z['clipped']
        for trial in record['annotations']:
            if trial['split']=='excluded':
                continue
            for end in base.window_ends(record,trial['prepare'],trial['context_end'],2800):
                start=end-2800
                label=base.training_label(trial,start,end,classes)
                if label is None:
                    excluded['ambiguous_transition']+=1
                    continue
                if np.any(clipped[(ends>start)&(ends-40<end)]):
                    excluded['clipping']+=1
                    continue
                rows.append(dict(recording=record['recording'],subject=record['subject'],split=trial['split'],
                                 trial_id=trial['trial_id'],block=trial['block'],label=label,
                                 start=start,end=end,offset=start-record['first'],
                                 fingerprint=f'{record["recording"]}:{start}:{end}'))
    indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','validation','test']}
    for s in indices:
        assert {rows[i]['label'] for i in indices[s]}==set(range(11))
    indices['fit']=[i for i,r in enumerate(rows) if r['split'] in ['train','validation']]
    indices['validation']=base.balanced_indices(rows,indices['validation'],200,20261022,False).tolist()
    indices['dev_calibration']=base.balanced_indices(rows,indices['train'],50,20261022,False).tolist()
    indices['final_calibration']=base.balanced_indices(rows,indices['fit'],50,20261022,False).tolist()
    assert set(indices['dev_calibration']) <= set(indices['train'])
    assert set(indices['final_calibration']) <= set(indices['fit'])
    assert not set(indices['fit']) & set(indices['test'])
    return rows, indices, dict(excluded)


def input_contract(classes, norm):
    return dict(classes=classes,normalization=norm,fs=2000,step_samples=200,input_shape=[1,2,1,2800],
                filter_sos=lda.filter_sos('notch50_bp500').tolist(),architecture=architecture(),decoder=DECODER,
                preprocessing='continuous causal filter; ring buffer; no padding; no activity gate; standardize using saved training statistics',
                initial_state='disarmed; requires predicted rest before command',
                note='FP32 I/O around internal INT8. Newly trained model has not been deployed or board-validated.')


def dependencies():
    files=base.previous.dependencies()+[Path(__file__).resolve(),Path(sweep.__file__).resolve(),
        Path(base.__file__).resolve(),ROOT/'experiments/sliding_speech.py',
        ROOT/'experiments/run_personal_sliding_finetune.py',ROOT/'semg/lda.py']
    # Include semg modules imported by the feature extractor and frontend.
    files+=list((ROOT/'semg').glob('*.py'))
    return sorted(set(p.resolve() for p in files))


def prepare(out, data_root):
    selections=inventory(data_root)
    feature_config=dict(segments=6,metrics=['log_mean_abs_diff','log_mav','zc_count_zero','ssc_count_zero'])
    valid, excluded, sources=lda.extract(selections,'notch50_bp500',print,feature_config,[0,1])
    # LDA extraction supplies the existing QC and boundaries only; no LDA is fitted.
    split_rows,classes=make_split(valid)
    io.write(out/'trial_manifest.json',split_rows)
    io.write_csv(out/'split_manifest.csv',[{k:r[k] for k in
        ['split','subject','recording','path','trial_id','block','word','prepare','action','rest','fingerprint']} for r in split_rows])
    io.write(out/'data_audit.json',dict(sources=sources,excluded=excluded,
        valid_by_subject=dict(Counter(r['subject'] for r in valid))))
    records=build_cache(out,sources,split_rows)
    io.write(out/'recordings.json',records)
    rows,indices,window_exclusions=make_windows(out,records,classes)
    io.write(out/'windows_1400.json',rows);io.write(out/'indices.json',indices)
    waves=load_waves(out,records)
    norm=base.normalization(records,waves,{'train'})
    final_norm=base.normalization(records,waves,{'train','validation'})
    counts=dict(Counter(r['split'] for r in split_rows))
    per_subject={s:dict(Counter(r['split'] for r in split_rows if r['subject']==s)) for s in sweep.SUBJECTS}
    assert len(classes)==11 and classes[-1]=='rest'
    template=out/'contract_template'/'export';template.mkdir(parents=True)
    io.write(template/'input_contract.json',input_contract(classes,final_norm))
    cfg=dict(schema=SCHEMA,data_root=str(data_root),inventory=selections,
        base=str(template.parent),architectures=[architecture()],seeds=sweep.SEEDS,
        window_ms=1400,step_ms=100,classes=classes,max_epochs=100,early_stop=10,batch=44,
        windows_per_class_per_epoch=400,dropout=.5,
        optimizer=dict(name='Adam',lr=.001,weight_decay=.0001),scheduler=dict(factor=.1,patience=2),
        dev_normalization=norm,final_normalization=final_norm,decoder=DECODER,counts=counts,by_subject=per_subject,
        split='Fresh current-data within_recording 80/20 seed20260924; one training round per recording validation seed20261010. Original algorithms.',
        architecture_selection='Only mid64_temporal3, fixed by user before this run; no nine-architecture search.',
        seed_epoch_selection='Three fresh starts; lowest fixed validation-window macro CE selects seed and epoch.',
        refit='From scratch on train+validation; selected epoch count and learning-rate trajectory.',
        quantization='MinMax,99.99,99.9,99.5; development-validation macro CE selection; training-only calibration.',
        normalization='New unique train contexts; final train+validation contexts only. No old statistics.',
        decoder_policy='Frozen probability0.5/confirm2/rest_frames2/margin0; no threshold tuning.',
        caveats=['Seen-person exploratory evaluation; S013 and S015 were selected using earlier LDA scores.',
                 'New source ordering can change round assignments, including retained recordings. Old81.16% is not a paired comparison.',
                 'No old cache or model is loaded. No personal calibration, no LOSO, no automatic deployment.',
                 'Cue timings are proxies; overlapping windows are not independent trials.'],
        versions=dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__))
    io.write(out/'protocol.json',cfg)
    io.write(out/'dataset_summary.json',dict(counts=counts,by_subject=per_subject,
        windows=len(rows),window_exclusions=window_exclusions,
        window_counts={s:dict(Counter(classes[r['label']] for r in rows if r['split']==s)) for s in counts},
        indices_counts={k:len(v) for k,v in indices.items()},real_training_started=False))
    io.write(out/'planned_resources.json',base.resource_estimate(sweep.SweepNet(architecture())))
    hashes={str(Path(s['path'])/n):h for s in sources for n,h in s['hashes'].items()}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in dependencies():
        hashes[str(p)]=io.digest(p)
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    for p in (ROOT/'experiments/third_party').glob('SilentWear-*'):
        target=snapshot/'third_party'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    io.write(out/'inputs_sha256.json',hashes)
    base.verify(out)
    files=[out/n for n in ['protocol.json','recordings.json','trial_manifest.json','split_manifest.csv',
        'data_audit.json','windows_1400.json','indices.json','dataset_summary.json','planned_resources.json','inputs_sha256.json']]
    sweep.seal(out,'prepared_sha256.json',files+list((out/'cache').glob('*.npz'))+[template/'input_contract.json'])
    print('PREPARED',counts,per_subject,'one architecture, three seeds; NO REAL TRAINING YET',flush=True)


def verify_prepared(out):
    base.check_seal(out,'prepared_sha256.json');base.verify(out)
    cfg=io.read(out/'protocol.json')
    if cfg['schema']!=SCHEMA or inventory(Path(cfg['data_root']))!=cfg['inventory']:
        raise ValueError('Dataset inventory or experiment type changed; start a new experiment')
    assert cfg['versions']==dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__)
    data=dict(rows=io.read(out/'windows_1400.json'),records=io.read(out/'recordings.json'),indices=io.read(out/'indices.json'))
    data['waves']=load_waves(out,data['records'])
    assert base.normalization(data['records'],data['waves'],{'train'})==cfg['dev_normalization']
    assert base.normalization(data['records'],data['waves'],{'train','validation'})==cfg['final_normalization']
    return data,cfg


def summarize(out, cfg):
    report=io.read(out/architecture()['name']/'test/comparison.json')
    table=[]
    lines=['新版goodone_plus：mid64_temporal3，五人共用模型，已见人员整轮留出连续测试。',
           '当前数据重新划分；不能与旧81.16%直接作同测试集比较。','']
    for kind,group in report.items():
        lines.append(kind.upper()+'：人员 | 正确/总数 | 正确率 | 严格成功率 | 错词输出 | 无响应 | 静息输出/分钟')
        for s,m in [('ALL',group['overall'])]+list(group['by_subject'].items()):
            table.append(dict(format=kind,subject=s,**m))
            lines.append(f'{s} | {m["correct_unique_trials"]}/{m["trials"]} | {m["end_to_end_recall"]:.2%} | '
                         f'{m["clean_success_rate"]:.2%} | {m["wrong_label_outputs"]} | {m["no_response_trials"]} | {m["rest_proxy_outputs_per_min"]:.3f}')
        lines.append('')
    lines+=['结构已固定；种子、轮数、PTQ只由开发验证选择；解码固定。',
            '按历史LDA成绩整理过的数据，属于探索性评估；不是全新盲测、跨人或跨日测试。',
            '新模型未部署。原模型和原始录制未修改。']
    io.write(out/'comparison.json',dict(model=architecture(),results=report,test_targets=cfg['counts']['test']))
    io.write_csv(out/'comparison.csv',table)
    text='\n'.join(lines)+'\n';(out/'results.txt').write_text(text,encoding='utf-8-sig');print(text,flush=True)
    sweep.seal(out,'results_sha256.json',[out/n for n in ['comparison.json','comparison.csv','results.txt']])


def run_all(out):
    data,cfg=verify_prepared(out)
    if (out/'completed.json').exists():
        for marker in ['selection_sha256.json','all_models_frozen_sha256.json','results_sha256.json']:
            base.check_seal(out,marker)
        folder=out/architecture()['name']
        for marker in ['development_sha256.json','refit_sha256.json','final_export_sha256.json']:
            base.check_seal(folder,marker)
        base.check_seal(folder/'test','test_sha256.json')
        print('Already complete:',out,flush=True);return
    spec=architecture();folder=out/spec['name']
    sweep.develop(folder,spec,data,cfg)
    sweep.freeze_recommendation(out,cfg)
    sweep.finalize(folder,spec,data,cfg)
    sweep.seal(out,'all_models_frozen_sha256.json',[folder/'refit_sha256.json',folder/'final_export_sha256.json',out/'selection_sha256.json'])
    base.verify(out)
    sweep.test_model(folder/'test',folder,data,cfg)
    summarize(out,cfg)
    base.check_seal(out,'prepared_sha256.json');base.verify(out)
    base.check_seal(out,'all_models_frozen_sha256.json')
    io.write(out/'completed.json',dict(complete=True,no_deployment=True,architecture=spec['name'],
        seeds=cfg['seeds'],test_targets=cfg['counts']['test'],schema=SCHEMA))


def preflight(out):
    base.cnn.set_seed(20261010);net=sweep.SweepNet(architecture())
    x=np.random.default_rng(42).normal(size=(6,2,2800)).astype(np.float32)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    loss=nn.functional.cross_entropy(net(torch.from_numpy(x[:2])),torch.tensor([0,10]))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    optimizer.step()
    pre=base.export_fp(net,out/'synthetic_fp32.onnx',x[:2])
    path=out/'synthetic_int8.onnx';base.auto.quantize(pre,path,x,np.arange(len(x)),99.9)
    base.previous.check_quantized_graph(path)
    got=base.auto.infer(path,x,np.arange(2));assert got.shape==(2,11) and np.isfinite(got).all()
    io.write(out/'preflight.json',dict(synthetic_only=True,real_training_started=False,
        architecture=architecture(),resources=base.resource_estimate(net),gradient_fp32_int8_passed=True))
    print('SYNTHETIC CHECK PASSED; no real training',flush=True)


@contextmanager
def output_lock(out):
    # OS lock is automatically released on process exit, including crashes.
    with (out/'.run.lock').open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if sys.platform=='win32':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform=='win32':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','prepare','check'],default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    if args.resume and (args.output is None or args.phase!='all'):parser.error('--resume requires --output and --phase all')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out=(args.output or ROOT/'models'/('cnn_goodone_current_'+stamp)).resolve()
    if args.resume:
        if not (out/'prepared_sha256.json').exists():parser.error('Not a fully prepared new-data experiment')
    else:out.mkdir(parents=True,exist_ok=False)
    with output_lock(out):
        tmp=out/'temporary';tmp.mkdir(exist_ok=True);tempfile.tempdir=str(tmp)
        with (out/'run.log').open('a',encoding='utf-8') as log:
            with redirect_stdout(base.previous.Tee(sys.stdout,log)),redirect_stderr(base.previous.Tee(sys.stderr,log)):
                try:
                    print('OUTPUT',out,'PHASE',args.phase,'RESUME',args.resume,flush=True)
                    if args.phase=='check':preflight(out)
                    else:
                        if not args.resume:prepare(out,DATA.resolve())
                        if args.phase=='all':run_all(out)
                    print('PHASE COMPLETE',args.phase,out,flush=True)
                except BaseException:
                    traceback.print_exc();raise


if __name__=='__main__':main()
