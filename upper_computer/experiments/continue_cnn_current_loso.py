"""Resume the 2026-10-30 CNN LOSO experiment after user-requested partial testing.

No old code or prepared split is changed. Completed seeds/models are verified
and reused; unfinished seeds restart. Default resumes S013--S015 training,
scores all five held-out people, and writes audited combined results.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import run_cnn_current_loso as run

base, sweep, io, torch, np = run.base, run.sweep, run.io, run.torch, run.np
DEFAULT = run.current.ROOT/'models/cnn_current_loso_20261030'
FIRST = ['S011', 'S012']


def amend(out):
    path=out/'user_continuation_policy.json'
    if path.exists():
        base.check_seal(out,'continuation_policy_sha256.json');return
    policy=dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        request='Finish current S011/S012 group; provide scripts for user to execute remaining folds.',
        original_protocol_unchanged=True, partial_subjects=FIRST, remaining_subjects=['S013','S014','S015'],
        override='S011/S012 are scored before remaining three folds finish training, at explicit user request.',
        all_five_models_frozen_before_first_test=False,
        controls='Architecture, splits, seeds, normalization, optimizer, PTQ, decoder and diagnostic rule stay fixed.',
        scope='No target calibration or new tuning after partial results; incomplete fold scores are not five-fold results.')
    io.write(path,policy)
    sweep.seal(out,'continuation_policy_sha256.json',[path,Path(__file__).resolve()])


def test_one(out, subject):
    fold=out/subject;folder=fold/run.current.architecture()['name']
    with run.current.output_lock(fold):
        for marker in ['development_sha256.json','refit_sha256.json','final_export_sha256.json']:
            base.check_seal(folder,marker)
        data,cfg=run.load_fold(out,subject)
        sweep.test_model(folder/'test',folder,data,cfg)


def pool(out, subjects, phase, jobs, threads):
    pending=list(subjects);active=[]
    try:
        while pending or active:
            while pending and len(active)<jobs:
                subject=pending.pop(0);log=(out/subject/('continue_'+phase+'.log')).open('a',encoding='utf-8')
                script=Path(run.__file__).resolve() if phase=='fit' else Path(__file__).resolve()
                cmd=[sys.executable,'-u','-X','utf8',str(script),'--output',str(out),
                     '--phase',phase if phase=='fit' else 'test-one','--fold',subject,'--threads',str(threads)]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                active.append((subject,proc,log));print('START',phase,subject,'PID',proc.pid,flush=True)
            for item in active[:]:
                subject,proc,log=item
                if proc.poll() is not None:
                    log.close();active.remove(item)
                    if proc.returncode:raise RuntimeError(f'{subject} {phase} failed; inspect continue_{phase}.log and rerun.')
                    print('FINISHED',phase,subject,flush=True)
            if active:time.sleep(2)
    finally:
        for _,proc,log in active:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()


def aggregate(out, subjects):
    full=subjects==sweep.SUBJECTS;prefix='' if full else 'partial_'
    cfg=io.read(out/'protocol.json');records=[];table=[];reports={};diagnostic={};files=[]
    base.check_seal(out,'diagnostic_policy_sha256.json')
    for subject in subjects:
        records += [r for r in io.read(out/subject/'recordings.json') if r['subject']==subject]
    expected=sum(cfg['valid_by_subject'][s] for s in subjects)
    for kind in ['fp32','int8']:
        all_events=[];per_subject={};word_trials=[]
        for subject in subjects:
            fold=out/subject;folder=fold/run.current.architecture()['name']
            base.check_seal(folder/'test','test_sha256.json')
            fold_records=io.read(fold/'recordings.json')
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            rows=io.read(fold/'windows_1400.json')
            assert cp['training_fingerprints']==[r['fingerprint'] for r in rows if r['split'] in ['train','validation']]
            assert all(r['subject']!=subject for r in rows if r['split'] in ['train','validation'])
            with np.load(folder/'test'/f'{kind}_trace.npz') as z:trace={k:z[k] for k in z.files}
            sweep.validate_grid(trace,fold_records,'test')
            events=base.decode(trace,run.current.DECODER,cfg['classes'])
            assert events==io.read(folder/'test'/f'{kind}_events.json')
            metrics,_,_=sweep.score(events,fold_records,cfg['classes'])
            assert metrics==io.read(folder/'test/comparison.json')[kind]['overall']
            per_subject[subject]=metrics;all_events+=events
            table.append(dict(format=kind,subject=subject,**metrics))
            # Identical predeclared cue-assisted rule from score_cnn_loso_trials.py.
            for r in fold_records:
                if r['subject']!=subject:continue
                rec_ix=np.flatnonzero(trace['recording']==r['recording']);ends=trace['end'][rec_ix];starts=ends-2800
                for t in r['annotations']:
                    if t['split']!='test':continue
                    overlap=np.maximum(0,np.minimum(ends,t['rest'])-np.maximum(starts,t['action']))
                    keep=(starts>=t['prepare']) & (ends<=t['context_end']) & (overlap>=2100) & \
                         (ends>=t['action']+300) & (ends<=t['rest'])
                    ix=rec_ix[keep];assert len(ix)>0
                    mean=trace['probabilities'][ix].astype(np.float64).mean(axis=0)
                    pred=cfg['classes'][int(np.argmax(mean[:10]))]
                    word_trials.append(dict(subject=subject,recording=r['recording'],trial_id=t['trial_id'],
                        word=t['word'],prediction=pred,correct=pred==t['word'],windows=len(ix),mean_rest_probability=float(mean[10])))
        overall,trials,confusion=sweep.score(all_events,records,cfg['classes'])
        assert overall['trials']==expected==len(word_trials)
        assert len({(r['recording'],r['trial_id']) for r in word_trials})==expected
        reports[kind]=dict(overall=overall,by_subject=per_subject,
            subject_mean_recall=float(np.mean([m['end_to_end_recall'] for m in per_subject.values()])))
        table.append(dict(format=kind,subject='ALL',**overall))
        diagnostic[kind]={}
        for subject in subjects+['ALL']:
            part=word_trials if subject=='ALL' else [t for t in word_trials if t['subject']==subject]
            diagnostic[kind][subject]=dict(trials=len(part),correct=sum(t['correct'] for t in part),
                accuracy=sum(t['correct'] for t in part)/len(part))
        for suffix,content in [('trials',trials),('confusion',confusion),('ten_word_trials',word_trials)]:
            path=out/f'{prefix}{kind}_{suffix}.csv';io.write_csv(path,content);files.append(path)
    for suffix,content in [('comparison',reports),('ten_word_comparison',diagnostic)]:
        path=out/f'{prefix}{suffix}.json';io.write(path,content);files.append(path)
    path=out/f'{prefix}comparison.csv';io.write_csv(path,table);files.append(path)
    lines=[('完整五折结果' if full else '仅 S011/S012 两折阶段结果，不能代表完整五折'),
        '当前 mid64_temporal3；每折其他四人训练，目标人无校准；每折3种子由源人员验证选择。',
        '用户要求先完成当前两折再自行运行后续，因此前两折先测试；所有训练规则保持固定。','']
    for kind,group in reports.items():
        lines.append(kind.upper()+'：目标人 | 十词分类准确率（提示分段） | 连续正确命令率 | 严格成功率 | 无响应')
        for subject,m in list(group['by_subject'].items())+[('ALL',group['overall'])]:
            d=diagnostic[kind][subject]
            lines.append(f'{subject} | {d["correct"]}/{d["trials"]} ({d["accuracy"]:.2%}) | '
                f'{m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) | '
                f'{m["clean_success_rate"]:.2%} | {m["no_response_trials"]}')
        lines.append('')
    lines += ['十词分类使用已知动作提示边界、平均概率并排除rest后强制选一个词，不是实时触发成功率。',
        '连续正确命令率允许同一试次还出现错词或重复；严格成功要求整试次仅一次正确输出。',
        '只有源人员全局归一化，无目标人标签训练、均值方差拟合或阈值调整。',
        '训练后量化的550窗口来自源人员，不是个人校准集。',
        'goodone_plus 已按历史实验整理，属于探索性分析；未部署或改动App/板端。']
    path=out/f'{prefix}results.txt';path.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig');files.append(path)
    print('\n'.join(lines),flush=True)
    base.verify(out);base.check_seal(out,'prepared_sha256.json');base.check_seal(out,'continuation_policy_sha256.json')
    sweep.seal(out,f'{prefix}results_sha256.json',files)
    io.write(out/f'{prefix}completed.json',dict(complete=True,full_five_fold=full,subjects=subjects,
        test_trials=expected,all_five_models_frozen_before_first_test=False,
        user_requested_staged_testing=True,no_deployment=True,trace_rescoring_passed=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--phase',choices=['all','first-two','test-one','status'],default='all')
    parser.add_argument('--fold',choices=sweep.SUBJECTS)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve()
    if args.jobs<1 or args.threads<1:parser.error('jobs and threads must be positive')
    if args.phase=='test-one' and not args.fold:parser.error('--fold required')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.phase=='status':
        for s in sweep.SUBJECTS:
            folder=out/s/run.current.architecture()['name']
            print(s,'seeds',sum((folder/'seeds'/str(seed)/'seed_sha256.json').exists() for seed in sweep.SEEDS),
                  'model_complete',(folder/'final_export_sha256.json').exists(),
                  'test_complete',(folder/'test/test_sha256.json').exists())
        return
    tmp=out/(args.fold or 'continuation')/'temporary';tmp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(tmp)
    if args.phase=='test-one':
        base.check_seal(out,'continuation_policy_sha256.json');test_one(out,args.fold);return
    with run.current.output_lock(out):
        base.check_seal(out,'prepared_sha256.json');base.verify(out);amend(out)
        subjects=FIRST if args.phase=='first-two' else sweep.SUBJECTS
        if args.phase=='all':
            pool(out,['S013','S014','S015'],'fit',args.jobs,args.threads)
        markers=[out/s/run.current.architecture()['name']/n for s in subjects
                 for n in ['refit_sha256.json','final_export_sha256.json']]
        for p in markers:base.check_seal(p.parent,p.name)
        sweep.seal(out,('all' if args.phase=='all' else 'partial')+'_models_frozen_sha256.json',
            markers+[Path(p) for marker in markers for p in io.read(marker)])
        pool(out,subjects,'test',args.jobs,args.threads)
        aggregate(out,subjects)


if __name__=='__main__':main()
