"""Five independent within-person CNNs, grouped 80/20 train/test, no pretraining.

Same mid64_temporal3 architecture and training recipe as the current CNN.
Default output is resumable. --phase prepare performs data preparation only.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import run_cnn_goodone_plus_current as current

sweep, base, io = current.sweep, current.base, current.io
torch, np, lda = current.torch, current.np, current.lda
SCHEMA = 'cnn_independent_people_grouped_80_20_v1'
DEFAULT = current.ROOT/'models/cnn_individual_80_20_20261030'


def subject_split(valid, subject):
    own = [r for r in valid if r['subject']==subject]
    if not own:
        raise ValueError(f'No valid trials for {subject}')
    rows, classes = current.make_split(own)
    if len(classes)!=11 or classes[-1]!='rest':
        raise ValueError(f'{subject}: expected all ten command classes plus rest')
    assert {r['subject'] for r in rows}=={subject}
    assert len(rows)==len(own) and len({r['fingerprint'] for r in rows})==len(own)
    # Each recording's test rounds are chosen before any window generation.
    for recording in {r['recording'] for r in rows}:
        part=[r for r in rows if r['recording']==recording]
        blocks={r['block'] for r in part}
        tests={r['block'] for r in part if r['split']=='test'}
        assert len(tests)==max(1,min(len(blocks)-1,int(np.ceil(len(blocks)*.2))))
    return rows, classes


def audit_person(data, subject):
    assert data['records'] and data['rows']
    assert {r['subject'] for r in data['records']}=={subject}
    assert {r['subject'] for r in data['rows']}=={subject}
    indices=data['indices']; rows=data['rows']
    assert set(indices['fit'])=={i for i,r in enumerate(rows) if r['split'] in ['train','validation']}
    assert not set(indices['fit']) & set(indices['test'])
    for name in ['train','validation','test']:
        assert all(rows[i]['split']==name for i in indices[name])
    assert set(indices['dev_calibration'])<=set(indices['train'])
    assert set(indices['final_calibration'])<=set(indices['fit'])
    # A full window must stay in its own trial context; actual samples may not
    # overlap train/validation/test even though within-split windows overlap.
    contexts={s:[] for s in ['train','validation','test']}
    for r in data['records']:
        by_trial={t['trial_id']:t for t in r['annotations']}
        for w in rows:
            if w['recording']!=r['recording']:continue
            t=by_trial[w['trial_id']]
            assert w['split']==t['split'] and t['prepare']<=w['start']<w['end']<=t['context_end']
        for t in r['annotations']:
            if t['split'] in contexts:contexts[t['split']].append((r['recording'],t['prepare'],t['context_end']))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert all(rec!=rec2 or end<=start2 or end2<=start
                   for rec,start,end in contexts[a] for rec2,start2,end2 in contexts[b])


def prepare(out):
    selections=current.inventory(current.DATA.resolve())
    config=dict(segments=6,metrics=['log_mean_abs_diff','log_mav','zc_count_zero','ssc_count_zero'])
    valid,excluded,sources=lda.extract(selections,'notch50_bp500',lambda s:print(s,flush=True),config,[0,1])
    plans={}; all_rows=[]; expected_classes=lda.ordered_classes(valid)+['rest']
    for subject in sweep.SUBJECTS:
        rows,classes=subject_split(valid,subject)
        assert classes==expected_classes
        plans[subject]=rows;all_rows.extend(rows)
    assert len(all_rows)==len(valid)
    records=current.build_cache(out,sources,all_rows)
    summaries={};files=list((out/'cache').glob('*.npz'))
    for subject in sweep.SUBJECTS:
        fold=out/subject;fold.mkdir()
        own_records=[r for r in records if r['subject']==subject]
        rows,indices,exclusions=current.make_windows(out,own_records,classes)
        data=dict(records=own_records,rows=rows,indices=indices);audit_person(data,subject)
        waves=current.load_waves(out,own_records)
        norm=base.normalization(own_records,waves,{'train'})
        final_norm=base.normalization(own_records,waves,{'train','validation'})
        counts=dict(Counter(r['split'] for r in plans[subject]))
        template=fold/'contract_template'/'export';template.mkdir(parents=True)
        io.write(template/'input_contract.json',current.input_contract(classes,final_norm))
        cfg=dict(schema=SCHEMA,subject=subject,base=str(template.parent),architectures=[current.architecture()],
            seeds=sweep.SEEDS,classes=classes,window_ms=1400,step_ms=100,max_epochs=100,early_stop=10,
            batch=44,windows_per_class_per_epoch=400,dropout=.5,
            optimizer=dict(name='Adam',lr=.001,weight_decay=.0001),scheduler=dict(factor=.1,patience=2),
            dev_normalization=norm,final_normalization=final_norm,decoder=current.DECODER,counts=counts,
            split='This person only. Per-recording whole-round outer 80/20 seed20260924; one source round per recording validation seed20261010.',
            architecture_selection='Fixed mid64_temporal3; no architecture search.',
            selection='Three random initializations; minimum own-training validation macro CE selects seed/epoch. No test used.',
            refit='Fresh random initialization on the whole own-person 80% partition, selected seed/epochs/LR schedule.',
            quantization='Training-only 550-window activation calibration; four PTQ candidates selected by development validation.',
            versions=dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__))
        summary=dict(subject=subject,valid=len(plans[subject]),development=counts,
            final_train=counts['train']+counts['validation'],test=counts['test'],
            test_fraction=counts['test']/len(plans[subject]),windows=len(rows),
            indices_counts={k:len(v) for k,v in indices.items()},window_exclusions=exclusions)
        summaries[subject]=summary
        for name,value in [('protocol',cfg),('recordings',own_records),('windows_1400',rows),
                           ('indices',indices),('trial_manifest',plans[subject]),('dataset_summary',summary)]:
            io.write(fold/(name+'.json'),value)
        io.write_csv(fold/'split_manifest.csv',[{k:r[k] for k in
            ['split','subject','recording','trial_id','block','word','prepare','action','rest','fingerprint']}
            for r in plans[subject]])
        files+=list(fold.glob('*.json'))+[fold/'split_manifest.csv',template/'input_contract.json']
        print('PREPARED',subject,summary,flush=True)
    io.write(out/'data_audit.json',dict(sources=sources,excluded=excluded))
    io.write(out/'dataset_summary.json',summaries)
    io.write(out/'protocol.json',dict(schema=SCHEMA,inventory=selections,subjects=sweep.SUBJECTS,classes=classes,
        architecture=current.architecture(),seeds=sweep.SEEDS,test_trials=sum(s['test'] for s in summaries.values()),
        independent_people=True,outer_train_percent=80,outer_seed=20260924,inner_seed=20261010,
        all_models_frozen_before_test=True,no_pretrained_weights=True,
        diagnostic='Mean original 11-class probabilities in cue-valid action windows; ten-word argmax excluding rest.',
        caveats=['Within-person, same-recording whole-round holdout; not cross-person or cross-session/day.',
                 'QC exclusions can make the valid-trial ratio slightly different from 80/20.',
                 'Adjacent rounds can remain correlated; this is not independent reattachment testing.',
                 'No deployment or change to existing models.']))
    files += [out/n for n in ['data_audit.json','dataset_summary.json','protocol.json']]
    hashes={str(Path(s['path'])/n):h for s in sources for n,h in s['hashes'].items()}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in sorted(set(current.dependencies()+[Path(__file__).resolve()])):
        hashes[str(p)]=io.digest(p)
        target=snapshot/p.relative_to(current.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    io.write(out/'inputs_sha256.json',hashes);files.append(out/'inputs_sha256.json')
    base.verify(out);sweep.seal(out,'prepared_sha256.json',files)


def load_person(out,subject):
    fold=out/subject;cfg=io.read(fold/'protocol.json')
    assert cfg['schema']==SCHEMA and cfg['subject']==subject
    assert cfg['versions']==dict(torch=torch.__version__,numpy=np.__version__,onnx=base.auto.onnx.__version__)
    data=dict(records=io.read(fold/'recordings.json'),rows=io.read(fold/'windows_1400.json'),indices=io.read(fold/'indices.json'))
    audit_person(data,subject)
    data['waves']=current.load_waves(out,data['records'])
    assert base.normalization(data['records'],data['waves'],{'train'})==cfg['dev_normalization']
    assert base.normalization(data['records'],data['waves'],{'train','validation'})==cfg['final_normalization']
    return data,cfg


def worker(out,subject,phase):
    fold=out/subject;spec=current.architecture();folder=fold/spec['name']
    with current.output_lock(fold):
        data,cfg=load_person(out,subject)
        if phase=='fit':
            sweep.develop(folder,spec,data,cfg);sweep.freeze_recommendation(fold,cfg)
            sweep.finalize(folder,spec,data,cfg)
        else:
            base.check_seal(out,'all_models_frozen_sha256.json')
            sweep.test_model(folder/'test',folder,data,cfg)


def pool(out,phase,jobs,threads):
    pending=list(sweep.SUBJECTS);active=[]
    try:
        while pending or active:
            while pending and len(active)<jobs:
                subject=pending.pop(0);log=(out/subject/(phase+'.log')).open('a',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-u','-X','utf8',str(Path(__file__).resolve()),
                    '--output',str(out),'--phase',phase,'--subject',subject,'--threads',str(threads)],
                    stdout=log,stderr=subprocess.STDOUT)
                active.append((subject,proc,log));print('START',phase,subject,'PID',proc.pid,flush=True)
            for item in active[:]:
                subject,proc,log=item
                if proc.poll() is not None:
                    log.close();active.remove(item)
                    if proc.returncode:raise RuntimeError(f'{subject} {phase} failed: inspect {out/subject/(phase+".log")}')
                    print('FINISHED',phase,subject,flush=True)
            if active:time.sleep(2)
    finally:
        for _,proc,log in active:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()


def summarize(out):
    cfg=io.read(out/'protocol.json');all_records=[];report={};table=[];files=[]
    for s in sweep.SUBJECTS:all_records+=io.read(out/s/'recordings.json')
    for kind in ['fp32','int8']:
        events=[];classification=[];per_subject={}
        for subject in sweep.SUBJECTS:
            fold=out/subject;folder=fold/current.architecture()['name'];records=io.read(fold/'recordings.json')
            base.check_seal(folder/'test','test_sha256.json')
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            windows=io.read(fold/'windows_1400.json')
            assert {r['subject'] for r in windows}=={subject}
            assert cp['training_fingerprints']==[r['fingerprint'] for r in windows if r['split'] in ['train','validation']]
            with np.load(folder/'test'/f'{kind}_trace.npz') as z:trace={k:z[k] for k in z.files}
            sweep.validate_grid(trace,records,'test')
            decoded=base.decode(trace,current.DECODER,cfg['classes'])
            assert decoded==io.read(folder/'test'/f'{kind}_events.json')
            metrics,_,_=sweep.score(decoded,records,cfg['classes'])
            assert metrics==io.read(folder/'test/comparison.json')[kind]['overall']
            own=[]
            for r in records:
                ix=np.flatnonzero(trace['recording']==r['recording']);ends=trace['end'][ix];starts=ends-2800
                for t in r['annotations']:
                    if t['split']!='test':continue
                    overlap=np.maximum(0,np.minimum(ends,t['rest'])-np.maximum(starts,t['action']))
                    keep=(starts>=t['prepare']) & (ends<=t['context_end']) & (overlap>=2100) & \
                         (ends>=t['action']+300) & (ends<=t['rest'])
                    selected=ix[keep];assert len(selected)>0
                    probabilities=trace['probabilities'][selected].astype(np.float64).mean(axis=0)
                    pred=cfg['classes'][int(probabilities[:10].argmax())]
                    own.append(dict(subject=subject,recording=r['recording'],trial_id=t['trial_id'],word=t['word'],
                        prediction=pred,correct=pred==t['word'],windows=len(selected)))
            assert len(own)==metrics['trials']
            per_subject[subject]=dict(**metrics,ten_word_correct=sum(t['correct'] for t in own),
                ten_word_accuracy=sum(t['correct'] for t in own)/len(own))
            events+=decoded;classification+=own
            table.append(dict(format=kind,subject=subject,**per_subject[subject]))
        overall,trials,confusion=sweep.score(events,all_records,cfg['classes'])
        assert overall['trials']==cfg['test_trials']==len(classification)
        assert len({(r['recording'],r['trial_id']) for r in classification})==len(classification)
        overall.update(ten_word_correct=sum(t['correct'] for t in classification),
            ten_word_accuracy=sum(t['correct'] for t in classification)/len(classification))
        report[kind]=dict(overall=overall,by_subject=per_subject)
        table.append(dict(format=kind,subject='ALL',**overall))
        for name,rows in [('trials',trials),('confusion',confusion),('ten_word_trials',classification)]:
            path=out/f'{kind}_{name}.csv';io.write_csv(path,rows);files.append(path)
    io.write(out/'comparison.json',report);io.write_csv(out/'comparison.csv',table)
    lines=['五个独立个人 CNN：每人只训练自己的数据，整轮 80/20；不是跨人测试。',
           '三种子在各自训练部分的内部验证集选择；最终用各自完整80%重训，再测试20%。','']
    for kind,result in report.items():
        lines.append(kind.upper()+'：人员 | 十词分类（提示分段） | 连续正确命令率 | 严格成功率')
        for subject,m in list(result['by_subject'].items())+[('ALL',result['overall'])]:
            lines.append(f'{subject} | {m["ten_word_correct"]}/{m["trials"]} ({m["ten_word_accuracy"]:.2%}) | '
                f'{m["end_to_end_recall"]:.2%} | {m["clean_success_rate"]:.2%}')
    lines+=['','十词分类借助提示边界平均概率，再排除rest强制选择一个词，不是在线触发成功率。',
        '连续正确命令率允许错词/重复；严格成功要求整试次只输出一次且该输出正确。',
        '同录制按整轮留出，可能仍有同次佩戴相关性；不能当作跨录制、跨日或重新佩戴成绩。',
        '与跨人实验的测试量、训练人数不同，不能把差值只归因于算法。未部署或修改原模型。']
    text='\n'.join(lines)+'\n';(out/'results.txt').write_text(text,encoding='utf-8-sig');print(text,flush=True)
    files += [out/n for n in ['comparison.json','comparison.csv','results.txt']]
    base.verify(out);base.check_seal(out,'prepared_sha256.json');base.check_seal(out,'all_models_frozen_sha256.json')
    sweep.seal(out,'results_sha256.json',files)
    io.write(out/'completed.json',dict(complete=True,schema=SCHEMA,subjects=sweep.SUBJECTS,
        test_trials=cfg['test_trials'],independent_people=True,trace_rescoring_passed=True,no_deployment=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--phase',choices=['all','prepare','fit','test','status'],default='all')
    parser.add_argument('--subject',choices=sweep.SUBJECTS)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve()
    if args.jobs<1 or args.threads<1:parser.error('jobs and threads must be positive')
    if args.phase in ['fit','test'] and not args.subject:parser.error('--subject required for a worker')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.phase=='status':
        print('OUTPUT',out,'prepared',(out/'prepared_sha256.json').exists(),'complete',(out/'completed.json').exists())
        for s in sweep.SUBJECTS:
            folder=out/s/current.architecture()['name']
            print(s,'seeds',sum((folder/'seeds'/str(seed)/'seed_sha256.json').exists() for seed in sweep.SEEDS),
                'model_complete',(folder/'final_export_sha256.json').exists(),'test_complete',(folder/'test/test_sha256.json').exists())
        return
    fresh=not out.exists()
    if fresh:out.mkdir(parents=True)
    if not fresh and not (out/'prepared_sha256.json').exists():
        parser.error('Preparation was interrupted. Keep this directory for diagnosis and use --output with a new directory.')
    tmp=out/(args.subject or 'orchestrator')/'temporary';tmp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(tmp)
    if args.phase in ['fit','test']:
        worker(out,args.subject,args.phase);return
    with current.output_lock(out):
        if fresh:prepare(out)
        base.check_seal(out,'prepared_sha256.json');base.verify(out)
        if args.phase=='prepare':print('PREPARATION ONLY COMPLETE; no real training started',flush=True);return
        if (out/'completed.json').exists():
            base.check_seal(out,'results_sha256.json');base.check_seal(out,'all_models_frozen_sha256.json')
            print('Already complete:',out/'results.txt');return
        pool(out,'fit',args.jobs,args.threads)
        markers=[out/s/current.architecture()['name']/n for s in sweep.SUBJECTS for n in ['refit_sha256.json','final_export_sha256.json']]
        for p in markers:base.check_seal(p.parent,p.name)
        sweep.seal(out,'all_models_frozen_sha256.json',markers+[Path(p) for marker in markers for p in io.read(marker)])
        base.verify(out);pool(out,'test',args.jobs,args.threads);summarize(out)


if __name__=='__main__':main()
