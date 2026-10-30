"""Five-person LOSO of the current mid64_temporal3 CNN, trained from scratch.

Uses the deployed-candidate architecture and existing training/decoder recipe.
No target-person calibration, normalization fitting, seed selection or tuning.
All five final models are frozen before any held-out continuous test replay.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import run_cnn_goodone_plus_current as current

sweep, base, io = current.sweep, current.base, current.io
np, torch, lda = current.np, current.torch, current.lda
SCHEMA = 'current_cnn_five_person_loso_v1'


def split_trials(valid, heldout):
    report = dict(train=[r for r in valid if r['subject'] != heldout],
                  test=[r for r in valid if r['subject'] == heldout],
                  classes=lda.ordered_classes(valid))
    rows, indices = base.cnn.split_data(report)
    for split, ix in indices.items():
        for i in ix:
            rows[int(i)] = dict(rows[int(i)], split=split)
    assert all((r['split'] == 'test') == (r['subject'] == heldout) for r in rows)
    sets = [{f for r in rows if r['split'] == s for f in r['signal_fingerprints']}
            for s in ['train', 'validation', 'test']]
    groups = [{(r['recording'], r['block']) for r in rows if r['split'] == s}
              for s in ['train', 'validation', 'test']]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not sets[a] & sets[b] and not groups[a] & groups[b]
    assert len({r['fingerprint'] for r in rows}) == len(valid)
    return rows, report['classes'] + ['rest']


def prepare(out, data_root):
    inventory = current.inventory(data_root)
    feature_config = dict(segments=6, metrics=['log_mean_abs_diff', 'log_mav', 'zc_count_zero', 'ssc_count_zero'])
    valid, excluded, sources = lda.extract(inventory, 'notch50_bp500', print, feature_config, [0, 1])
    # This extractor supplies QC and timing boundaries only. No LDA is trained.
    io.write(out/'data_audit.json', dict(sources=sources, excluded=excluded,
        valid_by_subject=dict(Counter(r['subject'] for r in valid))))
    first_rows, classes = split_trials(valid, sweep.SUBJECTS[0])
    records = current.build_cache(out, sources, first_rows)
    waves = current.load_waves(out, records)
    all_test = []
    files = [out/'data_audit.json'] + list((out/'cache').glob('*.npz'))
    for heldout in sweep.SUBJECTS:
        fold = out/heldout; fold.mkdir()
        trial_rows, fold_classes = split_trials(valid, heldout)
        assert fold_classes == classes
        lookup = {(r['recording'], r['trial_id']): r['split'] for r in trial_rows}
        fold_records = deepcopy(records)
        for record in fold_records:
            for trial in record['annotations']:
                trial['split'] = lookup.get((record['recording'], trial['trial_id']), 'excluded')
        rows, indices, window_exclusions = current.make_windows(out, fold_records, classes)
        assert all(rows[i]['subject'] != heldout for key in
            ['train', 'validation', 'fit', 'dev_calibration', 'final_calibration'] for i in indices[key])
        assert all(rows[i]['subject'] == heldout for i in indices['test'])
        dev_norm = base.normalization(fold_records, waves, {'train'})
        final_norm = base.normalization(fold_records, waves, {'train', 'validation'})
        counts = dict(Counter(r['split'] for r in trial_rows))
        template = fold/'contract_template'/'export'; template.mkdir(parents=True)
        io.write(template/'input_contract.json', current.input_contract(classes, final_norm))
        cfg = dict(schema=SCHEMA, heldout=heldout, data_root=str(data_root),
            base=str(template.parent), architectures=[current.architecture()], seeds=sweep.SEEDS,
            classes=classes, window_ms=1400, step_ms=100, max_epochs=100, early_stop=10,
            batch=44, windows_per_class_per_epoch=400, dropout=.5,
            optimizer=dict(name='Adam', lr=.001, weight_decay=.0001), scheduler=dict(factor=.1, patience=2),
            dev_normalization=dev_norm, final_normalization=final_norm, decoder=current.DECODER,
            counts=counts, by_subject={s:dict(Counter(r['split'] for r in trial_rows if r['subject']==s))
                                      for s in sweep.SUBJECTS},
            split='Outer: entire held-out person. Inner: one complete source round per recording, seed20261010.',
            architecture_selection='Fixed current mid64_temporal3; no architecture search.',
            seed_epoch_selection='Three fresh starts, minimum source validation-window macro CE; no target data.',
            refit='Fresh source-only train+validation refit using chosen epoch count and LR trajectory.',
            normalization='Global two-channel statistics of source contexts only, refitted for final source training.',
            quantization='Source-only PTQ calibration; source validation selects MinMax/99.99/99.9/99.5.',
            decoder_policy='Fixed probability0.5, confirm2, rest_frames2, margin0; no target threshold search.',
            versions=dict(torch=torch.__version__, numpy=np.__version__, onnx=base.auto.onnx.__version__))
        io.write(fold/'protocol.json', cfg)
        io.write(fold/'trial_manifest.json', trial_rows)
        io.write_csv(fold/'split_manifest.csv', [{k:r[k] for k in
            ['split','subject','recording','path','trial_id','block','word','prepare','action','rest','fingerprint']}
            for r in trial_rows])
        io.write(fold/'recordings.json', fold_records)
        io.write(fold/'windows_1400.json', rows)
        io.write(fold/'indices.json', indices)
        io.write(fold/'dataset_summary.json', dict(counts=counts, by_subject=cfg['by_subject'],
            windows=len(rows), indices_counts={k:len(v) for k,v in indices.items()},
            window_exclusions=window_exclusions))
        files += list(fold.glob('*.json')) + [fold/'split_manifest.csv', template/'input_contract.json']
        all_test += [r['fingerprint'] for r in trial_rows if r['split']=='test']
        print('PREPARED FOLD', heldout, counts, flush=True)
    assert len(all_test)==len(valid) and len(set(all_test))==len(valid)
    cfg = dict(schema=SCHEMA, inventory=inventory, subjects=sweep.SUBJECTS,
        architecture=current.architecture(), classes=classes, test_trials=len(valid),
        valid_by_subject=dict(Counter(r['subject'] for r in valid)),
        seeds=sweep.SEEDS, personal_calibration=False, all_models_frozen_before_test=True,
        evaluation='Continuous command decoding; correct-command recall and strict one-command success.',
        caveats=['Exploratory LOSO on previously curated goodone_plus, not a new blind cohort.',
                 'Validation people are source-training people; no nested held-out-subject hyperparameter search.',
                 'Continuous command scores differ from LDA ten-class isolated-trial accuracy.',
                 'PTQ calibration means source activation range estimation, NOT personal calibration.',
                 'No board access, deployment or app changes.'])
    io.write(out/'protocol.json', cfg)
    files += [out/'protocol.json']
    hashes = {str(Path(s['path'])/n):h for s in sources for n,h in s['hashes'].items()}
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for p in sorted(set(current.dependencies()+[Path(__file__).resolve()])):
        hashes[str(p)] = io.digest(p)
        target = snapshot/p.relative_to(current.ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    io.write(out/'inputs_sha256.json', hashes)
    files += [out/'inputs_sha256.json']
    base.verify(out)
    sweep.seal(out, 'prepared_sha256.json', files)


def load_fold(out, heldout):
    fold = out/heldout
    cfg = io.read(fold/'protocol.json')
    data = dict(rows=io.read(fold/'windows_1400.json'), records=io.read(fold/'recordings.json'),
                indices=io.read(fold/'indices.json'))
    data['waves'] = current.load_waves(out, data['records'])
    assert base.normalization(data['records'],data['waves'],{'train'})==cfg['dev_normalization']
    assert base.normalization(data['records'],data['waves'],{'train','validation'})==cfg['final_normalization']
    return data, cfg


def worker(out, subject, phase):
    fold = out/subject
    with current.output_lock(fold):
        data, cfg = load_fold(out, subject)
        spec = current.architecture(); folder = fold/spec['name']
        if phase=='fit':
            sweep.develop(folder,spec,data,cfg)
            sweep.freeze_recommendation(fold,cfg)
            sweep.finalize(folder,spec,data,cfg)
        else:
            base.check_seal(out,'all_models_frozen_sha256.json')
            sweep.test_model(folder/'test',folder,data,cfg)


def pool(out, phase, jobs, threads):
    pending = list(sweep.SUBJECTS); active = []
    try:
        while pending or active:
            while pending and len(active)<jobs:
                subject=pending.pop(0)
                stream=(out/subject/(phase+'.log')).open('a',encoding='utf-8')
                cmd=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--output',str(out),
                     '--phase',phase,'--fold',subject,'--threads',str(threads)]
                process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
                active.append((subject,process,stream))
                print('START',phase,subject,'PID',process.pid,flush=True)
            for item in active[:]:
                subject,process,stream=item
                if process.poll() is not None:
                    stream.close();active.remove(item)
                    if process.returncode:raise RuntimeError(f'{phase} {subject} failed; see {out/subject/(phase+".log")}')
                    print('FINISHED',phase,subject,flush=True)
            if active:time.sleep(2)
    finally:
        for _,process,stream in active:
            if process.poll() is None:process.terminate();process.wait()
            stream.close()


def summarize(out):
    cfg=io.read(out/'protocol.json'); records=[]; table=[]; aggregate={}
    for subject in sweep.SUBJECTS:
        records += [r for r in io.read(out/subject/'recordings.json') if r['subject']==subject]
    for kind in ['fp32','int8']:
        events=[]; per_subject={}
        for subject in sweep.SUBJECTS:
            folder=out/subject/current.architecture()['name']
            base.check_seal(folder/'test','test_sha256.json')
            cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            rows=io.read(out/subject/'windows_1400.json')
            assert cp['training_fingerprints']==[r['fingerprint'] for r in rows if r['split'] in ['train','validation']]
            training_fingerprints=set(cp['training_fingerprints'])
            assert all(r['subject']!=subject for r in rows if r['fingerprint'] in training_fingerprints)
            with np.load(folder/'test'/f'{kind}_trace.npz') as z:
                trace={k:z[k] for k in z.files}
            fold_records=io.read(out/subject/'recordings.json')
            sweep.validate_grid(trace,fold_records,'test')
            decoded=base.decode(trace,current.DECODER,cfg['classes'])
            assert decoded==io.read(folder/'test'/f'{kind}_events.json')
            metrics,_,_=sweep.score(decoded,fold_records,cfg['classes'])
            assert metrics==io.read(folder/'test/comparison.json')[kind]['overall']
            per_subject[subject]=metrics;events+=decoded
            table.append(dict(format=kind,subject=subject,**metrics))
        overall,trials,confusion=sweep.score(events,records,cfg['classes'])
        assert overall['trials']==cfg['test_trials']
        aggregate[kind]=dict(overall=overall,by_subject=per_subject,
            subject_mean_recall=float(np.mean([m['end_to_end_recall'] for m in per_subject.values()])))
        table.append(dict(format=kind,subject='ALL',**overall))
        io.write_csv(out/f'{kind}_trials.csv',trials);io.write_csv(out/f'{kind}_confusion.csv',confusion)
    io.write(out/'comparison.json',aggregate);io.write_csv(out/'comparison.csv',table)
    lines=['当前 CNN 五折跨人 LOSO：四人训练，整人留出；无目标人员校准。',
           'mid64_temporal3，1400ms/100ms步进，10命令+rest；每折3种子仅按源人员验证集选择。',
           '先完成全部五个最终模型，再统一测试。以下为连续命令指标，不等同于LDA十分类准确率。','']
    for kind,group in aggregate.items():
        lines.append(kind.upper()+'：人员 | 正确命令/试次 | 正确命令率 | 严格成功率 | 无响应 | 错词输出')
        for subject,m in list(group['by_subject'].items())+[('ALL',group['overall'])]:
            lines.append(f'{subject} | {m["correct_unique_trials"]}/{m["trials"]} | {m["end_to_end_recall"]:.2%} | '
                f'{m["clean_success_rate"]:.2%} | {m["no_response_trials"]} | {m["wrong_label_outputs"]}')
        lines.append('')
    lines += ['正确命令率：响应时段内至少一次正确命令；可能同时存在错词或重复输出。',
              '严格成功率：整个试次上下文内只输出一次，且该输出在响应时段内并正确。',
              '归一化只用其他四人的训练数据；PTQ也是源人员量化校准，不是目标人员校准。',
              '数据此前按历史实验整理过，属于探索性跨人测试；不是独立盲测新队列。',
              '未修改原始录制、现有应用或板端模型。']
    text='\n'.join(lines)+'\n';(out/'results.txt').write_text(text,encoding='utf-8-sig');print(text,flush=True)
    base.verify(out);base.check_seal(out,'prepared_sha256.json');base.check_seal(out,'all_models_frozen_sha256.json')
    sweep.seal(out,'results_sha256.json',[out/n for n in ['comparison.json','comparison.csv','results.txt',
        'fp32_trials.csv','int8_trials.csv','fp32_confusion.csv','int8_confusion.csv']])
    io.write(out/'completed.json',dict(complete=True,schema=SCHEMA,test_trials=cfg['test_trials'],
        no_deployment=True,all_five_models_frozen_before_test=True,trace_rescoring_passed=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','prepare','fit','test','summarize'],default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fold',choices=sweep.SUBJECTS)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.jobs<1 or args.threads<1:parser.error('jobs/threads must be positive')
    if args.phase in ['fit','test'] and not args.fold:parser.error('worker requires --fold')
    if (args.resume or args.phase in ['fit','test','summarize']) and not args.output:parser.error('existing run requires --output')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or current.ROOT/'models'/('cnn_current_loso_'+stamp)).resolve()
    if args.phase in ['all','prepare'] and not args.resume:out.mkdir(parents=True,exist_ok=False)
    tmp=out/(args.fold or 'orchestrator')/'temporary';tmp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(tmp)
    print('OUTPUT',out,'PHASE',args.phase,flush=True)
    if args.phase in ['fit','test']:worker(out,args.fold,args.phase);return
    with current.output_lock(out):
        if args.phase=='summarize':summarize(out);return
        if not args.resume:prepare(out,current.DATA.resolve())
        if args.phase=='prepare':return
        base.check_seal(out,'prepared_sha256.json');base.verify(out)
        pool(out,'fit',args.jobs,args.threads)
        markers=[out/s/current.architecture()['name']/n for s in sweep.SUBJECTS
                 for n in ['refit_sha256.json','final_export_sha256.json']]
        for p in markers:base.check_seal(p.parent,p.name)
        sweep.seal(out,'all_models_frozen_sha256.json',markers+
            [Path(p) for marker in markers for p in io.read(marker)])
        base.verify(out)
        pool(out,'test',args.jobs,args.threads)
        summarize(out)


if __name__=='__main__':main()
