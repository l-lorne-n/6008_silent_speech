"""LDA leave-one-subject-out evaluation of explicitly selected people.

Only reads the requested person directories. Extract once, then independently
fit a fresh training-only scaler and shrinkage LDA for each held-out person.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from semg import __version__, lda


def fit_fold(records, held_out, sources, excluded, config, output, variant='notch50'):
    train = [dict(r, role='train') for r in records if r['subject'] != held_out]
    test = [dict(r, role='test') for r in records if r['subject'] == held_out]
    train_subjects = sorted({r['subject'] for r in train})
    subject_count = len({r['subject'] for r in records})
    assert held_out not in train_subjects and len(train_subjects) == subject_count - 1
    classes = lda.ordered_classes(train)
    assert len(classes) == 10 and {r['word'] for r in test} == set(classes)
    assert min(Counter(r['word'] for r in train).values()) >= 2
    assert not {r['fingerprint'] for r in train} & {r['fingerprint'] for r in test}
    assert not {f for r in train for f in r['signal_fingerprints']} & {f for r in test for f in r['signal_fingerprints']}
    x = np.array([r['features'] for r in train])
    xt = np.array([r['features'] for r in test])
    y = np.array([classes.index(r['word']) for r in train])
    scaler = StandardScaler().fit(x)
    estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(scaler.transform(x), y)
    model = dict(schema=lda.SCHEMA, app_version=__version__, sklearn_version=sklearn.__version__,
                 created_utc=datetime.now(timezone.utc).isoformat(), classes=classes,
                 feature_names=lda.feature_names(config, [0, 1]), feature_config=config, channels=[0, 1],
                 variant=variant, sos=lda.filter_sos(variant).tolist(),
                 scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                 coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                 preprocessing=lda.preprocessing_config(variant),
                 training_fingerprints=[r['fingerprint'] for r in train], training_trials=len(train),
                 training_signal_fingerprints=sorted({f for r in train for f in r['signal_fingerprints']}),
                 training_subjects=train_subjects)
    lda.validate_model(model)
    prediction = lda.predict(model, xt)
    np.testing.assert_array_equal(prediction, estimator.predict(scaler.transform(xt)))
    np.testing.assert_allclose(scaler.mean_, x.mean(axis=0), rtol=0, atol=0)
    metrics = lda.metrics(test, prediction, classes)
    warnings = [
        '每折重新拟合仅训练人员的标准化及 LDA；测试人员不参与拟合或参数选择。',
        '使用用户指定的人员目录；本结果属于探索性评估，不证明对未筛选人群的无偏表现。',
        f'{subject_count}位受试者的留人测试，每折{subject_count-1}人训练、一人测试。',
        '仍使用提示动作窗口，未评价自动截段、静息误触发或实时端到端表现。',
    ]
    report = dict(scope=f'{subject_count}折留人测试：{", ".join(train_subjects)} 训练 → {held_out} 测试',
                  strategy='subject', evaluation_scope=f'{subject_count}_subject_loso', held_out_subject=held_out,
                  requested_train_percent=None, actual_train_percent=100*len(train)/len(records), seed=None,
                  classes=classes, variant=variant, feature_config=config,
                  feature_names=model['feature_names'], channels=[0, 1],
                  train=train, test=test, excluded=excluded, sources=sources, metrics=metrics, warnings=warnings)
    run = lda.save_run(output/'folds'/held_out, model, report, test, prediction, 'train')
    reloaded = lda.load_model(run['model_path'])
    np.testing.assert_array_equal(lda.predict(reloaded, xt), prediction)
    assert reloaded['training_subjects'] == train_subjects
    summary = dict(held_out_subject=held_out, train_subjects=train_subjects, train_count=len(train),
                   test_count=len(test), correct=metrics['correct'], accuracy=metrics['accuracy'],
                   macro_f1=metrics['macro_f1'], folder=run['folder'], model_reload_verified=True,
                   training_only_normalization_verified=True)
    predicted_rows = [dict(held_out_subject=held_out, subject=r['subject'], recording=r['recording'],
                           trial_id=r['trial_id'], fingerprint=r['fingerprint'], word=r['word'],
                           prediction=classes[int(p)], correct=int(classes[int(p)] == r['word']))
                      for r, p in zip(test, prediction)]
    return summary, predicted_rows, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'recordings/goodone')
    parser.add_argument('--people', nargs='+', default=['s1', 's2', 's4', 's5'])
    parser.add_argument('--bandpass-high', type=int, choices=[400, 450, 500], default=400)
    parser.add_argument('--exclude-recording', action='append', default=[], help='Exact recording directory name to exclude entirely')
    parser.add_argument('--feature-set', choices=['legacy', 'rms_mav_zcr_wl'], default='legacy')
    args = parser.parse_args()
    variant = {400: 'notch50', 450: 'notch50_bp450', 500: 'notch50_bp500'}[args.bandpass_high]
    if len(args.people) < 2 or len(set(args.people)) != len(args.people):
        raise ValueError('Specify at least two distinct person directories')
    selections, inventory, excluded_recordings = [], [], []
    for person in args.people:
        directory = (args.root/person).resolve()
        if not directory.is_relative_to(args.root.resolve()):
            raise ValueError('Person directory must be inside root')
        metadata = sorted(directory.rglob('metadata.json'))
        if not metadata:
            raise ValueError(f'No metadata found in {directory}')
        ids = set()
        for p in metadata:
            m = json.loads(p.read_text(encoding='utf-8-sig'))
            if p.parent.name in args.exclude_recording:
                excluded_recordings.append(dict(path=str(p.parent), subject=m.get('subject_id'), status=m.get('status'), reason='Explicitly excluded by user: no training or evaluation'))
                continue
            if m.get('status') not in ('completed', 'stopped') or m.get('condition') != 'silent' or m.get('articulation_mode') != 'silent_mouthing':
                raise ValueError(f'Unexpected acquisition condition: {p}')
            subject = m.get('subject_id')
            if not subject:
                raise ValueError(f'Missing subject ID: {p}')
            ids.add(subject)
            selections.append(dict(path=str(p.parent), role='train'))
            inventory.append(dict(path=str(p.parent), subject=subject, person_directory=person,
                                  planned_trials=len(m.get('trial_order', []))))
        if len(ids) != 1:
            raise ValueError(f'Inconsistent subject IDs in {directory}: {ids}')
    if {Path(r['path']).name for r in excluded_recordings} != set(args.exclude_recording):
        raise ValueError('An explicitly excluded recording was not found')
    subjects = sorted({i['subject'] for i in inventory})
    if len(subjects) != len(args.people):
        raise ValueError(f'Expected one distinct subject per directory, got {subjects}')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'models'/f'lda_loso_{len(subjects)}p_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir()
    shutil.copy2(__file__, output/'train_lda_loso.py')
    shutil.copy2(ROOT/'semg/lda.py', output/'lda_source_snapshot.py')
    config = dict(segments=6, metrics=['log_mean_abs_diff', 'log_mav', 'zc_count_zero', 'ssc_count_zero'])
    if args.feature_set == 'rms_mav_zcr_wl':
        config = dict(segments=6, metrics=['rms', 'mav', 'zcr_zero', 'waveform_length'])
    protocol = dict(subjects=subjects, directories=args.people, inventory=inventory, excluded_recordings=excluded_recordings,
                    evaluation=f'{len(subjects)}-fold LOSO: each subject held out exactly once; {len(subjects)-1} train and one test.',
                    sampling_rate_hz=lda.variant_fs(variant), preprocessing=lda.preprocessing_config(variant),
                    channels=[0, 1], feature_config=config, variant=variant,
                    feature_set=args.feature_set,
                    bandpass_hz=[20, args.bandpass_high], notch_hz=50, notch_q=30,
                    estimator='StandardScaler + LinearDiscriminantAnalysis(lsqr, shrinkage=auto)',
                    selection='No tuning or result-dependent exclusion. All eligible trials in requested folders.',
                    recording_policy='Completed and stopped recordings; only complete trials passing existing quality checks.', sklearn_version=sklearn.__version__)
    lda.write_json(output/'protocol.json', protocol)
    print('OUTPUT', output, flush=True)
    records, excluded, sources = lda.extract(selections, variant,
        progress=lambda s: print(s, flush=True), config=config, channels=[0, 1])
    assert sorted({r['subject'] for r in records}) == subjects
    lda.write_json(output/'data_audit.json', dict(sources=sources, excluded=excluded,
                   valid_by_subject=dict(Counter(r['subject'] for r in records)),
                   words_by_subject={s:dict(Counter(r['word'] for r in records if r['subject']==s)) for s in subjects}))
    print('VALID_COUNTS', dict(Counter(r['subject'] for r in records)), 'EXCLUDED', len(excluded), flush=True)
    folds, predictions = [], []
    for subject in subjects:
        fold, rows, classes = fit_fold(records, subject, sources, excluded, config, output, variant)
        folds.append(fold)
        predictions.extend(rows)
        print('FOLD', subject, fold['correct'], '/', fold['test_count'], f"{fold['accuracy']:.2%}", flush=True)
    assert len(predictions) == len(records)
    assert len({r['fingerprint'] for r in predictions}) == len(records)
    assert {r['fingerprint'] for r in predictions} == {r['fingerprint'] for r in records}
    pooled = lda.metrics(predictions, [classes.index(r['prediction']) for r in predictions], classes)
    for source in sources:
        for name, expected in source['hashes'].items():
            actual = hashlib.sha256((Path(source['path'])/name).read_bytes()).hexdigest()
            assert actual == expected, f'Input changed: {source["path"]}/{name}'
    result = dict(schema='semg_lda_loso_v1', folder=str(output), protocol=protocol, folds=folds,
                  subject_mean_accuracy=float(np.mean([f['accuracy'] for f in folds])),
                  subject_mean_macro_f1=float(np.mean([f['macro_f1'] for f in folds])),
                  pooled_metrics=pooled, excluded=excluded, sources_unchanged=True,
                  every_trial_tested_exactly_once=True, subjects_disjoint_each_fold=True)
    lda.write_json(output/'report.json', result)
    with (output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    text = [f'# {len(subjects)}人 LDA 跨受试者测试', '',
            f'带通：20–{args.bandpass_high} Hz；50 Hz 陷波 Q=30。', '',
            '每通道特征：'+lda.describe_features(config)+'。', '',
            f'使用 {", ".join(subjects)}，每折{len(subjects)-1}人训练、一人测试。', '',
            '| 留出测试人员 | 训练人员 | 训练动作数 | 正确/测试动作数 | 准确率 | 宏 F1 |',
            '|---|---|---:|---:|---:|---:|']
    for f in folds:
        text.append(f"| {f['held_out_subject']} | {', '.join(f['train_subjects'])} | {f['train_count']} | {f['correct']}/{f['test_count']} | {f['accuracy']:.2%} | {f['macro_f1']:.4f} |")
    text += ['', f"受试者等权平均准确率：{result['subject_mean_accuracy']:.2%}。",
             f"全部留出预测合计：{pooled['correct']}/{pooled['total']} = {pooled['accuracy']:.2%}；整体宏 F1：{pooled['macro_f1']:.4f}。", '',
             f'整组排除 {len(excluded_recordings)} 份录制（用户指定），详见 protocol.json。', '',
             f'固定双通道、6 段、4 特征，48 维；每折独立拟合训练人员的 StandardScaler 和收缩 LDA。共排除 {len(excluded)} 次，详细原因在 data_audit.json。', '',
             '原始输入哈希前后不变，各折模型重载通过，每个有效动作恰好作为测试一次。原始录制及已有模型均未覆盖。', '',
             '这是用户指定人员上的探索性留人评估，不是连续实时准确率。没有根据本次测试结果调整设置或追加筛选。']
    (output/'结果汇总.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('SUMMARY', result['subject_mean_accuracy'], pooled['correct'], '/', pooled['total'], flush=True)
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
