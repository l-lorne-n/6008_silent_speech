"""Offline, paired LOSO comparison of whole-trial waveform normalization.

Uses a frozen four-person LDA baseline and verifies original bytes, trial
fingerprints and features. Custom checkpoints deliberately cannot load in GUI.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from semg import lda

MODES = ('baseline', 'per_channel', 'shared_scale')
EPS = 1e-6
LABELS = {'baseline': '原方案', 'per_channel': '每动作两路分别标准化',
          'shared_scale': '每动作两路共同缩放'}


def normalize_action(action, mode):
    """Input T x 2, filtered and trimmed, no padding or other trials."""
    if mode == 'baseline':
        return action.copy(), dict(mean=[0., 0.], scale=[1., 1.], floor_used=False)
    centered = action - action.mean(axis=0, keepdims=True)
    if mode == 'per_channel':
        scale = np.sqrt(np.mean(centered**2, axis=0))
    elif mode == 'shared_scale':
        scale = np.repeat(np.sqrt(np.mean(centered**2)), 2)
    else:
        raise ValueError(mode)
    floor_used = bool(np.any(scale < EPS))
    scale = np.maximum(scale, EPS)
    result = centered / scale
    assert np.isfinite(result).all()
    return result, dict(mean=action.mean(axis=0).tolist(), scale=scale.tolist(), floor_used=floor_used)


def features(action, config):
    return np.concatenate([np.array([lda.segment_metric(part, metric)
        for metric in config['metrics'] for part in np.array_split(action[:, ch], config['segments'])])
        for ch in range(2)])


def predict_checkpoint(model, x):
    assert model['schema'] == 'semg_lda_trial_normalization_experiment_v1'
    scores = ((x-np.array(model['scaler_mean']))/np.array(model['scaler_scale'])) @ np.array(model['coef']).T
    return np.argmax(scores + np.array(model['intercept']), axis=1)


def self_check():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(600, 2)) * [2., 5.] + [4., -3.]
    z, _ = normalize_action(a, 'per_channel')
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(z.std(0), 1)
    z2, _ = normalize_action(a*[3., .4]+[10., -4.], 'per_channel')
    np.testing.assert_allclose(z, z2, atol=1e-13)
    shared, _ = normalize_action(a, 'shared_scale')
    np.testing.assert_allclose(shared.std(0)[0]/shared.std(0)[1], a.std(0)[0]/a.std(0)[1])
    np.testing.assert_allclose(shared, normalize_action(a*7+[2., 8.], 'shared_scale')[0], atol=1e-13)
    for mode in MODES:
        assert np.isfinite(normalize_action(np.zeros((600, 2)), mode)[0]).all()
    print('Normalization invariance, channel-ratio and zero-signal checks passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
        default=ROOT/'models/lda_loso_goodone_20261014_130607_0c55eb')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.check_only:
        return
    baseline = args.baseline.resolve()
    baseline_hashes = {str(p): lda.digest(p.read_bytes()) for p in baseline.rglob('*') if p.is_file()}
    old = json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    first = json.loads((Path(old['folds'][0]['folder'])/'report.json').read_text(encoding='utf-8'))
    records = sorted(first['train']+first['test'], key=lambda r:(r['subject'], r['recording'], r['trial_id']))
    subjects = sorted({r['subject'] for r in records})
    assert subjects == ['S01', 'S02', 'S04', 'S05'] and len(records) == 1998
    assert len({r['fingerprint'] for r in records}) == len(records)
    sources, config = first['sources'], first['feature_config']
    assert first['variant'] == 'notch50'
    assert config == dict(segments=6, metrics=['log_mean_abs_diff', 'log_mav', 'zc_count_zero', 'ssc_count_zero'])
    old_predictions = {}
    with (baseline/'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        old_predictions = {r['fingerprint']: r['prediction'] for r in csv.DictReader(f)}
    assert set(old_predictions) == {r['fingerprint'] for r in records}
    for fold in old['folds']:
        saved = json.loads((Path(fold['folder'])/'report.json').read_text(encoding='utf-8'))
        assert {r['fingerprint'] for r in saved['test']} == {r['fingerprint'] for r in records if r['subject']==fold['held_out_subject']}
        assert {r['fingerprint'] for r in saved['train']} == {r['fingerprint'] for r in records if r['subject']!=fold['held_out_subject']}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'models'/f'lda_normalization_loso_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir()
    print('OUTPUT', output, flush=True)
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(ROOT/'semg/lda.py', output/'lda_source_snapshot.py')
    protocol = dict(baseline=str(baseline), subjects=subjects, feature_config=config,
        modes=list(MODES), bandpass_hz=[20,400], notch_hz=50, notch_q=30,
        preprocessing='Per-trial causal filtering with preparation warmup; trim 150 samples at both action ends; normalize entire remaining action before 6 segments; no padding.',
        normalization=dict(baseline='No additional waveform normalization',
            per_channel='Subtract each channel action mean; divide by its population std (ddof=0).',
            shared_scale='Subtract each channel action mean; divide both channels by sqrt(mean of squared centered samples over time and both channels)).',
            scale_floor=EPS, unit='Filtered ADC counts',
            scope='Each trial uses only its own waveform, without labels, other trials or subject-level test statistics.'),
        estimator='Training-person-only StandardScaler + LDA(solver=lsqr, shrinkage=auto)',
        note='Exploratory four-person LOSO. Three predeclared methods, no tuning. ADC clipping exclusions frozen to baseline. Full prompted windows required; no activity detection evaluation.',
        sklearn_version=sklearn.__version__, numpy_version=np.__version__)
    lda.write_json(output/'protocol.json', protocol)
    xs = {mode:np.empty((len(records),48)) for mode in MODES}
    visits = np.zeros(len(records), dtype=int)
    stats = []
    sos = lda.filter_sos('notch50')
    for index, source in enumerate(sources):
        folder = Path(source['path'])
        for name, expected in source['hashes'].items():
            assert lda.digest((folder/name).read_bytes()) == expected, f'Source changed: {folder/name}'
        content = (folder/'samples.csv').read_bytes()
        reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
        data = np.array([(int(r['device_sample_idx']),int(r['raw_ch0']),int(r['raw_ch1'])) for r in reader],dtype=np.int64)
        assert np.all(np.diff(data[:,0])==1)
        for i,r in enumerate(records):
            if Path(r['path']) != folder:
                continue
            visits[i] += 1
            raw = data[r['prepare']-data[0,0]:r['rest']-data[0,0],1:]
            offset = r['action']-r['prepare']
            assert not np.any((raw<=1)|(raw>=65534))
            assert lda.digest(raw.astype('<u2').tobytes()+str(offset).encode('ascii')) == r['fingerprint']
            assert [lda.digest(raw[:,ch].astype('<u2').tobytes()+str(offset).encode('ascii')) for ch in range(2)] == r['signal_fingerprints']
            filtered = []
            for ch in range(2):
                x = raw[:,ch].astype(float)-32768
                y,_ = sosfilt(sos,x,zi=sosfilt_zi(sos)*x[0])
                filtered.append(y[offset+150:-150])
            action = np.column_stack(filtered)
            assert len(action)>=400 and len(action)>=config['segments']*50
            for mode in MODES:
                z, scale = normalize_action(action, mode)
                xs[mode][i] = features(z,config)
                stats.append(dict(fingerprint=r['fingerprint'], subject=r['subject'], mode=mode,
                    length=len(action), **scale))
            np.testing.assert_allclose(xs['baseline'][i],r['features'],rtol=1e-12,atol=1e-12)
        print(f'EXTRACT {index+1}/{len(sources)} {folder.name}',flush=True)
    assert np.all(visits == 1), 'Every frozen trial must be extracted exactly once'
    assert all(np.isfinite(x).all() for x in xs.values())
    np.savez_compressed(output/'features.npz',**xs)
    lda.write_json(output/'trials.json',[{k:v for k,v in r.items() if k!='features'} for r in records])
    lda.write_json(output/'normalization_stats.json',stats)
    lda.write_json(output/'data_audit.json',dict(sources=sources,excluded=first['excluded'],
        valid_by_subject=dict(Counter(r['subject'] for r in records)),
        normalization_floor_uses=dict(Counter(s['mode'] for s in stats if s['floor_used']))))
    classes = lda.ordered_classes(records)
    y = np.array([classes.index(r['word']) for r in records])
    results, predictions = {}, []
    for mode in MODES:
        folds, all_pred = [], np.full(len(records),-1,dtype=int)
        for held_out in subjects:
            test = np.array([r['subject']==held_out for r in records])
            train = ~test
            scaler = StandardScaler().fit(xs[mode][train])
            estimator = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(xs[mode][train]),y[train])
            pred = estimator.predict(scaler.transform(xs[mode][test]))
            all_pred[test] = pred
            rows = [r for r,keep in zip(records,test) if keep]
            metrics = lda.metrics(rows,pred,classes)
            model = dict(schema='semg_lda_trial_normalization_experiment_v1', mode=mode,
                protocol=protocol, classes=classes, feature_names=lda.feature_names(config,[0,1]),
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                training_subjects=[s for s in subjects if s!=held_out],
                training_fingerprints=[r['fingerprint'] for r,keep in zip(records,train) if keep])
            directory = output/mode/held_out
            directory.mkdir(parents=True)
            lda.write_json(directory/'model.json',model)
            reloaded = json.loads((directory/'model.json').read_text(encoding='utf-8'))
            np.testing.assert_array_equal(predict_checkpoint(reloaded,xs[mode][test]),pred)
            if mode=='baseline':
                assert [classes[int(p)] for p in pred] == [old_predictions[r['fingerprint']] for r in rows]
            fold = dict(subject=held_out,train_count=int(train.sum()),test_count=int(test.sum()),
                metrics=metrics, model_reload_verified=True)
            folds.append(fold)
            lda.write_json(directory/'report.json',fold)
            print('FOLD',mode,held_out,metrics['correct'],'/',len(rows),flush=True)
        assert np.all(all_pred>=0)
        results[mode] = dict(folds=folds,subject_mean_accuracy=float(np.mean([f['metrics']['accuracy'] for f in folds])),
            pooled_metrics=lda.metrics(records,all_pred,classes))
        for r,p in zip(records,all_pred):
            predictions.append(dict(mode=mode,subject=r['subject'],recording=r['recording'],trial_id=r['trial_id'],
                fingerprint=r['fingerprint'],word=r['word'],prediction=classes[int(p)],correct=int(classes[int(p)]==r['word'])))
    for source in sources:
        for name, expected in source['hashes'].items():
            assert lda.digest((Path(source['path'])/name).read_bytes())==expected
    for file,expected in baseline_hashes.items():
        assert lda.digest(Path(file).read_bytes())==expected
    baseline_correct = {r['fingerprint']:r['correct'] for r in predictions if r['mode']=='baseline'}
    for mode in MODES[1:]:
        current = [r for r in predictions if r['mode']==mode]
        results[mode]['paired_vs_baseline'] = dict(
            new_only_correct=sum(r['correct'] and not baseline_correct[r['fingerprint']] for r in current),
            baseline_only_correct=sum(not r['correct'] and baseline_correct[r['fingerprint']] for r in current))
    report = dict(protocol=protocol,results=results,checks=dict(original_sources_unchanged=True,
        baseline_files_unchanged=True,baseline_features_reproduced=True,baseline_predictions_reproduced=True,
        identical_trial_sets=True,subject_disjoint_folds=True,each_trial_tested_once_per_method=True))
    lda.write_json(output/'report.json',report)
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    summary=['# LDA 四人跨人：动作归一化对照','',
        '| 留出人员 | 原方案 | 两路分别标准化 | 两路共同缩放 |','|---|---:|---:|---:|']
    for i,s in enumerate(subjects):
        summary.append('| '+s+' | '+' | '.join(f"{results[m]['folds'][i]['metrics']['accuracy']:.2%}" for m in MODES)+' |')
    summary.append('| 四人等权平均 | '+' | '.join(f"{results[m]['subject_mean_accuracy']:.2%}" for m in MODES)+' |')
    for mode in MODES:
        metric=results[mode]['pooled_metrics']
        summary.append(f"\n{LABELS[mode]}：{metric['correct']}/{metric['total']}；合并宏 F1 {metric['macro_f1']:.4f}。")
    summary += ['', '原始1998次有效动作、39份录制及2次削顶排除不变。每折三人训练，一人测试。',
        '归一化发生在因果滤波和首尾各150 ms裁剪之后、六段特征提取之前；分别减去每通道整动作均值，分别或共同除以尺度。无补零，不按小段归一化。',
        '仅当前动作参与自身归一化；不使用目标人标签或其他目标动作的统计量。后续StandardScaler和LDA仅拟合训练人员。',
        '基线特征和全部逐次预测复现；源文件及基线文件哈希不变；各折模型重载预测一致。',
        '实验模型使用独立schema，不能直接加载到现有GUI。未修改GUI、固件或原模型。',
        '这是已查看四人数据上的探索性对照，不是完整五人最终测试，也不代表实时自动截段表现。']
    (output/'结果汇总.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print(json.dumps({m:results[m]['subject_mean_accuracy'] for m in MODES}),flush=True)
    print('OUTPUT',output,flush=True)


if __name__=='__main__':
    main()
