"""Paired CH0 / CH1 / dual-channel LDA LOSO on a frozen four-person cohort."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
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
from semg import lda

MODES = {'ch0': [0], 'ch1': [1], 'dual': [0, 1]}
SCHEMA = 'semg_lda_channel_ablation_v1'


def predict(model, x):
    assert model['schema'] == SCHEMA
    scores = ((x-model['scaler_mean'])/model['scaler_scale']) @ np.array(model['coef']).T + model['intercept']
    return scores.argmax(axis=1)


def verify_sources(sources):
    for source in sources:
        for name, expected in source['hashes'].items():
            assert lda.digest((Path(source['path'])/name).read_bytes()) == expected, f'Changed source: {source["path"]}/{name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'models/lda_loso_goodone_20261014_130607_0c55eb')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    baseline_hashes = {str(p):lda.digest(p.read_bytes()) for p in baseline.rglob('*') if p.is_file()}
    old = json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    first = json.loads((Path(old['folds'][0]['folder'])/'report.json').read_text(encoding='utf-8'))
    original = {r['fingerprint']:r for r in first['train']+first['test']}
    config = first['feature_config']
    assert config == dict(segments=6,metrics=['log_mean_abs_diff','log_mav','zc_count_zero','ssc_count_zero'])
    assert first['variant']=='notch50' and first['channels']==[0,1]
    with (baseline/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        baseline_predictions = {r['fingerprint']:r['prediction'] for r in csv.DictReader(f)}
    assert set(baseline_predictions)==set(original) and len(original)==1998
    verify_sources(first['sources'])
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'models'/f'lda_channels_loso_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir()
    shutil.copy2(__file__,output/Path(__file__).name)
    shutil.copy2(ROOT/'semg/lda.py',output/'lda_source_snapshot.py')
    protocol = dict(baseline=str(baseline),modes=MODES,feature_config=config,
        channel_sites={'0':'喉咙（按记录的通道定义）','1':'脸颊/下嘴角（按记录的通道定义）'},
        bandpass_hz=[20,400],notch_hz=50,notch_q=30,
        preprocessing='Original per-trial causal filter, preparation warmup, trim 150 samples at both ends, six segments. No trial amplitude normalization.',
        evaluation='Four folds: S01/S02/S04/S05 each held out once. Fixed three modes, no tuning.',
        exclusions='Keep dual-channel baseline exclusions for ALL modes, including when only the unused channel clipped.',
        estimator='Training-person-only StandardScaler + LDA(lsqr, shrinkage=auto)',
        warning='Exploratory previously examined cohort; not an independent final or real-time evaluation.',
        sklearn_version=sklearn.__version__,numpy_version=np.__version__)
    lda.write_json(output/'protocol.json',protocol)
    print('OUTPUT',output,flush=True)
    selections=[dict(path=s['path'],role='train') for s in first['sources']]
    records,excluded,sources=lda.extract(selections,'notch50',config=config,channels=[0,1],progress=lambda s:print(s,flush=True))
    records.sort(key=lambda r:(r['subject'],r['recording'],r['trial_id']))
    assert len(records)==1998 and {r['fingerprint'] for r in records}==set(original)
    identity=lambda r:(r['recording'],r['trial_id'],r['subject'],r['word'])
    assert {identity(r) for r in excluded}=={identity(r) for r in first['excluded']}
    for r in records:
        ref=original[r['fingerprint']]
        assert identity(r)==identity(ref)
        np.testing.assert_allclose(r['features'],ref['features'],rtol=1e-12,atol=1e-12)
    subjects=sorted({r['subject'] for r in records})
    assert subjects==['S01','S02','S04','S05']
    for fold in old['folds']:
        saved=json.loads((Path(fold['folder'])/'report.json').read_text(encoding='utf-8'))
        for role in ('train','test'):
            expected={r['fingerprint'] for r in records if (r['subject']==fold['held_out_subject'])==(role=='test')}
            assert {r['fingerprint'] for r in saved[role]}==expected
    x=np.array([r['features'] for r in records])
    names=lda.feature_names(config,[0,1])
    assert x.shape==(1998,48)
    columns={mode:[j for j,n in enumerate(names) if any(n.startswith(f'ch{ch}_') for ch in channels)] for mode,channels in MODES.items()}
    assert columns=={'ch0':list(range(24)),'ch1':list(range(24,48)),'dual':list(range(48))}
    np.savez_compressed(output/'features.npz',dual=x)
    lda.write_json(output/'trials.json',[{k:v for k,v in r.items() if k!='features'} for r in records])
    lda.write_json(output/'data_audit.json',dict(sources=sources,excluded=excluded,counts=dict(Counter(r['subject'] for r in records))))
    classes=lda.ordered_classes(records)
    y=np.array([classes.index(r['word']) for r in records])
    results,predictions={},[]
    audit=[]
    for mode,channels in MODES.items():
        xm=x[:,columns[mode]]
        folds,all_pred=[],np.full(len(records),-1,dtype=int)
        for subject in subjects:
            test=np.array([r['subject']==subject for r in records])
            train=~test
            scaler=StandardScaler().fit(xm[train])
            estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(xm[train]),y[train])
            pred=estimator.predict(scaler.transform(xm[test]))
            all_pred[test]=pred
            rows=[r for r,keep in zip(records,test) if keep]
            metrics=lda.metrics(rows,pred,classes)
            directory=output/mode/subject
            directory.mkdir(parents=True)
            model=dict(schema=SCHEMA,protocol=protocol,mode=mode,channels=channels,classes=classes,
                feature_columns=columns[mode],feature_names=[names[i] for i in columns[mode]],
                scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                coef=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),
                training_subjects=[s for s in subjects if s!=subject],
                training_fingerprints=[r['fingerprint'] for r,keep in zip(records,train) if keep])
            lda.write_json(directory/'model.json',model)
            loaded=json.loads((directory/'model.json').read_text(encoding='utf-8'))
            np.testing.assert_array_equal(predict(loaded,xm[test]),pred)
            np.testing.assert_allclose(loaded['scaler_mean'],xm[train].mean(0),atol=1e-12)
            np.testing.assert_allclose(loaded['scaler_scale'],xm[train].std(0),atol=1e-12)
            if mode=='dual':
                assert [classes[int(p)] for p in pred]==[baseline_predictions[r['fingerprint']] for r in rows]
            fold=dict(subject=subject,train_count=int(train.sum()),test_count=int(test.sum()),metrics=metrics)
            folds.append(fold)
            lda.write_json(directory/'report.json',fold)
            audit.append(dict(mode=mode,subject=subject,model_reload_verified=True,training_only_scaler_verified=True))
            print('FOLD',mode,subject,metrics['correct'],'/',len(rows),flush=True)
        assert np.all(all_pred>=0)
        results[mode]=dict(channels=channels,dimensions=xm.shape[1],folds=folds,
            subject_mean_accuracy=float(np.mean([f['metrics']['accuracy'] for f in folds])),
            pooled_metrics=lda.metrics(records,all_pred,classes))
        for r,p in zip(records,all_pred):
            predictions.append(dict(mode=mode,subject=r['subject'],recording=r['recording'],trial_id=r['trial_id'],
                fingerprint=r['fingerprint'],word=r['word'],prediction=classes[int(p)],correct=int(classes[int(p)]==r['word'])))
    base_correct={r['fingerprint']:r['correct'] for r in predictions if r['mode']=='dual'}
    for mode in ('ch0','ch1'):
        current=[r for r in predictions if r['mode']==mode]
        results[mode]['paired_vs_dual']=dict(
            single_only_correct=sum(r['correct'] and not base_correct[r['fingerprint']] for r in current),
            dual_only_correct=sum(not r['correct'] and base_correct[r['fingerprint']] for r in current))
    verify_sources(sources)
    for name,expected in baseline_hashes.items():
        assert lda.digest(Path(name).read_bytes())==expected
    checks=dict(baseline_features_reproduced=True,dual_predictions_reproduced=True,
        identical_trial_sets=True,original_sources_unchanged=True,baseline_files_unchanged=True,
        each_trial_tested_once_per_method=True,subjects_disjoint=True)
    lda.write_json(output/'report.json',dict(protocol=protocol,results=results,checks=checks))
    lda.write_json(output/'saved_models_audit.json',dict(status='passed',folds=audit,checks=checks))
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    text=['# LDA 四人跨人：单通道与双通道对照','',
        '| 留出人员 | 喉咙 CH0 | 脸部 CH1 | 双通道 |','|---|---:|---:|---:|']
    for i,subject in enumerate(subjects):
        text.append('| '+subject+' | '+' | '.join(f"{results[m]['folds'][i]['metrics']['accuracy']:.2%}" for m in MODES)+' |')
    text.append('| 四人等权平均 | '+' | '.join(f"{results[m]['subject_mean_accuracy']:.2%}" for m in MODES)+' |')
    for mode,result in results.items():
        metric=result['pooled_metrics']
        text.append(f"\n{mode}: {metric['correct']}/{metric['total']}，合并宏 F1 {metric['macro_f1']:.4f}。")
    text += ['', '三组均使用同一1998次动作、39份录制及原双通道规则排除的2次动作，不因切换单通道新增或删除样本。',
        '20–400 Hz + 50 Hz陷波，准备期预热，动作两端各裁150 ms；每路六段四特征。单通道24维，双通道48维。无逐动作幅度归一化。',
        '每折只用三位训练人员拟合StandardScaler与收缩LDA；第四位全部测试。模型重载和训练侧尺度核验通过；双通道全部逐次预测与原基线一致。',
        '模型为独立实验schema，未接入GUI。未运行CNN、修改采集或烧录固件。',
        '这是已查看数据上的探索性四折对照。单通道更好不等于另一通道损坏；不能按每个测试人的最佳通道拼接一个部署分数。']
    (output/'结果汇总.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({m:r['subject_mean_accuracy'] for m,r in results.items()}),flush=True)
    print('OUTPUT',output,flush=True)


if __name__=='__main__':
    main()
