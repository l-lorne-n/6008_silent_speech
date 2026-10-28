"""Frozen paired LDA learning curves: repetitions and source-person diversity.

Reuses previously independently verified 48-dimensional original features.
No architecture/parameter search, CNN, or target-person calibration.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from semg import lda

SEEDS=list(range(20261014,20261034))
VOLUMES=(10,25,50)
BASELINE=ROOT/'models/lda_loso_goodone_20261014_130607_0c55eb'
CACHE=ROOT/'models/lda_channels_loso_20261014_151634_22ec18'


def verify_sources(sources):
    for source in sources:
        for name,expected in source['hashes'].items():
            assert lda.digest((Path(source['path'])/name).read_bytes())==expected


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    inputs=[p for p in BASELINE.rglob('*') if p.is_file()]
    inputs += [CACHE/name for name in ('features.npz','trials.json','data_audit.json','report.json','independent_prediction_audit.json')]
    input_hashes={str(p):lda.digest(p.read_bytes()) for p in inputs}
    old=json.loads((BASELINE/'report.json').read_text(encoding='utf-8'))
    first=json.loads((Path(old['folds'][0]['folder'])/'report.json').read_text(encoding='utf-8'))
    trials=json.loads((CACHE/'trials.json').read_text(encoding='utf-8'))
    audit=json.loads((CACHE/'data_audit.json').read_text(encoding='utf-8'))
    x=np.load(CACHE/'features.npz')['dual']
    original={r['fingerprint']:r for r in first['train']+first['test']}
    assert x.shape==(1998,48) and len(trials)==len(original)==1998
    for i,r in enumerate(trials):
        ref=original[r['fingerprint']]
        assert (r['subject'],r['word'],r['recording'],r['block'],r['trial_id'])==tuple(ref[k] for k in ('subject','word','recording','block','trial_id'))
        np.testing.assert_allclose(x[i],ref['features'],rtol=1e-12,atol=1e-12)
    verify_sources(audit['sources'])
    with (BASELINE/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        old_pred={r['fingerprint']:r['prediction'] for r in csv.DictReader(f)}
    subjects=sorted({r['subject'] for r in trials})
    assert subjects==['S01','S02','S04','S05']
    classes=lda.ordered_classes(trials)
    y=np.array([classes.index(r['word']) for r in trials])
    groups=defaultdict(list)
    for i,r in enumerate(trials):
        groups[(r['subject'],r['recording'],r['block'])].append(i)
    keys={s:sorted(k for k in groups if k[0]==s) for s in subjects}
    complete={s:[k for k in keys[s] if len(groups[k])==10 and set(y[groups[k]])==set(range(10))] for s in subjects}
    assert [len(keys[s]) for s in subjects]==[50]*4
    assert [len(complete[s]) for s in subjects]==[50,50,50,48]
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=ROOT/'models'/f'lda_learning_curves_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir()
    print('OUTPUT',output,flush=True)
    shutil.copy2(__file__,output/Path(__file__).name)
    shutil.copy2(ROOT/'semg/lda.py',output/'lda_source_snapshot.py')
    protocol=dict(subjects=subjects,seeds=SEEDS,classes=classes,channels=[0,1],
        baseline=str(BASELINE),feature_cache=str(CACHE),feature_config=first['feature_config'],
        filter='20-400 Hz bandpass, 50 Hz notch Q30; causal per-trial filtering, 150 sample end trims',
        model='Training-only StandardScaler + LDA(lsqr, shrinkage=auto); no trial normalization',
        volume=dict(planned_rounds_per_person=list(VOLUMES),source_people=3,
            sampling='Uniform random whole-round permutation per source person and seed, nested 10/25/50 prefixes; original invalid actions stay excluded.',
            note='Counts are planned repetitions per word. S05 has two 9-action rounds; 50 rounds gives 498 valid actions.'),
        diversity=dict(total_training_actions=300,total_actions_per_word=30,source_people=[1,2,3],
            rounds_per_person={'1':30,'2':15,'3':10},combinations='All subsets of the three available source people, equally weighted within each held-out person.',
            sampling='Whole complete ten-word rounds only, nested per-person prefixes across number-of-people settings.',
            incomplete_training_rounds_omitted=[dict(subject=k[0],recording=k[1],block=k[2],valid_actions=len(groups[k])) for k in groups if k not in complete[k[0]]],
            note='18 valid S05 actions in two incomplete rounds are unavailable ONLY for diversity training. Every original valid test action remains.'),
        test='All original valid actions of the held-out person, identical across all settings and seeds.',
        aggregation='Average source combinations within each held-out person, then equal average of four test people per seed. Report mean and sample SD across 20 seeds.',
        interpretation='SD describes subsampling variability on the same four people; not uncertainty for the population or independent experimental replications. Exploratory data already examined.',
        sklearn_version=sklearn.__version__,numpy_version=np.__version__)
    lda.write_json(output/'protocol.json',protocol)
    lda.write_json(output/'input_hashes.json',input_hashes)
    lda.write_json(output/'trials.json',trials)
    lda.write_json(output/'data_audit.json',audit)
    np.savez_compressed(output/'features.npz',x=x,y=y)
    runs,manifests,models,predictions=[],[],{},{}
    fitted_cache={}

    def fit(experiment,level,held_out,source_people,seed,selected_groups):
        train=np.array(sorted(i for k in selected_groups for i in groups[k]),dtype=np.int64)
        test=np.array([i for i,r in enumerate(trials) if r['subject']==held_out],dtype=np.int64)
        assert len(set(train))==len(train) and not set(train)&set(test)
        assert {trials[i]['subject'] for i in train}==set(source_people) and held_out not in source_people
        assert set(y[train])==set(range(10))
        if experiment=='diversity':
            assert len(train)==300 and Counter(y[train])=={k:30 for k in range(10)}
        cache_key=(held_out,tuple(train))
        if cache_key not in fitted_cache:
            scaler=StandardScaler().fit(x[train])
            estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(x[train]),y[train])
            pred=estimator.predict(scaler.transform(x[test]))
            np.testing.assert_allclose(scaler.mean_,x[train].mean(0),atol=1e-12)
            fitted_cache[cache_key]=(scaler,estimator,pred)
        scaler,estimator,pred=fitted_cache[cache_key]
        if experiment=='volume' and level==50:
            assert [classes[int(p)] for p in pred]==[old_pred[trials[i]['fingerprint']] for i in test]
        run_id=f'r{len(runs):04d}'
        metric=lda.metrics([trials[i] for i in test],pred,classes)
        runs.append(dict(run_id=run_id,experiment=experiment,level=level,test_subject=held_out,
            train_subjects='+'.join(source_people),seed=seed,train_count=len(train),test_count=len(test),
            correct=metric['correct'],accuracy=metric['accuracy'],macro_f1=metric['macro_f1']))
        manifests.append(dict(run_id=run_id,train_indices=train.tolist(),test_indices=test.tolist(),
            selected_groups=[list(k) for k in selected_groups]))
        models[run_id+'_mean']=scaler.mean_
        models[run_id+'_scale']=scaler.scale_
        models[run_id+'_coef']=estimator.coef_
        models[run_id+'_intercept']=estimator.intercept_
        predictions[run_id]=pred.astype(np.int8)

    for seed in SEEDS:
        # Same per-person nested order across test people and settings for pairing.
        volume_order={s:[keys[s][i] for i in np.random.default_rng(np.random.SeedSequence([seed,j,1])).permutation(len(keys[s]))] for j,s in enumerate(subjects)}
        diversity_order={s:[complete[s][i] for i in np.random.default_rng(np.random.SeedSequence([seed,j,2])).permutation(len(complete[s]))] for j,s in enumerate(subjects)}
        for held_out in subjects:
            source=[s for s in subjects if s!=held_out]
            for rounds in VOLUMES:
                fit('volume',rounds,held_out,source,seed,[k for s in source for k in volume_order[s][:rounds]])
            for count in (1,2,3):
                for combo in itertools.combinations(source,count):
                    fit('diversity',count,held_out,list(combo),seed,[k for s in combo for k in diversity_order[s][:30//count]])
        print('SEED',seed,'RUNS',len(runs),flush=True)
    assert len(runs)==800
    write_csv(output/'runs.csv',runs)
    lda.write_json(output/'split_manifest.json',manifests)
    np.savez_compressed(output/'models.npz',**models)
    np.savez_compressed(output/'predictions.npz',**predictions)
    # Verify all serialized coefficients and predictions against the saved splits.
    saved_models=np.load(output/'models.npz')
    saved_pred=np.load(output/'predictions.npz')
    for run,manifest in zip(runs,manifests):
        rid=run['run_id'];test=manifest['test_indices'];train=manifest['train_indices']
        scores=((x[test]-saved_models[rid+'_mean'])/saved_models[rid+'_scale'])@saved_models[rid+'_coef'].T+saved_models[rid+'_intercept']
        np.testing.assert_array_equal(scores.argmax(1),saved_pred[rid])
        np.testing.assert_allclose(x[train].std(0),saved_models[rid+'_scale'],atol=1e-12)
        assert int(np.sum(saved_pred[rid]==y[test]))==run['correct']
    seed_rows,summary=[],[]
    for experiment,levels in [('volume',VOLUMES),('diversity',(1,2,3))]:
        for level in levels:
            selected=[r for r in runs if r['experiment']==experiment and r['level']==level]
            person_stats={}
            for person in subjects:
                vals=[float(np.mean([r['accuracy'] for r in selected if r['test_subject']==person and r['seed']==seed])) for seed in SEEDS]
                combos={c:float(np.mean([r['accuracy'] for r in selected if r['test_subject']==person and r['train_subjects']==c])) for c in sorted({r['train_subjects'] for r in selected if r['test_subject']==person})}
                person_stats[person]=dict(mean=float(np.mean(vals)),sampling_sd=float(np.std(vals,ddof=1)),combination_means=combos)
            values=[]
            for seed in SEEDS:
                value=float(np.mean([np.mean([r['accuracy'] for r in selected if r['seed']==seed and r['test_subject']==person]) for person in subjects]))
                values.append(value)
                seed_rows.append(dict(experiment=experiment,level=level,seed=seed,subject_mean_accuracy=value))
            summary.append(dict(experiment=experiment,level=level,mean=float(np.mean(values)),sampling_sd=float(np.std(values,ddof=1)),
                seed_min=min(values),seed_max=max(values),by_subject=person_stats,
                training_count_min=min(r['train_count'] for r in selected),training_count_max=max(r['train_count'] for r in selected)))
    contrasts=[]
    for experiment,pairs in [('volume',[(10,25),(25,50),(10,50)]),('diversity',[(1,2),(2,3),(1,3)])]:
        for a,b in pairs:
            differences=[next(r['subject_mean_accuracy'] for r in seed_rows if r['experiment']==experiment and r['level']==b and r['seed']==seed)-next(r['subject_mean_accuracy'] for r in seed_rows if r['experiment']==experiment and r['level']==a and r['seed']==seed) for seed in SEEDS]
            contrasts.append(dict(experiment=experiment,from_level=a,to_level=b,mean_difference=float(np.mean(differences)),sampling_sd=float(np.std(differences,ddof=1))))
    verify_sources(audit['sources'])
    for name,expected in input_hashes.items():assert lda.digest(Path(name).read_bytes())==expected
    checks=dict(inputs_unchanged=True,full_data_predictions_match_baseline=True,all_800_saved_models_verified=True,
        each_split_people_disjoint=True,whole_round_sampling=True,diversity_always_300_balanced_actions=True,
        fixed_test_sets=True,unique_fits=len(fitted_cache))
    lda.write_json(output/'report.json',dict(protocol=protocol,summary=summary,contrasts=contrasts,checks=checks))
    write_csv(output/'seed_averages.csv',seed_rows)
    text=['# LDA 数据量与训练人数学习曲线','',
        '固定双通道、原48维特征和LDA设置；20个预定种子；每个设置轮流测试S01/S02/S04/S05，测试集始终为原1998次动作。','',
        '均值对测试人员等权；人数实验先对训练人员组合等权。±为20次抽样的标准差，不是新人的泛化置信区间。','',
        '## 实验一：固定三个训练人，增加每人每词计划次数','',
        '| 每人每词计划次数 | 实际训练动作数 | 平均准确率 ± 抽样标准差 | S01 | S02 | S04 | S05 |',
        '|---|---|---|---|---|---|---|']
    for s in summary:
        if s['experiment']=='volume':text.append(f"| {s['level']} | {s['training_count_min']}–{s['training_count_max']} | {s['mean']:.2%} ± {100*s['sampling_sd']:.2f}个百分点 | "+' | '.join(f"{s['by_subject'][p]['mean']:.2%}" for p in subjects)+' |')
    text+=['','S05两次削顶动作沿用原排除，因此10/25/50为计划轮次；最后一档复现原1998次测试结果。50轮已使用全部源数据，抽样波动为零，不代表对新人没有不确定性。','',
        '## 实验二：固定300次训练动作，改变训练人数','',
        '| 训练人数 | 每人每词次数 | 每个测试人的训练组合数 | 平均准确率 ± 抽样标准差 | S01 | S02 | S04 | S05 |',
        '|---|---|---|---|---|---|---|---|']
    for s in summary:
        if s['experiment']=='diversity':text.append(f"| {s['level']} | {30//s['level']} | {3 if s['level']<3 else 1} | {s['mean']:.2%} ± {100*s['sampling_sd']:.2f}个百分点 | "+' | '.join(f"{s['by_subject'][p]['mean']:.2%}" for p in subjects)+' |')
    text+=['','人数实验只从完整十词轮次选训练数据。S05两个不完整轮次中的18次有效动作只在这组实验的训练候选池中不用，测试时仍保留；没有按识别结果筛选。',
        '每个设置包含所有可用训练人员组合，禁止从中挑最高组合代表整体；各组合平均分见report.json。',
        '模型及预测保存于models.npz/predictions.npz，run ID关联runs.csv和split_manifest.json；模型均重新拟合，重复的完全相同训练集复用拟合结果。',
        '原录制、旧报告及缓存哈希前后不变；800份模型参数重载预测核验通过。未运行CNN，未修改GUI或采集。',
        '这些学习曲线只覆盖现有四人和最多50次/词，不能外推100次/词或8人的具体准确率。']
    (output/'结果汇总.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps([dict(experiment=s['experiment'],level=s['level'],mean=s['mean'],sd=s['sampling_sd']) for s in summary]),flush=True)
    print('OUTPUT',output,flush=True)


if __name__=='__main__':main()
