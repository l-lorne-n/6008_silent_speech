"""Recording-to-LDA pipeline. Trial-isolated features and auditable holdouts."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

from . import __version__
from .session import WORDS
from .vocabulary import validate_words

FEATURES = [f"log_rms_quarter_{i}" for i in range(1, 5)] + [f"log_mean_abs_diff_quarter_{i}" for i in range(1, 5)]
SCHEMA = "semg_lda_v2"
VARIANTS = {"notch50": [50], "notch50_100_200": [50, 100, 200]}


def recording_words(meta):
    # Pre-vocabulary recordings without this field used the original four words.
    return validate_words(meta.get("words", list(WORDS)))


def ordered_classes(rows):
    present = {r['word'] for r in rows}
    return [w for w in WORDS if w in present] + sorted(present - set(WORDS))


def lines(content):
    return [json.loads(s) for s in content.decode("utf-8-sig").splitlines() if s.strip()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scan_recordings(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("数据目录不存在")
    files = [root / "metadata.json"] if (root / "metadata.json").exists() else sorted(root.rglob("metadata.json"))
    catalog, errors = [], []
    for file in files:
        try:
            meta = json.loads(file.read_text(encoding="utf-8-sig"))
            words = recording_words(meta)
            reason = ""
            if meta.get("mode") != "binary" or meta.get("sampling_rate_hz") != 1000:
                reason = "仅支持真实 1 kHz 完整采样"
            elif meta.get("status") not in ("completed", "stopped", "error"):
                reason = "录制尚未正常结束"
            trials = lines((file.parent / "trials.jsonl").read_bytes())
            trials = [t for t in trials if t.get("status") == "completed" and t.get("word") in words]
            catalog.append(dict(path=str(file.parent.resolve()), subject=str(meta.get("subject_id", "未知")),
                                session=meta.get("session_id"), condition=f"{meta.get('condition', '?')}/{meta.get('articulation_mode', '?')}",
                                site=meta.get("electrode_sites", ""), notes=meta.get("notes", ""),
                                status=meta.get("status"), disabled_reason=reason, trials=trials))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{file.parent}: {exc}")
    return catalog, errors


def filter_sos(variant):
    if variant not in VARIANTS:
        raise ValueError("未知滤波方案")
    sos = butter(4, [20, 400], fs=1000, btype="bandpass", output="sos")
    for hz in VARIANTS[variant]:
        b, a = iirnotch(hz, Q=hz/(50/30), fs=1000)
        sos = np.vstack((sos, tf2sos(b, a)))
    return sos


def feature_vector(raw, action_offset, sos):
    """Warm up on preparation; stop at rest onset. No future/trial sharing."""
    x = np.asarray(raw, dtype=float)-32768
    y, _ = sosfilt(sos, x, zi=sosfilt_zi(sos)*x[0])
    action = y[action_offset+150:len(y)-150]
    if len(action) < 400:
        raise ValueError("动作区间过短：裁去两端各150 ms后至少需要400 ms")
    quarters = np.array_split(action, 4)
    return np.array([np.log(np.sqrt(np.mean(q*q))+1e-8) for q in quarters] +
                    [np.log(np.mean(np.abs(np.diff(q)))+1e-8) for q in quarters])


def extract_recording(selection, variant):
    folder = Path(selection["path"])
    snapshots = {name: (folder/name).read_bytes() for name in ("metadata.json", "samples.csv", "events.jsonl", "trials.jsonl")}
    hashes = {name: digest(data) for name, data in snapshots.items()}
    meta = json.loads(snapshots["metadata.json"])
    words = recording_words(meta)
    if meta.get("mode") != "binary" or meta.get("sampling_rate_hz") != 1000 or meta.get("status") not in ("completed", "stopped", "error"):
        raise ValueError(f"{folder.name} 不是已结束的真实 1 kHz 完整采样记录")
    reader = csv.DictReader(io.StringIO(snapshots["samples.csv"].decode("utf-8-sig")))
    data = np.array([(int(r["device_sample_idx"]), int(r["raw_ch0"])) for r in reader], dtype=np.int64)
    if not len(data) or data.ndim != 2 or np.any(np.diff(data[:, 0]) != 1):
        raise ValueError(f"{folder.name} 样本序号不连续或为空，不能压缩时间后训练")
    if np.any((data[:, 1] < 0) | (data[:, 1] > 65535)):
        raise ValueError("ADC 数据超出16位范围")
    events = lines(snapshots["events.jsonl"])
    acks = {}
    for e in events:
        if e.get("kind") == "marker_ack":
            if e["event_id"] in acks:
                raise ValueError("重复的阶段应答，不能确定试次边界")
            acks[e["event_id"]] = e
    trials = lines(snapshots["trials.jsonl"])
    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("试次编号重复")
    selected = set(selection.get("trial_ids", trial_ids))
    if selected-set(trial_ids):
        raise ValueError("选择的试次已不存在，请重新扫描")
    sos = filter_sos(variant)
    accepted, excluded = [], []
    for trial in trials:
        tid = trial["trial_id"]
        if tid not in selected:
            continue
        base = dict(path=str(folder.resolve()), recording=folder.name, subject=str(meta.get("subject_id", "未知")),
                    trial_id=tid, word=trial.get("word"), block=trial.get("block"),
                    role=selection.get("role", "train"), source_hash=hashes["samples.csv"])
        try:
            if trial.get("status") != "completed" or trial.get("word") not in words:
                raise ValueError("未完成、已作废或词语不在本份录制词表中")
            if not isinstance(trial.get("block"), int):
                raise ValueError("缺少轮次编号")
            phases = [e for e in events if e.get("kind") == "phase" and e.get("trial_id") == tid and e.get("phase") in ("prepare", "action", "rest")]
            if len(phases) != 3 or {e['phase'] for e in phases} != {"prepare", "action", "rest"}:
                raise ValueError("准备、动作、休息阶段不完整或重复")
            marks = {e["phase"]: acks[e["event_id"]] for e in phases}
            prepare, action, rest = [int(marks[k]["device_sample_idx"]) for k in ("prepare", "action", "rest")]
            if not data[0, 0] <= prepare < action < rest <= data[-1, 0]+1 or action-prepare < 1000:
                raise ValueError("边界越界或准备期不足1秒")
            raw = data[prepare-data[0, 0]:rest-data[0, 0], 1]
            if np.any((raw <= 1) | (raw >= 65534)):
                raise ValueError("准备/动作期出现 ADC 削顶")
            features = feature_vector(raw, action-prepare, sos)
            # Stable even if a recording folder or CSV is copied/renamed.
            fingerprint = digest(raw.astype('<u2').tobytes() + str(action-prepare).encode('ascii'))
            accepted.append(dict(base, fingerprint=fingerprint, prepare=prepare, action=action, rest=rest,
                                 boundary_rtt_ms=max(float(marks[k].get('rtt_ms', 0)) for k in marks),
                                 features=features.tolist()))
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append(dict(base, reason=str(exc)))
    # Detect writes during read/analysis; no input files are modified by this module.
    ordered = sorted(accepted, key=lambda r:r['prepare'])
    if any(a['rest'] > b['prepare'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError("不同试次的采样区间重叠，不能作为独立样本划分")
    for name, expected in hashes.items():
        if digest((folder/name).read_bytes()) != expected:
            raise ValueError("录制文件正在变化，请结束采集并重新扫描")
    return accepted, excluded, dict(path=str(folder.resolve()), hashes=hashes,
                                    subject=meta.get("subject_id"), condition=meta.get("condition"),
                                    articulation_mode=meta.get("articulation_mode"), site=meta.get("electrode_sites"),
                                    notes=meta.get("notes"), status=meta.get("status"))


def extract(selections, variant, progress=lambda text: None):
    records, excluded, sources = [], [], []
    seen = set()
    for i, selection in enumerate(selections):
        progress(f"提取录制 {i+1}/{len(selections)}：{Path(selection['path']).name}")
        rows, bad, source = extract_recording(selection, variant)
        for row in rows:
            if row['fingerprint'] in seen:
                raise ValueError("选择中存在相同的原始试次（可能重复导入或复制目录），请去掉重复数据")
            seen.add(row['fingerprint'])
        records.extend(rows); excluded.extend(bad); sources.append(source)
    if not records:
        raise ValueError("没有可用的完整试次。" + (excluded[0]['reason'] if excluded else "请先勾选数据。"))
    return records, excluded, sources


def split_records(records, strategy, train_percent, seed):
    if strategy == "manual":
        train = [i for i, r in enumerate(records) if r['role'] == 'train']
        test = [i for i, r in enumerate(records) if r['role'] == 'test']
        if len(train)+len(test) != len(records):
            raise ValueError("手动划分必须指定训练/测试用途")
    else:
        if not 50 <= train_percent <= 90:
            raise ValueError("训练占比应在50%到90%之间")
        def key(r):
            if strategy == "block": return r['source_hash']+':'+str(r['block'])
            if strategy == "recording": return r['source_hash']
            if strategy == "subject": return r['subject']
            raise ValueError("未知划分策略")
        groups = sorted({key(r) for r in records})
        if len(groups) < 2:
            raise ValueError("该划分方式至少需要两个不同的组；单份录制请选按轮次划分")
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        count = max(1, min(len(groups)-1, int(np.ceil(len(groups)*(100-train_percent)/100))))
        test_groups = set(groups[:count])
        train = [i for i, r in enumerate(records) if key(r) not in test_groups]
        test = [i for i, r in enumerate(records) if key(r) in test_groups]
    if not train or not test:
        raise ValueError("训练集和测试集均不能为空")
    classes = ordered_classes([records[i] for i in train]) if strategy == 'manual' else ordered_classes(records)
    if len(classes) < 2:
        raise ValueError("LDA训练至少需要两个不同的词")
    checks = [(train, "训练", 2)]
    if strategy != 'manual':
        checks.append((test, "测试", 1))
    for indices, name, minimum in checks:
        counts = {w: sum(records[i]['word'] == w for i in indices) for w in classes}
        if min(counts.values()) < minimum:
            raise ValueError(f"{name}集每词至少需要{minimum}次，当前 {counts}。请增加数据或调整划分；不会按成绩自动挑选划分。")
    return np.array(train), np.array(test)


def metrics(rows, predictions, classes=WORDS):
    classes = list(classes)
    unknown_words = sorted({r['word'] for r in rows} - set(classes))
    row_labels = classes + unknown_words
    cm = np.zeros((len(row_labels), len(classes)), dtype=int)
    for row, pred in zip(rows, predictions):
        cm[row_labels.index(row['word']), int(pred)] += 1
    n = len(classes)
    support = cm[:n].sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(n), where=support > 0)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(n), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2*recall*precision, recall+precision, out=np.zeros(n), where=recall+precision > 0)
    known = int(support.sum())
    return dict(total=len(rows), correct=int(np.trace(cm)), accuracy=float(np.trace(cm)/len(rows)),
                macro_f1=float(np.mean(f1)), confusion_matrix=cm.tolist(),
                row_labels=row_labels, column_labels=classes,
                known_total=known, known_accuracy=float(np.trace(cm)/known) if known else None,
                unknown_total=len(rows)-known, unknown_words=unknown_words,
                missing_model_classes=[w for i,w in enumerate(classes) if support[i] == 0],
                per_word={w:dict(count=int(cm[i].sum()),correct=int(cm[i,i]) if i<n else 0,
                                recall=float(recall[i]) if i<n else 0., supported=i<n)
                          for i,w in enumerate(row_labels)})


def coverage_warnings(m):
    result = ["模型只会输出训练词表中的词，没有自动未知词拒识功能。"]
    if m['unknown_total']:
        result.append(f"模型未训练的词：{', '.join(m['unknown_words'])}（{m['unknown_total']}次）。仍输出模型内预测，全部准确率计为错误；已知词准确率单独计算。")
    if m['missing_model_classes']:
        result.append("本次未测试模型中的词：" + ', '.join(m['missing_model_classes']) + "；不能代表完整词表效果。")
    result.append("宏平均F1按全部模型类别计算；未训练词的误判计入对应预测类的假阳性。")
    return result


def predict(model, x):
    z = (np.asarray(x)-np.asarray(model['scaler_mean'])) / np.asarray(model['scaler_scale'])
    scores = z@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    if not np.isfinite(scores).all():
        raise ValueError("模型输出非有限值，拒绝预测")
    if len(model['classes']) == 2:
        return (scores[:, 0] > 0).astype(int)
    return scores.argmax(axis=1)


def validate_model(model):
    if not isinstance(model, dict) or model.get('schema') not in (SCHEMA, 'semg_lda_v1') or model.get('feature_names') != FEATURES:
        raise ValueError("不是本版支持的 LDA JSON 模型；旧分析文件需重新训练导出")
    classes = validate_words(model.get('classes'))
    if len(classes) < 2 or (model['schema'] == 'semg_lda_v1' and classes != list(WORDS)):
        raise ValueError("模型词表无效或与旧版格式不一致")
    outputs = 1 if len(classes) == 2 else len(classes)
    variant = model.get('variant')
    expected = filter_sos(variant)
    for field, shape in [('scaler_mean',(8,)), ('scaler_scale',(8,)), ('coef',(outputs,8)), ('intercept',(outputs,)), ('sos',expected.shape)]:
        a = np.asarray(model.get(field), dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"模型字段 {field} 无效")
    if np.any(np.asarray(model['scaler_scale']) <= 0) or not np.allclose(model['sos'],expected,rtol=1e-10,atol=1e-12):
        raise ValueError("模型标准化或滤波系数与格式定义不一致")
    if model.get('preprocessing') != dict(fs=1000,center=32768,trim_samples=150,prepare_min_samples=1000,trial_isolated=True):
        raise ValueError("不支持的模型预处理配置")
    fingerprints = model.get('training_fingerprints')
    if not isinstance(fingerprints,list) or not fingerprints or not all(isinstance(s,str) and len(s)==64 for s in fingerprints):
        raise ValueError("模型缺少训练数据指纹，无法排查训练测试重叠")
    if model.get('training_trials') != len(fingerprints) or not isinstance(model.get('training_subjects'),list):
        raise ValueError("模型训练来源信息不完整")
    return model


def load_model(path):
    if Path(path).stat().st_size > 10_000_000:
        raise ValueError("模型文件过大")
    return validate_model(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_run(output_root, model, report, rows, predictions, prefix):
    classes = report['classes']
    folder = Path(output_root)/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False)
    if model is not None:
        write_json(folder/'model.json',model)
    write_json(folder/'report.json',report)
    with (folder/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['subject','recording','trial_id','word','prediction','correct','known_to_model','fingerprint']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row,pred in zip(rows,predictions):
            writer.writerow({**{k:row[k] for k in fields if k not in ('prediction','correct','known_to_model')},
                             'prediction':classes[int(pred)],'correct':int(row['word']==classes[int(pred)]),
                             'known_to_model':row['word'] in classes})
    m=report['metrics']
    text=f"# LDA 结果\n\n{report['scope']}\n\n正确 {m['correct']}/{m['total']}，准确率 {m['accuracy']:.1%}，宏平均 F1 {m['macro_f1']:.3f}。\n\n"
    known_text = f"{m['known_accuracy']:.1%}" if m['known_accuracy'] is not None else '不适用（没有已知词）'
    text+=f"模型词表：{', '.join(classes)}。\n\n已知词 {m['known_total']} 次，准确率 {known_text}；未训练词 {m['unknown_total']} 次。\n\n"
    text+='## 混淆矩阵（行=真实，列=预测）\n\n|真实/预测|'+'|'.join(classes)+'|\n|---|'+'---:|'*len(classes)+'\n'
    text+='\n'.join('|'+w+'|'+'|'.join(map(str,row))+'|' for w,row in zip(m['row_labels'],m['confusion_matrix']))
    text+='\n\n'+ '\n'.join('- '+s for s in report.get('warnings',[]))
    text+='\n\n报告包含实际数据划分和逐试次预测；这是指定数据上的评估，不能直接代表未来录制效果。\n'
    (folder/'结果说明.md').write_text(text,encoding='utf-8')
    return dict(folder=str(folder),model_path=str(folder/'model.json') if model is not None else None,report=report)


def train(selections, output_root, strategy='block', train_percent=80, seed=20260924, variant='notch50', progress=lambda text: None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn import __version__ as sklearn_version
    records,excluded,sources=extract(selections,variant,progress)
    tr,te=split_records(records,strategy,train_percent,seed)
    classes = ordered_classes([records[i] for i in tr])
    x=np.array([r['features'] for r in records]);y=np.array([classes.index(r['word']) if r['word'] in classes else -1 for r in records])
    progress(f"训练 LDA：训练 {len(tr)} 次，测试 {len(te)} 次")
    scaler=StandardScaler().fit(x[tr])
    estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(x[tr]),y[tr])
    model=dict(schema=SCHEMA,app_version=__version__,sklearn_version=sklearn_version,
               created_utc=datetime.now(timezone.utc).isoformat(),classes=classes,feature_names=FEATURES,
               variant=variant,sos=filter_sos(variant).tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
               coef=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),
               preprocessing=dict(fs=1000,center=32768,trim_samples=150,prepare_min_samples=1000,trial_isolated=True),
               training_fingerprints=[records[i]['fingerprint'] for i in tr],training_trials=len(tr),
               training_subjects=sorted({records[i]['subject'] for i in tr}))
    validate_model(model)
    prediction=predict(model,x[te])
    if not np.array_equal(prediction,estimator.predict(scaler.transform(x[te]))):
        raise ValueError("保存模型与训练器预测不一致")
    warnings=["模型仅使用训练集拟合；保存的模型没有用测试集再次训练。", "动作条件来自录制元数据，历史默认值未必准确；不要混入出声对照。"]
    if strategy=='block': warnings.append("同一录制按整轮划分，只能反映组内表现，不是跨录制或跨人泛化。")
    if len(te)<40: warnings.append("测试试次数较少，百分比会随个别判断明显变化。")
    if excluded: warnings.append(f"排除 {len(excluded)} 次，原因见 report.json；没有按预测成绩剔除。")
    if any(r['boundary_rtt_ms']>100 for r in records): warnings.append("部分事件往返超过100 ms，标签时机存在额外不确定性。")
    scopes={'block':'按轮次留出测试（组内）','recording':'按整组录制留出测试（跨批次）',
            'subject':'按受试者留出测试（跨人）','manual':'手动指定留出测试'}
    if strategy=='manual':
        warnings.append("手动用途由操作者指定；同一个人的不同批次不等于跨人测试。")
    if any(s['status'] != 'completed' for s in sources):
        warnings.append("包含曾中止/异常结束的录制，仅使用其中通过检查的完整试次；请查看来源记录。")
    m = metrics([records[i] for i in te], prediction, classes)
    warnings.extend(coverage_warnings(m))
    report=dict(scope=scopes[strategy],classes=classes,
                strategy=strategy,requested_train_percent=train_percent if strategy!='manual' else None,seed=seed,
                actual_train_percent=100*len(tr)/len(records),variant=variant,
                train=[records[i] for i in tr],test=[records[i] for i in te],excluded=excluded,sources=sources,
                metrics=m,warnings=warnings)
    return save_run(output_root,model,report,[records[i] for i in te],prediction,'train')


def evaluate(model_path, selections, output_root, progress=lambda text: None):
    model=load_model(model_path)
    records,excluded,sources=extract(selections,model['variant'],progress)
    overlap=[r for r in records if r['fingerprint'] in set(model['training_fingerprints'])]
    if overlap:
        raise ValueError(f"检测到 {len(overlap)} 个试次已用于训练（包括复制目录）。请选择未训练过的试次/批次；本次未输出独立测试准确率。")
    prediction=predict(model,[r['features'] for r in records])
    warnings=['只排除了直接训练试次重叠；反复查看测试结果调参后，该数据不再是未见过的最终测试集。',
              '按模型内固定的滤波和特征进行测试。输入必须是含事件标记的完整录制目录。']
    m = metrics(records, prediction, model['classes'])
    warnings.extend(coverage_warnings(m))
    if excluded:warnings.append(f'排除 {len(excluded)} 次，原因见报告。')
    report=dict(scope='加载冻结模型测试（未重新训练）',classes=model['classes'],model_path=str(Path(model_path).resolve()),
                model_sha256=digest(Path(model_path).read_bytes()),variant=model['variant'],
                test=records,excluded=excluded,sources=sources,metrics=m,
                warnings=warnings)
    return save_run(output_root,None,report,records,prediction,'test')
