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
from .channels import validate_channels, require_channels, recording_channels, description as channel_description

FEATURES = [f"log_rms_quarter_{i}" for i in range(1, 5)] + [f"log_mean_abs_diff_quarter_{i}" for i in range(1, 5)]
SCHEMA = "semg_lda_v4"
FEATURE_LABELS = {
    'log_rms': 'RMS 幅度（log）',
    'log_mean_abs_diff': '平均绝对相邻差（log）',
    'log_mav': '平均绝对值 MAV（log）',
    'mean_frequency_hz': '平均频率 MNF（Hz）',
    'zc_count_zero': '过零次数 ZC（零阈值）',
    'ssc_count_zero': '斜率符号变化 SSC（零阈值）',
    'rms': '均方根 RMS',
    'mav': '平均绝对值 MAV',
    'zcr_zero': '过零率 ZCR（零阈值）',
    'waveform_length': '波形长度 WL',
}
DEFAULT_FEATURE_CONFIG = dict(segments=4, metrics=['log_rms', 'log_mean_abs_diff'])
VARIANTS = {"notch50": [50], "notch50_100_200": [50, 100, 200], "notch50_bp450": [50], "notch50_bp500": [50]}


def variant_fs(variant):
    if variant not in VARIANTS:
        raise ValueError('未知滤波方案')
    return 2000 if variant == 'notch50_bp500' else 1000


def preprocessing_config(variant):
    fs = variant_fs(variant)
    return dict(fs=fs, center=32768, trim_samples=round(.15*fs), prepare_min_samples=fs, trial_isolated=True)


def feature_config(value=None):
    value = DEFAULT_FEATURE_CONFIG if value is None else value
    if not isinstance(value, dict) or set(value) != {'segments', 'metrics'}:
        raise ValueError('特征配置必须包含分段数和特征种类')
    segments, metrics = value['segments'], value['metrics']
    if type(segments) is not int or not 1 <= segments <= 12:
        raise ValueError('动作分段数应为1–12')
    if (not isinstance(metrics, list) or not metrics or not all(isinstance(k,str) for k in metrics)
            or len(set(metrics)) != len(metrics) or any(k not in FEATURE_LABELS for k in metrics)):
        raise ValueError('请至少选择一种有效特征，不能重复')
    return dict(segments=segments, metrics=list(metrics))


def feature_names(config=None, channels=None):
    c = feature_config(config)
    unit = 'quarter' if c['segments'] == 4 else 'segment'
    names = [f'{metric}_{unit}_{i}' for metric in c['metrics'] for i in range(1, c['segments']+1)]
    selected = validate_channels(channels)
    return names if selected == [0] else [f'ch{ch}_{name}' for ch in selected for name in names]


def model_channels(model):
    if model.get('schema') == SCHEMA:
        if 'channels' not in model: raise ValueError('模型缺少通道配置')
        return validate_channels(model['channels'])
    if model.get('channels', [0]) != [0]: raise ValueError('旧模型仅支持CH0')
    return [0]


def model_feature_config(model):
    if model.get('schema') in (SCHEMA, 'semg_lda_v3'):
        if 'feature_config' not in model:
            raise ValueError('模型缺少特征配置')
        return feature_config(model['feature_config'])
    # v1/v2 always used four quarters and these exact two features.
    if 'feature_config' in model and feature_config(model['feature_config']) != DEFAULT_FEATURE_CONFIG:
        raise ValueError('旧版模型不能使用可变特征配置')
    return feature_config()


def describe_features(config=None):
    c = feature_config(config)
    return (f"{c['segments']}段 × {len(c['metrics'])}类 = {len(feature_names(c))}维；"
            + '、'.join(FEATURE_LABELS[k] for k in c['metrics']))


def segment_metric(x, metric, fs=1000):
    x = np.asarray(x, dtype=float)
    if metric == 'rms': return float(np.sqrt(np.mean(x*x)))
    if metric == 'mav': return float(np.mean(np.abs(x)))
    if metric == 'waveform_length': return float(np.sum(np.abs(np.diff(x))))
    if metric == 'zcr_zero':
        # Fraction of adjacent pairs that strictly cross zero; dimensionless.
        return segment_metric(x, 'zc_count_zero') / (len(x)-1) if len(x) > 1 else 0.
    if metric == 'log_rms': return np.log(np.sqrt(np.mean(x*x))+1e-8)
    if metric == 'log_mean_abs_diff': return np.log(np.mean(np.abs(np.diff(x)))+1e-8)
    if metric == 'log_mav': return np.log(np.mean(np.abs(x))+1e-8)
    # Strict changes only: exact zeros / flat slopes do not count. No log.
    if metric == 'zc_count_zero':
        return float(np.count_nonzero(((x[:-1] < 0) & (x[1:] > 0)) | ((x[:-1] > 0) & (x[1:] < 0))))
    if metric == 'ssc_count_zero':
        d = np.diff(x)
        return float(np.count_nonzero(((d[:-1] < 0) & (d[1:] > 0)) | ((d[:-1] > 0) & (d[1:] < 0))))
    if metric != 'mean_frequency_hz':
        raise ValueError('未知特征')
    # Hann-windowed, demeaned spectrum centroid in the analysis band; no padding.
    power = np.abs(np.fft.rfft((x-x.mean())*np.hanning(len(x))))**2
    freq = np.fft.rfftfreq(len(x), d=1/fs)
    band = (freq >= 20) & (freq <= (500 if fs == 2000 else 400))
    total = power[band].sum()
    return float(np.sum(freq[band]*power[band])/total) if total > 1e-20 else 0.


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
            if meta.get("mode") != "binary" or meta.get("sampling_rate_hz") not in (1000, 2000):
                reason = "仅支持真实 1/2 kHz 完整采样"
            elif meta.get("status") not in ("completed", "stopped", "error"):
                reason = "录制尚未正常结束"
            trials = lines((file.parent / "trials.jsonl").read_bytes())
            trials = [t for t in trials if t.get("status") == "completed" and t.get("word") in words]
            catalog.append(dict(path=str(file.parent.resolve()), subject=str(meta.get("subject_id", "未知")),
                                session=meta.get("session_id"), condition=f"{meta.get('condition', '?')}/{meta.get('articulation_mode', '?')} · "+'+'.join(f'CH{c}' for c in recording_channels(meta)),
                                site=meta.get("electrode_sites", ""), notes=meta.get("notes", ""), sampling_rate_hz=meta.get('sampling_rate_hz'),
                                status=meta.get("status"), disabled_reason=reason, trials=trials))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{file.parent}: {exc}")
    return catalog, errors


def filter_sos(variant):
    if variant not in VARIANTS:
        raise ValueError("未知滤波方案")
    fs = variant_fs(variant)
    high = 500 if fs == 2000 else 450 if variant == 'notch50_bp450' else 400
    sos = butter(4, [20, high], fs=fs, btype="bandpass", output="sos")
    for hz in VARIANTS[variant]:
        b, a = iirnotch(hz, Q=hz/(50/30), fs=fs)
        sos = np.vstack((sos, tf2sos(b, a)))
    return sos


def feature_vector(raw, action_offset, sos, config=None, fs=1000):
    """Warm up on preparation; stop at rest onset. No future/trial sharing."""
    x = np.asarray(raw, dtype=float)-32768
    y, _ = sosfilt(sos, x, zi=sosfilt_zi(sos)*x[0])
    trim = round(.15*fs)
    action = y[action_offset+trim:len(y)-trim]
    if len(action) < round(.4*fs):
        raise ValueError("动作区间过短：裁去两端各150 ms后至少需要400 ms")
    c = feature_config(config)
    if len(action) < c['segments']*round(.05*fs):
        raise ValueError('动作分段过短：每段至少50 ms，请减少分段数或增加动作时长')
    parts = np.array_split(action, c['segments'])
    return np.array([segment_metric(q, metric, fs) for metric in c['metrics'] for q in parts])


def extract_recording(selection, variant, config=None, channels=None):
    config = feature_config(config)
    channels = validate_channels(channels)
    folder = Path(selection["path"])
    snapshots = {name: (folder/name).read_bytes() for name in ("metadata.json", "samples.csv", "events.jsonl", "trials.jsonl")}
    hashes = {name: digest(data) for name, data in snapshots.items()}
    meta = json.loads(snapshots["metadata.json"])
    require_channels(meta, channels)
    words = recording_words(meta)
    fs = variant_fs(variant)
    if meta.get("mode") != "binary" or meta.get("status") not in ("completed", "stopped", "error"):
        raise ValueError(f"{folder.name} 不是已结束的真实完整采样记录")
    if meta.get('sampling_rate_hz') != fs:
        raise ValueError(f"{folder.name} 采样率{meta.get('sampling_rate_hz')} Hz与滤波/模型要求{fs} Hz不一致；2 kHz请选20–500 Hz，不可混用新旧采样率")
    reader = csv.DictReader(io.StringIO(snapshots["samples.csv"].decode("utf-8-sig")))
    available = [c for c in (0,1) if f'raw_ch{c}' in reader.fieldnames]
    if not set(channels).issubset(available): raise ValueError('CSV缺少模型要求的原始通道列')
    data = np.array([(int(r["device_sample_idx"]), *[int(r[f'raw_ch{c}']) for c in available]) for r in reader], dtype=np.int64)
    if not len(data) or data.ndim != 2 or np.any(np.diff(data[:, 0]) != 1):
        raise ValueError(f"{folder.name} 样本序号不连续或为空，不能压缩时间后训练")
    if np.any((data[:, 1:] < 0) | (data[:, 1:] > 65535)):
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
            if not data[0, 0] <= prepare < action < rest <= data[-1, 0]+1 or action-prepare < fs:
                raise ValueError("边界越界或准备期不足1秒")
            all_raw = data[prepare-data[0, 0]:rest-data[0, 0], 1:]
            raw = all_raw[:, [available.index(c) for c in channels]]
            if np.any((raw <= 1) | (raw >= 65534)):
                raise ValueError("准备/动作期出现 ADC 削顶")
            features = np.concatenate([feature_vector(raw[:,i], action-prepare, sos, config, fs=fs) for i in range(len(channels))])
            # Stable even if a recording folder or CSV is copied/renamed.
            fingerprint = digest(all_raw.astype('<u2').tobytes() + str(action-prepare).encode('ascii'))
            signal_fingerprints = [digest(raw[:,i].astype('<u2').tobytes()+str(action-prepare).encode('ascii')) for i in range(len(channels))]
            base.update(channels=channels, signal_fingerprints=signal_fingerprints)
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
                                    notes=meta.get("notes"), status=meta.get("status"),channels=channels,channel_sites=meta.get('channel_sites',{}))


def extract(selections, variant, progress=lambda text: None, config=None, channels=None):
    config = feature_config(config)
    records, excluded, sources = [], [], []
    seen = set()
    for i, selection in enumerate(selections):
        progress(f"提取录制 {i+1}/{len(selections)}：{Path(selection['path']).name}")
        rows, bad, source = extract_recording(selection, variant, config, channels)
        for row in rows:
            identity = tuple(row['signal_fingerprints'])
            if identity in seen:
                raise ValueError("选择中存在相同的原始试次（可能重复导入或复制目录），请去掉重复数据")
            seen.add(identity)
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
        if strategy == 'within_recording':
            rng = np.random.default_rng(seed)
            train, test = [], []
            batches = sorted({(r['subject'], r['source_hash']) for r in records})
            for subject, source in batches:
                indices = [i for i,r in enumerate(records) if (r['subject'],r['source_hash']) == (subject,source)]
                groups = sorted({records[i]['block'] for i in indices})
                if len(groups) < 2:
                    row = records[indices[0]]
                    raise ValueError(f"每批次内划分需要至少两个有效轮次：{subject} / {row.get('recording',source)} 仅有{len(groups)}轮。请增加数据、取消该批次或选择其他划分方式。")
                rng.shuffle(groups)
                count = max(1, min(len(groups)-1, int(np.ceil(len(groups)*(100-train_percent)/100))))
                test_groups = set(groups[:count])
                train.extend(i for i in indices if records[i]['block'] not in test_groups)
                test.extend(i for i in indices if records[i]['block'] in test_groups)
            train.sort(); test.sort()
        else:
            train, test = split_pooled_groups(records, strategy, train_percent, seed)
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


def split_pooled_groups(records, strategy, train_percent, seed):
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
    return train, test


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


def group_metrics(train, rows, predictions, classes):
    """Keep pooled sample accuracy distinct from equally weighted subject accuracy."""
    result = {}
    for level in ('subject', 'recording'):
        def key(r):
            return (r['subject'],) if level == 'subject' else (r['subject'], r['source_hash'])
        groups = []
        for identity in sorted({key(r) for r in train + rows}):
            training = [r for r in train if key(r) == identity]
            indices = [i for i,r in enumerate(rows) if key(r) == identity]
            testing = [rows[i] for i in indices]
            entry = dict(subject=identity[0], train_total=len(training), test_total=len(testing),
                         metrics=metrics(testing, [predictions[i] for i in indices], classes) if testing else None)
            if level == 'recording':
                representative = (training + testing)[0]
                entry.update(recording=representative['recording'], source_hash=identity[1])
            groups.append(entry)
        result[level] = groups
    accuracies = [g['metrics']['accuracy'] for g in result['subject'] if g['metrics'] is not None]
    result['subject_mean_accuracy'] = float(np.mean(accuracies)) if accuracies else None
    return result


def group_summary_lines(report):
    grouped = report.get('group_metrics')
    if not grouped:
        return []
    lines = ['总体准确率按所有测试试次汇总；受试者平均准确率让每位有测试数据的人权重相同。']
    mean = grouped['subject_mean_accuracy']
    if mean is not None:
        lines.append(f'受试者平均准确率：{mean:.1%}（仅包含有测试数据的受试者）')
    for level in ('subject','recording'):
        for g in grouped[level]:
            name = g['subject'] + ((' / ' + g['recording']) if level == 'recording' else '')
            m = g['metrics']
            score = f"正确 {m['correct']}/{m['total']}，准确率 {m['accuracy']:.1%}" if m else '无测试数据'
            training = f"训练 {g['train_total']} / " if 'train' in report else ''
            lines.append(f"{name}：{training}测试 {g['test_total']}；{score}")
    return lines


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
    if not isinstance(model, dict) or model.get('schema') not in (SCHEMA, 'semg_lda_v3', 'semg_lda_v2', 'semg_lda_v1'):
        raise ValueError("不是本版支持的 LDA JSON 模型；旧分析文件需重新训练导出")
    config = model_feature_config(model)
    names = feature_names(config, model_channels(model))
    if model.get('feature_names') != names:
        raise ValueError('模型特征名称或顺序与配置不一致')
    dimensions = len(names)
    classes = validate_words(model.get('classes'))
    if len(classes) < 2 or (model['schema'] == 'semg_lda_v1' and classes != list(WORDS)):
        raise ValueError("模型词表无效或与旧版格式不一致")
    outputs = 1 if len(classes) == 2 else len(classes)
    variant = model.get('variant')
    expected = filter_sos(variant)
    for field, shape in [('scaler_mean',(dimensions,)), ('scaler_scale',(dimensions,)), ('coef',(outputs,dimensions)), ('intercept',(outputs,)), ('sos',expected.shape)]:
        a = np.asarray(model.get(field), dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"模型字段 {field} 无效")
    if np.any(np.asarray(model['scaler_scale']) <= 0) or not np.allclose(model['sos'],expected,rtol=1e-10,atol=1e-12):
        raise ValueError("模型标准化或滤波系数与格式定义不一致")
    if model.get('preprocessing') != preprocessing_config(variant):
        raise ValueError("不支持的模型预处理配置")
    fingerprints = model.get('training_fingerprints')
    if model.get('schema') == SCHEMA:
        signals = model.get('training_signal_fingerprints')
        if not isinstance(signals,list) or not signals or not all(isinstance(s,str) and len(s)==64 for s in signals):
            raise ValueError('模型缺少通道级训练指纹')
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
    report['group_metrics'] = group_metrics(report.get('train', []), rows, predictions, classes)
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
    text+='特征设置：'+describe_features(report.get('feature_config'))+'。\n\n'
    text+='输入通道：'+channel_description(report.get('channels',[0]))+'（按通道拼接特征）。\n\n'
    text+='## 人员与批次评估\n\n'+'\n\n'.join(group_summary_lines(report))+'\n\n'
    text+='## 混淆矩阵（行=真实，列=预测）\n\n|真实/预测|'+'|'.join(classes)+'|\n|---|'+'---:|'*len(classes)+'\n'
    text+='\n'.join('|'+w+'|'+'|'.join(map(str,row))+'|' for w,row in zip(m['row_labels'],m['confusion_matrix']))
    text+='\n\n'+ '\n'.join('- '+s for s in report.get('warnings',[]))
    text+='\n\n报告包含实际数据划分和逐试次预测；这是指定数据上的评估，不能直接代表未来录制效果。\n'
    (folder/'结果说明.md').write_text(text,encoding='utf-8')
    return dict(folder=str(folder),model_path=str(folder/'model.json') if model is not None else None,report=report)


def train(selections, output_root, strategy='block', train_percent=80, seed=20260924, variant='notch50', progress=lambda text: None, feature_settings=None, channels=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn import __version__ as sklearn_version
    config = feature_config(feature_settings)
    channels = validate_channels(channels)
    records,excluded,sources=extract(selections,variant,progress,config,channels)
    tr,te=split_records(records,strategy,train_percent,seed)
    classes = ordered_classes([records[i] for i in tr])
    x=np.array([r['features'] for r in records]);y=np.array([classes.index(r['word']) if r['word'] in classes else -1 for r in records])
    progress(f"训练 LDA：训练 {len(tr)} 次，测试 {len(te)} 次")
    scaler=StandardScaler().fit(x[tr])
    estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(x[tr]),y[tr])
    model=dict(schema=SCHEMA,app_version=__version__,sklearn_version=sklearn_version,
               created_utc=datetime.now(timezone.utc).isoformat(),classes=classes,feature_names=feature_names(config,channels),feature_config=config,channels=channels,
               variant=variant,sos=filter_sos(variant).tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
               coef=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),
               preprocessing=preprocessing_config(variant),
               training_fingerprints=[records[i]['fingerprint'] for i in tr],training_trials=len(tr),
               training_signal_fingerprints=sorted({f for i in tr for f in records[i]['signal_fingerprints']}),
               training_subjects=sorted({records[i]['subject'] for i in tr}))
    validate_model(model)
    prediction=predict(model,x[te])
    if not np.array_equal(prediction,estimator.predict(scaler.transform(x[te]))):
        raise ValueError("保存模型与训练器预测不一致")
    warnings=["模型仅使用训练集拟合；保存的模型没有用测试集再次训练。", "动作条件来自录制元数据，历史默认值未必准确；不要混入出声对照。"]
    if len(feature_names(config,channels)) >= len(tr)//2:
        warnings.append('特征维数相对训练样本较高；增加特征不保证改善，需用独立数据验证。')
    warnings.append('分段数和特征种类属于特征提取设置；分类器保持lsqr + 自动收缩。反复比较本测试集后应另留最终测试数据。')
    if strategy in ('block','within_recording'): warnings.append("同一录制按整轮划分，只能反映组内表现，不是跨录制或跨人泛化。")
    if strategy == 'within_recording':
        warnings.append('各批次分别留出完整轮次，然后合并训练一个模型。比例按轮次数取整；保证每个有效批次两边都有数据，不保证手动删减后每批次每词均覆盖。')
    if len(te)<40: warnings.append("测试试次数较少，百分比会随个别判断明显变化。")
    if excluded: warnings.append(f"排除 {len(excluded)} 次，原因见 report.json；没有按预测成绩剔除。")
    if any(r['boundary_rtt_ms']>100 for r in records): warnings.append("部分事件往返超过100 ms，标签时机存在额外不确定性。")
    scopes={'block':'按轮次留出测试（组内）','within_recording':'每批次内按轮次划分（多人混合，已见人员及批次）','recording':'按整组录制留出测试（跨批次）',
            'subject':'按受试者留出测试（跨人）','manual':'手动指定留出测试'}
    if strategy=='manual':
        warnings.append("手动用途由操作者指定；同一个人的不同批次不等于跨人测试。")
    if any(s['status'] != 'completed' for s in sources):
        warnings.append("包含曾中止/异常结束的录制，仅使用其中通过检查的完整试次；请查看来源记录。")
    m = metrics([records[i] for i in te], prediction, classes)
    warnings.extend(coverage_warnings(m))
    report=dict(scope=scopes[strategy],classes=classes,
                strategy=strategy,requested_train_percent=train_percent if strategy!='manual' else None,seed=seed,
                actual_train_percent=100*len(tr)/len(records),variant=variant,feature_config=config,feature_names=feature_names(config,channels),channels=channels,
                train=[records[i] for i in tr],test=[records[i] for i in te],excluded=excluded,sources=sources,
                metrics=m,warnings=warnings)
    return save_run(output_root,model,report,[records[i] for i in te],prediction,'train')


def evaluate(model_path, selections, output_root, progress=lambda text: None):
    model=load_model(model_path)
    config = model_feature_config(model)
    channels = model_channels(model)
    records,excluded,sources=extract(selections,model['variant'],progress,config,channels)
    seen = set(model.get('training_signal_fingerprints',model['training_fingerprints']))
    overlap=[r for r in records if r['fingerprint'] in set(model['training_fingerprints']) or set(r['signal_fingerprints']) & seen]
    if overlap:
        raise ValueError(f"检测到 {len(overlap)} 个试次已用于训练（包括复制目录）。请选择未训练过的试次/批次；本次未输出独立测试准确率。")
    prediction=predict(model,[r['features'] for r in records])
    warnings=['只排除了直接训练试次重叠；反复查看测试结果调参后，该数据不再是未见过的最终测试集。',
              '按模型内固定的滤波和特征进行测试。输入必须是含事件标记的完整录制目录。']
    m = metrics(records, prediction, model['classes'])
    warnings.extend(coverage_warnings(m))
    if excluded:warnings.append(f'排除 {len(excluded)} 次，原因见报告。')
    report=dict(scope='加载冻结模型测试（未重新训练）',classes=model['classes'],model_path=str(Path(model_path).resolve()),
                model_sha256=digest(Path(model_path).read_bytes()),variant=model['variant'],feature_config=config,feature_names=feature_names(config,channels),channels=channels,
                test=records,excluded=excluded,sources=sources,metrics=m,
                warnings=warnings)
    return save_run(output_root,None,report,records,prediction,'test')
