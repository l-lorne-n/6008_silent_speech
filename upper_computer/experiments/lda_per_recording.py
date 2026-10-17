"""Evaluate each timestamped recording independently with a frozen LDA setup."""
import argparse
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
from semg import lda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    recordings = []
    for session in args.sessions:
        paths = sorted(session.resolve().glob('*/metadata.json'))
        if not paths:
            raise ValueError(f'No recordings in {session}')
        recordings.extend(p.parent for p in paths)
    if len(set(recordings)) != len(recordings):
        raise ValueError('Duplicate recordings requested')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'models'/f'lda_per_recording_{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir()
    shutil.copy2(__file__, output/'lda_per_recording.py')
    config = dict(segments=6, metrics=['log_mean_abs_diff', 'log_mav', 'zc_count_zero', 'ssc_count_zero'])
    protocol = dict(unit='One timestamped recording, not a session folder or a single internal block.',
                    strategy='block', train_percent=80, seed=20260924, variant='notch50',
                    feature_config=config, channels=[0, 1], threshold='accuracy > 0.5 (strict)',
                    recordings=[str(p) for p in recordings])
    lda.write_json(output/'protocol.json', protocol)
    print('OUTPUT', output, flush=True)
    results = []
    for folder in recordings:
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
        trials = [json.loads(s) for s in (folder/'trials.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        hashes = {n: hashlib.sha256((folder/n).read_bytes()).hexdigest()
                  for n in ['metadata.json', 'samples.csv', 'events.jsonl', 'trials.jsonl']}
        item = dict(session=folder.parent.name, recording=folder.name, path=str(folder),
                    subject=meta.get('subject_id'), recording_status=meta.get('status'),
                    recorded_trial_rows=len(trials), completed_trial_rows=sum(t.get('status')=='completed' for t in trials),
                    source_hashes=hashes, status='not_evaluable')
        try:
            run = lda.train([dict(path=str(folder), role='train')], output/'runs',
                            strategy='block', train_percent=80, seed=20260924,
                            variant='notch50', feature_settings=config, channels=[0, 1])
            report = run['report']
            if len(report['classes']) != 10:
                raise ValueError('Incomplete ten-word coverage; not comparable with ten-word recordings')
            model = lda.load_model(Path(run['model_path']))
            prediction = lda.predict(model, [r['features'] for r in report['test']])
            assert lda.metrics(report['test'], prediction, report['classes']) == report['metrics']
            assert not {r['block'] for r in report['train']} & {r['block'] for r in report['test']}
            item.update(status='evaluated', model_folder=run['folder'],
                        train_count=len(report['train']), test_count=len(report['test']),
                        correct=report['metrics']['correct'], accuracy=report['metrics']['accuracy'],
                        macro_f1=report['metrics']['macro_f1'],
                        above_50=report['metrics']['accuracy']>0.5,
                        test_blocks=sorted({r['block'] for r in report['test']}),
                        excluded=report['excluded'], model_reload_verified=True)
            print(folder.parent.name, folder.name, item['correct'], '/', item['test_count'],
                  f"{item['accuracy']:.1%}", flush=True)
        except ValueError as exc:
            item['reason'] = str(exc)
            print(folder.parent.name, folder.name, 'NOT_EVALUABLE', str(exc), flush=True)
        for name, expected in hashes.items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected
        item['source_hashes_unchanged'] = True
        results.append(item)
        lda.write_json(output/'results.json', dict(protocol=protocol, recordings=results))
    valid = [r for r in results if r['status']=='evaluated']
    counts = dict(recordings_total=len(results), evaluated=len(valid),
                  not_evaluable=len(results)-len(valid),
                  above_50=sum(r['above_50'] for r in valid),
                  equal_50=sum(r['accuracy']==0.5 for r in valid),
                  below_50=sum(r['accuracy']<0.5 for r in valid))
    lda.write_json(output/'results.json', dict(protocol=protocol, counts=counts, recordings=results))
    fields = ['session', 'recording', 'recording_status', 'status', 'train_count', 'test_count',
              'correct', 'accuracy', 'above_50', 'reason', 'model_folder']
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# S03 逐次独立录制 LDA 结果', '',
             f"共 {counts['recordings_total']} 次独立录制，可评估 {counts['evaluated']} 次；严格大于 50% 的有 {counts['above_50']} 次，恰好 50% 的有 {counts['equal_50']} 次，低于 50% 的有 {counts['below_50']} 次；无法评估 {counts['not_evaluable']} 次。", '',
             '每个带时间戳的录制单独建模，不合并 session。双通道、6 段、4 特征；固定种子 20260924，按内部完整轮次做 80/20 划分。训练集拟合 StandardScaler，LDA 使用 lsqr + shrinkage=auto。', '',
             '| Session | 独立录制 | 正确/测试 | 准确率 | >50% |', '|---|---|---:|---:|---|']
    for r in results:
        if r['status']=='evaluated':
            lines.append(f"| {r['session']} | {r['recording']} | {r['correct']}/{r['test_count']} | {r['accuracy']:.1%} | {'是' if r['above_50'] else '否'} |")
        else:
            lines.append(f"| {r['session']} | {r['recording']} | — | 无法评估 | — |")
    lines += ['', '## 无法评估的录制', '']
    for r in results:
        if r['status']!='evaluated':
            lines.append(f"- {r['session']}/{r['recording']}：{r['reason']}")
    lines += ['', '每次录制的测试集较小，单次正确与否会明显改变百分比；本表用于描述这次固定划分，不能把低分录制直接当作坏数据删除，也不能以筛选后的成绩替代独立评估。原始文件哈希前后不变，成功模型均完成重载核对。']
    (output/'结果汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('COUNTS', json.dumps(counts), flush=True)
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
