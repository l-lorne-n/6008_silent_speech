"""Run the fixed small CNN on all four frozen LDA subject holdouts."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
from semg import lda
import train_small_cnn as cnn


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_folds(baseline):
    summary = read(baseline/'report.json')
    assert summary['protocol']['subjects'] == ['S01', 'S02', 'S04', 'S05']
    assert summary['protocol']['directories'] == ['s1', 's2', 's4', 's5']
    assert summary['protocol']['variant'] == 'notch50'
    seen, all_fps, folds = set(), None, []
    for fold in summary['folds']:
        path = Path(fold['folder'])
        report = read(path/'report.json')
        held_out = fold['held_out_subject']
        assert report['channels'] == [0, 1] and len(report['classes']) == 10
        assert report['variant'] == 'notch50'
        assert {r['subject'] for r in report['test']} == {held_out}
        training_people = {r['subject'] for r in report['train']}
        assert training_people == set(summary['protocol']['subjects'])-{held_out}
        rows, split = cnn.split_data(report)
        fps = {r['fingerprint'] for r in rows}
        assert len(fps) == len(rows)
        if all_fps is None:
            all_fps = fps
        assert all_fps == fps
        test_fps = {rows[i]['fingerprint'] for i in split['test']}
        assert not seen & test_fps
        seen.update(test_fps)
        for key in ['train', 'validation']:
            assert {rows[i]['subject'] for i in split[key]} == training_people
        folds.append(dict(subject=held_out, baseline=str(path),
                          split_counts={k:len(v) for k,v in split.items()},
                          baseline_report_sha256=cnn.digest(path/'report.json'),
                          baseline_model_sha256=cnn.digest(path/'model.json')))
    assert seen == all_fps and len(folds) == 4
    assert len(seen) == summary['pooled_metrics']['total']
    assert sum(p.numel() for p in cnn.SmallCNN().parameters()) == 23498
    return summary, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'models/lda_loso_goodone_20261014_130607_0c55eb')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', type=Path, help='Existing incomplete run; reuse completed folds and frozen selection.')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    reference, folds = verify_folds(baseline)
    if args.check_only:
        print(json.dumps(folds, indent=2), flush=True)
        return
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = args.resume.resolve() if args.resume else ROOT/'models'/f'cnn_loso_goodone_{stamp}_{uuid.uuid4().hex[:6]}'
    if args.resume:
        assert (output/'protocol.json').is_file() and not (output/'report.json').exists()
    else:
        output.mkdir()
    for source in [Path(__file__), ROOT/'experiments/train_small_cnn.py', ROOT/'semg/lda.py']:
        shutil.copy2(source, output/('resume_'+source.name if args.resume else source.name))
    protocol = dict(baseline=str(baseline), baseline_sha256=cnn.digest(baseline/'report.json'),
                    evaluation='Four-fold LOSO, three training subjects and one held-out subject.',
                    parameters=23498, subjects=reference['protocol']['subjects'], excluded_subjects=['S03'],
                    bandpass_hz=[20,400], notch_hz=50, notch_q=30, folds=folds,
                    seeds=cnn.SEEDS, max_epochs=cnn.MAX_EPOCHS, patience=cnn.PATIENCE,
                    batch_size=cnn.BATCH, selection='Source-only validation cross entropy; fresh refit on all three source subjects.',
                    architecture_selection='Fixed before running any fold; no adaptation to held-out scores.',
                    caveat='Previously inspected goodone subset; exploratory comparison, not an untouched final test.')
    if args.resume:
        assert read(output/'protocol.json') == protocol
    else:
        cnn.write_json(output/'protocol.json', protocol)
    print('OUTPUT', output, flush=True)
    results, predictions = [], []
    for fold in folds:
        subject = fold['subject']
        folder = output/'folds'/subject
        print('START_FOLD', subject, fold['split_counts'], flush=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        command = [sys.executable, str(ROOT/'experiments/train_small_cnn.py'),
                   '--baseline', fold['baseline'], '--cross-subject', '--output', str(folder)]
        if not (folder/'report.json').exists():
            if folder.exists():
                assert args.resume and (folder/'selected_hyperparameters.json').is_file()
                command.append('--resume-refit')
            with (output/f'{subject}_training.log').open('a' if args.resume else 'w', encoding='utf-8') as log:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8') as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(f'[{subject}] {line}', end='', flush=True)
                    if process.wait():
                        raise RuntimeError(f'{subject} training failed; see {log.name}')
        else:
            assert args.resume
            print('REUSE_COMPLETED_FOLD', subject, flush=True)
        r = read(folder/'report.json')
        b = read(Path(fold['baseline'])/'report.json')
        assert r['protocol']['test_subjects'] == [subject]
        assert r['protocol']['split_counts'] == fold['split_counts']
        assert r['model_roundtrip_verified'] and r['input_hashes_unchanged']
        assert not r['protocol']['test_used_for_selection']
        for name in ['report', 'model']:
            assert cnn.digest(Path(fold['baseline'])/f'{name}.json') == fold[f'baseline_{name}_sha256']
        lookup = {(t['subject'],t['recording'],str(t['trial_id'])): t for t in b['test']}
        with (folder/'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(lookup)
        assert {(t['subject'],t['recording'],t['trial_id']) for t in rows} == set(lookup)
        for t in rows:
            original = lookup[(t['subject'],t['recording'],t['trial_id'])]
            assert t['word'] == original['word'] and t['subject'] == subject
            assert int(t['correct']) == int(t['word'] == t['prediction'])
            predictions.append(dict(held_out_subject=subject, fingerprint=original['fingerprint'], **t))
        result = dict(subject=subject, folder=str(folder), train_count=r['final_train_trials'],
                      test_count=r['test_trials'], cnn_metrics=r['metrics'], lda_metrics=r['baseline_metrics'],
                      selected=r['selected'], paired=r['paired'], elapsed_seconds=r['elapsed_seconds'])
        results.append(result)
        cnn.write_json(output/'completed_folds.json', results)
        print('DONE_FOLD', subject, r['metrics']['correct'], '/', r['test_trials'], flush=True)
    assert len(predictions) == len({r['fingerprint'] for r in predictions}) == reference['pooled_metrics']['total']
    classes = b['classes']
    pooled = lda.metrics(predictions, [classes.index(r['prediction']) for r in predictions], classes)
    old = lda.metrics(predictions, [classes.index(r['lda_prediction']) for r in predictions], classes)
    assert old == reference['pooled_metrics']
    assert cnn.digest(baseline/'report.json') == protocol['baseline_sha256']
    for source in b['sources']:
        for name, expected in source['hashes'].items():
            assert hashlib.sha256((Path(source['path'])/name).read_bytes()).hexdigest() == expected
    summary = dict(schema='semg_small_cnn_loso_v1', folder=str(output), protocol=protocol, folds=results,
                   subject_mean_accuracy=sum(r['cnn_metrics']['accuracy'] for r in results)/len(results),
                   lda_subject_mean_accuracy=reference['subject_mean_accuracy'], pooled_metrics=pooled,
                   lda_pooled_metrics=old, every_trial_tested_once=True, inputs_unchanged=True)
    cnn.write_json(output/'report.json', summary)
    with (output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    note=['小型 CNN 四人跨人测试：每折三人训练、一人测试，排除 S03。',
          '20–400 Hz；50 Hz 陷波 Q=30；2×2048 波形；16→32→64；23498 参数。',
          '每折从训练人员各录制抽一整轮验证；三个预定种子；验证损失选种子和轮数后，用全部训练人员数据从头重训。', '']
    for r in results:
        note.append(f"{r['subject']}：CNN {r['cnn_metrics']['correct']}/{r['test_count']} = {r['cnn_metrics']['accuracy']:.2%}；LDA {r['lda_metrics']['accuracy']:.2%}")
    note.extend(['', f"四人平均：CNN {summary['subject_mean_accuracy']:.2%}；LDA {summary['lda_subject_mean_accuracy']:.2%}",
                 f"合并：CNN {pooled['correct']}/{pooled['total']} = {pooled['accuracy']:.2%}；宏 F1 {pooled['macro_f1']:.4f}",
                 '同一试次及划分，模型重载、原始哈希、测试隔离检查通过。',
                 '这是已查看数据上的探索性对照；未部署 STM32，未验证连续实时识别。'])
    (output/'结果汇总.txt').write_text('\n'.join(note)+'\n', encoding='utf-8')
    print('\n'.join(note), flush=True)
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
