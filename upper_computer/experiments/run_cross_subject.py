"""Source-only LDA baseline and a directional held-out-subject test.

Run with the project .runtime Python. Original recordings and models are read
only; both baselines and CNN outputs are saved in new model directories.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semg import lda


def session_selections(session, role):
    session = session.resolve()
    files = sorted(session.glob('*/metadata.json'))
    if not files:
        raise ValueError(f'No recording metadata in {session}')
    subjects = set()
    selections = []
    for path in files:
        meta = json.loads(path.read_text(encoding='utf-8-sig'))
        if meta.get('status') != 'completed':
            raise ValueError(f'Recording is not complete: {path.parent}')
        if meta.get('articulation_mode') != 'silent_mouthing' or meta.get('condition') != 'silent':
            raise ValueError(f'Expected silent mouthing: {path.parent}')
        subject = meta.get('subject_id')
        if not subject:
            raise ValueError(f'Missing subject identity: {path}')
        subjects.add(subject)
        selections.append(dict(path=str(path.parent), role=role))
    if len(subjects) != 1:
        raise ValueError(f'A session must contain exactly one subject: {subjects}')
    return selections, subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-session', type=Path, required=True)
    parser.add_argument('--test-session', type=Path, required=True)
    parser.add_argument('--with-cnn', action='store_true', help='Also run an optional fresh CNN experiment; default is LDA only.')
    args = parser.parse_args()
    training, source = session_selections(args.train_session, 'train')
    testing, target = session_selections(args.test_session, 'test')
    if source & target:
        raise ValueError('Training and test subjects must differ')
    feature_settings = dict(segments=6, metrics=[
        'log_mean_abs_diff', 'log_mav', 'zc_count_zero', 'ssc_count_zero'])
    result = lda.train(training + testing, ROOT/'models', strategy='manual',
                       variant='notch50', feature_settings=feature_settings,
                       channels=[0, 1], progress=lambda s: print(s, flush=True))
    folder = Path(result['folder'])
    report = result['report']
    assert {r['subject'] for r in report['train']} == source
    assert {r['subject'] for r in report['test']} == target
    assert len(report['classes']) == 10
    assert {r['word'] for r in report['test']} == set(report['classes'])
    report['scope'] = '单向跨受试者测试：' + ','.join(sorted(source)) + ' → ' + ','.join(sorted(target))
    report['evaluation_scope'] = 'cross_subject'
    report['warnings'].append('该方向仅训练源受试者，未使用目标受试者拟合；不是完整五折留人验证。数据曾用于历史开发，属于探索性跨人评估。')
    lda.write_json(folder/'report.json', report)
    note = folder/'结果说明.md'
    note.write_text(note.read_text(encoding='utf-8').replace('手动指定留出测试', report['scope']), encoding='utf-8')
    shutil.copy2(__file__, folder/'run_cross_subject.py')
    lda.write_json(folder/'cross_subject_protocol.json', dict(
        train_session=str(args.train_session.resolve()), test_session=str(args.test_session.resolve()),
        train_subjects=sorted(source), test_subjects=sorted(target),
        train_trials=len(report['train']), test_trials=len(report['test']),
        validation='LDA uses all source trials, without target-based tuning.',
        optional_cnn_requested=args.with_cnn,
        prior_exposure='Both sessions have been used in earlier mixed-person experiments.'))
    print('CROSS_SUBJECT_BASELINE', folder, flush=True)
    print('Train', sorted(source), len(report['train']), 'Test', sorted(target), len(report['test']), flush=True)
    print('LDA', report['metrics']['correct'], '/', report['metrics']['total'],
          'accuracy', report['metrics']['accuracy'], flush=True)
    if args.with_cnn:
        subprocess.run([sys.executable, str(ROOT/'experiments/train_small_cnn.py'),
                        '--baseline', str(folder), '--cross-subject'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
