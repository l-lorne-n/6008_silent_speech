"""Four or five seen subjects: freeze 80/20 rounds, then train SmallCNN."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semg import lda


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-s3', action='store_true', help='Use all five goodone people, including S03.')
    args = parser.parse_args()
    reference = ROOT/'models/lda_loso_goodone_20261014_130607_0c55eb'
    original = read(Path(read(reference/'report.json')['folds'][0]['folder'])/'report.json')
    old_subjects = ['S01', 'S02', 'S04', 'S05']
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05'] if args.include_s3 else old_subjects
    assert sorted({r['subject'] for r in original['train']+original['test']}) == old_subjects
    for source in original['sources']:
        for name, expected in source['hashes'].items():
            assert hashlib.sha256((Path(source['path'])/name).read_bytes()).hexdigest() == expected
    selections = [dict(path=s['path'], role='train') for s in original['sources']]
    inventory = []
    if args.include_s3:
        selections = []
        for number in range(1, 6):
            metadata = sorted((ROOT/f'recordings/goodone/s{number}').rglob('metadata.json'))
            assert metadata, f'Missing recordings for S{number:02d}'
            for path in metadata:
                meta = json.loads(path.read_text(encoding='utf-8-sig'))
                assert meta['subject_id'] == f'S{number:02d}', path
                assert (meta['status'], meta['condition'], meta['articulation_mode']) == ('completed', 'silent', 'silent_mouthing'), path
                selections.append(dict(path=str(path.parent.resolve()), role='train'))
                inventory.append(dict(path=str(path.parent.resolve()), subject=meta['subject_id'], planned_trials=len(meta['trial_order'])))
        assert {Path(s['path']) for s in original['sources']} <= {Path(s['path']) for s in selections}
    prefix = 'cnn_mixed_5p_goodone_' if args.include_s3 else 'cnn_mixed_goodone_'
    output = ROOT/'models'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir()
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(ROOT/'semg/lda.py', output/'lda_source_snapshot.py')
    print('OUTPUT', output, flush=True)
    baseline = lda.train(
        selections, output/'lda',
        strategy='within_recording', train_percent=80, seed=20261010,
        variant=original['variant'], feature_settings=original['feature_config'], channels=[0, 1],
        progress=lambda message: print(message, flush=True))
    report = baseline['report']
    old = {r['fingerprint']: r for r in original['train']+original['test']}
    new = {r['fingerprint']: r for r in report['train']+report['test']}
    assert set(old) == {k for k, r in new.items() if r['subject'] in old_subjects} and len(old) == 1998
    assert len(new) == len(report['train']) + len(report['test'])
    assert sorted({r['subject'] for r in new.values()}) == subjects
    assert all(new[k]['features'] == old[k]['features'] for k in old)
    assert len([r for r in report['excluded'] if r['subject'] in old_subjects]) == len(original['excluded']) == 2
    counts = []
    group = lambda r: (r['subject'], r['recording'], r['block'])
    assert not {group(r) for r in report['train']} & {group(r) for r in report['test']}
    for subject in subjects:
        train = [r for r in report['train'] if r['subject'] == subject]
        test = [r for r in report['test'] if r['subject'] == subject]
        assert len({group(r) for r in train}) == 40
        assert len({group(r) for r in test}) == 10
        assert {r['word'] for r in train} == {r['word'] for r in test} == set(report['classes'])
        counts.append(dict(subject=subject, train=len(train), test=len(test), train_rounds=40, test_rounds=10,
                           train_by_word=dict(Counter(r['word'] for r in train)),
                           test_by_word=dict(Counter(r['word'] for r in test))))
    protocol = dict(evaluation='seen-subject within-recording mixed 80/20 whole-round holdout',
                    subjects=subjects, split_seed=20261010, counts=counts,
                    reference=str(reference), baseline=baseline['folder'],
                    original_four_1998_trials_and_features_unchanged=True, inventory=inventory,
                    split_note='Same whole-round splitting algorithm and seed on the current participant list; adding S03 can change later subjects test rounds.',
                    excluded=report['excluded'],
                    rounds_disjoint=True, test_used_for_selection=False)
    lda.write_json(output/'protocol.json', protocol)
    print('FROZEN_SPLIT', counts, flush=True)
    command = [sys.executable, '-u', str(ROOT/'experiments/train_small_cnn.py'),
               '--baseline', baseline['folder'], '--output', str(output/'cnn')]
    with (output/'training.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if child.wait() != 0:
            raise RuntimeError('CNN subprocess failed; see training.log')
    result = read(output/'cnn/report.json')
    assert result['final_train_trials'] == len(report['train'])
    assert result['test_trials'] == len(report['test'])
    assert result['model_roundtrip_verified'] and result['input_hashes_unchanged']
    assert result['protocol']['train_subjects'] == result['protocol']['test_subjects'] == subjects
    lda.write_json(output/'report.json', dict(protocol=protocol, cnn=result, lda_metrics=report['metrics']))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
