"""Independently audit saved mixed CNN splits, membership, and metrics."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.cnn-deps'))
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    folder = parser.parse_args().run.resolve()
    protocol = read(folder/'protocol.json')
    report = read(folder/'cnn/report.json')
    baseline = read(Path(protocol['baseline'])/'report.json')
    with (folder/'cnn/split_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest = list(csv.DictReader(f))
    sets = {s: {r['fingerprint'] for r in manifest if r['split'] == s} for s in ['train', 'validation', 'test']}
    total = len(baseline['train']) + len(baseline['test'])
    assert sum(map(len, sets.values())) == len(manifest) == total
    assert len(set.union(*sets.values())) == total
    groups = {s: {(r['subject'], r['recording'], r['block']) for r in manifest if r['split'] == s} for s in sets}
    assert not groups['train'] & groups['validation']
    assert not (groups['train'] | groups['validation']) & groups['test']
    assert sets['test'] == {r['fingerprint'] for r in baseline['test']}
    checkpoint = torch.load(folder/'cnn/model.pt', map_location='cpu', weights_only=True)
    assert set(checkpoint['training_fingerprints']) == sets['train'] | sets['validation'] == {r['fingerprint'] for r in baseline['train']}
    assert len(checkpoint['training_fingerprints']) == checkpoint['normalization']['fit_trials'] == len(baseline['train'])
    assert checkpoint['selected'] == report['selected']
    assert checkpoint['protocol'] == report['protocol']
    assert report['protocol']['train_subjects'] == report['protocol']['test_subjects'] == protocol['subjects']
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    with (folder/'cnn/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        predictions = list(csv.DictReader(f))
    identity = lambda r: (r['subject'], r['recording'], int(r['trial_id']))
    expected = {identity(r): r['word'] for r in baseline['test']}
    assert len(predictions) == len({identity(r) for r in predictions}) == len(expected)
    assert {identity(r): r['word'] for r in predictions} == expected
    classes = checkpoint['classes']
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for r in predictions:
        assert int(r['correct']) == (r['word'] == r['prediction'])
        assert int(r['lda_correct']) == (r['word'] == r['lda_prediction'])
        matrix[classes.index(r['word']), classes.index(r['prediction'])] += 1
    assert matrix.tolist() == report['metrics']['confusion_matrix']
    assert int(matrix.trace()) == report['metrics']['correct']
    assert sum(int(r['lda_correct']) for r in predictions) == report['baseline_metrics']['correct']
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    macro_f1 = float(np.mean(np.divide(2*matrix.diagonal(), denominator, out=np.zeros(len(classes)), where=denominator > 0)))
    assert abs(macro_f1 - report['metrics']['macro_f1']) < 1e-12
    for g in report['group_metrics']['subject']:
        rows = [r for r in predictions if r['subject'] == g['subject']]
        assert len(rows) == g['test_total']
        assert sum(int(r['correct']) for r in rows) == g['metrics']['correct']
        planned = next(c for c in protocol['counts'] if c['subject'] == g['subject'])
        assert g['train_total'] == planned['train'] and g['test_total'] == planned['test']
        for key, round_key in [('test', 'test_rounds'), ('train', 'train_rounds')]:
            group_set = groups['test'] if key == 'test' else groups['train'] | groups['validation']
            assert sum(s == g['subject'] for s, _, _ in group_set) == planned[round_key]
    audit = dict(status='passed', trials=total, final_train=len(baseline['train']), test=len(expected),
                 whole_rounds_disjoint=True, checkpoint_training_membership_verified=True,
                 stored_predictions_labels_and_confusion_matrix_verified=True,
                 per_subject_counts_verified=True, macro_f1_recomputed=True,
                 model_roundtrip_verified_by_training_runner=report['model_roundtrip_verified'],
                 source_hashes_verified_by_training_runner=report['input_hashes_unchanged'])
    assert audit['model_roundtrip_verified_by_training_runner'] and audit['source_hashes_verified_by_training_runner']
    (folder/'saved_outputs_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
