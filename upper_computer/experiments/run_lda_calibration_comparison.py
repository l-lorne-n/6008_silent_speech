"""Reproduce the two frozen 2026-10-29 offline LDA calibration experiments.

Run from any working directory. All input paths are relative to this repository.
No board, App, neural-network training, or per-subject normalization is involved.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(__file__).resolve().parent / 'calibration_data_20261029'
METHODS = ('fixed50_x40', 'pooled_k3_x10')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(sources):
    for source in sources:
        for name, expected in source['hashes'].items():
            path = ROOT / source['path'] / name
            require(path.is_file() and digest(path) == expected, f'Source changed/missing: {path}')


def metrics(truth, pred):
    return dict(correct=int(np.sum(truth == pred)), total=len(truth),
                accuracy=float(np.mean(truth == pred)),
                macro_f1=float(f1_score(truth, pred, labels=list(range(10)), average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(truth, pred, labels=list(range(10))).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--reextract', action='store_true', help='Also re-extract and verify all features from the 49 raw recordings.')
    parser.add_argument('--output', type=Path, help='New output directory; must not exist.')
    args = parser.parse_args()
    manifest = read(BUNDLE / 'manifest.json')
    for name, expected in manifest['files'].items():
        require(digest(BUNDLE / name) == expected, f'Frozen bundle changed: {name}')
    protocol = read(BUNDLE / 'protocol.json')
    rows = read(BUNDLE / 'trials.json')
    sources = read(BUNDLE / 'sources.json')
    verify_sources(sources)
    x = np.load(BUNDLE / 'features.npy', allow_pickle=False)
    require(x.shape == (2496, 48) and np.isfinite(x).all(), 'Unexpected feature matrix')
    require(len({r['fingerprint'] for r in rows}) == len(rows) == len(x), 'Duplicate/missing trial identity')
    reextracted = False
    if args.reextract:
        sys.path.insert(0, str(ROOT))
        from semg import lda
        extracted, excluded, _ = lda.extract(
            [dict(path=str(ROOT / s['path']), role='train') for s in sources],
            protocol['variant'], config=protocol['feature_config'], channels=[0, 1],
            progress=lambda message: print(message, flush=True))
        extracted.sort(key=lambda r: (r['subject'], r['recording'], r['trial_id']))
        require([r['fingerprint'] for r in extracted] == [r['fingerprint'] for r in rows], 'Re-extracted trial IDs changed')
        np.testing.assert_array_equal(np.array([r['features'] for r in extracted]), x)
        reextracted = True
        print('RAW FEATURE CHECK PASSED', len(extracted), 'excluded', len(excluded), flush=True)
    output = args.output or ROOT / 'models' / f'lda_calibration_release_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    methods = METHODS if args.method == 'all' else (args.method,)
    all_results = {}
    for method in methods:
        fixed = method == 'fixed50_x40'
        repeat = 40 if fixed else 10
        classes = protocol['fixed50_classes'] if fixed else sorted({r['word'] for r in rows})
        splits = read(BUNDLE / ('fixed50_splits.json' if fixed else 'pooled_k3_splits.json'))
        y = np.array([classes.index(r['word']) for r in rows])
        with (BUNDLE / f'{method}_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            reference = {r['fingerprint']: r['prediction'] for r in csv.DictReader(f)}
        predictions, folds = [], []
        dest = output / method
        dest.mkdir()
        for subject in protocol['subjects']:
            sp = splits[subject]
            if fixed:
                src, cal, test = sp['source_train'], sp['calibration'], sp['test']
                require(len(cal) == 50 and Counter(rows[i]['word'] for i in cal) == Counter({c: 5 for c in classes}), 'Fixed50 balance changed')
                require(not {rows[i]['recording'] for i in cal} & {rows[i]['recording'] for i in test}, 'Fixed50 recording overlap')
            else:
                src = [i for i, r in enumerate(rows) if r['subject'] != subject]
                cal = [i for group in sp.values() for i in group['calibration']]
                test = [i for group in sp.values() for i in group['test']]
                for group in sp.values():
                    require(Counter(rows[i]['word'] for i in group['calibration']) == Counter({c: 3 for c in classes}), 'K3 balance changed')
                    for word in classes:
                        word_ids = sorted([i for i in group['calibration'] + group['test'] if rows[i]['word'] == word], key=lambda i: rows[i]['prepare'])
                        require(set(word_ids[:3]) == {i for i in group['calibration'] if rows[i]['word'] == word}, 'K3 selection changed')
            require(len(src + cal + test) == len(rows) and set(src + cal + test) == set(range(len(rows))), 'Split overlap or omission')
            require({rows[i]['subject'] for i in cal + test} == {subject} and subject not in {rows[i]['subject'] for i in src}, 'Subject overlap')
            require(not {f for i in src + cal for f in rows[i]['signal_fingerprints']} & {f for i in test for f in rows[i]['signal_fingerprints']}, 'Raw signal overlap')
            fit = src + list(np.repeat(cal, repeat))
            scaler = StandardScaler().fit(x[fit])
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(scaler.transform(x[fit]), y[fit])
            pred = model.predict(scaler.transform(x[test]))
            state = dict(schema='semg_lda_offline_calibration_release_v1', method=method, subject=subject,
                         classes=classes, repeat=repeat, source_train_indices=src, calibration_indices=cal, test_indices=test,
                         mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=model.coef_.tolist(), intercept=model.intercept_.tolist())
            model_path = dest / f'{subject}_model.json'
            write(model_path, state)
            saved = read(model_path)
            restored = (((x[test] - saved['mean']) / saved['scale']) @ np.array(saved['coef']).T + saved['intercept']).argmax(1)
            np.testing.assert_array_equal(pred, restored)
            # Independent moment check: unique source trials weight 1, calibration trials weight repeat.
            unique = src + cal
            weights = np.array([1] * len(src) + [repeat] * len(cal))
            mean = np.average(x[unique], axis=0, weights=weights)
            var = np.average((x[unique] - mean) ** 2, axis=0, weights=weights)
            np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(scaler.var_, var, rtol=1e-10, atol=1e-10)
            for i, p in zip(test, pred):
                row = rows[i]
                require(reference[row['fingerprint']] == classes[int(p)], f'Historical prediction mismatch: {method}/{row["fingerprint"]}')
                predictions.append(dict(subject=subject, recording=row['recording'], trial_id=row['trial_id'],
                                        fingerprint=row['fingerprint'], word=row['word'], prediction=classes[int(p)], correct=int(y[i] == p)))
            result = dict(subject=subject, source_train_count=len(src), unique_calibration_count=len(cal),
                          calibration_rows=len(cal) * repeat, training_rows=len(fit), unique_training_count=len(unique), metrics=metrics(y[test], pred))
            folds.append(result)
            print(method, subject, result['metrics']['correct'], '/', len(test), flush=True)
        require(set(reference) == {p['fingerprint'] for p in predictions} and len(reference) == len(predictions), 'Reference test set mismatch')
        pooled = metrics(np.array([classes.index(r['word']) for r in predictions]), np.array([classes.index(r['prediction']) for r in predictions]))
        expected = manifest['expected'][method]
        require((pooled['correct'], pooled['total']) == (expected['correct'], expected['total']), 'Aggregate mismatch')
        with (dest / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        result = dict(method=method, classes=classes, folds=folds, pooled=pooled)
        write(dest / 'report.json', result)
        all_results[method] = result
        print(method, 'TOTAL', pooled['correct'], '/', pooled['total'], f"{pooled['accuracy']:.4%}", flush=True)
    verify_sources(sources)
    write(output / 'completed.json', dict(passed=True, methods=list(methods), numpy_version=np.__version__, sklearn_version=sklearn.__version__,
          bundle_manifest_sha256=digest(BUNDLE / 'manifest.json'), raw_sources_verified=True, raw_features_reextracted=reextracted,
          exact_reference_predictions=True, saved_models_verified=True, weighted_scaler_moments_verified=True,
          results={k: v['pooled'] for k, v in all_results.items()}, limitations=protocol['limitations']))
    print('OUTPUT', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
