"""Verify published raw bytes and both models' historical data inputs, offline."""
import hashlib
import json
from pathlib import Path, PureWindowsPath


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    data = Path(__file__).resolve().parent
    root = data.parents[1]
    manifest = json.loads((data / 'DATASET_MANIFEST.json').read_text(encoding='utf-8'))
    mapping = json.loads((data / 'MODEL_DATASETS.json').read_text(encoding='utf-8'))
    errors, hashes, checked = [], {}, {}
    for relative, expected in manifest['files'].items():
        path = data / relative
        if not path.is_file():
            errors.append(f'missing: {relative}')
            continue
        hashes[relative] = digest(path)
        if hashes[relative] != expected['sha256'] or path.stat().st_size != expected['bytes']:
            errors.append(f'data mismatch: {relative}')
    for model in mapping['models']:
        inputs = json.loads((root / model['run'] / 'inputs_sha256.json').read_text(encoding='utf-8'))
        directories = {row['recording']: root / row['directory'] for row in model['recordings']}
        count = 0
        for original, expected in inputs.items():
            parts = PureWindowsPath(original).parts
            if 'recordings' not in parts:
                continue
            recording, filename = parts[-2:]
            directory = directories.get(recording)
            relative = (directory / filename).relative_to(data).as_posix() if directory else None
            if hashes.get(relative) != expected:
                errors.append(f'historical input mismatch: {model["model_id"]}: {original}')
            count += 1
        checked[model['model_id']] = count
        if count != model['historical_input_files_verified']:
            errors.append(f'unexpected input count: {model["model_id"]}: {count}')
    replacement = json.loads((data / 'S013_REPLACEMENT_20261027.json').read_text(encoding='utf-8'))
    correction_count = 0
    for field, folder in [('target_before_sha256', 'S013/session_001'),
                          ('target_after_sha256', 'goodone_plus/s013')]:
        for relative, expected in replacement[field].items():
            name = (Path(folder) / Path(*PureWindowsPath(relative).parts)).as_posix()
            if hashes.get(name) != expected.lower():
                errors.append(f'S013 replacement evidence mismatch: {name}')
            correction_count += 1
    print(json.dumps({'data_files_checked': len(hashes), 'model_inputs_checked': checked,
                      's013_correction_files_checked': correction_count, 'errors': errors},
                     ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
