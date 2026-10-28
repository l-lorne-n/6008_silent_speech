"""Verify the two published model bundles and four HEX files without hardware."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'upper_computer/models/ASSET_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file():
            errors.append(f'missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'checksum mismatch: {name}')
    print(json.dumps({'checked_files': len(manifest['files']), 'models': manifest['models'], 'errors': errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
