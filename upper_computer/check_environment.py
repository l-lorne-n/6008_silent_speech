"""Read-only Windows/Python/dependency check; does not open a serial port."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-only', action='store_true', help='Check the base interpreter before setup')
    args = parser.parse_args()
    if sys.platform != 'win32' or struct.calcsize('P') != 8 or not (3, 12) <= sys.version_info[:2] < (3, 15):
        print('FAIL: Use 64-bit Windows Python 3.12 (recommended); supported setup range is 3.12-3.14.')
        return 1
    print(f'Python {sys.version.split()[0]} / 64-bit: {sys.executable}')
    if args.python_only:
        return 0
    modules = {'PySide6':'PySide6.QtWidgets', 'pyqtgraph':'pyqtgraph', 'pyserial':'serial',
               'numpy':'numpy', 'scipy':'scipy.signal', 'scikit-learn':'sklearn.discriminant_analysis'}
    errors = []
    for line in (Path(__file__).resolve().parent/'requirements.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        package, expected = line.split('==')
        try:
            actual = importlib.metadata.version(package)
            importlib.import_module(modules[package])
            print(f'{package}: {actual}')
            if actual != expected:
                errors.append(f'{package}: expected {expected}, found {actual}')
        except Exception as exc:
            errors.append(f'{package}: {exc}')
    if errors:
        print('FAIL: Run setup.cmd in this checkout. Details:')
        for error in errors:
            print('  '+error)
        return 1
    print('PASS: application dependencies import successfully. No serial port was opened.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
