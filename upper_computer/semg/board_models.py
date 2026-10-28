"""Versioned board model sources shared by export, monitor and media bridge.

No ML dependencies: importing the registry never trains or deploys a model.
Existing identifiers retain their original frozen exports.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_MODEL = 'mid64_temporal3_20261027'
_SWEEP = Path('models/sliding_arch_sweep_20261022_125638_517426')
MODEL_EXPORTS = {
    'mid64_temporal3': _SWEEP / 'mid64_temporal3/export',
    NEW_MODEL: Path('models/cnn_goodone_current_20261027_214651_552433/mid64_temporal3/export'),
    'current_base': Path('models/sliding_rest_20261022_103406/export'),
    **{name: _SWEEP / name / 'export' for name in (
        'baseline', 'tail64', 'mid64', 'wide2x', 'temporal3', 'pool2',
        'mid64_pool2', 'mid64_temporal3_pool2')},
}
NAMES = list(MODEL_EXPORTS)


def model_source(name, root=None):
    """Resolve only registered IDs; never turn a serial string into a path."""
    if name not in MODEL_EXPORTS:
        raise ValueError(f'Unknown board model: {name!r}')
    return (ROOT if root is None else Path(root)) / MODEL_EXPORTS[name]
