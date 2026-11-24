"""Deterministic, versioned identity for checkpointed MACE backends."""
import hashlib
import json
import re
from collections.abc import Mapping

import numpy as np


def _canonical(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic)) or (
        hasattr(value, 'shape') and hasattr(value, 'dtype')
    ):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError('Cannot fingerprint object arrays')
        return ['array', str(array.dtype), list(array.shape),
                hashlib.sha256(array.tobytes(order='C')).hexdigest()]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        pairs = [(_canonical(k), _canonical(v)) for k, v in value.items()]
        return ['mapping', sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_canonical(v) for v in value]]
    # MACE configuration can contain e3nn irreps, classes and torch dtypes.
    if isinstance(value, type) or (callable(value) and hasattr(value, '__qualname__')):
        return ['symbol', value.__module__, value.__qualname__]
    representation = repr(value)
    if re.search(r'0x[0-9a-fA-F]+', representation):
        raise ValueError(f'Cannot fingerprint unstable MACE configuration: {type(value)}')
    return ['repr', type(value).__module__, type(value).__qualname__, representation]


def fingerprint(*values):
    payload = json.dumps(_canonical(values), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return 'mace-v1:' + hashlib.sha256(payload).hexdigest()
