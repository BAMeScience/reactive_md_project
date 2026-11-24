import numpy as np
import pytest

from reactive_md.mace_identity import fingerprint


def test_fingerprint_is_order_independent_and_content_sensitive():
    weights = np.array([1., 2.], dtype=np.float32)
    a = fingerprint({'weights': weights, 'r_max': 5.})
    assert a == fingerprint({'r_max': 5., 'weights': weights.copy()})
    assert a != fingerprint({'weights': weights + 1, 'r_max': 5.})
    assert a != fingerprint({'weights': weights.astype(np.float64), 'r_max': 5.})
    assert a != fingerprint({'weights': weights.reshape(1, 2), 'r_max': 5.})
    assert a != fingerprint({'weights': weights, 'r_max': 6.})


def test_fingerprint_rejects_unstable_objects():
    with pytest.raises(ValueError, match='unstable MACE configuration'):
        fingerprint(object())
