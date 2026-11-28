"""Cache ownership and invalidation, independent of the real integrator."""
import jax.numpy as jnp
import pytest
from reactive_md import relaxation


@pytest.mark.parametrize('change', ['energy', 'neighbor', 'shift', 'dtype', 'dt', 'kT', 'gamma'])
def test_cache_rebuilds_when_backend_or_integrator_settings_change(monkeypatch, change):
    calls = []
    def build(*args):
        calls.append(args)
        assert args[0] is None  # No classical trial FF captured.
        return object(), object(), object()
    monkeypatch.setattr(relaxation, '_build_langevin_kernels', build)
    cache = relaxation.LangevinKernelCache()
    settings = dict(energy_fn=object(), neighbor_fn=object(), shift_fn=object(),
                    dtype=jnp.float32, dt=.00025, kT=.8, gamma=100.)
    first = cache.get(**settings)
    assert cache.get(**settings) is first
    key = {'energy': 'energy_fn', 'neighbor': 'neighbor_fn', 'shift': 'shift_fn'}.get(change, change)
    if change in ('energy', 'neighbor', 'shift'):
        settings[key] = object()
    elif change == 'dtype':
        settings[key] = jnp.float16
    else:
        settings[key] *= 2
    second = cache.get(**settings)
    assert first is not second and len(calls) == 2
    assert cache.get(**settings) is second
    assert relaxation.LangevinKernelCache().get(**settings) is not second
    assert len(calls) == 3
