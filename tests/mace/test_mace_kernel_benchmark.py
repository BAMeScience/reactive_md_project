from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from validation.benchmark_mace_kernels import benchmark_kernel, compare_reports, array_digest


def test_benchmark_kernel_synchronizes_and_compiles():
    result, value = benchmark_kernel(lambda x: jnp.sum(x*x), (jnp.ones(3),), 1, 2)
    assert float(value) == 3.
    assert len(result['samples_s']) == 2
    assert result['compile_s'] >= 0 and result['median_s'] >= 0
    assert result['dtype'] == str(value.dtype)


def reports():
    reference = dict(model_id='model', source='mp', variant='small-0b2',
                     positions_sha256='p', box_sha256='b', species_sha256='z',
                     neighbor_sha256='n', neighbor_shape=[2, 4], versions={},
                     backend='cpu', dtype='float64', capacity_multiplier=1.25,
                     dr_threshold=.5, device_kind=['CPU'], mode='legacy', energy_eV=1.,
                     forces_eV_per_A=[[1., -2., 0.]],
                     kernels={k: dict(dtype='float64', median_s=2.) for k in ('energy', 'force')})
    candidate = deepcopy(reference)
    candidate['mode'] = 'energy-only'
    candidate['kernels']['force']['median_s'] = 1.
    return reference, candidate


def compare(a, b):
    return compare_reports(a, b, energy_atol=1e-8, force_atol=1e-8, force_rtol=1e-10)


def test_comparison_reports_speed_and_checks_hashes():
    a, b = reports()
    result = compare(a, b)
    assert result['energy_abs_error_eV'] == 0 and result['force_max_abs_error_eV_per_A'] == 0
    assert result['median_speedup']['force'] == 2.
    assert array_digest(np.ones(3, np.float64)) != array_digest(np.ones(3, np.float32))


@pytest.mark.parametrize('field,value', [('neighbor_sha256','other'), ('model_id','other'),
                                         ('energy_eV', 1.1), ('forces_eV_per_A', [[1.,-2.,.001]])])
def test_comparison_rejects_inconsistent_inputs_or_outputs(field, value):
    a, b = reports()
    b[field] = value
    with pytest.raises((ValueError, AssertionError)):
        compare(a, b)
