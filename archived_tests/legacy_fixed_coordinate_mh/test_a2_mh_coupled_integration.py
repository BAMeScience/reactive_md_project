"""Real JAX-MD NVT + production A2 MH integration tests (run on HPC)."""
import pytest

from validation.run_a2_mh_coupled import check_replay, run_case


@pytest.mark.parametrize('initial_dimers', [0, 2])
def test_coupled_md_mh_invariants(initial_dimers):
    result = run_case(initial_dimers=initial_dimers, steps=12, check_every=1,
                      seed=31)
    assert result['checks'] == 12
    assert result['accepted'] + result['rejected'] + result['null'] == 12


def test_coupled_md_mh_reproducible():
    first = run_case(initial_dimers=2, steps=8, check_every=2, seed=13)
    second = run_case(initial_dimers=2, steps=8, check_every=2, seed=13)
    check_replay(first, second)
