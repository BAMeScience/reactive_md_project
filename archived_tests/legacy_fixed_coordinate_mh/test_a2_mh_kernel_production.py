"""HPC-only production MH checks; requires JAX-MD and the moving pilot files."""

import pytest

from validation.a2_mh_moving_reference import TwoAtomModel
from validation.run_a2_mh_kernel_audit import audit_case


@pytest.mark.parametrize('r,periodic_edge', [(2.0, False), (3.05, False)])
def test_real_two_atom_kernel(r, periodic_edge):
    result = audit_case(TwoAtomModel(), r, periodic_edge=periodic_edge)
    assert result['status'] == 'PASS'
