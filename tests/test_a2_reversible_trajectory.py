"""Integration checks for the reversible A2 NVT demonstration.

Place in tests/test_a2_reversible_trajectory.py.
"""

import numpy as np

from scripts.run_a2_reversible import simulate_case, check_topology_and_lj


def test_forced_dissociation_restores_free_pair():
    result, _, _ = simulate_case(mode="dissociation", steps=1)
    assert result.steps_done == 1
    assert result.population_history == ((0, 1), (1, 0))
    assert [record.event.reaction_type for record in result.event_history] == [
        "dissociation"
    ]
    check_topology_and_lj(result, expect_no_bonds=True)
    assert np.isfinite(result.samples[-1].potential_energy_kcal_mol)
    assert np.isfinite(result.samples[-1].kinetic_energy_kcal_mol)


def test_association_then_dissociation_roundtrip():
    result, _, _ = simulate_case(mode="roundtrip", steps=2)
    assert result.steps_done == 2
    assert result.population_history == ((0, 0), (1, 1), (2, 0))
    assert [record.event.reaction_type for record in result.event_history] == [
        "association", "dissociation"
    ]
    check_topology_and_lj(result, expect_no_bonds=True)
    assert all(np.isfinite(s.potential_energy_kcal_mol) for s in result.samples)
    assert all(np.isfinite(s.kinetic_energy_kcal_mol) for s in result.samples)

