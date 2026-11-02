"""Regression test for A + A -> A2 -> A + A topology and LJ exclusions.

Place in the repository's tests/ directory and run from the repository root:
    python -m pytest -q tests/test_a2_topology_roundtrip.py

Uses the production A2 rate cycle and real force-field rebuilds. The Langevin
relaxation is replaced with an identity operation so this is a topology test,
not a molecular-dynamics or equilibration test.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from jax_md.mm_forcefields import neighbor

from reactive_md.reactions import a2_rate
from reactive_md.reactions.a2 import A2Reaction
from scripts.run_a2_small import (
    KB_REAL,
    K_BOND_JAX,
    make_forcefield,
    make_positions,
    make_system,
)


def exclusion_mask(system):
    return np.asarray(
        neighbor.make_exclusion_mask(
            len(system.molecule_id),
            np.asarray(system.bonds[0]),
            np.asarray(system.angles[0]),
            np.asarray(system.molecule_id),
        )
    )


def potential_energy(ff, positions):
    nlist = ff.neighbor_fn.allocate(positions)
    assert not bool(np.asarray(nlist.did_buffer_overflow))
    return float(ff.energy_fn(positions, nlist)["total"])


def test_association_dissociation_restores_lj(monkeypatch):
    # Do not integrate or attempt reactions during relaxation. Preserve the
    # newly rebuilt force field and the positions returned by the reaction.
    def no_relaxation(key, positions, velocities, *, ff, **kwargs):
        return key, positions, velocities, ff.nlist

    monkeypatch.setattr(a2_rate, "langevin_relax_with_nlist", no_relaxation)

    R, box = make_positions()
    disp_fn, shift_fn = space.periodic(box)
    system_0 = make_system(len(R))
    ff_0 = make_forcefield(R, box, system_0)
    reaction = A2Reaction(
        a_indices=np.array([0, 1], dtype=np.int32),
        k_bond_jax=K_BOND_JAX,
        r0_bond=1.5,
        association_cutoff=2.7,
    )
    key = jax.random.PRNGKey(1)
    rng = np.random.default_rng(1)
    velocities = jnp.zeros_like(R)
    masses = jnp.ones((len(R),), dtype=jnp.float32)

    def cycle(positions, velocities, ff, system, *, assoc_rate, dissoc_rate):
        return a2_rate.a2_rate_reactive_cycle(
            key=key,
            R=positions,
            velocities=velocities,
            box=box,
            system=system,
            ff=ff,
            reaction=reaction,
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            masses=masses,
            association_rate_ps=assoc_rate,
            dissociation_rate_ps=dissoc_rate,
            reactive_interval_ps=1.0,
            temperature_k=188.0,
            kb_real=KB_REAL,
            rng=rng,
            relaxation_time_ps=1.0,
        )

    # Initially, the free pair must interact through LJ.
    assert np.asarray(system_0.bonds[0]).shape == (0, 2)
    assert system_0.molecule_id[0] != system_0.molecule_id[1]
    assert not exclusion_mask(system_0)[0, 1]

    # Force precisely one association (p_assoc=1, p_diss=0).
    associated = cycle(R, velocities, ff_0, system_0, assoc_rate=1.0, dissoc_rate=0.0)
    assert len(associated.events) == 1
    assert associated.events[0].reaction_type == "association"
    np.testing.assert_array_equal(np.asarray(associated.system.bonds[0]), [[0, 1]])
    ids_associated = np.asarray(associated.system.molecule_id)
    assert ids_associated[0] == ids_associated[1]
    assert exclusion_mask(associated.system)[0, 1]
    assert exclusion_mask(associated.system)[1, 0]
    assert associated.ff is not ff_0

    # Force precisely one dissociation (p_assoc=0, p_diss=1).
    dissociated = cycle(
        associated.positions,
        associated.velocities,
        associated.ff,
        associated.system,
        assoc_rate=0.0,
        dissoc_rate=1.0,
    )
    assert len(dissociated.events) == 1
    assert dissociated.events[0].reaction_type == "dissociation"
    assert np.asarray(dissociated.system.bonds[0]).shape == (0, 2)
    ids_dissociated = np.asarray(dissociated.system.molecule_id)
    assert ids_dissociated[0] != ids_dissociated[1]
    assert not exclusion_mask(dissociated.system)[0, 1]
    assert not exclusion_mask(dissociated.system)[1, 0]
    assert dissociated.ff is not associated.ff

    # Energy-based check of the ACTUAL rebuilt force fields, at identical
    # coordinates. At r=1.5 A, an unexcluded sigma=3.4 A LJ pair is highly
    # repulsive. The bonded field should exclude it; the dissociated field
    # should reproduce the original unbonded potential.
    probe = R.at[1].set(R[0] + jnp.array([1.5, 0.0, 0.0]))
    energy_free_before = potential_energy(ff_0, probe)
    energy_bonded = potential_energy(associated.ff, probe)
    energy_free_after = potential_energy(dissociated.ff, probe)

    assert energy_free_before - energy_bonded > 1000.0
    np.testing.assert_allclose(energy_free_after, energy_free_before, rtol=1e-5, atol=1e-3)

    print("A+A -> A2 -> A+A: bond, molecule IDs, LJ mask, and LJ energy PASS")
