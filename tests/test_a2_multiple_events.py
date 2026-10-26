"""Regression tests for multiple accepted A2 rate events.

Use the original kinetic model: each reacting atom translates by 0.25 A.
The initial pair separation is 2.65 A, so dissociation gives 3.15 A and
association gives 2.15 A. Relaxation is disabled to test pre-relaxation
geometry and topology. The force-field rebuild is real.
"""

from dataclasses import replace

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


class ControlledRNG:
    """Deterministic candidate order and acceptance draws."""

    def __init__(self, *, reverse=False, draws=()):
        self.reverse = reverse
        self.draws = iter(draws)

    def permutation(self, n):
        order = np.arange(n)
        return order[::-1] if self.reverse else order

    def random(self):
        return float(next(self.draws))


def make_case(monkeypatch, bonded_pairs):
    def no_relaxation(key, positions, velocities, *, ff, **kwargs):
        return key, positions, velocities, ff.nlist

    monkeypatch.setattr(a2_rate, "langevin_relax_with_nlist", no_relaxation)
    # Diagnostic energy logging is not used by the production rate cycle.
    # Only relaxation is replaced; topology, force-field rebuild, and
    # coordinate translation remain under test.

    R, box = make_positions()
    R = R.at[1].set(R[0] + jnp.array([2.65, 0.0, 0.0]))
    R = R.at[3].set(R[2] + jnp.array([2.65, 0.0, 0.0]))

    sys = make_system(len(R))
    pairs = np.asarray(bonded_pairs, dtype=np.int32).reshape(-1, 2)
    molecule_id = np.arange(len(R), dtype=np.int32)
    for i, j in pairs:
        molecule_id[j] = molecule_id[i]

    sys = replace(
        sys,
        bonds=(
            pairs,
            np.full(len(pairs), K_BOND_JAX, dtype=np.float32),
            np.full(len(pairs), 1.5, dtype=np.float32),
        ),
        molecule_id=molecule_id,
    )
    ff = make_forcefield(R, box, sys)
    disp_fn, shift_fn = space.periodic(box)
    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        k_bond_jax=K_BOND_JAX,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

    def run(*, assoc_rate, dissoc_rate, rng):
        return a2_rate.a2_rate_reactive_cycle(
            key=jax.random.PRNGKey(1),
            R=R,
            velocities=jnp.zeros_like(R),
            box=box,
            system=sys,
            ff=ff,
            reaction=reaction,
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            masses=jnp.ones(len(R), dtype=jnp.float32),
            association_rate_ps=assoc_rate,
            dissociation_rate_ps=dissoc_rate,
            reactive_interval_ps=1.0,
            temperature_k=188.0,
            kb_real=KB_REAL,
            rng=rng,
            relaxation_time_ps=1.0,
        )

    return R, disp_fn, run


def bond_pairs(system):
    return {tuple(sorted(map(int, pair))) for pair in np.asarray(system.bonds[0])}


def distance(R, disp_fn, i, j):
    return float(jnp.linalg.norm(disp_fn(R[i], R[j])))


def excluded(system, i, j):
    mask = neighbor.make_exclusion_mask(
        len(system.molecule_id),
        np.asarray(system.bonds[0]),
        np.asarray(system.angles[0]),
        np.asarray(system.molecule_id),
    )
    return bool(np.asarray(mask)[i, j])


def test_two_dissociations_translate_both_pairs(monkeypatch):
    R, disp_fn, run = make_case(monkeypatch, [(0, 1), (2, 3)])
    result = run(assoc_rate=0.0, dissoc_rate=1.0,
                 rng=ControlledRNG(draws=[0.0, 0.0]))

    assert len(result.events) == 2
    assert all(e.reaction_type == "dissociation" for e in result.events)
    assert bond_pairs(result.system) == set()
    for i, j in [(0, 1), (2, 3)]:
        np.testing.assert_allclose(distance(result.positions, disp_fn, i, j), 3.15, atol=1e-5)
        assert not excluded(result.system, i, j)
    assert distance(R, disp_fn, 0, 1) < 3.15
    assert distance(R, disp_fn, 2, 3) < 3.15


def test_mixed_events_translate_both_accepted_pairs(monkeypatch):
    R, disp_fn, run = make_case(monkeypatch, [(0, 1)])
    # Reverse candidate order: dissociation is accepted BEFORE association,
    # so the last loop variable is "association" in the original buggy code.
    result = run(assoc_rate=1.0, dissoc_rate=1.0,
                 rng=ControlledRNG(reverse=True, draws=[0.0, 0.0]))

    assert {e.reaction_type for e in result.events} == {"association", "dissociation"}
    assert bond_pairs(result.system) == {(2, 3)}
    np.testing.assert_allclose(distance(result.positions, disp_fn, 0, 1), 3.15, atol=1e-5)
    np.testing.assert_allclose(distance(result.positions, disp_fn, 2, 3), 2.15, atol=1e-5)
    assert not excluded(result.system, 0, 1)
    assert excluded(result.system, 2, 3)


def test_rejected_last_dissociation_does_not_translate(monkeypatch):
    R, disp_fn, run = make_case(monkeypatch, [(0, 1)])
    # Association (2,3) accepted; dissociation (0,1) examined last but rejected.
    result = run(assoc_rate=1.0, dissoc_rate=0.0,
                 rng=ControlledRNG(draws=[0.0, 0.5]))

    assert [e.reaction_type for e in result.events] == ["association"]
    assert bond_pairs(result.system) == {(0, 1), (2, 3)}
    # Accepted association moves only (2,3); rejected dissociation leaves
    # the originally bonded (0,1) pair at its initial coordinates.
    np.testing.assert_allclose(np.asarray(result.positions[:2]), np.asarray(R[:2]))
    np.testing.assert_allclose(distance(result.positions, disp_fn, 2, 3), 2.15, atol=1e-5)
    assert excluded(result.system, 0, 1)
    assert excluded(result.system, 2, 3)


def test_accepted_first_dissociation_rejected_last(monkeypatch):
    R, disp_fn, run = make_case(monkeypatch, [(0, 1), (2, 3)])
    # Only (0,1) accepted; the last candidate (2,3) is rejected.
    result = run(assoc_rate=0.0, dissoc_rate=0.5,
                 rng=ControlledRNG(draws=[0.0, 0.9]))

    assert len(result.events) == 1
    assert (result.events[0].i, result.events[0].j) == (0, 1)
    assert bond_pairs(result.system) == {(2, 3)}
    np.testing.assert_allclose(distance(result.positions, disp_fn, 0, 1), 3.15, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result.positions[2:4]), np.asarray(R[2:4]))
    assert not excluded(result.system, 0, 1)
    assert excluded(result.system, 2, 3)
