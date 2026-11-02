"""Exercise the *production* A2 MH cycle against an independent exact model.

Requires the HPC environment with JAX-MD. No production modules are patched;
only the expensive force-field constructor is replaced in the 10-state
kernel test by a deterministic topology-energy fake. A separate test exercises
actual JAX-MD force-field construction and energy offsets.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.forcefield import build_forcefield
from reactive_md.reaction import SystemState, build_trial_forcefield
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_rate import make_a2_system_state_from_trial
from reactive_md.reactions import a2_metropolis as mh
from validation.a2_mh_exact_reference import STATES, energy, free_pairs


POSITIONS = jnp.asarray([[0., 0., 0.], [2., 0., 0.],
                         [0., 2., 0.], [2., 2., 0.]], dtype=jnp.float32)
BOX = jnp.asarray([30., 30., 30.], dtype=jnp.float32)
DISP = lambda a, b: a - b


def empty_system():
    bonds = (jnp.empty((0, 2), dtype=jnp.int32),
             jnp.empty((0,), dtype=jnp.float32),
             jnp.empty((0,), dtype=jnp.float32))
    angles = (jnp.empty((0, 3), dtype=jnp.int32),
              jnp.empty((0,), dtype=jnp.float32),
              jnp.empty((0,), dtype=jnp.float32))
    four = (jnp.empty((0, 4), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32),
            jnp.empty((0,), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32))
    return SystemState(
        bonds=bonds, angles=angles, torsions=four, impropers=four,
        charges=jnp.zeros(4), sigmas=jnp.ones(4) * 1.2,
        epsilons=jnp.ones(4) * 0.2,
        molecule_id=jnp.arange(4), pf6_reacted=jnp.zeros((0,), dtype=bool),
    )


def reaction_model(offset=0.0):
    return A2Reaction(a_indices=np.arange(4), k_bond_jax=1.5,
                      r0_bond=2.0, association_cutoff=3.0,
                      bond_energy_offset=offset)


def matching(system):
    return tuple(sorted(tuple(sorted(map(int, pair)))
                        for pair in np.asarray(system.bonds[0]).reshape(-1, 2)))


def make_state(state, reaction):
    system = empty_system()
    for pair in state:
        candidate = next(c for c in reaction.find_association_candidates(
            POSITIONS, DISP, system=system) if (c.i, c.j) == pair)
        trial = reaction.build_association_trial(system, candidate)
        trial = mh._finalize_trial(trial, reaction, 4)
        system = make_a2_system_state_from_trial(trial, system)
    return system


class FakeNeighbor:
    def allocate(self, positions, **kwargs):
        return SimpleNamespace(did_buffer_overflow=False)

    def update(self, positions, nlist):
        return nlist


class FakeFF:
    def __init__(self, state):
        self.state = state
        self.neighbor_fn = FakeNeighbor()
        self.nlist = SimpleNamespace(did_buffer_overflow=False)

    def energy_fn(self, positions, nlist):
        return {'total': energy(self.state)}


class ForcedRNG:
    def __init__(self, *, association, index):
        self.association = association
        self.index = index
        self.calls = 0

    def random(self):
        self.calls += 1
        if self.calls == 1:
            return 0.0 if self.association else 0.999999
        return 0.0  # accept whenever the probability is strictly positive

    def integers(self, high):
        assert 0 <= self.index < high
        return self.index


def test_production_mh_all_24_directed_edges(monkeypatch):
    """Check candidate counts, proposal ratio, acceptance and actual states."""
    def fake_trial_ff(R, box, trial, ff):
        return FakeFF(tuple(sorted(tuple(sorted(map(int, pair)))
                                   for pair in np.asarray(trial['bonds'][0]).reshape(-1, 2))))

    monkeypatch.setattr(mh, 'build_trial_forcefield', fake_trial_ff)
    reaction = reaction_model()
    p_a, kT = 0.37, 1.0
    edges = 0
    for state in STATES:
        system = make_state(state, reaction)
        ff = FakeFF(state)
        assoc = reaction.find_association_candidates(POSITIONS, DISP, system=system)
        diss = reaction.find_dissociation_candidates(POSITIONS, DISP, system=system)
        assert len(assoc) == len(free_pairs(state))
        assert len(diss) == len(state)
        for association, candidates in ((True, assoc), (False, diss)):
            for index, candidate in enumerate(candidates):
                rng = ForcedRNG(association=association, index=index)
                result = mh.a2_metropolis_reactive_cycle(
                    key=None, R=POSITIONS, velocities=jnp.zeros_like(POSITIONS),
                    box=BOX, system=system, ff=ff, reaction=reaction,
                    disp_fn=DISP, temperature_k=1.0, kb_real=kT,
                    rng=rng, association_proposal_probability=p_a,
                )
                target = tuple(sorted((*state, (candidate.i, candidate.j)))) if association \
                    else tuple(pair for pair in state if set(pair) != {candidate.i, candidate.j})
                assert target in STATES
                reverse_count = len(target) if association else len(free_pairs(target))
                forward_count = len(candidates)
                q_f = (p_a if association else 1.0 - p_a) / forward_count
                q_r = (1.0 - p_a if association else p_a) / reverse_count
                expected_ratio = math.log(q_r / q_f)
                expected_delta = energy(target) - energy(state)
                expected_p = math.exp(min(0.0, -expected_delta / kT + expected_ratio))
                assert len(result.attempts) == 1
                attempt = result.attempts[0]
                assert attempt.accepted
                assert attempt.probability == pytest.approx(expected_p, abs=1e-12)
                assert attempt.log_proposal_ratio == pytest.approx(expected_ratio, abs=1e-12)
                assert attempt.delta_energy == pytest.approx(expected_delta, abs=1e-12)
                assert matching(result.system) == target
                assert result.positions is POSITIONS
                assert len(result.events) == 1
                edges += 1
    assert edges == 24


def test_unavailable_reverse_is_rejected_without_forcefield_build(monkeypatch):
    reaction = reaction_model()
    system = make_state(((0, 1),), reaction)
    positions = POSITIONS.at[1].set(jnp.asarray([4., 0., 0.]))
    # All four atoms are used, but only the bonded (0,1) is selected below.
    # Force dissociation of that bond; the reverse association is outside 3 A.
    def should_not_build(*args, **kwargs):
        raise AssertionError('No energy evaluation is allowed for a zero reverse proposal')

    monkeypatch.setattr(mh, 'build_trial_forcefield', should_not_build)
    result = mh.a2_metropolis_reactive_cycle(
        key=None, R=positions, velocities=jnp.zeros_like(positions), box=BOX,
        system=system, ff=FakeFF(((0, 1),)), reaction=reaction,
        disp_fn=DISP, temperature_k=1.0, kb_real=1.0,
        rng=ForcedRNG(association=False, index=0),
    )
    assert not result.events
    assert len(result.attempts) == 1
    assert result.attempts[0].reason == 'reverse move unavailable'
    assert result.attempts[0].probability == 0.0
    assert result.system is system


def test_null_direction_and_invalid_inputs():
    reaction = reaction_model()
    system = empty_system()
    ff = FakeFF(())
    kwargs = dict(key=None, R=POSITIONS, velocities=jnp.zeros_like(POSITIONS),
                  box=BOX, system=system, ff=ff, reaction=reaction,
                  disp_fn=DISP, temperature_k=1.0, kb_real=1.0)
    result = mh.a2_metropolis_reactive_cycle(
        **kwargs, rng=ForcedRNG(association=False, index=0))
    assert result.attempts == result.events == ()
    assert result.ff is ff and result.system is system
    for invalid_p in (0.0, 1.0, -0.1, float('nan')):
        with pytest.raises(ValueError):
            mh.a2_metropolis_reactive_cycle(
                **kwargs, rng=ForcedRNG(association=True, index=0),
                association_proposal_probability=invalid_p)
    with pytest.raises(ValueError):
        mh.a2_metropolis_reactive_cycle(
            **{**kwargs, 'temperature_k': 0.0},
            rng=ForcedRNG(association=True, index=0))



def test_rejected_finite_and_nonfinite_trials_preserve_state(monkeypatch):
    reaction = reaction_model()
    state = ((0, 1),)
    system = make_state(state, reaction)
    ff = FakeFF(state)

    def trial_ff(R, box, trial, ff_ref):
        return FakeFF(tuple(sorted(tuple(sorted(map(int, pair)))
                                   for pair in np.asarray(trial['bonds'][0]).reshape(-1, 2))))

    monkeypatch.setattr(mh, 'build_trial_forcefield', trial_ff)

    class RejectRNG(ForcedRNG):
        def random(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 0.999999999

    kwargs = dict(key=None, R=POSITIONS, velocities=jnp.zeros_like(POSITIONS),
                  box=BOX, system=system, ff=ff, reaction=reaction,
                  disp_fn=DISP, temperature_k=1.0, kb_real=1.0,
                  association_proposal_probability=0.37)
    rejected = mh.a2_metropolis_reactive_cycle(
        **kwargs, rng=RejectRNG(association=True, index=0))
    assert len(rejected.attempts) == 1
    assert 0.0 < rejected.attempts[0].probability < 1.0
    assert not rejected.attempts[0].accepted
    assert rejected.system is system and rejected.ff is ff
    assert rejected.positions is POSITIONS and rejected.events == ()

    class NonfiniteFF(FakeFF):
        def energy_fn(self, positions, nlist):
            return {'total': float('nan')}

    monkeypatch.setattr(mh, 'build_trial_forcefield',
                        lambda *args, **kwargs: NonfiniteFF(()))
    nonfinite = mh.a2_metropolis_reactive_cycle(
        **kwargs, rng=ForcedRNG(association=True, index=0))
    assert not nonfinite.attempts[0].accepted
    assert nonfinite.attempts[0].probability == 0.0
    assert nonfinite.system is system and nonfinite.ff is ff


def test_two_dimer_offset_count():
    reaction = reaction_model(offset=-1.25)
    first = make_state(((0, 1),), reaction)
    candidate = next(c for c in reaction.find_association_candidates(
        POSITIONS, DISP, system=first) if (c.i, c.j) == (2, 3))
    trial = mh._finalize_trial(reaction.build_association_trial(first, candidate),
                               reaction, 4)
    assert trial['energy_offset'] == pytest.approx(-2.5)
    assert len(set(trial['molecule_id'])) == 2


def _real_ff(system, *, energy_offset=None):
    return build_forcefield(
        POSITIONS, BOX, bond_idx=np.asarray(system.bonds[0]),
        k_b=np.asarray(system.bonds[1]), r0=np.asarray(system.bonds[2]),
        angle_idx=np.asarray(system.angles[0]),
        k_theta=np.asarray(system.angles[1]), theta0=np.asarray(system.angles[2]),
        torsions=tuple(np.asarray(x) for x in system.torsions),
        impropers=tuple(np.asarray(x) for x in system.impropers),
        charges=np.asarray(system.charges), sigmas=np.asarray(system.sigmas),
        epsilons=np.asarray(system.epsilons),
        molecule_id=np.asarray(system.molecule_id),
        r_cut=9.0, dr_threshold=0.5, energy_offset=energy_offset,
    )


def test_real_jaxmd_bond_offset_and_reverse_energy_roundtrip():
    """HPC integration: actual topology masks, JAX-MD energy, and offset."""
    reaction = reaction_model(offset=-1.25)
    system = empty_system()
    ff = _real_ff(system, energy_offset=0.0)
    candidate = reaction.find_association_candidates(POSITIONS, DISP, system=system)[0]
    trial = mh._finalize_trial(reaction.build_association_trial(system, candidate), reaction, 4)
    assert trial['energy_offset'] == pytest.approx(-1.25)
    bonded = make_a2_system_state_from_trial(trial, system)
    ff_bond = build_trial_forcefield(POSITIONS, BOX, trial, ff)
    ff_bond_no_offset = _real_ff(bonded, energy_offset=0.0)
    u_bond = float(ff_bond.energy_fn(POSITIONS, ff_bond.nlist)['total'])
    u_without_offset = float(ff_bond_no_offset.energy_fn(POSITIONS, ff_bond_no_offset.nlist)['total'])
    assert math.isfinite(u_bond)
    assert u_bond - u_without_offset == pytest.approx(-1.25, abs=2e-4)
    diss = reaction.find_dissociation_candidates(POSITIONS, DISP, system=bonded)[0]
    reverse = mh._finalize_trial(reaction.build_dissociation_trial(bonded, diss), reaction, 4)
    assert reverse['energy_offset'] == pytest.approx(0.0)
    ff_reverse = build_trial_forcefield(POSITIONS, BOX, reverse, ff_bond)
    u_start = float(ff.energy_fn(POSITIONS, ff.nlist)['total'])
    u_reverse = float(ff_reverse.energy_fn(POSITIONS, ff_reverse.nlist)['total'])
    assert u_reverse == pytest.approx(u_start, abs=2e-4)

    # Run the production acceptance code with *real* JAX-MD energies.
    # For four free atoms there are six association proposals; the reverse
    # bonded state has one dissociation proposal.
    forward = mh.a2_metropolis_reactive_cycle(
        key=None, R=POSITIONS, velocities=jnp.zeros_like(POSITIONS),
        box=BOX, system=system, ff=ff, reaction=reaction,
        disp_fn=DISP, temperature_k=1.0, kb_real=1.0,
        rng=ForcedRNG(association=True, index=0),
    )
    assert forward.attempts[0].delta_energy == pytest.approx(u_bond - u_start, abs=2e-4)
    assert forward.attempts[0].log_proposal_ratio == pytest.approx(math.log(6.0))
    assert matching(forward.system) == matching(bonded)

    backward = mh.a2_metropolis_reactive_cycle(
        key=None, R=POSITIONS, velocities=jnp.zeros_like(POSITIONS),
        box=BOX, system=bonded, ff=ff_bond, reaction=reaction,
        disp_fn=DISP, temperature_k=1.0, kb_real=1.0,
        rng=ForcedRNG(association=False, index=0),
    )
    assert backward.attempts[0].delta_energy == pytest.approx(u_start - u_bond, abs=2e-4)
    assert backward.attempts[0].log_proposal_ratio == pytest.approx(-math.log(6.0))
    assert matching(backward.system) == ()
