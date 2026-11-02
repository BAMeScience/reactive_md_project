"""Paper rs@md[MC] Figure 3a transactional algorithm tests.

Run in the project's JAX-MD environment. Expensive Langevin integration and
force-field construction are replaced with deterministic fakes, so these
assertions test algorithm order and state handoff rather than thermostat physics.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions import a2_metropolis as mc
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from reactive_md.engine.contracts import ReactionContext


@dataclass
class FakeNeighbor:
    def allocate(self, R, **kwargs):
        return SimpleNamespace(did_buffer_overflow=False)


class FakeFF:
    def __init__(self, energy):
        self.energy = energy
        self.neighbor_fn = FakeNeighbor()
        self.nlist = self.neighbor_fn.allocate(None)

    def energy_fn(self, R, nlist):
        return {"total": self.energy}


class FakeRNG:
    def __init__(self, candidate_index, acceptance_draw):
        self.candidate_index = candidate_index
        self.acceptance_draw = acceptance_draw
        self.calls = []

    def integers(self, n):
        self.calls.append(("candidate", n))
        assert 0 <= self.candidate_index < n
        return self.candidate_index

    def random(self):
        self.calls.append(("acceptance",))
        return self.acceptance_draw


def make_system(bonded=False):
    bonds = (jnp.asarray([[2, 3]] if bonded else [], dtype=jnp.int32).reshape(-1, 2),
             jnp.asarray([1.] if bonded else []),
             jnp.asarray([1.8] if bonded else []))
    angles = (jnp.empty((0, 3), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    four = (jnp.empty((0, 4), dtype=jnp.int32), jnp.empty(0),
            jnp.empty(0, dtype=jnp.int32), jnp.empty(0))
    return SystemState(bonds, angles, four, four, jnp.zeros(4),
                       jnp.ones(4), jnp.ones(4), jnp.arange(4),
                       jnp.zeros(0, dtype=bool))


@pytest.fixture
def setup(monkeypatch):
    R = jnp.asarray([[0., 0., 0.], [1.8, 0., 0.],
                     [6., 0., 0.], [7.8, 0., 0.]])
    v = jnp.zeros_like(R)
    reaction = A2Reaction(a_indices=np.arange(4), k_bond_jax=1.,
                          r0_bond=1.8, association_cutoff=2.0,
                          bond_energy_offset=-0.5)
    steps = []
    old = FakeFF(0.)
    trial = FakeFF(-2.)

    def translate(R_in, i, j, **kwargs):
        steps.append("translate")
        return R_in + 0.25

    def build(R_in, box, trial_dict, ff):
        steps.append("build")
        assert steps[-2] == "translate"
        assert float(trial_dict["energy_offset"]) == pytest.approx(
            -0.5 * len(trial_dict["bonds"][0]))
        return trial

    def relax(key, R_in, v_in, **kwargs):
        steps.append("relax")
        assert steps[-2] == "build"
        assert kwargs["dt_ps"] == 0.00025
        assert kwargs["relaxation_time_ps"] == 1.0
        assert kwargs["thermostat_tau_ps"] == 0.01
        return "advanced-key", R_in + 0.1, v_in + 0.2, trial.nlist

    monkeypatch.setattr(mc, "translate_reacting_pair", translate)
    monkeypatch.setattr(mc, "build_trial_forcefield", build)
    monkeypatch.setattr(mc, "langevin_relax_with_nlist", relax)
    return SimpleNamespace(R=R, v=v, reaction=reaction, steps=steps,
                           old=old, trial=trial)


def run(s, rng, system=None):
    return mc.a2_metropolis_reactive_cycle(
        key="original-key", R=s.R, velocities=s.v, box=jnp.ones(3)*20,
        system=system or make_system(), ff=s.old, reaction=s.reaction,
        disp_fn=lambda a, b: a-b, shift_fn=lambda x, dx: x+dx,
        masses=jnp.ones(4), temperature_k=1., kb_real=1., rng=rng,
        energy_mode="full")


def test_paper_mc_relaxes_before_accepting_and_commits(setup):
    s = setup
    rng = FakeRNG(candidate_index=0, acceptance_draw=0.99)
    result = run(s, rng)
    assert len(result.events) == 1
    assert result.attempts[0].probability == 1.0
    assert result.attempts[0].delta_energy == pytest.approx(-2.)
    assert result.ff is s.trial
    np.testing.assert_allclose(result.positions, s.R + 0.35)
    np.testing.assert_allclose(result.velocities, s.v + 0.2)
    assert s.steps == ["translate", "build", "relax"]
    assert rng.calls == [("candidate", 2), ("acceptance",)]
    assert result.key == "advanced-key"


def test_paper_mc_rejection_rolls_back_every_physical_field(setup):
    s = setup
    s.trial.energy = 4.
    rng = FakeRNG(candidate_index=0, acceptance_draw=0.999)
    original_system = make_system()
    result = run(s, rng, original_system)
    assert not result.events
    assert not result.attempts[0].accepted
    assert result.attempts[0].probability == pytest.approx(np.exp(-4.))
    assert result.positions is s.R
    assert result.velocities is s.v
    assert result.ff is s.old
    assert result.system is original_system
    assert result.key == "advanced-key"
    assert s.steps == ["translate", "build", "relax"]


def test_paper_mc_selects_from_union_not_direction_first(setup):
    s = setup
    # With 2 free atoms and 1 existing bond, the combined list has two entries.
    system = make_system(bonded=True)
    rng = FakeRNG(candidate_index=1, acceptance_draw=0.0)
    result = run(s, rng, system)
    assert rng.calls[0] == ("candidate", 2)
    assert result.attempts[0].reaction_type == "dissociation"
    assert result.n_association_candidates == 1
    assert result.n_dissociation_candidates == 1


def test_paper_mc_handler_accounts_for_accepted_relaxation(monkeypatch, setup):
    s = setup
    handler = A2MetropolisHandler(reaction=s.reaction, rng=FakeRNG(0, 0.0),
                                  energy_mode="full")
    context = ReactionContext(
        key="original-key", step=10, interval_ps=0.1,
        positions=s.R, velocities=s.v, system=make_system(), ff=s.old,
        box=jnp.ones(3)*20, shift_fn=lambda x, dx: x+dx,
        masses=jnp.ones(4), temperature_k=1., kb_real=1.)
    # FakeFF lacks disp_fn; attach the normal engine contract.
    s.old.disp_fn = lambda a, b: a-b
    outcome = handler.attempt(context)
    assert outcome.events
    assert outcome.reinitialize_thermostat is True
    assert outcome.relaxation_time_ps == 1.0
    assert outcome.preserve_momenta is False
