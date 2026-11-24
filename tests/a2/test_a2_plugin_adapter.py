"""A2 output and compatibility-contract tests for the generic MD engine."""
from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from reactive_md.config import SimConfig
from reactive_md.md_driver import MDObservation, MDRunResult, ReactionOutcome
from reactive_md.reactions import a2_driver


@dataclass
class Event:
    reaction_type: str
    i: int = 0
    j: int = 1


class System:
    def __init__(self, bonds):
        self.bonds = (np.asarray(bonds, dtype=np.int32).reshape(-1, 2),)


class FakeFF:
    nlist = None

    @staticmethod
    def energy_fn(position, nlist):
        return {'total': jnp.asarray(3.0)}


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


def _state(x):
    positions = jnp.asarray([[x, 0., 0.], [x + 1., 0., 0.]])
    return SimpleNamespace(position=positions, velocity=jnp.zeros_like(positions))


def test_a2_observer_records_pre_and_post_topology_and_time():
    frames, bonds, rdf = StringIO(), StringIO(), Recorder()
    dumps = []

    def writer(**kwargs):
        dumps.append(kwargs['step'])

    observer = a2_driver.A2OutputObserver(
        reaction=SimpleNamespace(a_indices=np.asarray([0, 1])),
        cfg=SimConfig(dt=0.01),
        box=np.asarray([10., 10., 10.]), masses=np.ones(2),
        rdf_observer=rdf, dump_file=frames, dump_writer=writer,
        atom_ids=np.asarray([1, 2]), atom_types=np.asarray([1, 1]),
        dump_every_checks=5, dump_bonds_file=bonds, dump_frames_file=frames,
    )
    before, after = System([]), System([[0, 1]])
    ff = FakeFF()
    observer.on_observation(MDObservation(
        step=0, md_time_ps=0., relaxation_time_ps=0.,
        state=_state(0.), ff=ff, system=before, outcome=None,
    ))
    event = Event('association')
    outcome = ReactionOutcome(
        key=None, positions=_state(0.2).position,
        velocities=_state(0.2).velocity, system=after, ff=ff,
        events=(event,), relaxation_time_ps=1.,
        reinitialize_thermostat=True,
        metadata={'n_association_candidates': 4, 'n_dissociation_candidates': 0},
    )
    observer.on_observation(MDObservation(
        step=100, md_time_ps=1., relaxation_time_ps=1.,
        state=_state(0.2), ff=ff, system=after, outcome=outcome,
        pre_reaction_positions=_state(0.1).position,
    ))
    assert dumps == [0, 100, 100]
    assert observer.population_history == [(0, 0), (100, 1)]
    assert observer.samples[-1].n_association == 1
    assert observer.samples[-1].n_association_candidates == 4
    assert observer.event_history[0].reaction_time_ps == 1.
    assert observer.event_history[0].relaxation_end_time_ps == 2.
    assert len(rdf.calls) == 2
    assert '0,0,0.000000000,initial,0' in frames.getvalue()
    assert '1,100,1.000000000,pre_check,0' in frames.getvalue()
    assert '2,100,1.000000000,post_relaxation,1' in frames.getvalue()
    assert '2,100,post_relaxation,1,2' in bonds.getvalue()


def test_a2_runner_uses_generic_driver(monkeypatch):
    calls = []
    ff, system = FakeFF(), System([])

    def fake_run(key, *, cfg, init_positions, masses, shift_fn, ff, sys,
                 reaction_handler, box, observers):
        calls.append(reaction_handler)
        state = _state(0.)
        observers[0].on_observation(MDObservation(
            step=0, md_time_ps=0., relaxation_time_ps=0.,
            state=state, ff=ff, system=sys, outcome=None,
        ))
        return MDRunResult(state, ff, sys, 0, 0, key, 0.)

    monkeypatch.setattr(a2_driver, 'run_md_nvt', fake_run)
    result = a2_driver.run_md_nvt_a2(
        key=None, cfg=SimConfig(steps=0), init_positions=_state(0.).position,
        masses=jnp.ones(2), box=jnp.asarray([10., 10., 10.]),
        shift_fn=None, ff=ff, sys=system,
        reaction=SimpleNamespace(a_indices=np.asarray([0, 1])),
        rng=np.random.default_rng(1),
        association_rate_ps=1.e-4, dissociation_rate_ps=1.e-4,
    )
    assert isinstance(calls[0], a2_driver.A2RateHandler)
    assert result.steps_done == 0
    assert result.population_history == ((0, 0),)
    assert len(result.samples) == 1


def test_a2_observer_no_event_periodic_dump_and_mc_clock():
    dumps = []

    def writer(**kwargs):
        dumps.append(kwargs['step'])

    observer = a2_driver.A2OutputObserver(
        reaction=SimpleNamespace(a_indices=np.asarray([0, 1])),
        cfg=SimConfig(dt=0.01), box=np.asarray([10., 10., 10.]),
        masses=np.ones(2), dump_file=StringIO(), dump_writer=writer,
        atom_ids=np.asarray([1, 2]), atom_types=np.asarray([1, 1]),
        dump_every_checks=2,
    )
    system = System([])
    ff = FakeFF()
    observer.on_observation(MDObservation(
        step=0, md_time_ps=0., relaxation_time_ps=0.,
        state=_state(0.), ff=ff, system=system, outcome=None,
    ))
    empty = ReactionOutcome(
        key=None, positions=_state(0.1).position,
        velocities=_state(0.1).velocity, system=system, ff=ff,
        metadata={'attempts': (SimpleNamespace(accepted=False),)},
    )
    observer.on_observation(MDObservation(
        step=10, md_time_ps=.1, relaxation_time_ps=0.,
        state=_state(0.1), ff=ff, system=system, outcome=empty,
        pre_reaction_positions=_state(0.1).position,
    ))
    observer.on_observation(MDObservation(
        step=20, md_time_ps=.2, relaxation_time_ps=0.,
        state=_state(0.2), ff=ff, system=system, outcome=empty,
        pre_reaction_positions=_state(0.2).position,
    ))
    assert dumps == [0, 20]
    assert observer.attempted_mc_moves == 2
    assert observer.rejected_mc_moves == 2
    assert len(observer.samples) == 3
    assert observer.samples[-1].cumulative_relaxation_time_ps == 0.
