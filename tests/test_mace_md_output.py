
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from reactive_md.config import SimConfig
from reactive_md.engine.contracts import MDObservation
from reactive_md.reactions import lipf6_driver
from reactive_md.reactions.lipf6_output import LiPF6OutputObserver


class FakeNeighbor:
    def __init__(self, overflow):
        self.did_buffer_overflow = jnp.asarray(overflow)

    def update(self, positions):
        return self


class FakeNeighborFunction:
    def __init__(self, always_overflow=False):
        self.capacities = []
        self.always_overflow = always_overflow

    def allocate(self, positions, *, extra_capacity=0):
        self.capacities.append(extra_capacity)
        overflow = self.always_overflow or extra_capacity < 8
        return FakeNeighbor(overflow)


def make_observer(neighbor_fn):
    return LiPF6OutputObserver(
        cfg=SimConfig(check_every=10),
        box=jnp.array([10.0, 10.0, 10.0]),
        masses=jnp.ones((1,)),
        md_energy_fn=lambda positions, neighbors: jnp.asarray(7.0),
        md_neighbor_fn=neighbor_fn,
    )


def make_observation():
    state = SimpleNamespace(
        position=jnp.zeros((1, 3)),
        velocity=jnp.zeros((1, 3)),
    )

    ff = SimpleNamespace(
        disp_fn=lambda a, b: a - b,
        energy_fn=lambda *args: (_ for _ in ()).throw(
            AssertionError("Classical energy must not be used")
        ),
        nlist=None,
    )

    return MDObservation(
        step=1,
        md_time_ps=0.001,
        relaxation_time_ps=0.0,
        state=state,
        ff=ff,
        system=SimpleNamespace(
            pf6_reacted=jnp.array([False])
        ),
        outcome=None,
    )


def test_mace_observer_reports_mace_energy(capsys):
    neighbor_fn = FakeNeighborFunction()
    observer = make_observer(neighbor_fn)

    observer.on_before_reaction(make_observation())

    output = capsys.readouterr().out

    assert "PE= 7.000000" in output
    assert neighbor_fn.capacities == [2, 8]


def test_mace_observer_rejects_persistent_overflow():
    neighbor_fn = FakeNeighborFunction(always_overflow=True)
    observer = make_observer(neighbor_fn)

    with pytest.raises(RuntimeError, match="overflowed"):
        observer.on_before_reaction(make_observation())

    assert len(neighbor_fn.capacities) == 13


def test_lipf6_wrapper_keeps_observer_with_mace(monkeypatch):
    captured = {}
    ff = SimpleNamespace()
    sys = SimpleNamespace(pf6_reacted=jnp.array([False]))

    md_energy_fn = lambda positions, neighbors: jnp.asarray(1.0)
    md_neighbor_fn = object()

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            final_md_state=None,
            ff=ff,
            sys=sys,
            accepted_events=0,
        )

    monkeypatch.setattr(lipf6_driver, "run_md_nvt", fake_run)

    lipf6_driver.run_md_nvt_with_reactions(
        jnp.array([0, 1], dtype=jnp.uint32),
        cfg=SimConfig(),
        init_positions=jnp.zeros((1, 3)),
        masses=jnp.ones((1,)),
        shift_fn=lambda positions, displacement: positions + displacement,
        ff=ff,
        sys=sys,
        reaction_step_fn=lambda *args: None,
        md_energy_fn=md_energy_fn,
        md_neighbor_fn=md_neighbor_fn,
    )

    assert len(captured["observers"]) == 1

    observer = captured["observers"][0]

    assert isinstance(observer, LiPF6OutputObserver)
    assert observer.md_energy_fn is md_energy_fn
    assert observer.md_neighbor_fn is md_neighbor_fn
