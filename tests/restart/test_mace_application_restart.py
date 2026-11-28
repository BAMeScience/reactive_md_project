"""LiPF6 application restart with real reaction/MD code and a surrogate MACE model."""
from dataclasses import replace
import sys
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.apps import lipf6
from reactive_md.checkpoint import load_checkpoint
from reactive_md.config import SimConfig
from reactive_md.mace_identity import fingerprint


class MaceNeighbors(NamedTuple):
    idx: object
    did_buffer_overflow: object

    def update(self, positions):
        return self


class MaceNeighborFn:
    def allocate(self, positions, *, extra_capacity=0):
        return MaceNeighbors(
            jnp.full((len(positions), max(1, extra_capacity)),
                     len(positions), dtype=jnp.int32),
            jnp.asarray(False),
        )

    def update(self, positions, neighbor):
        assert isinstance(neighbor, MaceNeighbors)
        return neighbor.update(positions)


@pytest.fixture
def surrogate_mace(monkeypatch):
    # One PF6 and one Li; keep the original reference topology on every load.
    def parse(*args):
        R = np.array([[0., 0., 0.], [1.6, 0., 0.], [-1.6, 0., 0.],
                      [0., 1.6, 0.], [0., -1.6, 0.], [0., 0., -1.6],
                      [0., 0., 1.6], [0., 0., 4.]]) + 10.
        bonds = (np.array([[0, f] for f in range(1, 7)], np.int32),
                 np.full(6, 370.4589), np.full(6, 1.606))
        angles = (np.array([[i, 0, j] for i in range(1, 7)
                            for j in range(i + 1, 7)], np.int32),
                  np.full(15, 139.2209), np.full(15, 1.5707963))
        empty = (np.empty((0, 4), np.int32), np.empty(0),
                 np.empty(0, np.int32), np.empty(0))
        nb = (np.array([1.34] + [-.39]*6 + [1.]),
              np.array([3.74] + [3.118]*6 + [2.12645]),
              np.array([.2] + [.061]*6 + [.0182792]), None, None)
        return (R, bonds, angles, empty, empty, nb,
                np.array([1]*7 + [2]), np.full(3, 30.),
                np.array([31.] + [19.]*6 + [7.]),
                np.array([6] + [7]*6 + [8]))

    weights = np.array([1.], dtype=np.float32)
    setup = types.ModuleType('reactive_md.mace_setup')
    energy = types.ModuleType('reactive_md.mace_energy')

    def build_model(**kwargs):
        assert kwargs['return_backend_id'] is True
        return weights.copy(), {}, None, fingerprint(weights)

    def build_energy(**kwargs):
        weight = float(kwargs['jax_model'][0])
        def make_energy(neighbor):
            assert isinstance(neighbor, MaceNeighbors)
            # Constant differentiable energy guarantees MC acceptance when
            # the sigma gate is open. Actual NVT and Langevin remain active.
            return lambda R, **kw: weight + 0. * jnp.sum(R)
        return None, None, MaceNeighborFn(), make_energy

    setup.build_mace_model = build_model
    energy.build_mace_energy_system = build_energy
    monkeypatch.setitem(sys.modules, setup.__name__, setup)
    monkeypatch.setitem(sys.modules, energy.__name__, energy)
    monkeypatch.setattr(lipf6, 'parse_lammps_data', parse)
    return weights


def assert_tree_equal(a, b):
    aa, ta = jax.tree_util.tree_flatten(a)
    bb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('mode', ['metropolis', 'rate'])
@pytest.mark.parametrize('accept', [False, True])
def test_application_restart_equivalence(tmp_path, surrogate_mace, mode, accept):
    cfg = SimConfig(steps=4, check_every=2, dt=.001, r_cut=8.,
                    reaction_mode=mode, use_mace_md=True,
                    use_mace_mc=mode == 'metropolis',
                    reaction_rate_ps=500. if mode == 'rate' else None,
                    sigma_mid=-1000. if accept else 1000.,
                    relaxation_dt_ps=.00025, relaxation_time_ps=.001)

    def paths(name):
        directory = tmp_path / name
        directory.mkdir()
        outputs = [directory / x for x in ('traj.dump', 'events.csv', 'candidates.csv')]
        return (replace(cfg, dump_file=str(outputs[0]), event_log_file=str(outputs[1]),
                        candidate_log_file=str(outputs[2])), directory / 'state.pkl', outputs)

    full_cfg, full_path, full_outputs = paths('full')
    split_cfg, split_path, split_outputs = paths('split')
    full = lipf6.main(full_cfg, checkpoint_path=str(full_path), checkpoint_every_steps=2)
    first = lipf6.main(replace(split_cfg, steps=2), checkpoint_path=str(split_path),
                       checkpoint_every_steps=2)
    saved = load_checkpoint(split_path)
    assert saved['steps_done'] == 2
    assert first.accepted_events == saved['accepted_events'] == int(accept)
    assert bool(first.sys.pf6_reacted[0]) == accept
    assert saved['relaxation_elapsed_ps'] == (.001 if accept else 0.)

    # Simulate output written after the last durable checkpoint.
    for path in split_outputs:
        with path.open('ab') as stream:
            stream.write(b'uncommitted output\n')
    before = [path.read_bytes() for path in split_outputs]
    checkpoint_bytes = split_path.read_bytes()
    surrogate_mace[0] = 2.
    with pytest.raises(ValueError, match='md_backend_id'):
        lipf6.main(split_cfg, restart_path=str(split_path))
    assert split_path.read_bytes() == checkpoint_bytes
    assert [path.read_bytes() for path in split_outputs] == before
    surrogate_mace[0] = 1.
    with pytest.raises(ValueError, match='metadata'):
        lipf6.main(replace(split_cfg, mace_energy_only=True), restart_path=str(split_path))
    assert split_path.read_bytes() == checkpoint_bytes
    assert [path.read_bytes() for path in split_outputs] == before

    resumed = lipf6.main(split_cfg, restart_path=str(split_path), checkpoint_every_steps=2)
    expected, actual = load_checkpoint(full_path), load_checkpoint(split_path)
    assert actual['steps_done'] == expected['steps_done'] == 4
    assert resumed.accepted_events == full.accepted_events == int(accept)
    for field in ('md_state', 'system_state', 'key', 'accepted_events',
                  'relaxation_elapsed_ps'):
        assert_tree_equal(actual[field], expected[field])
    assert not isinstance(resumed.ff.nlist, MaceNeighbors)
    for a, b in zip(full_outputs, split_outputs):
        assert a.read_bytes() == b.read_bytes()
