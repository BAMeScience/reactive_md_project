"""Topology-only round trip; does not require a JAX-MD forcefield runtime."""
from types import SimpleNamespace

import numpy as np

from reactive_md.reactions.lipf6 import LiPF6Reaction, ReactionCandidate
from reactive_md.reactions.templates_pf5 import make_lif_template, make_pf5_template


def _system():
    # Synthetic PF6 + Li with all fifteen F-P-F angles.
    bond_idx = np.array([[0, f] for f in range(1, 7)], dtype=np.int32)
    angle_idx = np.array([[i, 0, j] for i in range(1, 7) for j in range(i+1, 7)], dtype=np.int32)
    empty = lambda n: np.zeros((0, n), dtype=np.int32)
    return SimpleNamespace(
        bonds=(bond_idx, np.full(6, 370.4589, np.float32), np.full(6, 1.606, np.float32)),
        angles=(angle_idx, np.full(15, 139.2209, np.float32), np.full(15, 1.5707963, np.float32)),
        torsions=(empty(4), np.zeros(0), np.zeros(0, np.int32), np.zeros(0)),
        impropers=(empty(4), np.zeros(0), np.zeros(0, np.int32), np.zeros(0)),
        charges=np.array([1.34]+[-.39]*6+[1.0], dtype=np.float32),
        sigmas=np.array([3.74]+[3.118]*6+[2.12645], dtype=np.float32),
        epsilons=np.array([.2]+[.061]*6+[.0182792], dtype=np.float32),
        molecule_id=np.array([1]*7+[2], dtype=np.int32),
    )


def test_forward_reverse_topology_roundtrip():
    sys = _system()
    reaction = LiPF6Reaction(
        pf6_atoms=np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int32),
        li_atoms=np.array([7], dtype=np.int32),
        atom_types=np.array([6]+[7]*6+[8], dtype=np.int32),
        pf5=make_pf5_template(), lif=make_lif_template(),
        p_type=6, f_type=7, li_type=8,
    )
    reaction.set_reference_state(sys)
    candidate = ReactionCandidate(0, 7, 6, 1.9, 1.6)
    forward, _ = reaction.build_trial(sys, candidate)
    assert len(forward['bonds'][0]) == 5
    assert len(forward['angles'][0]) == 10
    assert forward['molecule_id'][6] != sys.molecule_id[6]
    assert np.isclose(forward['charges'].sum(), sys.charges.sum(), atol=1e-6)
    reverse = ReactionCandidate(0, 7, 6, 1.9, 2.3, 'reverse')
    backward, _ = reaction.build_trial(SimpleNamespace(**forward), reverse)
    for name in ('charges', 'sigmas', 'epsilons', 'molecule_id'):
        np.testing.assert_allclose(backward[name], getattr(sys, name))
    for name in ('bonds', 'angles', 'torsions', 'impropers'):
        for actual, expected in zip(backward[name], getattr(sys, name)):
            np.testing.assert_allclose(actual, expected)


def test_reverse_candidates_require_recorded_partners_and_nearby_geometry():
    sys = _system()
    reaction = LiPF6Reaction(
        pf6_atoms=np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int32),
        li_atoms=np.array([7], dtype=np.int32),
        atom_types=np.array([6]+[7]*6+[8], dtype=np.int32),
        pf5=make_pf5_template(), lif=make_lif_template(),
        p_type=6, f_type=7, li_type=8,
    )
    positions = np.zeros((8, 3), np.float32)
    positions[6] = [2.5, 0, 0]
    positions[7] = [3.2, 0, 0]
    candidates = reaction.find_reverse_candidates(
        positions, lambda a,b: a-b,
        pf6_reacted=[True], pf6_leave_f=[6], pf6_li_partner=[7],
    )
    assert len(candidates) == 1 and candidates[0].direction == 'reverse'
    assert reaction.find_reverse_candidates(
        positions, lambda a,b: a-b,
        pf6_reacted=[True], pf6_leave_f=[-1], pf6_li_partner=[-1],
    ) == []


def test_langevin_velocities_propagate_through_handler():
    import jax.numpy as jnp
    from reactive_md.md_driver import ReactionContext
    from reactive_md.reactions.lipf6_handler import LiPF6Handler

    original_v = jnp.zeros((2, 3))
    relaxed_v = jnp.ones((2, 3))
    def callback(key, positions, ff, system, velocities):
        np.testing.assert_allclose(velocities, original_v)
        return key, True, ff, system, {
            'accepted_event': {'direction': 'reverse'},
            '_relaxed_velocities': relaxed_v,
            'relaxation_time_ps': 0.5,
        }, positions

    handler = LiPF6Handler(callback, passes_velocities=True)
    context = ReactionContext(
        key=0, step=1, interval_ps=.01,
        positions=jnp.zeros((2, 3)), velocities=original_v,
        system=object(), ff=object(), box=None, shift_fn=None,
        masses=jnp.ones(2), temperature_k=300., kb_real=.001987,
    )
    outcome = handler.attempt(context)
    np.testing.assert_allclose(outcome.velocities, relaxed_v)
    assert outcome.reinitialize_thermostat
    assert outcome.relaxation_time_ps == .5
    assert '_relaxed_velocities' not in outcome.metadata
