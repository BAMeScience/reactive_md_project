"""Real JAX-MD integration checks for the published rs@md[MC] workflow.

These tests intentionally use a *two-step* Langevin relaxation to exercise the
production integrator without turning the test suite into a simulation. The
production default remains 1 ps / 0.25 fs, as specified by the paper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.forcefield import build_forcefield
from reactive_md.md_driver import run_md_nvt
from reactive_md.reaction import SystemState, build_trial_forcefield
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from reactive_md.reactions.a2_metropolis import a2_metropolis_reactive_cycle, _finalize_trial


class ForcedRNG:
    def __init__(self, draw):
        self.draw = draw
        self.choices = 0
        self.draws = 0

    def integers(self, n):
        assert n == 1
        self.choices += 1
        return 0

    def random(self):
        self.draws += 1
        return self.draw


def _system(bonded=False):
    bonds = (jnp.asarray([[0, 1]] if bonded else [], dtype=jnp.int32).reshape(-1, 2),
             jnp.asarray([1.5] if bonded else []),
             jnp.asarray([2.] if bonded else []))
    angles = (jnp.empty((0, 3), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    four = (jnp.empty((0, 4), dtype=jnp.int32), jnp.empty(0),
            jnp.empty(0, dtype=jnp.int32), jnp.empty(0))
    return SystemState(bonds, angles, four, four, jnp.zeros(2),
                       jnp.full(2, 1.2), jnp.full(2, 0.2),
                       jnp.asarray([0, 0] if bonded else [0, 1]),
                       jnp.zeros(0, dtype=bool))


def _forcefield(R, box, system, offset):
    return build_forcefield(
        R, box, bond_idx=np.asarray(system.bonds[0]),
        k_b=np.asarray(system.bonds[1]), r0=np.asarray(system.bonds[2]),
        angle_idx=np.asarray(system.angles[0]),
        k_theta=np.asarray(system.angles[1]), theta0=np.asarray(system.angles[2]),
        torsions=tuple(np.asarray(v) for v in system.torsions),
        impropers=tuple(np.asarray(v) for v in system.impropers),
        charges=np.asarray(system.charges), sigmas=np.asarray(system.sigmas),
        epsilons=np.asarray(system.epsilons),
        molecule_id=np.asarray(system.molecule_id), r_cut=6.,
        dr_threshold=0.5, use_shift_lj=True, energy_offset=offset)


def _setup(*, bonded=False, offset=-50.):
    R = jnp.asarray([[3., 3., 3.], [5.2, 3., 3.]], dtype=jnp.float32)
    box = jnp.full(3, 20.)
    disp, shift = space.periodic(box)
    system = _system(bonded)
    ff = _forcefield(R, box, system, offset if bonded else 0.)
    reaction = A2Reaction(a_indices=np.arange(2), k_bond_jax=1.5,
                          r0_bond=2., association_cutoff=3.,
                          bond_energy_offset=offset)
    return R, box, disp, shift, system, ff, reaction


def _cycle(*, bonded=False, offset=-50., draw=0., energy_mode="local_solvation"):
    R, box, disp, shift, system, ff, reaction = _setup(
        bonded=bonded, offset=offset)
    rng = ForcedRNG(draw)
    result = a2_metropolis_reactive_cycle(
        key=jax.random.PRNGKey(19), R=R, velocities=jnp.zeros_like(R),
        box=box, system=system, ff=ff, reaction=reaction, disp_fn=disp,
        shift_fn=shift, masses=jnp.asarray([2., 3.]),
        temperature_k=188., kb_real=0.0019872041, rng=rng,
        relaxation_dt_ps=0.00025, relaxation_time_ps=0.0005,
        energy_mode=energy_mode)
    assert rng.choices == 1 and rng.draws == 1
    return result, R, system, ff


def test_real_forcefield_offset_association_dissociation_roundtrip():
    R, box, disp, _, system, ff, reaction = _setup()
    candidate = reaction.find_association_candidates(R, disp, system=system)[0]
    trial = _finalize_trial(reaction.build_association_trial(system, candidate),
                            reaction, 2)
    assert trial['energy_offset'] == pytest.approx(-50.)
    bonded_ff = build_trial_forcefield(R, box, trial, ff)
    bonded_system = _system(bonded=True)
    bonded_no_offset = _forcefield(R, box, bonded_system, 0.)
    u_bonded = float(bonded_ff.energy_fn(R, bonded_ff.nlist)['total'])
    u_no_offset = float(bonded_no_offset.energy_fn(R, bonded_no_offset.nlist)['total'])
    assert u_bonded - u_no_offset == pytest.approx(-50., abs=1.e-4)
    reverse_candidate = reaction.find_dissociation_candidates(
        R, disp, system=bonded_system)[0]
    reverse = _finalize_trial(reaction.build_dissociation_trial(
        bonded_system, reverse_candidate), reaction, 2)
    free_ff = build_trial_forcefield(R, box, reverse, bonded_ff)
    u_free = float(free_ff.energy_fn(R, free_ff.nlist)['total'])
    assert u_free == pytest.approx(float(ff.energy_fn(R, ff.nlist)['total']),
                                   abs=1.e-4)


def test_real_mc_acceptance_hands_off_relaxed_state():
    result, R, system, ff = _cycle(bonded=False, offset=-50., draw=0.)
    assert len(result.events) == 1
    assert result.events[0].reaction_type == 'association'
    assert result.attempts[0].accepted
    assert len(np.asarray(result.system.bonds[0])) == 1
    assert result.ff is not ff
    assert not bool(np.asarray(result.ff.nlist.did_buffer_overflow))
    assert np.isfinite(float(result.ff.energy_fn(
        result.positions, result.ff.nlist)['total']))
    assert bool(np.asarray(jnp.all(jnp.isfinite(result.velocities))))
    assert not np.array_equal(np.asarray(result.positions), np.asarray(R))
    assert len(np.asarray(system.bonds[0])) == 0


def test_real_mc_rejection_rolls_back_all_physical_state():
    result, R, system, ff = _cycle(bonded=True, offset=-50., draw=0.999999)
    assert len(result.events) == 0
    assert result.attempts[0].reaction_type == 'dissociation'
    assert not result.attempts[0].accepted
    assert result.positions is R
    assert result.system is system
    assert result.ff is ff
    assert len(np.asarray(result.system.bonds[0])) == 1
    assert result.attempts[0].delta_energy > 0
    assert float(result.attempts[0].probability) < 0.999999


def test_real_nvt_engine_uses_committed_mc_state_and_refreshes_force():
    R, box, _, shift, system, ff, reaction = _setup()
    observer = []

    class Capture:
        def on_observation(self, obs):
            if obs.outcome is not None and obs.outcome.events:
                observer.append(obs)

    cfg = SimConfig(steps=1, check_every=1, max_events=2, dt=0.00025,
                    temperature_k=188., kb_real=0.0019872041, tau_T=0.5)
    handler = A2MetropolisHandler(reaction, ForcedRNG(0.),
                                  relaxation_time_ps=0.0005)
    run = run_md_nvt(
        jax.random.PRNGKey(33), cfg=cfg, init_positions=R,
        masses=jnp.asarray([2., 3.]), shift_fn=shift, ff=ff, sys=system,
        reaction_handler=handler, box=box, observers=(Capture(),))
    assert run.accepted_events == 1
    assert len(observer) == 1
    obs = observer[0]
    assert obs.ff is run.ff
    assert obs.system is run.sys
    assert len(np.asarray(obs.system.bonds[0])) == 1
    assert not bool(np.asarray(obs.ff.nlist.did_buffer_overflow))
    np.testing.assert_allclose(np.asarray(obs.state.position),
                               np.asarray(obs.outcome.positions), atol=1.e-6)
    np.testing.assert_allclose(np.asarray(obs.state.velocity),
                               np.asarray(obs.outcome.velocities), atol=1.e-5)
    calculated_force = -jax.grad(
        lambda x: obs.ff.energy_fn(x, obs.ff.nlist)['total'])(obs.state.position)
    np.testing.assert_allclose(np.asarray(obs.state.force),
                               np.asarray(calculated_force), rtol=1.e-4, atol=1.e-4)

@pytest.mark.parametrize(
    'bonded,offset,expected_reaction',
    [(False, -50., 'association'), (True, 50., 'dissociation')],
)
def test_real_mc_delta_energy_accounting_with_relaxation(
    bonded, offset, expected_reaction
):
    """Audit the original C++ optional full-system convention.

    For an accepted trial, the reported Delta V must equal the difference
    between the real JAX-MD force-field energies at the pre-trial and
    post-relaxation coordinates. The per-bond V0 term must enter exactly once.
    This does not establish whether the published local-energy convention
    excludes solvent-solvent energy changes during relaxation.
    """
    result, initial_R, _, initial_ff = _cycle(
        bonded=bonded, offset=offset, draw=0., energy_mode="full"
    )
    assert len(result.events) == 1
    assert result.events[0].reaction_type == expected_reaction

    before_terms = initial_ff.energy_fn(initial_R, initial_ff.nlist)
    after_terms = result.ff.energy_fn(result.positions, result.ff.nlist)
    before_total = float(np.asarray(before_terms['total']))
    after_total = float(np.asarray(after_terms['total']))
    before_offset = float(np.asarray(before_terms.get('offset', 0.)))
    after_offset = float(np.asarray(after_terms.get('offset', 0.)))

    measured_delta = after_total - before_total
    np.testing.assert_allclose(
        result.events[0].delta_energy, measured_delta, atol=2.e-4, rtol=1.e-5
    )
    np.testing.assert_allclose(
        result.attempts[0].delta_energy, measured_delta, atol=2.e-4, rtol=1.e-5
    )
    expected_offset_change = offset if not bonded else -offset
    np.testing.assert_allclose(
        after_offset - before_offset, expected_offset_change, atol=2.e-4
    )

    # Verify the offset is not added twice or omitted. The raw term includes
    # *all* force-field contributions other than the explicit energy offset.
    raw_delta = ((after_total - after_offset)
                 - (before_total - before_offset))
    np.testing.assert_allclose(
        measured_delta, raw_delta + expected_offset_change,
        atol=2.e-4, rtol=1.e-5,
    )


def test_real_mc_local_solvation_energy_matches_author_partition():
    """Four-atom A2 + solvent: exclude B-B but include A-B interactions.

    In the source C++ this is the combined
    computeLocalPotentialEnergy + computeSolvationPotentialEnergy mode.
    """
    from reactive_md.reactions.a2_energy import local_solvation_energy
    R = jnp.asarray([[3., 3., 3.], [5.2, 3., 3.],
                     [7.5, 3., 3.], [9.5, 3., 3.]], dtype=jnp.float32)
    box = jnp.full(3, 20.)
    disp, shift = space.periodic(box)
    empty_bonds = (jnp.empty((0, 2), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    empty_angles = (jnp.empty((0, 3), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    empty_four = (jnp.empty((0, 4), dtype=jnp.int32), jnp.empty(0),
                  jnp.empty(0, dtype=jnp.int32), jnp.empty(0))
    sys = SystemState(empty_bonds, empty_angles, empty_four, empty_four,
                      jnp.zeros(4), jnp.full(4, 1.2), jnp.full(4, 0.2),
                      jnp.arange(4), jnp.zeros(0, dtype=bool))
    ff = _forcefield(R, box, sys, 0.)
    reaction = A2Reaction(a_indices=np.arange(2), k_bond_jax=1.5,
                          r0_bond=2., association_cutoff=3.,
                          bond_energy_offset=-50.)
    result = a2_metropolis_reactive_cycle(
        key=jax.random.PRNGKey(22), R=R, velocities=jnp.zeros_like(R),
        box=box, system=sys, ff=ff, reaction=reaction, disp_fn=disp,
        shift_fn=shift, masses=jnp.ones(4) * 2.,
        temperature_k=188., kb_real=0.0019872041, rng=ForcedRNG(0.),
        relaxation_dt_ps=0.00025, relaxation_time_ps=0.0005,
        energy_mode='local_solvation')
    assert result.events and result.events[0].reaction_type == 'association'
    pair = (0, 1)
    before = local_solvation_energy(ff, R, sys, pair, box)
    after = local_solvation_energy(result.ff, result.positions,
                                   result.system, pair, box)
    np.testing.assert_allclose(result.events[0].delta_energy,
                               after - before, rtol=1.e-5, atol=2.e-4)

    # Compute B-B potential separately and verify the exact algebraic
    # difference from the full-system mode, including the V0 correction.
    def bb_energy(coords):
        B = coords[2:]
        Bsys = SystemState(empty_bonds, empty_angles, empty_four, empty_four,
                           jnp.zeros(2), jnp.full(2, 1.2), jnp.full(2, 0.2),
                           jnp.arange(2), jnp.zeros(0, dtype=bool))
        bff = _forcefield(B, box, Bsys, 0.)
        return float(bff.energy_fn(B, bff.nlist)['total'])

    full_before = float(ff.energy_fn(R, ff.nlist)['total'])
    full_after = float(result.ff.energy_fn(result.positions, result.ff.nlist)['total'])
    expected = (full_after - full_before) - (bb_energy(result.positions) - bb_energy(R))
    np.testing.assert_allclose(result.events[0].delta_energy,
                               expected, rtol=1.e-5, atol=2.e-4)
