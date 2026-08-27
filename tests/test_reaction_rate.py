from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reactive_md import reaction as reaction_module
from reactive_md.reaction import SystemState, maybe_react_rate_events
from reactive_md.reactions import lipf6


def test_rate_mode_multiple_events_relaxes_once(monkeypatch):
    """A rate sweep may accept several events but must relax only once."""

    candidates = [
        lipf6.ReactionCandidate(
            k_pf6=0,
            li_idx=14,
            leave_F=1,
            d_lif=1.8,
            d_pf=1.6,
        ),
        lipf6.ReactionCandidate(
            k_pf6=1,
            li_idx=15,
            leave_F=8,
            d_lif=1.9,
            d_pf=1.6,
        ),
    ]

    class TestReaction:
        pf6_atoms = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
            ],
            dtype=np.int32,
        )

        def find_candidates(
            self,
            R,
            disp_fn,
            *,
            pf6_reacted,
        ):
            del R, disp_fn

            return [
                candidate
                for candidate in candidates
                if not pf6_reacted[candidate.k_pf6]
            ]

        def phosphorus_index(self, candidate):
            return int(self.pf6_atoms[candidate.k_pf6, 0])

        def build_trial(self, system, candidate):
            trial = {
                "bonds": tuple(
                    np.asarray(value)
                    for value in system.bonds
                ),
                "angles": tuple(
                    np.asarray(value)
                    for value in system.angles
                ),
                "torsions": tuple(
                    np.asarray(value)
                    for value in system.torsions
                ),
                "impropers": tuple(
                    np.asarray(value)
                    for value in system.impropers
                ),
                "charges": np.asarray(system.charges).copy(),
                "sigmas": np.asarray(system.sigmas).copy(),
                "epsilons": np.asarray(system.epsilons).copy(),
                "molecule_id": np.asarray(
                    system.molecule_id,
                    dtype=np.int32,
                ).copy(),
            }

            return trial, candidate.k_pf6

        def prepare_probe(
            self,
            R,
            candidate,
            *,
            sigma_p,
            sigma_f,
            disp_fn,
            shift_fn,
        ):
            del sigma_p, sigma_f, disp_fn, shift_fn

            displacement = jnp.array(
                [0.1 * (candidate.k_pf6 + 1), 0.0, 0.0],
                dtype=R.dtype,
            )

            return R.at[candidate.leave_F].add(displacement)

    n_atoms = 16

    R = jnp.zeros((n_atoms, 3), dtype=jnp.float32)

    empty_bonds = (
        jnp.empty((0, 2), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )
    empty_angles = (
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )
    empty_four_body = (
        jnp.empty((0, 4), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    system = SystemState(
        bonds=empty_bonds,
        angles=empty_angles,
        torsions=empty_four_body,
        impropers=empty_four_body,
        charges=jnp.zeros((n_atoms,), dtype=jnp.float32),
        sigmas=jnp.ones((n_atoms,), dtype=jnp.float32),
        epsilons=jnp.ones((n_atoms,), dtype=jnp.float32),
        molecule_id=jnp.arange(n_atoms, dtype=jnp.int32),
        pf6_reacted=jnp.array([False, False]),
    )

    class NeighborFn:
        def allocate(self, positions):
            return {"positions": positions}

    initial_ff = SimpleNamespace(
        disp_fn=lambda a, b: b - a,
        neighbor_fn=NeighborFn(),
        nlist=None,
    )

    built_forcefields = []

    def fake_build_trial_forcefield(
        positions,
        box,
        trial,
        ff_ref,
    ):
        del positions, box, trial

        new_ff = SimpleNamespace(
            disp_fn=ff_ref.disp_fn,
            neighbor_fn=ff_ref.neighbor_fn,
            nlist=None,
        )
        built_forcefields.append(new_ff)
        return new_ff

    relax_calls = []

    def fake_fire_relax_with_nlist(
        R0,
        *,
        ff_trial,
        shift_fn,
        **kwargs,
    ):
        del shift_fn, kwargs

        relax_calls.append(
            {
                "R0": np.asarray(R0),
                "ff_trial": ff_trial,
            }
        )

        return R0, {"final": True}

    monkeypatch.setattr(
        reaction_module,
        "build_trial_forcefield",
        fake_build_trial_forcefield,
    )
    monkeypatch.setattr(
        reaction_module,
        "fire_relax_with_nlist",
        fake_fire_relax_with_nlist,
    )

    (
        _key,
        accepted,
        ff_new,
        sys_new,
        info,
        R_new,
    ) = maybe_react_rate_events(
        jax.random.PRNGKey(0),
        R,
        jnp.eye(3, dtype=jnp.float32),
        shift_fn=lambda position, displacement: (
            position + displacement
        ),
        ff=initial_ff,
        sys=system,
        reaction=TestReaction(),
        reaction_rate_ps=1.0,
        activation_energy_eV=None,
        temperature_k=300.0,
        prefactor_ps=None,
        reactive_interval_ps=1.0,
        max_reactions_per_check=2,
        candidate_log_top_n=10,
        sigma_mid=0.0,
        sigma_width=0.2,
    )

    assert accepted
    assert info["n_accepted_this_check"] == 2
    assert len(info["accepted_events"]) == 2

    assert [
        event["k_pf6"]
        for event in info["accepted_events"]
    ] == [0, 1]

    assert [
        event["li_idx"]
        for event in info["accepted_events"]
    ] == [14, 15]

    np.testing.assert_array_equal(
        np.asarray(sys_new.pf6_reacted),
        np.array([True, True]),
    )

    # One force-field update per accepted topology change.
    assert len(built_forcefields) == 2

    # The complete batch must be relaxed exactly once.
    assert len(relax_calls) == 1

    # Both probe displacements must already be present when FIRE starts.
    R_before_relaxation = relax_calls[0]["R0"]

    np.testing.assert_allclose(
        R_before_relaxation[1],
        [0.1, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        R_before_relaxation[8],
        [0.2, 0.0, 0.0],
    )

    assert jnp.all(jnp.isfinite(R_new))
    np.testing.assert_allclose(
        np.asarray(R_new),
        R_before_relaxation,
    )

    assert ff_new is built_forcefields[-1]
    assert ff_new.nlist == {"final": True}
