"""Reusable Nose-Hoover NVT integrator and neighbor-list recovery.

This module is independent of all reaction chemistries and output formats.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ..config import SimConfig
from .contracts import ReactionOutcome

if TYPE_CHECKING:
    from ..forcefield import FFBundle


class NVTStepper:
    """Nose-Hoover NVT stepping with transactional neighbor-list recovery.

    A chunk is accepted only when *every* intermediate neighbor-list update,
    including the final coordinates, is overflow-free. Retried chunks start
    from the same thermostat state and the same positions.
    """

    def __init__(self, *, cfg: SimConfig, ff: FFBundle, shift_fn, masses, max_retries=12):
        self.cfg = cfg
        self.ff = ff
        self.shift_fn = shift_fn
        self.mass = jnp.asarray(masses)
        self.max_retries = int(max_retries)
        if self.max_retries < 0:
            raise ValueError('max_retries must be nonnegative')
        if not np.isfinite(cfg.dt) or cfg.dt <= 0 or cfg.check_every <= 0:
            raise ValueError('MD dt and check_every must be positive')
        self.set_forcefield(ff)

    def energy_scalar(self, positions, neighbor):
        return self.ff.energy_fn(positions, neighbor)['total']

    def set_forcefield(self, ff: FFBundle):
        self.ff = ff
        cfg = self.cfg
        from jax_md import simulate

        self.init_nvt, apply_nvt = simulate.nvt_nose_hoover(
            self.energy_scalar,
            self.shift_fn,
            dt=cfg.dt,
            kT=cfg.kb_real * cfg.temperature_k,
            tau=cfg.tau_T,
        )
        neighbor_fn = ff.neighbor_fn

        @partial(jax.jit, static_argnames=('chunk',))
        def md_chunk(md_state, nlist, *, chunk):
            def body(carry, _):
                state, nl, overflow = carry
                nl = neighbor_fn.update(state.position, nl)
                overflow = jnp.logical_or(overflow, nl.did_buffer_overflow)
                state = apply_nvt(state, neighbor=nl)
                return (state, nl, overflow), None

            (state, nl, overflow), _ = jax.lax.scan(
                body, (md_state, nlist, jnp.asarray(False)), xs=None, length=chunk
            )
            nl = neighbor_fn.update(state.position, nl)
            overflow = jnp.logical_or(overflow, nl.did_buffer_overflow)
            return state, nl, overflow

        self.md_chunk = md_chunk

    def allocate_valid_nlist(self, positions, *, extra_capacity=0):
        for attempt in range(self.max_retries + 1):
            nl = self.ff.neighbor_fn.allocate(
                positions, extra_capacity=extra_capacity
            )
            if not bool(np.asarray(nl.did_buffer_overflow)):
                return nl
            extra_capacity = max(8, 2 * extra_capacity)
        raise RuntimeError('Neighbor-list allocation overflowed after capacity growth')

    def initialize(self, key, positions):
        self.ff.nlist = self.allocate_valid_nlist(positions)
        return self.init_nvt(key, positions,mass=self.mass ,neighbor=self.ff.nlist)

    def refresh_force(self, state, nlist):
        if not hasattr(state, 'force'):
            raise RuntimeError('NVT state does not expose its cached force')
        return replace(
            state,
            force=-jax.grad(self.energy_scalar)(state.position, neighbor=nlist),
        )

    def advance(self, state, nlist, *, chunk):
        if bool(np.asarray(nlist.did_buffer_overflow)):
            raise RuntimeError('MD chunk started with an overflowing neighbor list')
        extra_capacity = max(8, int(nlist.idx.shape[-1]))
        for attempt in range(self.max_retries + 1):
            trial_state, trial_nlist, overflow = self.md_chunk(
                state, nlist, chunk=int(chunk)
            )
            if not bool(np.asarray(overflow)):
                if not bool(np.asarray(jnp.all(jnp.isfinite(trial_state.position)))):
                    raise FloatingPointError('Nonfinite positions in accepted NVT chunk')
                if not bool(np.asarray(jnp.all(jnp.isfinite(trial_state.momentum)))):
                    raise FloatingPointError('Nonfinite momenta in accepted NVT chunk')
                return trial_state, trial_nlist
            if attempt == self.max_retries:
                raise RuntimeError(
                    f'Neighbor-list overflow persisted after {self.max_retries} retries'
                )
            nlist = self.allocate_valid_nlist(
                state.position, extra_capacity=extra_capacity
            )
            state = self.refresh_force(state, nlist)
            extra_capacity *= 2
        raise AssertionError('Unreachable neighbor-list recovery state')

    def synchronize(self, state, outcome: ReactionOutcome, *, key=None):
        """Commit a reaction and refresh the force under the new topology.

        Existing momenta and Nose-Hoover chain state are retained by default.
        A paper A2 reaction with accepted Langevin relaxation explicitly
        requests a thermostat reset and supplies relaxed velocities.
        """
        self.set_forcefield(outcome.ff)
        nlist = self.allocate_valid_nlist(outcome.positions)
        if outcome.reinitialize_thermostat:
            if key is None:
                raise ValueError('A PRNG key is required for thermostat reinitialization')
            state = self.init_nvt(key, outcome.positions, mass=self.mass, neighbor=nlist)
        else:
            state = replace(state, position=outcome.positions)
        # For topology-only handlers, preserve_momenta avoids a needless
        # velocity-to-momentum conversion. Paper A2 handlers supply the
        # relaxed velocities and explicitly request their handoff.
        if not outcome.preserve_momenta:
            state = replace(
                state,
                momentum=jnp.asarray(outcome.velocities) * self.mass[:, None],
            )
        state = self.refresh_force(state, nlist)
        self.ff.nlist = nlist
        return state, nlist
