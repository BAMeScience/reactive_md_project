"""Compatibility entry point for LiPF6, now using the generic MD engine.

This module contains no NVT stepping or reaction loop. It configures a
reaction handler and chemistry-specific observer, then delegates to run_md_nvt.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..config import SimConfig
from ..forcefield import FFBundle
from ..reaction import SystemState
from ..md_driver import run_md_nvt
from .lipf6_handler import LiPF6Handler
from .lipf6_output import LiPF6OutputObserver


@dataclass
class RunResult:
    final_md_state: object
    ff: FFBundle
    sys: SystemState
    accepted_events: int


def run_md_nvt_with_reactions(
    key, *, cfg: SimConfig, init_positions, masses, shift_fn,
    ff: FFBundle, sys: SystemState, reaction_step_fn,
    box=None, atom_ids=None, atom_types=None, dump_file=None,
    dump_writer=None, event_file=None, candidate_file=None,
):
    """Maintain the legacy LiPF6 public signature and return type."""
    observer = LiPF6OutputObserver(
        cfg=cfg, box=box, masses=masses,
        atom_ids=atom_ids, atom_types=atom_types,
        dump_file=dump_file, dump_writer=dump_writer,
        event_file=event_file, candidate_file=candidate_file,
    )
    result = run_md_nvt(
        key, cfg=cfg, init_positions=init_positions, masses=masses,
        shift_fn=shift_fn, ff=ff, sys=sys, box=box,
        reaction_handler=LiPF6Handler(reaction_step_fn),
        observers=(observer,),
    )
    print('Done.')
    print(f'Total accepted events: {result.accepted_events}')
    print(f'PF6 reacted count: {int(jnp.sum(result.sys.pf6_reacted))}')
    return RunResult(
        final_md_state=result.final_md_state,
        ff=result.ff, sys=result.sys,
        accepted_events=result.accepted_events,
    )
