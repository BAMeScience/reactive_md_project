"""Stable chemistry-independent entry point for reactive NVT dynamics.

Core data contracts live in :mod:`reactive_md.engine.contracts`; the
Nose-Hoover stepper lives in :mod:`reactive_md.engine.nvt`. The generic
orchestration below deliberately knows nothing about A2 or LiPF6.

Historical chemistry-specific entry points remain available lazily.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .config import SimConfig
from .engine import (
    ReactionContext, ReactionOutcome, ReactionHandler,
    MDObservation, MDObserver, MDRunResult, NVTStepper,
)
from .md_units import md_unit_factors


if TYPE_CHECKING:
    from .forcefield import FFBundle


def _observer_checkpoint_state(observers):
    """Collect state from observers, never including open output handles."""
    saved = {}
    for index, observer in enumerate(observers):
        getter = getattr(observer, 'checkpoint_state', None)
        if not callable(getter):
            raise TypeError(
                f'Observer {index} ({type(observer).__name__}) must implement '
                'checkpoint_state() for checkpoint/restart'
            )
        saved[str(index)] = getter()
    return saved


def _restore_observers(observers, saved):
    if saved is None:
        if observers:
            raise ValueError('Checkpoint has no observer state')
        return
    if set(saved) != {str(i) for i in range(len(observers))}:
        raise ValueError('Checkpoint observer count differs from current run')
    for index, observer in enumerate(observers):
        restore = getattr(observer, 'restore_checkpoint_state', None)
        if not callable(restore):
            raise TypeError(
                f'Observer {index} ({type(observer).__name__}) must implement '
                'restore_checkpoint_state(state) for restart'
            )
        restore(saved[str(index)])


def _restart_metadata(cfg, box, masses, float_dtype):
    """Inputs that must remain unchanged across a continuation."""
    return {
        'dt': float(cfg.dt),
        'md_unit_system': str(cfg.md_unit_system),
        'tau_T': float(cfg.tau_T),
        'float_dtype': np.dtype(float_dtype).name,
        'temperature_k': float(cfg.temperature_k),
        'kb_real': float(cfg.kb_real),
        'check_every': int(cfg.check_every),
        'max_events': int(cfg.max_events),
        'box': None if box is None else np.asarray(jax.device_get(box)).copy(),
        'masses': np.asarray(jax.device_get(masses)).copy(),
    }


def _validate_restart_metadata(saved, current):
    if set(saved) != set(current):
        raise ValueError('Checkpoint simulation metadata is incompatible')
    for name, value in current.items():
        previous = saved[name]
        if name in ('box', 'masses'):
            equal = (previous is None and value is None) or (
                previous is not None and value is not None
                and np.array_equal(np.asarray(previous), np.asarray(value))
            )
        else:
            equal = previous == value
        if not equal:
            raise ValueError(f'Checkpoint does not match current {name}')


def _rebuild_forcefield(md_state, system, options, energy_offset, box):
    """Reconstruct executable functions from saved numerical topology."""
    from .forcefield import build_forcefield

    return build_forcefield(
        md_state.position, box,
        bond_idx=system.bonds[0], k_b=system.bonds[1], r0=system.bonds[2],
        angle_idx=system.angles[0], k_theta=system.angles[1],
        theta0=system.angles[2],
        torsions=system.torsions, impropers=system.impropers,
        charges=system.charges, sigmas=system.sigmas,
        epsilons=system.epsilons, molecule_id=system.molecule_id,
        r_cut=float(options['r_cut']),
        dr_threshold=float(options['dr_threshold']),
        use_shift_lj=options['use_shift_lj'],
        energy_offset=energy_offset,
    )


def run_md_nvt(
    key,
    *,
    cfg: SimConfig,
    init_positions,
    masses,
    shift_fn,
    ff: FFBundle,
    sys: object,
    reaction_handler: ReactionHandler,
    box=None,
    observers: tuple[MDObserver, ...] = (),
    checkpoint_path: str | None = None,
    checkpoint_every_steps: int | None = None,
    restart_path: str | None = None,
    energy_offset: float | None = None,
    md_energy_fn=None,
    md_neighbor_fn=None,
) -> MDRunResult:
    """Generic NVT driver with optional checkpoint/restart.

    ``cfg.steps`` is the TOTAL desired step count, including completed steps.
    Checkpoints occur only after a complete reaction-check/observer cycle.
    For continuous reaction timing, the checkpoint interval must be a multiple
    of ``cfg.check_every``. Observer classes must implement checkpoint_state()
    and restore_checkpoint_state(state) when checkpointing is enabled.

    The caller must construct the reaction handler (including its unreacted
    reference topology) from the original input before invoking this driver.
    """
    if (md_energy_fn is None) != (md_neighbor_fn is None):
        raise ValueError(
            "md_energy_fn and md_neighbor_fn must be provided together"
        )

    mace_md = md_energy_fn is not None

    if mace_md:
        if checkpoint_path is not None or restart_path is not None:
            raise NotImplementedError(
                "Checkpoint/restart is not yet supported for MACE-MD"
            )

        if observers:
            raise NotImplementedError(
                "MACE-MD observer output is not yet supported"
            )

    if cfg.steps < 0 or cfg.max_events < 0:
        raise ValueError('steps and max_events must be nonnegative')
    if int(cfg.check_every) <= 0:
        raise ValueError('check_every must be positive')
    if checkpoint_every_steps is not None:
        if checkpoint_path is None:
            raise ValueError('checkpoint_every_steps requires checkpoint_path')
        if (checkpoint_every_steps <= 0
                or checkpoint_every_steps % int(cfg.check_every) != 0):
            raise ValueError('checkpoint_every_steps must be a positive multiple of check_every')

    if (checkpoint_path is not None or restart_path is not None) and observers:
        for index, observer in enumerate(observers):
            if not callable(getattr(observer, 'checkpoint_state', None)) or not callable(
                getattr(observer, 'restore_checkpoint_state', None)
            ):
                raise TypeError(
                    f'Observer {index} ({type(observer).__name__}) needs both '
                    'checkpoint_state() and restore_checkpoint_state(state)'
                )

    # Select a single floating-point dtype for the MD inputs. This must be
    # configured before the caller constructs the force field and shift_fn;
    # casting here cannot retroactively change their captured parameters.
    float_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    init_positions = jnp.asarray(init_positions, dtype=float_dtype)
    masses = jnp.asarray(masses, dtype=float_dtype)
    if box is not None:
        box = jnp.asarray(box, dtype=float_dtype)

    metadata = _restart_metadata(cfg, box, masses, float_dtype)
    accepted_events = 0
    steps_done = 0
    relaxation_elapsed = 0.0
    active_energy_offset = energy_offset

    if restart_path is not None:
        from .checkpoint import load_checkpoint

        saved = load_checkpoint(restart_path)
        _validate_restart_metadata(saved['metadata'], metadata)
        steps_done = int(saved['steps_done'])
        if steps_done > int(cfg.steps):
            raise ValueError('Checkpoint is beyond requested total cfg.steps')
        accepted_events = int(saved['accepted_events'])
        relaxation_elapsed = float(saved['relaxation_elapsed_ps'])
        key = saved['key']
        state = saved['md_state']
        # Never silently cast a restored thermostat/MD state: doing so would
        # invalidate a strict continuation and could change the trajectory.
        if np.dtype(state.position.dtype) != np.dtype(float_dtype):
            raise ValueError(
                f'Checkpoint position dtype {state.position.dtype} does not '
                f'match requested {np.dtype(float_dtype).name}'
            )
        sys = type(sys)(**saved['system_state'])
        active_energy_offset = saved['energy_offset']
        ff = _rebuild_forcefield(
            state, sys, saved['forcefield_options'], active_energy_offset, box
        )
        # Do not call initialize(): that would reset Nose-Hoover variables.
        stepper = NVTStepper(cfg=cfg, ff=ff, shift_fn=shift_fn, masses=masses, md_energy_fn=md_energy_fn, md_neighbor_fn=md_neighbor_fn,)
        md_nlist = ff.nlist
        _restore_observers(observers, saved['observer_state'])
    else:
        stepper = NVTStepper(cfg=cfg, ff=ff, shift_fn=shift_fn, masses=masses, md_energy_fn=md_energy_fn, md_neighbor_fn=md_neighbor_fn,)
        key, sub = jax.random.split(key)
        state = stepper.initialize(sub, init_positions)
        md_nlist = stepper.nlist

    def notify(outcome=None, pre_positions=None):
        observation = MDObservation(
            step=steps_done,
            md_time_ps=steps_done * float(cfg.dt),
            relaxation_time_ps=relaxation_elapsed,
            state=state,
            ff=ff,
            system=sys,
            outcome=outcome,
            pre_reaction_positions=pre_positions,
        )
        for observer in observers:
            observer.on_observation(observation)

    def save_now():
        if checkpoint_path is None:
            return
        from .checkpoint import save_checkpoint

        options = ff.nb_options
        save_checkpoint(
            checkpoint_path,
            md_state=state,
            system_state=sys,
            key=key,
            steps_done=steps_done,
            accepted_events=accepted_events,
            relaxation_elapsed_ps=relaxation_elapsed,
            observer_state=_observer_checkpoint_state(observers),
            forcefield_options={
                'r_cut': float(options.r_cut),
                'dr_threshold': float(options.dr_threshold),
                'use_shift_lj': getattr(options, 'use_shift_lj', None),
            },
            energy_offset=active_energy_offset,
            metadata=metadata,
        )

    if restart_path is None:
        notify()
    # On restart, the saved step has already been emitted. Do not duplicate it.
    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(int(cfg.check_every), int(cfg.steps - steps_done))

        state, md_nlist = stepper.advance(
            state, md_nlist, chunk=chunk
        )

        steps_done += chunk

        if mace_md:
            # Reaction code still uses the classical force-field object.
            # Its neighbor list must be valid at the current MACE-MD positions,
            # and must never be replaced by the MACE neighbor list.
            extra_capacity = 0
            for _ in range(13):
                classical_nlist = ff.neighbor_fn.allocate(
                    state.position, extra_capacity=extra_capacity
                )
                if not bool(np.asarray(classical_nlist.did_buffer_overflow)):
                    ff.nlist = classical_nlist
                    break
                extra_capacity = max(8, 2 * extra_capacity)
            else:
                raise RuntimeError(
                    'Classical reaction neighbor-list allocation overflowed'
                )
        else:
            # Preserve the existing classical MD neighbor-list behavior.
            ff.nlist = md_nlist

        before = state.position
        # Optional pre-reaction observation: chemistry-specific diagnostics and
        # periodic trajectory frames must see the old topology and force field.
        # The regular on_observation callback remains post-reaction so existing
        # observers (including A2) retain their original contract.
        before_observation = MDObservation(
            step=steps_done,
            md_time_ps=steps_done * float(cfg.dt),
            relaxation_time_ps=relaxation_elapsed,
            state=state,
            ff=ff,
            system=sys,
            outcome=None,
            pre_reaction_positions=before,
        )
        for observer in observers:
            before_callback = getattr(observer, 'on_before_reaction', None)
            if before_callback is not None:
                before_callback(before_observation)

        _, kb_md = md_unit_factors(cfg.md_unit_system)        
        outcome = reaction_handler.attempt(ReactionContext(
            key=key,
            step=steps_done,
            interval_ps=chunk * float(cfg.dt),
            positions=state.position,
            velocities=state.velocity,
            system=sys,
            ff=ff,
            box=box,
            shift_fn=shift_fn,
            masses=stepper.mass,
            temperature_k=cfg.temperature_k,
            kb_real=kb_md,
        ))
        key = outcome.key
        if outcome.events:
            accepted_events += len(outcome.events)
            ff, sys = outcome.ff, outcome.system
            # The current reaction trial builder does not propagate offsets.
            active_energy_offset = None
            if outcome.reinitialize_thermostat:
                key, sub = jax.random.split(key)
            else:
                sub = None
            state, md_nlist = stepper.synchronize(state, outcome, key=sub)
            if not mace_md:
                ff.nlist = md_nlist
            relaxation_elapsed += float(outcome.relaxation_time_ps)
        notify(outcome, before)
        if (checkpoint_every_steps is not None
                and steps_done % checkpoint_every_steps == 0):
            save_now()

    # Also persist the final complete boundary (including early max_events stop).
    save_now()
    return MDRunResult(
        state, ff, sys, accepted_events, steps_done, key, relaxation_elapsed
    )


# Compatibility with imports from versions where chemistry lived in this file.
# Kept lazy to avoid importing chemistry into the generic MD engine.
def __getattr__(name):
    if name == 'run_md_nvt_with_reactions' or name == 'RunResult':
        from .reactions import lipf6_driver
        return getattr(lipf6_driver, name)
    if name in ('run_md_nvt_a2', 'A2MDRecordedEvent', 'A2MDSample', 'A2MDRunResult'):
        from .reactions import a2_driver
        return getattr(a2_driver, name)
    raise AttributeError(name)

