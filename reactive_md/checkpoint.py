"""Atomic checkpoint I/O for LiPF6 NVT simulations.

This module handles serialization only. The driver is responsible for choosing a
safe checkpoint boundary, rebuilding the force field and reaction definition,
restoring the observer, and reconciling output files on restart.

SECURITY: Load only checkpoint files you trust. pickle is not a safe format for
untrusted inputs. Restart with the same code and compatible JAX/JAX-MD versions.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

CHECKPOINT_VERSION = 1
_SYSTEM_FIELDS = (
    "bonds", "angles", "torsions", "impropers",
    "charges", "sigmas", "epsilons", "molecule_id",
    "pf6_reacted", "pf6_leave_f", "pf6_li_partner",
)


def _host(value: Any) -> Any:
    """Transfer array leaves to host without altering pytree structure."""
    return jax.device_get(value)


def _device_arrays(value: Any) -> Any:
    """Convert NumPy array leaves back to JAX arrays, retaining other leaves."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        value,
    )


def _system_snapshot(system_state: Any) -> dict[str, Any]:
    missing = [name for name in _SYSTEM_FIELDS if not hasattr(system_state, name)]
    if missing:
        raise TypeError(f"SystemState is missing fields: {', '.join(missing)}")
    return {name: _host(getattr(system_state, name)) for name in _SYSTEM_FIELDS}


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    md_state: Any,
    system_state: Any,
    key: Any,
    steps_done: int,
    accepted_events: int,
    relaxation_elapsed_ps: float,
    observer_state: Mapping[str, Any] | None,
    forcefield_options: Mapping[str, Any],
    energy_offset: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write one checkpoint, replacing the previous complete file.

    ``observer_state`` must contain plain serializable data, not file handles.
    ``forcefield_options`` must include r_cut, dr_threshold, use_shift_lj.
    ``metadata`` can hold restart compatibility data such as box, dt, and T.
    """
    if int(steps_done) < 0 or int(accepted_events) < 0:
        raise ValueError("steps_done and accepted_events must be nonnegative")
    required_options = {"r_cut", "dr_threshold", "use_shift_lj"}
    missing = required_options.difference(forcefield_options)
    if missing:
        raise ValueError(f"Missing force-field options: {sorted(missing)}")

    payload = {
        "version": CHECKPOINT_VERSION,
        "md_state": _host(md_state),
        "system_state": _system_snapshot(system_state),
        "key": _host(key),
        "steps_done": int(steps_done),
        "accepted_events": int(accepted_events),
        "relaxation_elapsed_ps": float(relaxation_elapsed_ps),
        "observer_state": None if observer_state is None else _host(dict(observer_state)),
        "forcefield_options": dict(forcefield_options),
        "energy_offset": energy_offset,
        "metadata": {} if metadata is None else dict(metadata),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary_name = stream.name
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        # Best-effort directory sync for crash consistency on POSIX filesystems.
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a trusted checkpoint, restoring array leaves as JAX arrays.

    The driver should construct ``SystemState(**data['system_state'])`` and
    rebuild FFBundle using the saved parameters and force-field options.
    """
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format/version")
    expected = {
        "md_state", "system_state", "key", "steps_done", "accepted_events",
        "relaxation_elapsed_ps", "observer_state", "forcefield_options",
        "energy_offset", "metadata",
    }
    missing = expected.difference(payload)
    if missing:
        raise ValueError(f"Incomplete checkpoint: {sorted(missing)}")
    if set(payload["system_state"]) != set(_SYSTEM_FIELDS):
        raise ValueError("Checkpoint has an incompatible SystemState schema")
    for name in ("md_state", "system_state", "key", "observer_state"):
        payload[name] = _device_arrays(payload[name])
    return payload

