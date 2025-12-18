# reactive_md/md_driver.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional
import logging
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax_md import simulate, space

from .config import SimConfig
from .forcefield import FFBundle
from .reaction import SystemState


@dataclass
class RunResult:
    final_md_state: object
    ff: FFBundle
    sys: SystemState
    accepted_events: int


def run_md_nvt_with_reactions(
    key,
    *,
    cfg: SimConfig,
    init_positions,
    masses,
    shift_fn,
    ff: FFBundle,
    sys: SystemState,
    reaction_step_fn,
    logger: Optional[logging.Logger] = None,
    # Dump inputs
    box=None,
    atom_types=None,
    dump_filename: Optional[str] = "trajectory.lammpstrj",
    write_every: int = 50,
):
    """
    Chunked NVT Nose-Hoover integration + periodic reaction attempts.

    Logs (tab-separated):
      step    KE      PE      Etot    T       TimePerStep_s

    Writes a LAMMPS-style trajectory with wrapped + unwrapped coordinates.
    Unwrapped coordinates are accumulated using PBC-aware displacement (minimum image),
    which is robust when atoms cross periodic boundaries.
    """
    if logger is None:
        logger = logging.getLogger("reactive_md")

    if dump_filename is not None:
        if box is None:
            raise ValueError("dump_filename was set but no 'box' was provided to run_md_nvt_with_reactions.")
        if atom_types is None:
            raise ValueError("dump_filename was set but no 'atom_types' was provided to run_md_nvt_with_reactions.")

    box_arr = np.array(box) if box is not None else None
    atom_types_np = np.array(atom_types) if atom_types is not None else None

    # Build disp_fn from the true box for robust unwrapping
    disp_fn, _shift_local = space.periodic(box)

    # Vectorized displacement for all atoms
    disp_all = jax.jit(jax.vmap(disp_fn))

    kT = cfg.kb_real * cfg.temperature_k
    mass = jnp.array(masses)
    masses_np = np.array(masses)

    n_atoms = int(init_positions.shape[0])
    atom_ids = np.arange(1, n_atoms + 1)

    dump_file = open(dump_filename, "w") if dump_filename is not None else None

    def make_integrator(energy_fn):
        def energy_scalar(R, neighbor):
            return energy_fn(R, neighbor)["total"]

        init_nvt, apply_nvt = simulate.nvt_nose_hoover(
            energy_scalar,
            shift_fn,
            dt=cfg.dt,
            kT=kT,
            tau=cfg.tau_T,
            mass=mass,
        )
        return init_nvt, apply_nvt

    init_nvt, apply_nvt = make_integrator(ff.energy_fn)

    key, sub = jax.random.split(key)
    md_state = init_nvt(sub, init_positions, neighbor=ff.nlist)

    # Track wrapped prev positions in JAX, and unwrapped in numpy
    R_prev_wrapped = jnp.array(init_positions)
    R_unwrapped = np.array(init_positions)

    accepted_events = 0
    steps_done = 0

    logger.info("step\tKE\tPE\tEtot\tT\tTimePerStep_s")

    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(cfg.check_every, cfg.steps - steps_done)

        t0 = time.perf_counter()

        def body_fn(carry, _i):
            st, nlist = carry
            nlist = ff.neighbor_fn.update(st.position, nlist)
            st = apply_nvt(st, neighbor=nlist)
            return (st, nlist), None

        (md_state, ff.nlist), _ = jax.lax.scan(
            body_fn, (md_state, ff.nlist), jnp.arange(chunk)
        )
        steps_done += chunk

        R_wrapped = md_state.position  # jnp
        V_wrapped = md_state.velocity  # jnp

        # --- robust unwrap using minimum-image displacement ---
        dR = disp_all(R_prev_wrapped, R_wrapped)   # jnp (n_atoms, 3)
        R_unwrapped += np.array(dR)                # accumulate in numpy
        R_prev_wrapped = R_wrapped                 # update prev wrapped

        # Energies (use wrapped positions)
        E_dict = ff.energy_fn(R_wrapped, ff.nlist)
        PE = float(E_dict["total"])
        KE = float(0.5 * np.sum(masses_np[:, None] * np.array(V_wrapped) ** 2))
        Etot = KE + PE

        # Temperature estimate (simple 3N DOF)
        T_inst = (2.0 * KE) / (3.0 * n_atoms * cfg.kb_real)

        # Reaction attempt (based on current wrapped positions)
        key, accepted, ff_new, sys_new, info, R_new = reaction_step_fn(
            key, R_wrapped, ff, sys
        )

        t1 = time.perf_counter()
        time_per_step = (t1 - t0) / float(chunk)

        # 1) Thermo line first (requested ordering)
        logger.info(
            f"{steps_done}\t{KE:.6f}\t{PE:.6f}\t{Etot:.6f}\t{T_inst:.3f}\t{time_per_step:.6e}"
        )

        # Dump trajectory (state after MD chunk, before applying accepted reaction geometry)
        if dump_file is not None and (steps_done % write_every == 0):
            pos_wrapped_np = np.array(R_wrapped)

            f = dump_file
            f.write(f"ITEM: TIMESTEP\n{steps_done}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")

            if box_arr.ndim == 1 and box_arr.shape[0] == 3:
                f.write(f"0 {box_arr[0]}\n")
                f.write(f"0 {box_arr[1]}\n")
                f.write(f"0 {box_arr[2]}\n")
            elif box_arr.ndim == 2 and box_arr.shape == (3, 3):
                f.write(f"0 {box_arr[0,0]}\n")
                f.write(f"0 {box_arr[1,1]}\n")
                f.write(f"0 {box_arr[2,2]}\n")
            else:
                raise ValueError(f"Unexpected box shape: {box_arr.shape}")

            f.write("ITEM: ATOMS id type mass x y z xu yu zu\n")
            for i in range(n_atoms):
                f.write(
                    f"{atom_ids[i]} {atom_types_np[i]} {masses_np[i]} "
                    f"{pos_wrapped_np[i,0]:.6f} {pos_wrapped_np[i,1]:.6f} {pos_wrapped_np[i,2]:.6f} "
                    f"{R_unwrapped[i,0]:.6f} {R_unwrapped[i,1]:.6f} {R_unwrapped[i,2]:.6f}\n"
                )

        # 2) Reaction outcome lines after thermo
        if accepted:
            accepted_events += 1
            logger.info(
                f"Accepted event #{accepted_events}: {info.get('accepted_event')}, "
                f"dE={info.get('dE'):.4f}, p={info.get('p_acc'):.3f}"
            )

            # If positions jump due to relaxation, update unwrapped consistently:
            # add the PBC-aware displacement from pre-reaction wrapped R_wrapped to new wrapped R_new
            dR_relax = disp_all(R_wrapped, R_new)
            R_unwrapped += np.array(dR_relax)

            # Update state and bookkeeping
            ff = ff_new
            sys = sys_new
            md_state = replace(md_state, position=R_new)
            ff.nlist = ff.neighbor_fn.allocate(R_new)

            # Ensure next unwrap uses the new wrapped positions as "previous"
            R_prev_wrapped = R_new

            # refresh integrator to use new energy function
            init_nvt, apply_nvt = make_integrator(ff.energy_fn)

        else:
            if "candidate" in info:
                k_pf6, li_idx, leave_F, dmin = info["candidate"]
                logger.info(
                    f"Rejected candidate: pf6={k_pf6}, Li={li_idx}, F={leave_F}, "
                    f"d={dmin:.3f}, dE={info['dE']:.4f}, p={info['p_acc']:.3f}"
                )

    if dump_file is not None:
        dump_file.close()

    logger.info("Done.")
    logger.info(f"Total accepted events: {accepted_events}")
    logger.info(f"PF6 reacted count: {int(jnp.sum(sys.pf6_reacted))}")

    return RunResult(final_md_state=md_state, ff=ff, sys=sys, accepted_events=accepted_events)

