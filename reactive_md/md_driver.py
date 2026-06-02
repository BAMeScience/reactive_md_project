# reactive_md/md_driver.py
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax_md import simulate

from .config import SimConfig
from .forcefield import FFBundle
from .reaction import SystemState


@dataclass
class RunResult:
    final_md_state: object
    ff: FFBundle
    sys: SystemState
    accepted_events: int


def _write_dump_frame_if_requested(
    *,
    dump_file,
    dump_writer,
    step: int,
    box,
    atom_ids,
    atom_types,
    masses,
    wrapped_positions,
    unwrapped_positions,
):
    if dump_file is None or dump_writer is None:
        return

    dump_writer(
        file=dump_file,
        step=step,
        box=box,
        atom_ids=atom_ids,
        atom_types=atom_types,
        masses=np.asarray(masses),
        wrapped_positions=np.asarray(jax.device_get(wrapped_positions)),
        unwrapped_positions=np.asarray(jax.device_get(unwrapped_positions)),
    )

    dump_file.flush()


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
    box=None,
    atom_ids=None,
    atom_types=None,
    dump_file=None,
    dump_writer=None,
):
    kT = cfg.kb_real * cfg.temperature_k
    mass = jnp.asarray(masses)
    batched_disp_fn = jax.vmap(ff.disp_fn, in_axes=(0, 0))

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

    def make_md_chunk(apply_nvt, neighbor_fn):
        @partial(jax.jit, static_argnames=("chunk",))
        def md_chunk(md_state, nlist, chunk: int):
            def body_fn(carry, _):
                st, nl = carry
                nl = neighbor_fn.update(st.position, nl)
                st = apply_nvt(st, neighbor=nl)
                return (st, nl), None

            (st_out, nl_out), _ = jax.lax.scan(
                body_fn,
                (md_state, nlist),
                xs=None,
                length=chunk,
            )
            return st_out, nl_out

        return md_chunk

    init_nvt, apply_nvt = make_integrator(ff.energy_fn)
    md_chunk = make_md_chunk(apply_nvt, ff.neighbor_fn)

    key, sub = jax.random.split(key)
    md_state = init_nvt(sub, init_positions, neighbor=ff.nlist)

    unwrapped_position = jnp.array(init_positions)
    previous_position = jnp.array(init_positions)

    dump_every = cfg.dump_every
    if dump_every is None:
        dump_every = cfg.check_every

    _write_dump_frame_if_requested(
        dump_file=dump_file,
        dump_writer=dump_writer,
        step=0,
        box=box,
        atom_ids=atom_ids,
        atom_types=atom_types,
        masses=masses,
        wrapped_positions=md_state.position,
        unwrapped_positions=unwrapped_position,
    )

    accepted_events = 0
    steps_done = 0

    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(int(cfg.check_every), int(cfg.steps - steps_done))

        md_state, ff.nlist = md_chunk(md_state, ff.nlist, chunk)
        steps_done += chunk

        displacement = batched_disp_fn(previous_position, md_state.position)
        unwrapped_position = unwrapped_position + displacement
        previous_position = md_state.position

        positions = md_state.position
        velocities = md_state.velocity

        E_dict = ff.energy_fn(positions, ff.nlist)
        E_total = E_dict["total"]
        E_total.block_until_ready()
        PE = float(E_total)

        KE_arr = 0.5 * jnp.sum(mass[:, None] * velocities * velocities)
        KE_arr.block_until_ready()
        KE = float(KE_arr)

        reacted = int(jnp.sum(sys.pf6_reacted))
        print(f"[step {steps_done:6d}] PE={PE: .6f} KE={KE: .6f} reacted={reacted}")

        if steps_done % int(dump_every) == 0:
            _write_dump_frame_if_requested(
                dump_file=dump_file,
                dump_writer=dump_writer,
                step=steps_done,
                box=box,
                atom_ids=atom_ids,
                atom_types=atom_types,
                masses=masses,
                wrapped_positions=md_state.position,
                unwrapped_positions=unwrapped_position,
            )

        key, accepted, ff_new, sys_new, info, R_new = reaction_step_fn(
            key,
            positions,
            ff,
            sys,
        )

        if accepted:
            accepted_events += 1
            print(
                f" Accepted event #{accepted_events}: {info.get('accepted_event')}, "
                f"dE={info.get('dE'):.4f}, p={info.get('p_acc'):.3f}"
            )


            ff = ff_new
            sys = sys_new
            batched_disp_fn = jax.vmap(ff.disp_fn, in_axes=(0, 0))

            reaction_displacement = batched_disp_fn(previous_position, R_new)
            unwrapped_position = unwrapped_position + reaction_displacement
            previous_position = R_new

            md_state = replace(md_state, position=R_new)
            ff.nlist = ff.neighbor_fn.allocate(R_new)

            _write_dump_frame_if_requested(
                dump_file=dump_file,
                dump_writer=dump_writer,
                step=steps_done,
                box=box,
                atom_ids=atom_ids,
                atom_types=atom_types,
                masses=masses,
                wrapped_positions=md_state.position,
                unwrapped_positions=unwrapped_position,
            )

            init_nvt, apply_nvt = make_integrator(ff.energy_fn)
            md_chunk = make_md_chunk(apply_nvt, ff.neighbor_fn)

        else:
            if "candidate" in info:
                cand_info = info["candidate"]
                print(
                    " Rejected candidate: "
                    f"pf6={cand_info['k_pf6']}, Li={cand_info['li_idx']}, "
                    f"F={cand_info['leave_F']}, "
                    f"d_LiF={cand_info['d_lif']:.3f}, "
                    f"d_PF={cand_info['d_pf']:.3f}, "
                    f"dE={info['dE']:.4f}, p={info['p_acc']:.3f}"
                )
            elif "closest" in info:
                closest = info["closest"]
                print(
                    " No valid reaction candidate. "
                    f"Closest Li-F: Li={closest['li_idx']}, "
                    f"F={closest['leave_F']}, "
                    f"d_LiF={closest['d_lif']:.3f}, "
                    f"d_PF={closest['d_pf']:.3f}"
                )

    print("Done.")
    print(f"Total accepted events: {accepted_events}")
    print(f"PF6 reacted count: {int(jnp.sum(sys.pf6_reacted))}")

    return RunResult(
        final_md_state=md_state,
        ff=ff,
        sys=sys,
        accepted_events=accepted_events,
    )
