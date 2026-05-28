# reactive_md/md_driver.py
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

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
):
    """
    Chunked NVT Nose-Hoover integration plus periodic reaction attempts.

    Important implementation detail:
    the MD propagation chunk is JIT-compiled as a reusable function with a
    static chunk length. This avoids building a huge XLA program for very long
    runs and reduces the chance of GPU executable-memory OOMs.
    """
    kT = cfg.kb_real * cfg.temperature_k
    mass = jnp.asarray(masses)

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

    accepted_events = 0
    steps_done = 0

    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(int(cfg.check_every), int(cfg.steps - steps_done))

        md_state, ff.nlist = md_chunk(md_state, ff.nlist, chunk)
        steps_done += chunk

        positions = md_state.position
        velocities = md_state.velocity

        # Force synchronization here so errors occur at a clear boundary.
        E_dict = ff.energy_fn(positions, ff.nlist)
        E_total = E_dict["total"]
        E_total.block_until_ready()
        PE = float(E_total)

        KE_arr = 0.5 * jnp.sum(mass[:, None] * velocities * velocities)
        KE_arr.block_until_ready()
        KE = float(KE_arr)

        reacted = int(jnp.sum(sys.pf6_reacted))
        print(f"[step {steps_done:6d}] PE={PE: .6f} KE={KE: .6f} reacted={reacted}")

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

            md_state = replace(md_state, position=R_new)
            ff.nlist = ff.neighbor_fn.allocate(R_new)

            # The topology and energy function changed, so rebuild integrator/chunk.
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
