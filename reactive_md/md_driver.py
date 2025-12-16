# reactive_md/md_driver.py
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import replace

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
    reaction_step_fn,  # callable(key, positions, ff, sys) -> (key, accepted, ff, sys, info)
):
    """
    Chunked NVT Nose-Hoover integration + periodic reaction attempts.
    """
    kT = cfg.kb_real * cfg.temperature_k
    mass = jnp.array(masses)

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

    accepted_events = 0
    steps_done = 0

    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(cfg.check_every, cfg.steps - steps_done)

        def body_fn(carry, _i):
            st, nlist = carry
            nlist = ff.neighbor_fn.update(st.position, nlist)
            st = apply_nvt(st, neighbor=nlist)
            return (st, nlist), None

        (md_state, ff.nlist), _ = jax.lax.scan(body_fn, (md_state, ff.nlist), jnp.arange(chunk))
        steps_done += chunk

        positions = md_state.position
        V = md_state.velocity

        E_dict = ff.energy_fn(positions, ff.nlist)
        PE = float(E_dict["total"])
        KE = float(0.5 * jnp.sum(mass[:, None] * V * V))
        reacted = int(jnp.sum(sys.pf6_reacted))
        print(f"[step {steps_done:6d}] PE={PE: .6f}  KE={KE: .6f}  reacted={reacted}")

        key, accepted, ff_new, sys_new, info, R_new = reaction_step_fn(key, positions, ff, sys)
        if accepted:
            accepted_events += 1
            print(
                f"   Accepted event #{accepted_events}: {info.get('accepted_event')}, "
                f"dE={info.get('dE'):.4f}, p={info.get('p_acc'):.3f}"
            )
            ff = ff_new
            sys = sys_new
            md_state = replace(md_state, position=R_new)
            ff.nlist = ff.neighbor_fn.allocate(R_new)
            # refresh integrator to use new energy function
            init_nvt, apply_nvt = make_integrator(ff.energy_fn)
        else:
            if "candidate" in info:
                k_pf6, li_idx, leave_F, dmin = info["candidate"]
                print(
                    f"   Rejected candidate: pf6={k_pf6}, Li={li_idx}, F={leave_F}, "
                    f"d={dmin:.3f}, dE={info['dE']:.4f}, p={info['p_acc']:.3f}"
                )

    print("Done.")
    print(f"Total accepted events: {accepted_events}")
    print(f"PF6 reacted count: {int(jnp.sum(sys.pf6_reacted))}")

    return RunResult(final_md_state=md_state, ff=ff, sys=sys, accepted_events=accepted_events)

