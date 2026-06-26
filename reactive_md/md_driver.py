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


def _write_event_header(event_file):
    if event_file is None:
        return

    event_file.write(
        "step,event,event_index,mode,n_candidates,n_accepted_this_check,"
        "p_rate,k_rate_ps,dt_reactive_ps,"
        "pf6_index,li_idx,leave_F,"
        "d_lif,d_pf,sigma,dE,p_acc,reacted_count\n"
    )
    event_file.flush()


def _write_rate_check(
    *,
    event_file,
    step: int,
    info: dict,
    reacted_count: int,
):
    if event_file is None:
        return
    if info.get("mode") != "rate":
        return

    event_file.write(
        f"{step},rate_check,,rate,"
        f"{info.get('n_candidates', '')},"
        f"{info.get('n_accepted_this_check', 0)},"
        f"{info.get('p_rate', '')},"
        f"{info.get('k_rate_ps', '')},"
        f"{info.get('dt_reactive_ps', '')},"
        ",,,,,,,,"
        f"{reacted_count}\n"
    )
    event_file.flush()


def _write_accepted_event(
    *,
    event_file,
    step: int,
    event_index: int,
    info: dict,
    reacted_count: int,
):
    if event_file is None:
        return

    event = info.get("accepted_event", {})
    d_lif = event.get("d_lif")
    d_pf = event.get("d_pf")

    sigma = event.get("sigma", "")
    if sigma == "" and d_lif is not None and d_pf is not None:
        sigma = float(d_pf) - float(d_lif)

    event_file.write(
        f"{step},accepted,{event_index},"
        f"{info.get('mode', 'metropolis')},"
        f"{info.get('n_candidates', '')},"
        f"{info.get('n_accepted_this_check', 1)},"
        f"{info.get('p_rate', '')},"
        f"{info.get('k_rate_ps', '')},"
        f"{info.get('dt_reactive_ps', '')},"
        f"{event.get('k_pf6')},"
        f"{event.get('li_idx')},"
        f"{event.get('leave_F')},"
        f"{d_lif},"
        f"{d_pf},"
        f"{sigma},"
        f"{info.get('dE', '')},"
        f"{info.get('p_acc', '')},"
        f"{reacted_count}\n"
    )
    event_file.flush()


def _write_candidate_header(candidate_file):
    if candidate_file is None:
        return

    candidate_file.write(
        "step,mode,rank,pf6_index,li_idx,leave_F,"
        "d_lif,d_pf,sigma,accepted\n"
    )
    candidate_file.flush()


def _write_candidate_records(
    *,
    candidate_file,
    step: int,
    info: dict,
):
    if candidate_file is None:
        return

    mode = info.get("mode", "metropolis")

    accepted_event = info.get("accepted_event", {})
    accepted_key = (
        accepted_event.get("k_pf6"),
        accepted_event.get("li_idx"),
        accepted_event.get("leave_F"),
    )

    for rec in info.get("candidate_records", []):
        key = (
            rec.get("k_pf6"),
            rec.get("li_idx"),
            rec.get("leave_F"),
        )

        accepted = int(key == accepted_key)

        sigma = rec.get("sigma")
        if sigma is None:
            d_lif = rec.get("d_lif")
            d_pf = rec.get("d_pf")
            if d_lif is not None and d_pf is not None:
                sigma = float(d_pf) - float(d_lif)
            else:
                sigma = ""

        candidate_file.write(
            f"{step},{mode},{rec.get('rank')},"
            f"{rec.get('k_pf6')},"
            f"{rec.get('li_idx')},"
            f"{rec.get('leave_F')},"
            f"{rec.get('d_lif')},"
            f"{rec.get('d_pf')},"
            f"{sigma},"
            f"{accepted}\n"
        )

    candidate_file.flush()


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
    event_file=None,
    candidate_file=None,
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

    _write_event_header(event_file)
    _write_candidate_header(candidate_file)

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

        _write_rate_check(
            event_file=event_file,
            step=steps_done,
            info=info,
            reacted_count=int(jnp.sum(sys.pf6_reacted)),
        )

        _write_candidate_records(
            candidate_file=candidate_file,
            step=steps_done,
            info=info,
        )

        if accepted:
            accepted_events += 1
            event = info.get("accepted_event", {})

            reacted_count_new = int(jnp.sum(sys_new.pf6_reacted))

            d_lif = event.get("d_lif")
            d_pf = event.get("d_pf")
            sigma = event.get("sigma")
            if sigma is None and d_lif is not None and d_pf is not None:
                sigma = float(d_pf) - float(d_lif)

            sigma_text = ""
            if sigma is not None:
                sigma_text = f", sigma={float(sigma):.3f}"

            if info.get("mode") == "rate":
                print(
                    f"[step {steps_done}] RATE EVENT accepted "
                    f"event #{accepted_events}: "
                    f"pf6={event.get('k_pf6')}, "
                    f"Li={event.get('li_idx')}, "
                    f"F={event.get('leave_F')}, "
                    f"d_LiF={float(d_lif):.3f}, "
                    f"d_PF={float(d_pf):.3f}"
                    f"{sigma_text}, "
                    f"p_rate={float(info.get('p_rate')):.4f}, "
                    f"n_candidates={info.get('n_candidates')}, "
                    f"n_accepted_this_check={info.get('n_accepted_this_check')}"
                )
            else:
                print(
                    f"[step {steps_done}] TOPOLOGY CHANGE accepted "
                    f"event #{accepted_events}: "
                    f"pf6={event.get('k_pf6')}, "
                    f"Li={event.get('li_idx')}, "
                    f"F={event.get('leave_F')}, "
                    f"d_LiF={float(d_lif):.3f}, "
                    f"d_PF={float(d_pf):.3f}"
                    f"{sigma_text}, "
                    f"dE={float(info.get('dE')):.4f}, "
                    f"p={float(info.get('p_acc')):.3f}"
                )

            _write_accepted_event(
                event_file=event_file,
                step=steps_done,
                event_index=accepted_events,
                info=info,
                reacted_count=reacted_count_new,
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
                sigma = cand_info.get("sigma")
                if sigma is None:
                    sigma = float(cand_info["d_pf"]) - float(cand_info["d_lif"])

                extra = ""
                if "dE" in info:
                    extra += f", dE={float(info['dE']):.4f}"
                if "p_acc" in info:
                    extra += f", p_acc={float(info['p_acc']):.3f}"
                if "proposal_factor" in info:
                    extra += f", proposal_factor={float(info['proposal_factor']):.3f}"

                print(
                    " No reaction accepted for candidate: "
                    f"pf6={cand_info['k_pf6']}, Li={cand_info['li_idx']}, "
                    f"F={cand_info['leave_F']}, "
                    f"d_LiF={cand_info['d_lif']:.3f}, "
                    f"d_PF={cand_info['d_pf']:.3f}, "
                    f"sigma={float(sigma):.3f}"
                    f"{extra}"
                )
            elif "best_sigma_candidate" in info:
                best = info["best_sigma_candidate"]
                sigma = best.get("sigma")
                if sigma is None:
                    sigma = float(best["d_pf"]) - float(best["d_lif"])
                print(
                    " No reaction accepted. "
                    f"Best sigma candidate: Li={best['li_idx']}, "
                    f"F={best['leave_F']}, "
                    f"d_LiF={best['d_lif']:.3f}, "
                    f"d_PF={best['d_pf']:.3f}, "
                    f"sigma={float(sigma):.3f}"
                )
            elif "closest" in info:
                closest = info["closest"]
                sigma = closest.get("sigma")
                if sigma is None:
                    sigma = float(closest["d_pf"]) - float(closest["d_lif"])
                print(
                    " No reaction accepted. "
                    f"Diagnostic candidate: Li={closest['li_idx']}, "
                    f"F={closest['leave_F']}, "
                    f"d_LiF={closest['d_lif']:.3f}, "
                    f"d_PF={closest['d_pf']:.3f}, "
                    f"sigma={float(sigma):.3f}"
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
