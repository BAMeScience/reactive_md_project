"""LiPF6-specific diagnostics, CSV schemas, and unwrapped trajectory output.

All chemistry-specific output lives here; the NVT driver only emits snapshots.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
        "d_lif,d_pf,sigma,dE,p_acc,p_sigma,p_metropolis,p_total,reacted_count\n"
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
        ",,,,,,,,,,,"
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
        f"{info.get('p_sigma', '')},"
        f"{info.get('p_metropolis', '')},"
        f"{info.get('p_total', '')},"
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

    accepted_events = info.get("accepted_events")
    if accepted_events is None:
        accepted_event = info.get("accepted_event")
        accepted_events = (
            [accepted_event]
            if accepted_event is not None
            else []
        )

    accepted_keys = {
        (
            event.get("k_pf6"),
            event.get("li_idx"),
            event.get("leave_F"),
        )
        for event in accepted_events
    }

    for rec in info.get("candidate_records", []):
        key = (
            rec.get("k_pf6"),
            rec.get("li_idx"),
            rec.get("leave_F"),
        )

        accepted = int(key in accepted_keys)

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



class LiPF6OutputObserver:
    """Preserve legacy LiPF6 output ordering and CSV schemas.

    Called once for the initial state, before every reaction attempt (for
    pre-reaction PE/KE and periodic dumps), and after every attempt (for
    candidates, accepted events, and the post-reaction dump).
    """

    def __init__(self, *, cfg, box, masses, atom_ids=None, atom_types=None,
                 dump_file=None, dump_writer=None, event_file=None,
                 candidate_file=None):
        self.cfg = cfg
        self.box = box
        self.masses = jnp.asarray(masses)
        self.atom_ids = atom_ids
        self.atom_types = atom_types
        self.dump_file = dump_file
        self.dump_writer = dump_writer
        self.event_file = event_file
        self.candidate_file = candidate_file
        self.dump_every = cfg.dump_every if cfg.dump_every is not None else cfg.check_every
        if self.dump_every <= 0:
            raise ValueError('dump_every must be positive')
        self.unwrapped_position = None
        self.previous_position = None
        self.pre_reacted_count = 0
        self.accepted_events = 0
        self.disp_fn = None
        _write_event_header(event_file)
        _write_candidate_header(candidate_file)

    def _unwrap_to(self, positions):
        if self.previous_position is None:
            self.previous_position = jnp.array(positions)
            self.unwrapped_position = jnp.array(positions)
        else:
            delta = jax.vmap(self.disp_fn, in_axes=(0, 0))(
                self.previous_position, positions
            )
            self.unwrapped_position = self.unwrapped_position + delta
            self.previous_position = positions

    def _dump(self, step, positions):
        _write_dump_frame_if_requested(
            dump_file=self.dump_file, dump_writer=self.dump_writer,
            step=step, box=self.box, atom_ids=self.atom_ids,
            atom_types=self.atom_types, masses=self.masses,
            wrapped_positions=positions, unwrapped_positions=self.unwrapped_position,
        )

    def on_before_reaction(self, observation):
        """Called after the MD chunk, before chemistry changes the topology."""
        state = observation.state
        self.disp_fn = observation.ff.disp_fn
        self._unwrap_to(state.position)
        energy = observation.ff.energy_fn(state.position, observation.ff.nlist)['total']
        energy.block_until_ready()
        kinetic = 0.5 * jnp.sum(self.masses[:, None] * state.velocity ** 2)
        kinetic.block_until_ready()
        self.pre_reacted_count = int(jnp.sum(observation.system.pf6_reacted))
        print(f'[step {observation.step:6d}] PE={float(energy): .6f} '
              f'KE={float(kinetic): .6f} reacted={self.pre_reacted_count}')
        if observation.step % int(self.dump_every) == 0:
            self._dump(observation.step, state.position)

    def on_observation(self, observation):
        if observation.outcome is None:
            self.disp_fn = observation.ff.disp_fn
            self._unwrap_to(observation.state.position)
            self._dump(0, observation.state.position)
            return

        info = observation.outcome.metadata or {}
        events = observation.outcome.events
        reacted_count = int(jnp.sum(observation.system.pf6_reacted))
        step = observation.step
        _write_candidate_records(
            candidate_file=self.candidate_file, step=step, info=info,
        )

        if events:
            previous_accepted = self.accepted_events
            self.accepted_events += len(events)
            if info.get('mode') == 'rate':
                for batch_index, event in enumerate(events, start=1):
                    event_index = previous_accepted + batch_index
                    d_lif, d_pf = event.get('d_lif'), event.get('d_pf')
                    sigma = event.get('sigma')
                    if sigma is None and d_lif is not None and d_pf is not None:
                        sigma = float(d_pf) - float(d_lif)
                    sigma_text = '' if sigma is None else f', sigma={float(sigma):.3f}'
                    print(
                        f'[step {step}] RATE EVENT accepted event #{event_index}: '
                        f"pf6={event.get('k_pf6')}, Li={event.get('li_idx')}, "
                        f"F={event.get('leave_F')}, d_LiF={float(d_lif):.3f}, "
                        f'd_PF={float(d_pf):.3f}{sigma_text}, '
                        f"p_rate={float(event.get('p_rate')):.4f}, "
                        f"n_candidates={info.get('n_candidates')}, "
                        f"n_accepted_this_check={info.get('n_accepted_this_check')}"
                    )
                    event_info = dict(info)
                    event_info['accepted_event'] = event
                    event_info['p_rate'] = event.get('p_rate', info.get('p_rate'))
                    event_info['k_eff_ps'] = event.get('k_eff_ps', info.get('k_eff_ps'))
                    event_info['pf_rate_factor'] = event.get('pf_rate_factor', info.get('pf_rate_factor'))
                    _write_accepted_event(
                        event_file=self.event_file, step=step,
                        event_index=event_index, info=event_info,
                        reacted_count=self.pre_reacted_count + batch_index,
                    )
            else:
                event = events[0]
                d_lif, d_pf = event.get('d_lif'), event.get('d_pf')
                sigma = event.get('sigma')
                if sigma is None and d_lif is not None and d_pf is not None:
                    sigma = float(d_pf) - float(d_lif)
                sigma_text = '' if sigma is None else f', sigma={float(sigma):.3f}'
                print(
                    f'[step {step}] TOPOLOGY CHANGE accepted '
                    f'event #{self.accepted_events}: '
                    f"pf6={event.get('k_pf6')}, Li={event.get('li_idx')}, "
                    f"F={event.get('leave_F')}, d_LiF={float(d_lif):.3f}, "
                    f'd_PF={float(d_pf):.3f}{sigma_text}, '
                    f"dE={float(info.get('dE')):.4f}, "
                    f"p_sigma={float(info.get('p_sigma')):.3f}, "
                    f"pM={float(info.get('p_metropolis')):.3f}, "
                    f"p={float(info.get('p_total', info.get('p_acc'))):.3f}"
                )
                _write_accepted_event(
                    event_file=self.event_file, step=step,
                    event_index=self.accepted_events, info=info,
                    reacted_count=reacted_count,
                )

            self.disp_fn = observation.ff.disp_fn
            self._unwrap_to(observation.state.position)
            self._dump(step, observation.state.position)
        else:
            self._print_rejected(info)

        _write_rate_check(
            event_file=self.event_file, step=step, info=info,
            reacted_count=reacted_count,
        )

    @staticmethod
    def _print_rejected(info):
        if 'candidate' in info:
            cand = info['candidate']
            sigma = cand.get('sigma')
            if sigma is None:
                sigma = float(cand['d_pf']) - float(cand['d_lif'])
            extra = ''
            for name, label in (('dE', 'dE'), ('p_sigma', 'p_sigma'),
                                ('p_metropolis', 'pM'), ('p_total', 'p_total')):
                if name in info:
                    extra += f", {label}={float(info[name]):.{4 if name == 'dE' else 3}f}"
            if 'p_total' not in info and 'p_acc' in info:
                extra += f", p_acc={float(info['p_acc']):.3f}"
            print(
                ' No reaction accepted for candidate: '
                f"pf6={cand['k_pf6']}, Li={cand['li_idx']}, "
                f"F={cand['leave_F']}, d_LiF={cand['d_lif']:.3f}, "
                f"d_PF={cand['d_pf']:.3f}, sigma={float(sigma):.3f}{extra}"
            )
        elif 'best_sigma_candidate' in info or 'closest' in info:
            key = 'best_sigma_candidate' if 'best_sigma_candidate' in info else 'closest'
            cand = info[key]
            sigma = cand.get('sigma')
            if sigma is None:
                sigma = float(cand['d_pf']) - float(cand['d_lif'])
            prefix = ('Best sigma candidate' if key == 'best_sigma_candidate'
                      else 'Diagnostic candidate')
            print(
                f' No reaction accepted. {prefix}: '
                f"Li={cand['li_idx']}, F={cand['leave_F']}, "
                f"d_LiF={cand['d_lif']:.3f}, d_PF={cand['d_pf']:.3f}, "
                f'sigma={float(sigma):.3f}'
            )
