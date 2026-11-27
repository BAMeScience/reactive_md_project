"""LiPF6-specific diagnostics, CSV schemas, and unwrapped trajectory output.

All chemistry-specific output lives here; the NVT driver only emits snapshots.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from ..performance import PerformanceTimer

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
        "d_lif,d_pf,sigma,dE,p_acc,p_sigma,p_metropolis,p_total,reacted_count,direction\n"
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
        f"{reacted_count},\n"
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
        f"{reacted_count},{event.get('direction', 'forward')}\n"
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
                 candidate_file=None, resume=False,
                 md_energy_fn=None, md_neighbor_fn=None):
        self.cfg = cfg
        self.resume = bool(resume)
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
        if (md_energy_fn is None) != (md_neighbor_fn is None):
            raise ValueError(
                "md_energy_fn and md_neighbor_fn must be provided together"
            )
        self.md_energy_fn = md_energy_fn
        # Compile the complete scalar evaluation once per observer instead of
        # dispatching individual MACE operations and materializing intermediates.
        # Neighbor lists remain dynamic inputs, including after topology changes.
        self._compiled_md_energy = (
            jax.jit(md_energy_fn) if md_energy_fn is not None else None
        )
        self.md_neighbor_fn = md_neighbor_fn
        if self.resume:
            # A restart must not open an existing output with 'w' or 'x':
            # those modes destroy it before checkpoint restoration can run.
            for label, stream in self._output_streams().items():
                if stream is not None and any(c in getattr(stream, 'mode', '') for c in 'wx'):
                    raise ValueError(
                        f'{label} must be opened for append/update, not write/truncate, on restart'
                    )
        else:
            _write_event_header(event_file)
            _write_candidate_header(candidate_file)

    def _output_streams(self):
        return {
            'dump': self.dump_file,
            'events': self.event_file,
            'candidates': self.candidate_file,
        }

    @staticmethod
    def _stream_identity(stream):
        name = getattr(stream, 'name', None)
        return os.path.realpath(os.fspath(name)) if isinstance(name, (str, os.PathLike)) else None

    def checkpoint_state(self):
        """Capture unwrapping state and exact output boundaries after flushing."""
        outputs = {}
        for label, stream in self._output_streams().items():
            if stream is None:
                outputs[label] = None
                continue
            if stream.closed or not stream.seekable():
                raise ValueError(f'{label} output must be open and seekable for checkpointing')
            stream.flush()
            stream.seek(0, os.SEEK_END)
            outputs[label] = {
                'offset': stream.tell(),
                'path': self._stream_identity(stream),
            }
        return {
            'version': 1,
            'dump_every': int(self.dump_every),
            'previous_position': self.previous_position,
            'unwrapped_position': self.unwrapped_position,
            'pre_reacted_count': int(self.pre_reacted_count),
            'accepted_events': int(self.accepted_events),
            'outputs': outputs,
        }

    def restore_checkpoint_state(self, saved):
        """Restore observer and roll output files back to the saved boundary.

        Output files may contain records written after the last checkpoint.
        Truncating to saved offsets prevents duplicate records on replay.
        """
        if not self.resume:
            raise ValueError('Construct LiPF6OutputObserver(resume=True) for restart')
        if saved.get('version') != 1:
            raise ValueError('Unsupported LiPF6 observer checkpoint version')
        if int(saved['dump_every']) != int(self.dump_every):
            raise ValueError('dump_every differs from checkpoint')
        outputs = saved['outputs']
        streams = self._output_streams()
        if set(outputs) != set(streams):
            raise ValueError('Output stream configuration differs from checkpoint')

        # Validate every file before changing any of them.
        for label, stream in streams.items():
            record = outputs[label]
            if (stream is None) != (record is None):
                raise ValueError(f'{label} output presence differs from checkpoint')
            if stream is None:
                continue
            if stream.closed or not stream.seekable() or not stream.writable():
                raise ValueError(f'{label} output must be open, writable and seekable')
            if any(c in getattr(stream, 'mode', '') for c in 'wx'):
                raise ValueError(f'{label} output must not be opened in truncating mode')
            if record['path'] != self._stream_identity(stream):
                raise ValueError(f'{label} output path differs from checkpoint')
            stream.flush()
            stream.seek(0, os.SEEK_END)
            if stream.tell() < int(record['offset']):
                raise ValueError(f'{label} output is shorter than its checkpoint boundary')

        for label, stream in streams.items():
            if stream is not None:
                offset = int(outputs[label]['offset'])
                stream.seek(offset)
                stream.truncate(offset)
                stream.flush()

        self.previous_position = saved['previous_position']
        self.unwrapped_position = saved['unwrapped_position']
        self.pre_reacted_count = int(saved['pre_reacted_count'])
        self.accepted_events = int(saved['accepted_events'])
        # The driver will supply the restored FF displacement function on the
        # next on_before_reaction() call, before any unwrapping is performed.
        self.disp_fn = None

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
        perf = PerformanceTimer()
        with perf.measure(
            'trajectory_dump', step, requested=self.dump_file is not None and self.dump_writer is not None,
        ):
            _write_dump_frame_if_requested(
                dump_file=self.dump_file, dump_writer=self.dump_writer,
                step=step, box=self.box, atom_ids=self.atom_ids,
                atom_types=self.atom_types, masses=self.masses,
                wrapped_positions=positions, unwrapped_positions=self.unwrapped_position,
            )

    def on_before_reaction(self, observation):
        """Called after the MD chunk, before chemistry changes the topology."""
        perf = PerformanceTimer()
        state = observation.state
        self.disp_fn = observation.ff.disp_fn
        with perf.measure('observer_unwrap', observation.step) as wait:
            self._unwrap_to(state.position)
            wait(self.previous_position, self.unwrapped_position)
        with perf.measure('observer_potential_energy', step=observation.step):
            if self.md_energy_fn is None:
                energy = observation.ff.energy_fn(
                    state.position, observation.ff.nlist
                )['total']
            else:
                extra_capacity = 2
                for attempt in range(13):
                    with perf.measure(
                        'observer_mace_neighbor_allocate', attempt=attempt, extra_capacity=extra_capacity,
                    ) as wait:
                        neighbors = self.md_neighbor_fn.allocate(
                            state.position, extra_capacity=extra_capacity
                        )
                        neighbors = neighbors.update(state.position)
                        wait(neighbors)
                    if not bool(jnp.asarray(neighbors.did_buffer_overflow)):
                        break
                    extra_capacity = max(8, 2 * extra_capacity)
                else:
                    raise RuntimeError(
                        "MACE observer neighbor-list allocation overflowed"
                    )
                energy = self._compiled_md_energy(state.position, neighbors)
            energy.block_until_ready()
        with perf.measure('observer_kinetic_energy', step=observation.step):
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
                        reacted_count=reacted_count,
                    )
            
            else:
                event = events[0]
                d_lif = event.get('d_lif')
                d_pf = event.get('d_pf')
                sigma = event.get('sigma')

                if sigma is None and d_lif is not None and d_pf is not None:
                    sigma = float(d_pf) - float(d_lif)

                def fmt(value, digits=3):
                    return 'N/A' if value is None else f'{float(value):.{digits}f}'

                dE = event.get('dE', info.get('dE'))
                p_sigma = event.get('p_sigma', info.get('p_sigma'))
                p_metropolis = event.get(
                    'p_metropolis', info.get('p_metropolis')
                )
                p_total = event.get(
                    'p_total', info.get('p_total', info.get('p_acc'))
                )

                print(
                    f'[step {step}] TOPOLOGY CHANGE accepted '
                    f'event #{self.accepted_events}: '
                    f"direction={event.get('direction', 'unknown')}, "
                    f"pf6={event.get('k_pf6')}, Li={event.get('li_idx')}, "
                    f"F={event.get('leave_F')}, "
                    f"d_LiF={fmt(d_lif)}, "
                    f"d_PF={fmt(d_pf)}, "
                    f"sigma={fmt(sigma)}, "
                    f"dE={fmt(dE, 4)}, "
                    f"p_sigma={fmt(p_sigma)}, "
                    f"pM={fmt(p_metropolis)}, "
                    f"p={fmt(p_total)}",
                    flush=True,
                )

                _write_accepted_event(
                    event_file=self.event_file,
                    step=step,
                    event_index=self.accepted_events,
                    info=info,
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

