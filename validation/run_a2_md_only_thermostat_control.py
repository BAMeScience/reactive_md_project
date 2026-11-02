"""Reaction-free two-atom Nose-Hoover versus Langevin coordinate-mixing control.

Run on HPC as: python -m validation.run_a2_md_only_thermostat_control --steps 1000
This is a diagnostic, NOT a claim of equilibrium or thermostat correctness.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate, space

from reactive_md.config import SimConfig
from reactive_md.engine.nvt import NVTStepper
from validation.a2_mh_moving_reference import TwoAtomModel, integrate_reference
from validation.a2_mh_mixing_reference import radial_reference
from validation.a2_md_only_reference import (
    initial_displacement, kinetic_diagnostics, conditional_radial_summary,
)
from validation.run_a2_mh_moving_pilot import (
    system_two_atoms, make_forcefield, verify_real_energies,
)


class LangevinStepper(NVTStepper):
    """Same overflow-safe chunk semantics as the production NVTStepper.

    Only the integrator differs: JAX-MD's nvt_langevin, using the same
    force field, mass, timestep, neighbor lists and energy units.
    """
    def __init__(self, *, gamma, **kwargs):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('gamma must be positive and finite')
        self.gamma = float(gamma)
        super().__init__(**kwargs)

    def set_forcefield(self, ff):
        self.ff = ff
        self.init_nvt, apply_nvt = simulate.nvt_langevin(
            self.energy_scalar, self.shift_fn,
            dt=self.cfg.dt, kT=self.cfg.kb_real * self.cfg.temperature_k,
            gamma=self.gamma, mass=self.mass,
        )
        neighbor_fn = ff.neighbor_fn

        @partial(jax.jit, static_argnames=('chunk',))
        def md_chunk(md_state, nlist, *, chunk):
            def body(carry, _):
                state, nl, overflow = carry
                nl = neighbor_fn.update(state.position, nl)
                overflow = jnp.logical_or(overflow, nl.did_buffer_overflow)
                state = apply_nvt(state, neighbor=nl)
                return (state, nl, overflow), None

            (state, nl, overflow), _ = jax.lax.scan(
                body, (md_state, nlist, jnp.asarray(False)),
                xs=None, length=chunk,
            )
            nl = neighbor_fn.update(state.position, nl)
            overflow = jnp.logical_or(overflow, nl.did_buffer_overflow)
            return state, nl, overflow

        self.md_chunk = md_chunk


def run_case(model, *, thermostat, bonded, geometry, chain_number, seed,
             steps, sample_every, burn_in_samples, mass, dt, tau_T, gamma,
             radial, output):
    disp0 = initial_displacement(geometry)
    xyz = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5] + disp0], dtype=float)
    positions = jnp.asarray(xyz, dtype=jnp.float32)
    box = jnp.full((3,), model.box, dtype=jnp.float32)
    _, shift = space.periodic(box)
    system = system_two_atoms(bonded, model)
    ff = make_forcefield(system, model, positions, bonded)
    cfg = SimConfig(
        steps=steps, check_every=sample_every, dt=dt,
        temperature_k=model.temperature_k, kb_real=model.kb_real,
        tau_T=tau_T,
    )
    kwargs = dict(cfg=cfg, ff=ff, shift_fn=shift,
                  masses=jnp.full((2,), mass, dtype=jnp.float32))
    if thermostat == 'nose-hoover':
        stepper = NVTStepper(**kwargs)
    elif thermostat == 'langevin':
        stepper = LangevinStepper(gamma=gamma, **kwargs)
    else:
        raise ValueError('Unsupported thermostat')

    state = stepper.initialize(jax.random.PRNGKey(seed), positions)
    rows = []
    n_samples = steps // sample_every
    for sample_idx in range(1, n_samples + 1):
        state, ff.nlist = stepper.advance(state, ff.nlist, chunk=sample_every)
        pos = np.asarray(state.position)
        p = np.asarray(state.momentum)
        force = np.asarray(state.force)
        if not (np.isfinite(pos).all() and np.isfinite(p).all()
                and np.isfinite(force).all()):
            raise FloatingPointError('Nonfinite state in MD-only control')
        if bool(np.asarray(ff.nlist.did_buffer_overflow)):
            raise AssertionError('Committed neighbor list overflow')
        d = np.asarray(ff.disp_fn(state.position[0], state.position[1]))
        r = float(np.linalg.norm(d))
        if not 0 <= r <= model.box * np.sqrt(3.0) / 2 + 1e-3:
            raise AssertionError('Invalid periodic distance')
        k = kinetic_diagnostics(p, mass=mass, kt=model.kt)
        rows.append({
            'sample': sample_idx, 'step': sample_idx * sample_every,
            'time_ps': sample_idx * sample_every * dt,
            'r_A': r, 'dx_A': float(d[0]), 'dy_A': float(d[1]),
            'dz_A': float(d[2]),
            'kinetic_ratio_6dof': k['kinetic_ratio_6dof'],
            'relative_kinetic_ratio_3dof': k['relative_kinetic_ratio_3dof'],
            'com_momentum_norm': k['com_momentum_norm'],
            'force_norm': float(np.linalg.norm(force)),
            'post_burn_in': int(sample_idx > burn_in_samples),
        })

    trace = output.with_name(f'{output.stem}_{thermostat.replace("-", "_")}_'
                             f'{"bonded" if bonded else "unbound"}_{geometry}_'
                             f'chain{chain_number}.csv')
    with trace.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    kept = [row for row in rows if row['post_burn_in']]
    distances = np.asarray([row['r_A'] for row in kept])
    report = conditional_radial_summary(distances, reference=radial, bonded=bonded)
    report.update({
        'thermostat': thermostat, 'topology': 'bonded' if bonded else 'unbound',
        'start_geometry': geometry, 'chain': chain_number, 'seed': seed,
        'initial_r_A': float(np.linalg.norm(disp0)),
        'samples': len(rows), 'retained_samples': len(kept),
        'mean_kinetic_ratio_6dof': float(np.mean([r['kinetic_ratio_6dof'] for r in kept])),
        'mean_relative_kinetic_ratio_3dof': float(np.mean([
            r['relative_kinetic_ratio_3dof'] for r in kept])),
        'mean_com_momentum_norm': float(np.mean([r['com_momentum_norm'] for r in kept])),
        'signed_displacement_std_A': [float(np.std([r[k] for r in kept]))
                                      for k in ('dx_A', 'dy_A', 'dz_A')],
        'trace_csv': str(trace),
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--sample-every', type=int, default=20)
    parser.add_argument('--burn-in-samples', type=int, default=10)
    parser.add_argument('--chains-per-condition', type=int, default=1)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--mass', type=float, default=4.0)
    parser.add_argument('--dt', type=float, default=0.00025)
    parser.add_argument('--tau-T', type=float, default=0.1,
                        help='Nose-Hoover thermostat time constant (ps)')
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='Langevin friction coefficient in inverse simulation time')
    parser.add_argument('--radial-bins', type=int, default=12)
    parser.add_argument('--quadrature-order', type=int, default=96)
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_md_only_thermostat_control.json'))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.sample_every < 1 or args.steps % args.sample_every:
        parser.error('steps must be a positive multiple of sample-every')
    if args.chains_per_condition < 1 or args.mass <= 0 or args.dt <= 0:
        parser.error('chains-per-condition, mass and dt must be positive')
    if not (0 <= args.burn_in_samples < args.steps // args.sample_every):
        parser.error('burn-in-samples must be below the number of samples')
    if args.tau_T <= 0 or args.gamma <= 0 or args.radial_bins < 2:
        parser.error('tau-T, gamma must be positive and radial-bins >= 2')
    if args.quadrature_order < 8:
        parser.error('quadrature-order must be >= 8')

    model = TwoAtomModel()
    ref = integrate_reference(model, order=args.quadrature_order)
    radial = radial_reference(model, order=args.quadrature_order,
                              bins=args.radial_bins)
    checks = verify_real_energies(model)
    print('Real JAX-MD force-field spot checks: PASS', flush=True)
    print('MD-only controls: topology fixed; comparing P(r | topology)', flush=True)
    print(f'Joint-ensemble bonded reference (NOT the MD-only target): '
          f'{ref["p_bonded"]:.6f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for thermostat in ('nose-hoover', 'langevin'):
        for bonded in (False, True):
            for geometry in ('near', 'far'):
                for replicate in range(args.chains_per_condition):
                    idx = len(cases)
                    report = run_case(
                        model, thermostat=thermostat, bonded=bonded,
                        geometry=geometry, chain_number=replicate + 1,
                        seed=args.seed + 101 * idx, steps=args.steps,
                        sample_every=args.sample_every,
                        burn_in_samples=args.burn_in_samples, mass=args.mass,
                        dt=args.dt, tau_T=args.tau_T, gamma=args.gamma,
                        radial=radial, output=args.output,
                    )
                    cases.append(report)
                    print(
                        f'{thermostat}, {report["topology"]}, {geometry}, '
                        f'chain={replicate+1}: mean r={report["mean_r_A"]:.3f} A, '
                        f'r range=[{report["min_r_A"]:.3f},'
                        f'{report["max_r_A"]:.3f}] A, '
                        f'conditional radial TV={report["conditional_radial_tv"]:.4f}, '
                        f'relative kinetic ratio={report["mean_relative_kinetic_ratio_3dof"]:.3f}',
                        flush=True,
                    )
    result = {
        'stage': 'two_atom_md_only_thermostat_control',
        'baseline': 'develop commit 1d35ef8 plus additive validation files',
        'settings': {
            'steps': args.steps, 'sample_every': args.sample_every,
            'burn_in_samples': args.burn_in_samples,
            'chains_per_condition': args.chains_per_condition,
            'mass': args.mass, 'dt': args.dt, 'tau_T': args.tau_T,
            'gamma': args.gamma, 'seed': args.seed,
        },
        'model': vars(model), 'joint_reference': ref,
        'conditional_radial_reference': radial,
        'forcefield_spot_checks': checks, 'cases': cases,
        'equilibrium_validated': False,
        'interpretation': (
            'Reaction-free control; fixed-topology radial distributions must be '
            'compared with P(r|topology). Short runs, correlated samples and '
            'two-particle Nose-Hoover ergodicity limitations preclude an '
            'equilibrium claim. Momentum ratios are diagnostic, not formal '
            'thermostat tests.'
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('MD-only thermostat control: COMPLETE (equilibrium NOT yet validated)', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
