"""One-shot, zero-force JAX-MD thermostat calibration; no A2 forcefield or reactions.

Runs two masses x two temperatures in a single GPU job, with independent
Langevin chains. Tests initialization and stationary momenta against analytic
Gaussian expectations. Also audits Nose-Hoover *initialization only*; free
2-atom Nose-Hoover dynamics is not an ergodicity certification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate, space

from reactive_md.config import SimConfig
from a2_thermostat_calibration_math import (
    mass_from_amu, summarize_momenta, assess_langevin,
)


def zero_energy(R, **kwargs):
    # Depend explicitly on R so the autodiff force is exactly zero.
    return jnp.sum(R * 0.0)


def initial_states(kind, *, mass, kT, dt, gamma, tau, replicas, seed, box):
    _, shift = space.periodic(box)

    kw = dict(dt=dt, kT=kT)
    particle_masses = jnp.full((2,), mass)

    if kind == 'langevin':
        init, apply = simulate.nvt_langevin(
            zero_energy, shift, gamma=gamma, **kw
        )
    elif kind == 'nose-hoover':
        init, apply = simulate.nvt_nose_hoover(
            zero_energy, shift, tau=tau, **kw
        )
    else:
        raise ValueError(kind)

    keys = jax.random.split(jax.random.PRNGKey(seed), replicas)
    positions = jnp.asarray([
        [1.5, 1.5, 1.5],
        [3.5, 1.5, 1.5],
    ])

    states = jax.vmap(
        lambda key: init(key, positions, mass=particle_masses)
    )(keys)

    # Fail immediately if the integrator uses an unexpected mass.
    np.testing.assert_allclose(
        np.asarray(states.mass),
        mass,
        rtol=1e-5,
    )

    return states, jax.vmap(apply)

def run_case(*, mass, mass_label, temperature, cfg, args, case_id):
    kT = cfg.kb_real * temperature
    common = dict(mass=mass, kT=kT, dt=args.dt, gamma=args.gamma,
                  tau=args.tau_T, replicas=args.replicas,
                  box=jnp.full((3,), 7.0))
    state, apply = initial_states('langevin', seed=args.seed + 101 * case_id, **common)
    initial = summarize_momenta(np.asarray(state.momentum)[None], mass, kT)

    @jax.jit
    def advance(state):
        def one_step(s, _):
            return apply(s), None
        return jax.lax.scan(one_step, state, None, length=args.sample_every)[0]

    post = []
    first = None
    for step in range(args.sample_every, args.steps + 1, args.sample_every):
        state = advance(state)
        if step > args.burn_in_steps:
            p = np.asarray(state.momentum)
            if not np.isfinite(p).all():
                raise FloatingPointError('nonfinite momenta')
            post.append(p)
            if first is None:
                first = p
    samples = np.stack(post, axis=0)
    summary = summarize_momenta(samples, mass, kT)
    early = summarize_momenta(samples[:max(1, len(samples) // 2)], mass, kT)
    late = summarize_momenta(samples[len(samples) // 2:], mass, kT)
    nh_state, _ = initial_states('nose-hoover', seed=args.seed + 101 * case_id + 1,
                                  **common)
    nh_init = summarize_momenta(np.asarray(nh_state.momentum)[None], mass, kT)
    result = {
        'mass_label': mass_label, 'mass_integrator': mass,
        'temperature_K': temperature, 'kT_kcal_per_mol': kT,
        'langevin_initial': initial,
        'langevin_post_burn_in': summary,
        'langevin_first_half': early,
        'langevin_second_half': late,
        'langevin_assessment': assess_langevin(summary),
        'nose_hoover_initial_only': nh_init,
    }
    print(f'{mass_label}, T={temperature:g} K: '
          f'init K={initial["total"]["mean"]:.3f}; '
          f'Langevin K_total={summary["total"]["mean"]:.3f} '
          f'+/-{summary["total"]["replica_se"]:.3f}, '
          f'K_rel={summary["relative"]["mean"]:.3f}, '
          f'K_COM={summary["com"]["mean"]:.3f}; '
          f'NH init K_total={nh_init["total"]["mean"]:.3f}; '
          f'{result["langevin_assessment"]["status"]}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicas', type=int, default=64)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--sample-every', type=int, default=100)
    parser.add_argument('--burn-in-steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.00025)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--tau-T', type=float, default=0.1)
    parser.add_argument('--mass-amu', type=float, default=4.0)
    parser.add_argument('--effective-mass', type=float, default=4.0)
    parser.add_argument('--temperatures', type=float, nargs=2, default=[300., 600.])
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_thermostat_calibration.json'))
    args = parser.parse_args(argv)
    if args.replicas < 8 or args.steps < 1 or args.sample_every < 1:
        parser.error('replicas >=8 and positive steps/sample-every required')
    if args.steps % args.sample_every or args.burn_in_steps % args.sample_every:
        parser.error('steps and burn-in-steps must be multiples of sample-every')
    if not 0 <= args.burn_in_steps < args.steps:
        parser.error('burn-in-steps must be in [0, steps)')
    if not all(np.isfinite(v) and v > 0 for v in (
        args.dt, args.gamma, args.tau_T, args.mass_amu,
        args.effective_mass, *args.temperatures
    )):
        parser.error('all physical parameters must be positive and finite')

    cfg = SimConfig()
    physical_mass = mass_from_amu(args.mass_amu)
    print(f'Units: {args.mass_amu:g} amu -> {physical_mass:.9g} '
          'kcal/mol * ps^2 / A^2', flush=True)
    print(f'Effective mass {args.effective_mass:g} corresponds to '
          f'{args.effective_mass / physical_mass:.1f}x the mass of '
          f'{args.mass_amu:g} amu.', flush=True)
    print('Zero-force analytic target: total, relative and COM kinetic ratios = 1',
          flush=True)
    cases = []
    masses = [('effective_mass', args.effective_mass),
              ('converted_amu', physical_mass)]
    for label, mass in masses:
        for temperature in args.temperatures:
            cases.append(run_case(mass=mass, mass_label=label,
                                  temperature=temperature, cfg=cfg,
                                  args=args, case_id=len(cases)))
    result = {
        'stage': 'one_shot_zero_force_thermostat_calibration',
        'production_code_changed': False,
        'settings': vars(args) | {'output': str(args.output)},
        'kb_real': cfg.kb_real,
        'mass_conversion': {
            'amu': args.mass_amu,
            'mass_internal': physical_mass,
            'effective_mass': args.effective_mass,
            'effective_over_physical': args.effective_mass / physical_mass,
        },
        'cases': cases,
        'interpretation': (
            'Langevin zero-force independent-replica calibration of the momentum '
            'marginal, not a full configurational ensemble proof. Nose-Hoover '
            'initialization is reported but its free-particle propagation is '
            'deliberately not treated as an ergodicity test. Different mass '
            'conventions alter dynamical time scales but not normalized canonical '
            'momentum ratios. Assessment thresholds are exploratory.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('One-shot calibration: COMPLETE; no production physics modified', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
