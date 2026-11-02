"""Opt-in, reproducible four-atom production-MH equilibrium sampling check.

Run from the repository root in the HPC JAX-MD environment:
    python -m validation.run_a2_mh_sampling --mode synthetic
    python -m validation.run_a2_mh_sampling --mode real

No production files are changed. The fixed-position experiment calls the
production MH cycle for every proposal, but caches topology energies and
replaces the expensive trial force-field constructor *only in this process*.
The real mode calculates its cached energies using actual JAX-MD force fields
and separately runs uncached forward/reverse production force-field smoke checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from reactive_md.forcefield import build_forcefield
from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_rate import make_a2_system_state_from_trial
from reactive_md.reactions import a2_metropolis as mh
from validation.a2_mh_exact_reference import STATES, energy as synthetic_energy
from validation.a2_mh_sampling_reference import sampling_diagnostics


POSITIONS = jnp.asarray([[0., 0., 0.], [2., 0., 0.],
                         [0., 2., 0.], [2., 2., 0.]], dtype=jnp.float32)
BOX = jnp.asarray([30., 30., 30.], dtype=jnp.float32)
VELOCITIES = jnp.zeros_like(POSITIONS)
DISP = lambda a, b: a - b


def matching(system_or_pairs):
    pairs = system_or_pairs.bonds[0] if isinstance(system_or_pairs, SystemState) else system_or_pairs
    return tuple(sorted(tuple(sorted(map(int, pair))) for pair in
                        np.asarray(pairs, dtype=np.int32).reshape(-1, 2)))


def empty_system():
    bonds = (jnp.empty((0, 2), dtype=jnp.int32),
             jnp.empty((0,), dtype=jnp.float32), jnp.empty((0,), dtype=jnp.float32))
    angles = (jnp.empty((0, 3), dtype=jnp.int32),
              jnp.empty((0,), dtype=jnp.float32), jnp.empty((0,), dtype=jnp.float32))
    four = (jnp.empty((0, 4), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32), jnp.empty((0,), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32))
    return SystemState(bonds=bonds, angles=angles, torsions=four, impropers=four,
                       charges=jnp.zeros(4), sigmas=jnp.ones(4) * 1.2,
                       epsilons=jnp.ones(4) * 0.2, molecule_id=jnp.arange(4),
                       pf6_reacted=jnp.zeros((0,), dtype=bool))


def make_system(state, reaction):
    system = empty_system()
    for pair in state:
        candidates = reaction.find_association_candidates(POSITIONS, DISP, system=system)
        candidate = next(c for c in candidates if (c.i, c.j) == pair)
        trial = mh._finalize_trial(reaction.build_association_trial(system, candidate), reaction, 4)
        system = make_a2_system_state_from_trial(trial, system)
    if matching(system) != state:
        raise AssertionError('Incorrect initial topology')
    return system


class NoOverflowNeighbor:
    def allocate(self, positions, **kwargs):
        return SimpleNamespace(did_buffer_overflow=False)

    def update(self, positions, nlist):
        return nlist


class CachedEnergyFF:
    """Stand-in for a force field, backed by independently computed U(state)."""
    def __init__(self, total):
        self.total = float(total)
        self.neighbor_fn = NoOverflowNeighbor()
        self.nlist = SimpleNamespace(did_buffer_overflow=False)

    def energy_fn(self, positions, nlist):
        return {'total': self.total}


def real_forcefield(system, offset):
    return build_forcefield(
        POSITIONS, BOX, bond_idx=np.asarray(system.bonds[0]),
        k_b=np.asarray(system.bonds[1]), r0=np.asarray(system.bonds[2]),
        angle_idx=np.asarray(system.angles[0]),
        k_theta=np.asarray(system.angles[1]), theta0=np.asarray(system.angles[2]),
        torsions=tuple(np.asarray(x) for x in system.torsions),
        impropers=tuple(np.asarray(x) for x in system.impropers),
        charges=np.asarray(system.charges), sigmas=np.asarray(system.sigmas),
        epsilons=np.asarray(system.epsilons), molecule_id=np.asarray(system.molecule_id),
        r_cut=9.0, dr_threshold=0.5, energy_offset=offset,
    )


class ForcedDirectionRNG:
    def __init__(self, association):
        self.association = association
        self.calls = 0

    def random(self):
        self.calls += 1
        return (0.0 if self.association else 0.999999) if self.calls == 1 else 0.0

    def integers(self, high):
        return 0


def actual_forcefield_smoke(reaction, systems, energies, real_ff, *, temperature_k, kb_real, p):
    """Uncached actual trial construction, including a two-dimer initial state."""
    for start, association in ((STATES[0], True), (STATES[-1], False)):
        result = mh.a2_metropolis_reactive_cycle(
            key=None, R=POSITIONS, velocities=VELOCITIES, box=BOX,
            system=systems[start], ff=real_ff[start], reaction=reaction,
            disp_fn=DISP, temperature_k=temperature_k, kb_real=kb_real,
            rng=ForcedDirectionRNG(association), association_proposal_probability=p)
        if len(result.attempts) != 1:
            raise AssertionError('Expected one real-force-field trial')
        target = matching(result.system)
        attempt = result.attempts[0]
        # Forced RNG accepts every finite move with nonzero probability.
        if not attempt.accepted or target not in energies:
            raise AssertionError('Uncached MH trial failed')
        delta = energies[target] - energies[start]
        if not math.isclose(attempt.delta_energy, delta, abs_tol=2e-3):
            raise AssertionError(f'Real trial energy mismatch: {attempt.delta_energy} vs {delta}')


def calculate_energies(mode, reaction, systems, *, offset, temperature_k, kb_real, p):
    if mode == 'synthetic':
        return {state: float(synthetic_energy(state)) for state in STATES}
    real_ff = {}
    energies = {}
    for state in STATES:
        ff = real_forcefield(systems[state], len(state) * offset)
        value = float(ff.energy_fn(POSITIONS, ff.nlist)['total'])
        if not math.isfinite(value):
            raise AssertionError(f'Nonfinite JAX-MD energy for {state}')
        real_ff[state] = ff
        energies[state] = value
    # Check that the initial force field for a pre-bonded state includes the
    # entire n_dimers * bond_energy_offset contribution.
    for state in (STATES[1], STATES[-1]):
        ff_no_offset = real_forcefield(systems[state], 0.0)
        baseline = float(ff_no_offset.energy_fn(POSITIONS, ff_no_offset.nlist)['total'])
        if not math.isclose(energies[state] - baseline, len(state) * offset, abs_tol=2e-3):
            raise AssertionError(f'Incorrect initial energy offset for {state}')
    actual_forcefield_smoke(reaction, systems, energies, real_ff,
                            temperature_k=temperature_k, kb_real=kb_real, p=p)
    return energies


def sample(*, mode, samples, burn_in, chains, seed, p, temperature_k, kb_real, offset):
    if samples < 1 or burn_in < 0 or chains < 2:
        raise ValueError('Use positive samples, nonnegative burn-in and at least two chains')
    if not 0 < p < 1 or temperature_k <= 0 or kb_real <= 0:
        raise ValueError('Invalid proposal probability or thermal energy')
    reaction = A2Reaction(a_indices=np.arange(4), k_bond_jax=1.5,
                          r0_bond=2.0, association_cutoff=3.0,
                          bond_energy_offset=offset)
    systems = {state: make_system(state, reaction) for state in STATES}
    energies = calculate_energies(mode, reaction, systems, offset=offset,
                                  temperature_k=temperature_k, kb_real=kb_real, p=p)
    ff_cache = {state: CachedEnergyFF(energies[state]) for state in STATES}
    index = {state: i for i, state in enumerate(STATES)}
    counts = np.zeros(len(STATES), dtype=np.int64)
    by_chain = []
    accepted = 0
    attempts = 0
    original_build = mh.build_trial_forcefield

    def cached_trial_ff(R, box, trial, ff_ref):
        state = matching(trial['bonds'][0])
        expected_offset = len(state) * offset
        if not math.isclose(float(trial['energy_offset']), expected_offset, abs_tol=1e-9):
            raise AssertionError('Trial force field received incorrect total energy offset')
        return ff_cache[state]

    try:
        # This monkeypatch is confined to the standalone validation process.
        mh.build_trial_forcefield = cached_trial_ff
        for chain in range(chains):
            rng = np.random.default_rng(seed + 104729 * chain)
            start = STATES[0] if chain % 2 == 0 else STATES[-1]
            state = start
            system = systems[state]
            ff = ff_cache[state]
            chain_counts = np.zeros(len(STATES), dtype=np.int64)
            for step in range(burn_in + samples):
                result = mh.a2_metropolis_reactive_cycle(
                    key=None, R=POSITIONS, velocities=VELOCITIES, box=BOX,
                    system=system, ff=ff, reaction=reaction, disp_fn=DISP,
                    temperature_k=temperature_k, kb_real=kb_real,
                    rng=rng, association_proposal_probability=p)
                new_state = matching(result.system)
                if new_state not in index:
                    raise AssertionError(f'Invalid topology: {new_state}')
                if result.positions is not POSITIONS or result.velocities is not VELOCITIES:
                    raise AssertionError('Fixed-coordinate MH unexpectedly moved particles')
                if result.events and len(result.events) != 1:
                    raise AssertionError('More than one MH event per attempt')
                if step >= burn_in:
                    chain_counts[index[new_state]] += 1
                    attempts += len(result.attempts)
                    accepted += len(result.events)
                state, system, ff = new_state, result.system, result.ff
            counts += chain_counts
            by_chain.append(dict(start_matching=[list(pair) for pair in start],
                                 final_matching=[list(pair) for pair in state],
                                 counts=chain_counts.tolist()))
            print(f'Chain {chain+1}/{chains} finished (initial dimers={len(start)})', flush=True)
    finally:
        mh.build_trial_forcefield = original_build

    ordered_energies = [energies[state] for state in STATES]
    stats = sampling_diagnostics(counts, ordered_energies, p_association=p,
                                 kt=temperature_k * kb_real,
                                 samples_per_chain=samples, chains=chains)
    # 5 sigma (including exact Markov autocorrelation) per state; no
    # arbitrary i.i.d. multinomial error bars. Also require useful TV.
    passed = bool(stats['within_bounds'] and stats['total_variation'] < 0.10)
    report = dict(mode=mode, passed=passed, samples_per_chain=samples,
                  burn_in=burn_in, chains=chains, seed=seed,
                  association_proposal_probability=p,
                  temperature_k=temperature_k, kb_real=kb_real,
                  bond_energy_offset=offset,
                  attempted_non_null_moves=attempts, accepted_moves=accepted,
                  acceptance_rate=(accepted / attempts if attempts else None),
                  total_variation=stats['total_variation'],
                  states=[dict(matching=[list(pair) for pair in state],
                               dimers=len(state), energy=ordered_energies[i],
                               count=int(counts[i]), observed=float(stats['observed'][i]),
                               expected=float(stats['expected'][i]),
                               standard_error=float(stats['standard_error'][i]),
                               five_sigma_bound=float(stats['bounds'][i]))
                          for i, state in enumerate(STATES)],
                  chains_detail=by_chain,
                  caveats=['Fixed positions; this is conditional topology equilibrium, not coupled NVT/MC validation.',
                           'Production MH transition logic is exercised; topology energies are cached.',
                           'Real mode computes energies with JAX-MD and separately checks uncached trial construction.',
                           'Finite-chain confidence bounds assume burn-in is adequate.'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic', 'real'), default='synthetic')
    parser.add_argument('--samples', type=int, default=3000, help='Retained proposals per chain')
    parser.add_argument('--burn-in', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--p-association', type=float, default=0.37)
    parser.add_argument('--temperature-k', type=float, default=1.0)
    parser.add_argument('--kb-real', type=float, default=1.0)
    parser.add_argument('--bond-offset', type=float, default=-0.35)
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON report path')
    args = parser.parse_args(argv)
    report = sample(mode=args.mode, samples=args.samples, burn_in=args.burn_in,
                    chains=args.chains, seed=args.seed, p=args.p_association,
                    temperature_k=args.temperature_k, kb_real=args.kb_real,
                    offset=args.bond_offset)
    print('\nMode:', report['mode'])
    print('State                Expected    Observed    5sigma bound')
    for state in report['states']:
        print(f"{str(state['matching']):20.20s} {state['expected']:10.5f} "
              f"{state['observed']:10.5f} {state['five_sigma_bound']:12.5f}")
    print(f"Total variation: {report['total_variation']:.5f}")
    print(f"Non-null move acceptance: {report['acceptance_rate']}")
    print('Sampling validation:', 'PASS' if report['passed'] else 'FAIL')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Report:', args.output)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
