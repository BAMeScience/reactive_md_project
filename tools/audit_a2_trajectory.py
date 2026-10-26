"""Audit A2 kinetic trajectory/event CSVs without requiring JAX or JAX-MD.

Example:
    python tools/audit_a2_trajectory.py results/a2_trajectory.csv \\
        --events results/a2_events.csv --equilibration-ps 10000

Reports descriptive statistics, not proof of thermodynamic equilibrium.  Use
statistical tolerances rather than exact event histories across stochastic runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

KB_REAL = 0.0019872041
REQUIRED = (
    'step', 'md_time_ps', 'cumulative_relaxation_time_ps',
    'elapsed_time_ps', 'n_a2', 'n_association', 'n_dissociation',
    'potential_energy_kcal_mol', 'kinetic_energy_kcal_mol',
)


def _rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        yield from csv.DictReader(stream)


def audit_trajectory(path, *, events=None, equilibration_ps=10000.0,
                     n_atoms=1024, n_a=200, kb_real=KB_REAL):
    """Return validated trajectory statistics; raise ValueError on bad data."""
    if n_atoms <= 0 or n_a < 0 or n_a > n_atoms or kb_real <= 0:
        raise ValueError('Invalid atom counts or Boltzmann constant')
    if not math.isfinite(equilibration_ps) or equilibration_ps < 0:
        raise ValueError('equilibration_ps must be finite and nonnegative')
    rows = list(_rows(path))
    if not rows:
        raise ValueError('Trajectory CSV is empty')
    missing = set(REQUIRED) - set(rows[0])
    if missing:
        raise ValueError(f'Missing trajectory columns: {sorted(missing)}')

    last_step = -1
    last_md_time = -1.0
    last_relax = -1.0
    last_assoc = last_diss = -1
    samples = []
    for index, row in enumerate(rows):
        try:
            step = int(row['step'])
            n_a2 = int(row['n_a2'])
            assoc = int(row['n_association'])
            diss = int(row['n_dissociation'])
            md_time = float(row['md_time_ps'])
            relax = float(row['cumulative_relaxation_time_ps'])
            elapsed = float(row['elapsed_time_ps'])
            pe = float(row['potential_energy_kcal_mol'])
            ke = float(row['kinetic_energy_kcal_mol'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid numeric value at row {index + 2}') from exc
        if not all(map(math.isfinite, (md_time, relax, elapsed, pe, ke))):
            raise ValueError(f'Nonfinite trajectory data at row {index + 2}')
        if step <= last_step or md_time <= last_md_time or relax < last_relax:
            raise ValueError(f'Nonmonotonic trajectory at row {index + 2}')
        if assoc < last_assoc or diss < last_diss or n_a2 < 0 or n_a2 > n_a // 2:
            raise ValueError(f'Invalid reaction counters at row {index + 2}')
        if n_a2 != assoc - diss:
            raise ValueError(f'A2 population/counter mismatch at row {index + 2}')
        if not math.isclose(elapsed, md_time + relax, abs_tol=1e-4):
            raise ValueError(f'Elapsed-time mismatch at row {index + 2}')
        if ke < 0:
            raise ValueError(f'Negative kinetic energy at row {index + 2}')
        samples.append((step, md_time, relax, n_a2, assoc, diss, pe, ke))
        last_step, last_md_time, last_relax = step, md_time, relax
        last_assoc, last_diss = assoc, diss

    event_counts = None
    if events is not None:
        counts = Counter()
        per_step = Counter()
        for index, row in enumerate(_rows(events)):
            kind = row.get('reaction_type')
            if kind not in ('association', 'dissociation'):
                raise ValueError(f'Unknown event type at event row {index + 2}: {kind!r}')
            try:
                step = int(row['step'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid event step at row {index + 2}') from exc
            counts[kind] += 1
            per_step[(step, kind)] += 1
        if counts['association'] != last_assoc or counts['dissociation'] != last_diss:
            raise ValueError('Event CSV totals do not match trajectory counters')
        # Event rows are cumulative across all preceding checks; compare them
        # against each trajectory sample, not just the final totals.
        events_by_step = {}
        for (step, kind), count in per_step.items():
            events_by_step.setdefault(step, Counter())[kind] += count
        cumulative = Counter()
        for step, _, _, _, assoc, diss, _, _ in samples:
            cumulative.update(events_by_step.get(step, {}))
            if cumulative['association'] != assoc or cumulative['dissociation'] != diss:
                raise ValueError(f'Event CSV mismatch at trajectory step {step}')
        event_counts = dict(counts)

    retained = [r for r in samples if r[1] >= equilibration_ps]
    if not retained:
        raise ValueError('No trajectory samples at or after equilibration_ps')
    temperature = [2.0 * r[7] / (3.0 * n_atoms * kb_real) for r in retained]
    return {
        'samples': len(samples),
        'post_equilibration_samples': len(retained),
        'equilibration_ps': equilibration_ps,
        'md_duration_ps': samples[-1][1],
        'relaxation_duration_ps': samples[-1][2],
        'final_n_a2': samples[-1][3],
        'associations': last_assoc,
        'dissociations': last_diss,
        'mean_n_a2': mean(r[3] for r in retained),
        'mean_potential_energy_kcal_mol': mean(r[6] for r in retained),
        'mean_kinetic_temperature_k': mean(temperature),
        'event_counts_verified': event_counts is not None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--events', type=Path)
    parser.add_argument('--equilibration-ps', type=float, default=10000.)
    parser.add_argument('--n-atoms', type=int, default=1024)
    parser.add_argument('--n-a', type=int, default=200)
    args = parser.parse_args(argv)
    print(json.dumps(audit_trajectory(
        args.trajectory, events=args.events,
        equilibration_ps=args.equilibration_ps,
        n_atoms=args.n_atoms, n_a=args.n_a,
    ), indent=2))


if __name__ == '__main__':
    main()
