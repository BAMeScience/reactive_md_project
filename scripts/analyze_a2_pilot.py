"""Summarize a paired Phase 6 NVT diagnostic without claiming paper agreement.

Pure Python standard library; can be tested even when JAX-MD is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def analyze_run(folder: Path):
    metadata = json.loads((folder / 'metadata.json').read_text())
    samples = read_rows(folder / 'samples.csv')
    events = read_rows(folder / 'events.csv')
    attempts = read_rows(folder / 'mc_attempts.csv')
    if not samples:
        raise ValueError(f'{folder}: no samples')
    for row in samples:
        for field in ('step', 'n_a2', 'potential_energy_kcal_mol',
                      'kinetic_energy_kcal_mol', 'n_association_candidates',
                      'n_dissociation_candidates'):
            if not math.isfinite(float(row[field])):
                raise ValueError(f'{folder}: nonfinite {field}')
        if float(row['kinetic_energy_kcal_mol']) < 0:
            raise ValueError(f'{folder}: negative kinetic energy')
    if len(attempts) != int(metadata['attempted_mc_moves']):
        raise ValueError(f'{folder}: MC attempt count does not match log')
    if len(events) != int(metadata['accepted_events']):
        raise ValueError(f'{folder}: accepted event count does not match log')
    if metadata['mode'] == 'mc':
        rejected = sum(row['accepted'].lower() == 'false' for row in attempts)
        if rejected != int(metadata['rejected_mc_moves']):
            raise ValueError(f'{folder}: rejected attempt count does not match log')
    return {
        'mode': metadata['mode'],
        'metadata': metadata,
        'sample_count': len(samples),
        'reaction_checks': len(samples) - 1,
        'mc_attempts': len(attempts),
        'mc_rejected': int(metadata['rejected_mc_moves']),
        'mc_acceptance_fraction': ((len(attempts) - int(metadata['rejected_mc_moves'])) / len(attempts)
                                   if attempts else None),
        'accepted_events': len(events),
        'association_events': sum(r['type'] == 'association' for r in events),
        'dissociation_events': sum(r['type'] == 'dissociation' for r in events),
        'initial_dimers': int(samples[0]['n_a2']),
        'final_dimers': int(samples[-1]['n_a2']),
        'mean_dimers_all_samples': sum(int(r['n_a2']) for r in samples) / len(samples),
        'first_candidate_counts': [int(samples[1]['n_association_candidates']),
                                   int(samples[1]['n_dissociation_candidates'])] if len(samples) > 1 else None,
        'energy_start_kcal_mol': float(samples[0]['potential_energy_kcal_mol']),
        'energy_end_kcal_mol': float(samples[-1]['potential_energy_kcal_mol']),
        'mc_delta_energies_kcal_mol': [float(r['delta_energy_kcal_mol']) for r in attempts],
    }


def analyze_pair(root: Path):
    mc = analyze_run(root / 'mc')
    rate = analyze_run(root / 'rate')
    for key in ('initial_positions_sha256', 'seed', 'number_density_atoms_per_a3',
                'association_cutoff_a', 'v0_kj_mol', 'dt_ps', 'check_every',
                'temperature_k', 'relaxation_time_ps'):
        if mc['metadata'][key] != rate['metadata'][key]:
            raise ValueError(f'Paired run mismatch for {key}')
    if mc['mode'] != 'mc' or rate['mode'] != 'rate':
        raise ValueError('Expected mc/ and rate/ results')
    report = {
        'scope': 'Paired NVT PEL I pilot; NOT NPT/PEL II validation',
        'same_initial_positions': True,
        'same_seed_and_parameters': True,
        'rate_parameters_are_external': True,
        'rate_probability_per_candidate': {
            'association': rate['metadata']['rate_probability_association'],
            'dissociation': rate['metadata']['rate_probability_dissociation'],
        },
        'paper_relaxation': bool(mc['metadata']['relaxation_matches_paper']),
        'mc': mc, 'rate': rate,
    }
    (root / 'pilot_summary.json').write_text(json.dumps(report, indent=2))
    lines = [
        '# Paired A2 NVT pilot report', '',
        '**Scope:** Algorithmic/stability diagnostic, not a reproduction of published NPT results.',
        '**Initial positions and seed:** verified identical by stored checksum and metadata.',
        f"**Relaxation:** {mc['metadata']['relaxation_time_ps']} ps per attempted MC trial / accepted rate cycle; "
        + ('paper duration' if report['paper_relaxation'] else 'shortened diagnostic, not paper duration'),
        '', '| Diagnostic | MC | rate |', '|---|---:|---:|',
    ]
    for name, key in [('Reaction checks', 'reaction_checks'),
                      ('Accepted events', 'accepted_events'),
                      ('Associations', 'association_events'),
                      ('Dissociations', 'dissociation_events'),
                      ('Final dimers', 'final_dimers'),
                      ('Mean dimers (all samples)', 'mean_dimers_all_samples'),
                      ('Initial PE (kcal/mol)', 'energy_start_kcal_mol'),
                      ('Final PE (kcal/mol)', 'energy_end_kcal_mol')]:
        lines.append(f"| {name} | {mc[key]} | {rate[key]} |")
    lines.extend(['', f"**MC:** {mc['mc_attempts']} attempts, {mc['mc_rejected']} rejected; "
                  f"acceptance fraction {mc['mc_acceptance_fraction']}.",
                  f"**Rate:** Γ_assoc τ = {report['rate_probability_per_candidate']['association']}, "
                  f"Γ_diss τ = {report['rate_probability_per_candidate']['dissociation']}.",
                  '', '## Interpretation constraints',
                  '- No event in a short run is not evidence of a broken algorithm.',
                  '- The two modes have different kinetic meanings: do not compare MC time directly to rate time.',
                  '- Energy changes include NVT thermostat effects; PE is not expected to be conserved.',
                  '- Initial dimer populations and short-run means are not equilibrium estimates.',
                  '- Rates are externally supplied; they are not paper-derived unless separately established.',
                  '- The NVT density, mass calibration, and initial equilibration still require physical validation.',
                  ''])
    (root / 'pilot_report.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = analyze_pair(args.root)
    print((args.root / 'pilot_report.md').read_text())
    print('Machine-readable:', args.root / 'pilot_summary.json')


if __name__ == '__main__':
    main()
