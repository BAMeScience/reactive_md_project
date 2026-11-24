"""Standard-library pilot checks; do not require JAX-MD."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_a2_pilot import analyze_pair


def _write_csv(path, columns, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _make(folder, mode, *, checksum='same'):
    folder.mkdir(parents=True)
    meta = dict(mode=mode, attempted_mc_moves=1 if mode == 'mc' else 0,
                rejected_mc_moves=1 if mode == 'mc' else 0, accepted_events=0,
                initial_positions_sha256=checksum, seed=17,
                number_density_atoms_per_a3=0.0045, association_cutoff_a=3.0,
                v0_kj_mol=-1.9572, dt_ps=0.001, check_every=10,
                temperature_k=188., relaxation_time_ps=1.0,
                relaxation_matches_paper=True,
                rate_probability_association=0.02 if mode == 'rate' else None,
                rate_probability_dissociation=0.02 if mode == 'rate' else None)
    (folder / 'metadata.json').write_text(json.dumps(meta))
    fields = ['step', 'n_a2', 'potential_energy_kcal_mol',
              'kinetic_energy_kcal_mol', 'n_association_candidates',
              'n_dissociation_candidates']
    _write_csv(folder / 'samples.csv', fields, [
        dict(step=0, n_a2=0, potential_energy_kcal_mol=-10,
             kinetic_energy_kcal_mol=2, n_association_candidates=0,
             n_dissociation_candidates=0),
        dict(step=10, n_a2=0, potential_energy_kcal_mol=-11,
             kinetic_energy_kcal_mol=2.1, n_association_candidates=1,
             n_dissociation_candidates=0),
    ])
    _write_csv(folder / 'events.csv', ['step', 'type'], [])
    _write_csv(folder / 'mc_attempts.csv',
               ['step', 'accepted', 'delta_energy_kcal_mol'],
               [dict(step=10, accepted='False', delta_energy_kcal_mol=3.0)]
               if mode == 'mc' else [])


def test_analyze_paired_no_events_is_valid(tmp_path):
    _make(tmp_path / 'mc', 'mc')
    _make(tmp_path / 'rate', 'rate')
    report = analyze_pair(tmp_path)
    assert report['same_initial_positions']
    assert report['mc']['mc_rejected'] == 1
    assert report['rate']['accepted_events'] == 0
    assert (tmp_path / 'pilot_report.md').exists()
    assert report['mc']['mc_delta_energies_kcal_mol'] == [3.]


def test_analyze_rejects_different_initial_positions(tmp_path):
    _make(tmp_path / 'mc', 'mc')
    _make(tmp_path / 'rate', 'rate', checksum='different')
    with pytest.raises(ValueError, match='initial_positions_sha256'):
        analyze_pair(tmp_path)


def test_paired_runner_dry_run_without_jax_md(tmp_path):
    result = subprocess.run([
        sys.executable, '-m', 'scripts.run_a2_pilot_pair',
        '--density-number-a3', '0.0045', '--association-cutoff-a', '3.0',
        '--association-rate-ps', '2', '--dissociation-rate-ps', '2',
        '--dry-run', '--output-dir', str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--mode mc' in result.stdout
    assert '--mode rate' in result.stdout
    assert not (tmp_path / 'mc').exists()


def test_paired_runner_rejects_large_rate_interval(tmp_path):
    result = subprocess.run([
        sys.executable, '-m', 'scripts.run_a2_pilot_pair',
        '--density-number-a3', '0.0045', '--association-cutoff-a', '3.0',
        '--association-rate-ps', '20', '--dissociation-rate-ps', '2',
        '--dry-run', '--output-dir', str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert '0.1' in result.stderr
