"""Regression checks for the independent A2 CSV consistency auditor."""
import csv
import importlib.util
from pathlib import Path

import pytest

AUDIT_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'audit_a2_trajectory.py'
spec = importlib.util.spec_from_file_location('a2_trajectory_audit', AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _write_csv(path, headers, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def _sample(step, assoc, diss):
    return dict(step=step, md_time_ps=step * .01,
                cumulative_relaxation_time_ps=0., elapsed_time_ps=step * .01,
                n_a2=assoc-diss, n_association=assoc, n_dissociation=diss,
                potential_energy_kcal_mol=-5., kinetic_energy_kcal_mol=2.)


def _files(tmp_path):
    trajectory = tmp_path / 'trajectory.csv'
    events = tmp_path / 'events.csv'
    rows = [_sample(0, 0, 0), _sample(10, 1, 0), _sample(20, 1, 1)]
    _write_csv(trajectory, rows[0], rows)
    _write_csv(events, ['step', 'reaction_type'], [
        dict(step=10, reaction_type='association'),
        dict(step=20, reaction_type='dissociation'),
    ])
    return trajectory, events, rows


def test_validated_trajectory_and_events(tmp_path):
    trajectory, events, _ = _files(tmp_path)
    result = audit.audit_trajectory(trajectory, events=events,
                                    equilibration_ps=.1, n_atoms=4, n_a=2)
    assert result['post_equilibration_samples'] == 2
    assert result['mean_n_a2'] == .5
    assert result['final_n_a2'] == 0
    assert result['event_counts_verified']


def test_detects_population_inconsistency(tmp_path):
    trajectory, events, rows = _files(tmp_path)
    rows[1]['n_a2'] = 0
    _write_csv(trajectory, rows[0], rows)
    with pytest.raises(ValueError, match='population/counter mismatch'):
        audit.audit_trajectory(trajectory, events=events, equilibration_ps=0.,
                               n_atoms=4, n_a=2)


def test_detects_event_history_inconsistency(tmp_path):
    trajectory, events, _ = _files(tmp_path)
    _write_csv(events, ['step', 'reaction_type'], [
        dict(step=20, reaction_type='association'),
        dict(step=10, reaction_type='dissociation'),
    ])
    with pytest.raises(ValueError, match='Event CSV mismatch'):
        audit.audit_trajectory(trajectory, events=events, equilibration_ps=0.,
                               n_atoms=4, n_a=2)
