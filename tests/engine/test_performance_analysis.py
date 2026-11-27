"""Regression tests for nested and multi-process timing aggregation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('analyze_performance',
    Path(__file__).resolve().parents[2] / 'validation' / 'analyze_performance.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(span, parent, stage, elapsed, run='a', status='ok', **kw):
    return dict(run_id=run, span_id=span, parent_id=parent, stage=stage,
                elapsed_s=elapsed, status=status, **kw)


def test_nested_exclusive_and_multiple_runs():
    data = [row(2, 1, 'langevin_chunk', 2, chunk_steps=50),
            row(1, 0, 'trial_relaxation', 3), row(0, None, 'reaction_attempt', 5),
            row(0, None, 'md_chunk', 4, run='b', chunk_steps=100)]
    result = analysis.summarize(data)
    assert result['root_measured_s'] == 9 and result['runs'] == 2
    assert result['stages']['reaction_attempt']['exclusive_s'] == 2
    assert result['stages']['trial_relaxation']['exclusive_s'] == 1
    assert result['stages']['md_chunk']['steps_per_s'] == 25
    assert sum(r['exclusive_s'] for r in result['stages'].values()) == 9


def test_partial_failure_and_duplicate():
    data = [row(1, 0, 'md_chunk', 2, status='error', chunk_steps=100)]
    result = analysis.summarize(data)
    assert result['orphan_spans'] == 1 and result['root_measured_s'] == 0
    assert result['stages']['md_chunk']['errors'] == 1
    assert result['stages']['md_chunk']['successful_steps'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.summarize(data + data)


def test_log_and_jsonl(tmp_path):
    a = row(0, None, 'checkpoint', 1)
    b = row(1, None, 'checkpoint', 2)
    log = tmp_path / 'run.log'
    log.write_text('ordinary output\n' + analysis.PREFIX + json.dumps(a) + '\n' + json.dumps(b) + '\n')
    assert list(analysis.read_records([log])) == [a, b]
