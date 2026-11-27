#!/usr/bin/env python3
"""Summarize opt-in reactive_md timing logs without counting nested spans twice."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

PREFIX = '[REACTIVE_MD PERF] '


def read_records(paths):
    for path in paths:
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                line = line.strip()
                if line.startswith(PREFIX):
                    line = line[len(PREFIX):]
                elif not line.startswith('{'):
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or 'stage' not in row:
                        continue
                    yield row
                except ValueError as exc:
                    raise ValueError(f'{path}:{number}: invalid timing JSON') from exc


def summarize(records):
    spans = {}
    children = defaultdict(float)
    by_stage = defaultdict(list)
    for row in records:
        key = (row['run_id'], row['span_id'])
        elapsed = float(row['elapsed_s'])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f'Invalid elapsed time: {key}')
        if key in spans:
            raise ValueError(f'Duplicate span: {key}; avoid overlapping input logs')
        spans[key] = row
        by_stage[row['stage']].append(row)
        if row['parent_id'] is not None:
            children[(row['run_id'], row['parent_id'])] += elapsed
    orphans = sum(row['parent_id'] is not None and
                  (row['run_id'], row['parent_id']) not in spans
                  for row in spans.values())
    roots = [row for row in spans.values() if row['parent_id'] is None]
    summary = {}
    for stage, rows in sorted(by_stage.items()):
        good = [r for r in rows if r['status'] == 'ok']
        times = [float(r['elapsed_s']) for r in good]
        inclusive = sum(float(r['elapsed_s']) for r in rows)
        exclusive = sum(max(0.0, float(r['elapsed_s']) - children[(r['run_id'], r['span_id'])]) for r in rows)
        steps = sum(int(r.get('chunk_steps', 0)) for r in good) if stage in ('md_chunk', 'langevin_chunk') else 0
        summary[stage] = dict(calls=len(rows), errors=len(rows)-len(good),
                              inclusive_s=inclusive, exclusive_s=exclusive,
                              first_ok_s=times[0] if times else None,
                              median_ok_s=median(times) if times else None,
                              max_ok_s=max(times) if times else None,
                              successful_steps=steps,
                              steps_per_s=steps/sum(times) if steps and sum(times)>0 else None)
    return dict(records=len(spans), runs=len({k[0] for k in spans}), orphan_spans=orphans,
                root_measured_s=sum(float(r['elapsed_s']) for r in roots), stages=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', help='Full terminal logs or extracted JSONL')
    parser.add_argument('--output', type=Path, help='Optional JSON summary path')
    args = parser.parse_args()
    result = summarize(read_records(args.logs))
    if not result['records']:
        parser.error('No performance records found')
    print(f"{result['records']} spans across {result['runs']} processes; "
          f"measured root time {result['root_measured_s']:.3f} s; "
          f"orphan spans {result['orphan_spans']}")
    print(f"{'Stage':32} {'Calls':>8} {'Errors':>7} {'Incl. s':>12} {'Excl. s':>12} {'Median s':>12} {'Steps/s':>12}")
    for stage, row in result['stages'].items():
        med = '-' if row['median_ok_s'] is None else f"{row['median_ok_s']:.6f}"
        rate = '-' if row['steps_per_s'] is None else f"{row['steps_per_s']:.3f}"
        print(f"{stage:32} {row['calls']:8d} {row['errors']:7d} {row['inclusive_s']:12.3f} {row['exclusive_s']:12.3f} {med:>12} {rate:>12}")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
