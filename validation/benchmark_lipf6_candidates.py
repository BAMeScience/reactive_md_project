#!/usr/bin/env python3
"""Compare pre-vectorization scalar and installed LiPF6 forward search.
Run in a fresh Python process for each backend; no MD or MACE is loaded.
"""
import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reactive_md.reactions.lipf6 import (
    ReactionCandidate, discover_pf6_and_li, find_reaction_candidates,
    reaction_coordinate,
)


def scalar_search(R, pf6, li, disp, *, pf6_reacted_np):
    """Original scalar loop, for valid inputs (validation outside timings)."""
    R = jnp.asarray(R)
    candidates = []
    for k, row in enumerate(pf6):
        if pf6_reacted_np[k]:
            continue
        p = int(row[0])
        for atom in li:
            for fluorine in row[1:]:
                i, f = int(atom), int(fluorine)
                d_lif = float(np.linalg.norm(np.asarray(disp(R[i], R[f]))))
                d_pf = float(np.linalg.norm(np.asarray(disp(R[p], R[f]))))
                candidates.append(ReactionCandidate(k, i, f, d_lif, d_pf))
    candidates.sort(key=lambda c: reaction_coordinate(d_pf=c.d_pf, d_lif=c.d_lif), reverse=True)
    return candidates


def compare(a, b, expected, atol):
    identity = lambda c: (c.k_pf6, c.li_idx, c.leave_F, c.direction)
    if len(a) != expected or len(b) != expected:
        raise AssertionError(f'Candidate counts: {len(a)}, {len(b)}; expected {expected}')
    if [identity(c) for c in a] != [identity(c) for c in b]:
        raise AssertionError('Candidate identities or ranking differ')
    def values(cs):
        return np.asarray([(c.d_lif, c.d_pf, c.d_pf-c.d_lif) for c in cs], dtype=np.float64).reshape(-1, 3)
    x, y = values(a), values(b)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError('Nonfinite candidate geometry')
    np.testing.assert_allclose(x, y, rtol=0, atol=atol)
    return dict(zip(('d_lif', 'd_pf', 'sigma'), np.max(np.abs(x-y), axis=0, initial=0).tolist()))


def timed(fn):
    start = time.perf_counter()
    result = fn()  # Both implementations return host-side scalar distances.
    return time.perf_counter()-start, result


def stats(samples):
    return {'mean_s': statistics.mean(samples), 'median_s': statistics.median(samples),
            'stdev_s': statistics.stdev(samples) if len(samples)>1 else 0.0,
            'min_s': min(samples), 'max_s': max(samples), 'samples_s': samples}


def benchmark(R, pf6, li, disp, reacted, warmup, repeats, atol, first_order):
    funcs = {'scalar': lambda: scalar_search(R, pf6, li, disp, pf6_reacted_np=reacted),
             'vectorized': lambda: find_reaction_candidates(R, pf6, li, disp, pf6_reacted_np=reacted)}
    order = [first_order, 'vectorized' if first_order == 'scalar' else 'scalar']
    expected = int((~reacted).sum()) * len(li) * 6
    first, reference = {}, {}
    for name in order:
        first[name], reference[name] = timed(funcs[name])
        print(f'  {name} first call: {first[name]:.6f} s', flush=True)
    errors = compare(reference['scalar'], reference['vectorized'], expected, atol)
    for _ in range(warmup):
        for name in order:
            funcs[name]()
    samples = {name: [] for name in order}
    for i in range(repeats):
        # Alternate order to reduce systematic timing bias.
        for name in (order if i % 2 == 0 else order[::-1]):
            elapsed, result = timed(funcs[name])
            samples[name].append(elapsed)
            compare(reference['scalar'], result, expected, atol)
        print(f'  timed pair {i+1}/{repeats}', flush=True)
    summaries = {name: stats(samples[name]) for name in order}
    speedup = summaries['scalar']['median_s']/summaries['vectorized']['median_s']
    print(f'  median speedup: {speedup:.3f}x; max errors (A): {errors}', flush=True)
    return {'reacted_pf6': int(reacted.sum()), 'candidate_count': expected,
            'first_call_s': first, 'timings': summaries, 'median_speedup': speedup,
            'first_call_max_abs_error_A': errors, 'ranking_identical_all_calls': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--settings', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--reacted-counts', type=int, nargs='+', default=[0, 5, 15])
    p.add_argument('--atol', type=float, default=1e-12)
    p.add_argument('--first-order', choices=['scalar', 'vectorized'], default='scalar')
    p.add_argument('--p-type', type=int, default=6)
    p.add_argument('--f-type', type=int, default=7)
    p.add_argument('--li-type', type=int, default=8)
    args = p.parse_args()
    if not jax.config.jax_enable_x64:
        p.error('Set JAX_ENABLE_X64=True before starting Python')
    if args.repeats < 1 or args.warmup < 0 or not np.isfinite(args.atol) or args.atol < 0:
        p.error('Invalid repeats, warmup or tolerance')
    if args.output.exists():
        p.error('Output already exists; choose a new path')
    from jax_md import space
    from reactive_md.extract_params_oplsaa import parse_lammps_data
    parsed = parse_lammps_data(args.data, args.settings)
    R = jnp.asarray(parsed[0], dtype=jnp.float64)
    molid, box, types = np.asarray(parsed[6], dtype=np.int32), np.asarray(parsed[7], dtype=np.float64), np.asarray(parsed[9], dtype=np.int32)
    del parsed
    R.block_until_ready()
    pf6, li = discover_pf6_and_li(types, molid, p_type=args.p_type, f_type=args.f_type, li_type=args.li_type)
    if not len(pf6) or not len(li):
        p.error('No PF6 or Li discovered; check atom type IDs')
    if any(n < 0 or n >= len(pf6) for n in args.reacted_counts):
        p.error('Reacted counts must be between zero and n_pf6-1')
    disp, _ = space.periodic(box)
    source = inspect.getsource(find_reaction_candidates)
    if 'jax.vmap' not in source:
        raise RuntimeError('Installed candidate search does not contain expected vectorization')
    report = {'jax_version': jax.__version__, 'backend': jax.default_backend(),
              'devices': [str(d) for d in jax.devices()], 'dtype': str(R.dtype),
              'atoms': len(R), 'pf6': len(pf6), 'li': len(li), 'box_A': box.tolist(),
              'data': str(Path(args.data).resolve()), 'settings': str(Path(args.settings).resolve()),
              'search_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'search_source_file': inspect.getfile(find_reaction_candidates),
              'warmup': args.warmup, 'repeats': args.repeats, 'atol_A': args.atol,
              'first_order': args.first_order, 'cases': []}
    print(json.dumps(report, indent=2), flush=True)
    for n in args.reacted_counts:
        reacted = np.arange(len(pf6)) < n
        print(f'Case: {n} masked PF6; {(len(pf6)-n)*len(li)*6} candidates', flush=True)
        report['cases'].append(benchmark(R, pf6, li, disp, reacted, args.warmup, args.repeats, args.atol, args.first_order))
        # Save each completed case even if a later case fails.
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
