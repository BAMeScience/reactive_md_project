#!/usr/bin/env python3
"""Benchmark the actual X64 MACE scalar energy and force kernels in isolation.

Run each mode in a separate process. Use --compare-to on the energy-only run
with the legacy JSON to check energy, force, model and graph equivalence.
This does not modify production inputs, checkpoints, or simulation outputs.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def array_digest(array):
    a = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def synchronized_call(fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    jax.block_until_ready(value)
    return time.perf_counter() - start, value


def compiled_memory(executable):
    try:
        analysis = executable.memory_analysis()
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        return {}
    if analysis is None:
        return {}
    fields = ('argument_size_in_bytes', 'output_size_in_bytes',
              'temp_size_in_bytes', 'alias_size_in_bytes', 'generated_code_size_in_bytes')
    return {name: int(getattr(analysis, name)) for name in fields
            if getattr(analysis, name, None) is not None}


def benchmark_kernel(fn, args, warmup, repeats):
    jax.block_until_ready(args)
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lowering_s = time.perf_counter() - start
    start = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - start
    first_s, value = synchronized_call(executable, *args)
    first_value = np.asarray(value)
    if not np.isfinite(first_value).all():
        raise FloatingPointError('Nonfinite kernel output')
    for _ in range(warmup):
        synchronized_call(executable, *args)
    samples = []
    for _ in range(repeats):
        seconds, value = synchronized_call(executable, *args)
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError('Nonfinite kernel output')
        samples.append(seconds)
    return ({'lowering_s': lowering_s, 'compile_s': compile_s,
             'first_execution_s': first_s, 'median_s': statistics.median(samples),
             'mean_s': statistics.mean(samples), 'samples_s': samples,
             'memory_analysis': compiled_memory(executable),
             'dtype': str(first_value.dtype)}, first_value)


def compare_reports(reference, candidate, *, energy_atol, force_atol, force_rtol):
    for field in ('model_id', 'source', 'variant', 'positions_sha256', 'box_sha256',
                  'species_sha256', 'neighbor_sha256', 'neighbor_shape', 'versions',
                  'backend', 'device_kind', 'dtype', 'capacity_multiplier', 'dr_threshold'):
        if reference[field] != candidate[field]:
            raise ValueError(f'Benchmark input mismatch: {field}')
    if reference['mode'] != 'legacy' or candidate['mode'] != 'energy-only':
        raise ValueError('Comparison requires legacy reference and energy-only candidate')
    for key in ('energy', 'force'):
        if reference['kernels'][key]['dtype'] != candidate['kernels'][key]['dtype']:
            raise ValueError(f'Kernel output dtype changed: {key}')
    e0, e1 = float(reference['energy_eV']), float(candidate['energy_eV'])
    f0, f1 = np.asarray(reference['forces_eV_per_A']), np.asarray(candidate['forces_eV_per_A'])
    if f0.shape != f1.shape or not np.isfinite([e0, e1]).all() or not (
            np.isfinite(f0).all() and np.isfinite(f1).all()):
        raise ValueError('Nonfinite or shape-mismatched benchmark output')
    np.testing.assert_allclose(e1, e0, rtol=0., atol=energy_atol)
    np.testing.assert_allclose(f1, f0, rtol=force_rtol, atol=force_atol)
    diff = f1 - f0
    return {'energy_abs_error_eV': abs(e1-e0),
            'force_max_abs_error_eV_per_A': float(np.max(np.abs(diff))),
            'force_rms_error_eV_per_A': float(np.sqrt(np.mean(diff**2))),
            'energy_atol_eV': energy_atol, 'force_atol_eV_per_A': force_atol,
            'force_rtol': force_rtol,
            'median_speedup': {key: reference['kernels'][key]['median_s'] /
                               candidate['kernels'][key]['median_s']
                               for key in ('energy', 'force')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='Optional trusted immutable checkpoint for coordinates; uses original data for species')
    parser.add_argument('--source', default='mp')
    parser.add_argument('--variant', default='small-0b2')
    parser.add_argument('--mode', choices=('legacy', 'energy-only'), required=True)
    parser.add_argument('--capacity-multiplier', type=float, default=1.25)
    parser.add_argument('--dr-threshold', type=float, default=.5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-to', type=Path)
    parser.add_argument('--energy-atol', type=float, default=1e-8)
    parser.add_argument('--force-atol', type=float, default=1e-8)
    parser.add_argument('--force-rtol', type=float, default=1e-10)
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        parser.error('Set JAX_ENABLE_X64=True before running this benchmark')
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.repeats < 1 or args.warmup < 0 or not all(
            np.isfinite(x) and x >= 0 for x in
            (args.energy_atol, args.force_atol, args.force_rtol)):
        parser.error('Invalid timing or comparison parameters')
    if args.compare_to is not None and args.mode != 'energy-only':
        parser.error('--compare-to requires --mode energy-only')
    from reactive_md.extract_params_oplsaa import parse_lammps_data
    from reactive_md.mace_setup import build_mace_model
    from reactive_md.mace_energy import build_mace_energy_system

    parsed = parse_lammps_data(args.data, args.settings)
    positions = jnp.asarray(parsed[0], dtype=jnp.float64)
    box = np.asarray(parsed[7], dtype=np.float64)
    type_to_z = {1: 6, 2: 8, 3: 8, 4: 6, 5: 1, 6: 15, 7: 9, 8: 3}
    z = np.array([type_to_z[int(t)] for t in parsed[9]], dtype=np.int32)
    del parsed
    checkpoint_step = None
    if args.checkpoint is not None:
        # Only trusted local checkpoints: load_checkpoint uses pickle.
        from reactive_md.checkpoint import load_checkpoint
        saved = load_checkpoint(args.checkpoint)
        r = saved['md_state'].position
        if r.shape != positions.shape or r.dtype != positions.dtype:
            raise ValueError('Checkpoint coordinates do not match initial system size/precision')
        if not np.array_equal(np.asarray(saved['metadata']['box']), box):
            raise ValueError('Checkpoint box does not match original input box')
        positions = r
        checkpoint_step = int(saved['steps_done'])
        del saved
    model, config, _, model_id = build_mace_model(
        source=args.source, variant=args.variant, return_backend_id=True)
    _, _, neighbor_fn, make_energy = build_mace_energy_system(
        jax_model=model, jax_model_config=config, z_atomic=z, box0=box,
        ensemble='nve', dr_threshold=args.dr_threshold,
        capacity_multiplier=args.capacity_multiplier,
        energy_only=args.mode == 'energy-only')
    neighbors = neighbor_fn.allocate(positions)
    neighbors = neighbor_fn.update(positions, neighbors)
    if bool(np.asarray(neighbors.did_buffer_overflow)):
        raise RuntimeError('Initial benchmark neighbor list overflowed; choose a sufficient capacity')

    def energy(R, nl):
        return make_energy(nl)(R, box=box)

    # Match NVT/Langevin force construction: negative derivative of scalar energy.
    def force(R, nl):
        return -jax.grad(energy)(R, nl)

    print(f'{args.mode}: {len(positions)} atoms, neighbors {neighbors.idx.shape}', flush=True)
    energy_stats, E = benchmark_kernel(energy, (positions, neighbors), args.warmup, args.repeats)
    print(f'Energy median: {energy_stats["median_s"]:.6f} s', flush=True)
    force_stats, F = benchmark_kernel(force, (positions, neighbors), args.warmup, args.repeats)
    print(f'Force median: {force_stats["median_s"]:.6f} s', flush=True)
    report = {'mode': args.mode, 'source': args.source, 'variant': args.variant,
              'model_id': model_id, 'dtype': str(positions.dtype),
              'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
              'device_kind': [getattr(d, 'device_kind', str(d)) for d in jax.devices()],
              'versions': {name: version(name) for name in
                           ('jax', 'jaxlib', 'jax-md', 'mace-jax', 'mace-torch', 'flax')},
              'atoms': len(positions), 'checkpoint_step': checkpoint_step, 'positions_sha256': array_digest(positions),
              'box_sha256': array_digest(box), 'species_sha256': array_digest(z),
              'neighbor_sha256': array_digest(neighbors.idx),
              'neighbor_shape': list(neighbors.idx.shape),
              'capacity_multiplier': args.capacity_multiplier, 'dr_threshold': args.dr_threshold,
              'warmup': args.warmup, 'repeats': args.repeats,
              'kernels': {'energy': energy_stats, 'force': force_stats},
              'energy_eV': float(E), 'forces_eV_per_A': F.tolist()}
    # Keep raw results even if equivalence validation fails.
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if args.compare_to is not None:
        reference = json.loads(args.compare_to.read_text())
        report['comparison'] = compare_reports(reference, report, energy_atol=args.energy_atol,
                                               force_atol=args.force_atol, force_rtol=args.force_rtol)
        print(json.dumps(report['comparison'], indent=2), flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
