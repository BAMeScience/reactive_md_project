"""Run a controlled, paired MC/rate NVT pilot from identical initial conditions.

Supply density, candidate cutoff, and independently obtained or explicitly
*illustrative* rate parameters.  This script does not infer them from the paper.
Use --dry-run to inspect the two commands without needing JAX-MD.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from .analyze_a2_pilot import analyze_pair


def build_commands(args):
    common = [
        sys.executable, '-m', 'scripts.run_a2_paper_nvt',
        '--density-number-a3', str(args.density_number_a3),
        '--association-cutoff-a', str(args.association_cutoff_a),
        '--v0-kjmol', str(args.v0_kjmol),
        '--steps', str(args.steps), '--check-every', str(args.check_every),
        '--dt-ps', str(args.dt_ps), '--mass', str(args.mass),
        '--seed', str(args.seed),
        '--relaxation-time-ps', str(args.relaxation_time_ps),
    ]
    mc = common + ['--mode', 'mc', '--mc-energy-mode', 'local_solvation',
                   '--output-dir', str(args.output_dir / 'mc')]
    rate = common + ['--mode', 'rate',
                     '--association-rate-ps', str(args.association_rate_ps),
                     '--dissociation-rate-ps', str(args.dissociation_rate_ps),
                     '--output-dir', str(args.output_dir / 'rate')]
    return mc, rate


def validate(args, parser):
    for name in ('density_number_a3', 'association_cutoff_a', 'dt_ps', 'mass',
                 'relaxation_time_ps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    for name in ('association_rate_ps', 'dissociation_rate_ps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'{name} must be finite and nonnegative')
    if args.steps <= 0 or args.check_every <= 0:
        parser.error('steps and check-every must be positive')
    if args.steps % args.check_every:
        parser.error('For this pilot, steps must be divisible by check-every')
    tau = args.check_every * args.dt_ps
    for name in ('association_rate_ps', 'dissociation_rate_ps'):
        if getattr(args, name) * tau > 0.1:
            parser.error(f'{name}*tau > 0.1; use a smaller rate or reaction interval for the pilot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-number-a3', type=float, required=True)
    parser.add_argument('--association-cutoff-a', type=float, required=True)
    parser.add_argument('--association-rate-ps', type=float, required=True,
                        help='Explicit external rate in ps^-1; no paper value assumed')
    parser.add_argument('--dissociation-rate-ps', type=float, required=True)
    parser.add_argument('--v0-kjmol', type=float, default=-1.9572,
                        choices=(-0.4144, -1.9572, -3.5))
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--check-every', type=int, default=10)
    parser.add_argument('--dt-ps', type=float, default=0.001)
    parser.add_argument('--mass', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--relaxation-time-ps', type=float, default=1.0)
    parser.add_argument('--output-dir', type=Path, default=Path('results/paper_phase6_pilot'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    validate(args, parser)
    mc, rate = build_commands(args)
    if args.dry_run:
        import shlex
        print('MC:', shlex.join(mc))
        print('rate:', shlex.join(rate))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'pilot_protocol.json').write_text(json.dumps({
        'scope': 'NVT PEL I, NOT published NPT benchmark',
        'parameters': {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()},
        'rate_origin': 'User-supplied; may be illustrative, not inferred from paper',
        'commands': [mc, rate],
    }, indent=2))
    for cmd in (mc, rate):
        subprocess.run(cmd, check=True)
    analyze_pair(args.output_dir)
    print('Report:', args.output_dir / 'pilot_report.md')
    print('Summary:', args.output_dir / 'pilot_summary.json')


if __name__ == '__main__':
    main()
