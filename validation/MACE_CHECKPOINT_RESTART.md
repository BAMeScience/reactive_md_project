# MACE-MD checkpoint and restart validation

Phase 6 development validation, 2026-11-25. Results below are from the
reported HPC runs, not a new execution of the simulations for this document.

## Implementation and regression coverage

MACE-MD restart support extends the generic MD checkpoint workflow. It
restores the saved MD and random state, validates the backend fingerprint,
and reconstructs the MACE and classical neighbor lists independently.
The fingerprint includes model identity and parameters, relevant software
versions and simulation configuration. Only trusted pickle checkpoints
should be loaded.

Application regression tests exercise the actual MD and reaction drivers
with a differentiable MACE surrogate, including accepted and rejected
proposals and backend mismatch handling. The final reported CPU suite was:

```bash
JAX_PLATFORMS=cpu python -m pytest -q --strict-markers
```

Result: **309 passed** (294 before Phase 6). These tests do not replace
foundation-model integration checks.

## Actual foundation-model validation

The benchmark uses a 5,264-atom LiPF6-containing system, the Materials
Project MACE `small-0b2` model, and seed 0. The original HPC inputs were
`mixture_classical_after_300K_NPT.data` and `mixture.in.settings`.
The exact input bytes, GPU model, dependency versions and final Git commit
were not captured in the supplied validation record; preserve these for
future reproducibility.

Common CLI options: `--use-mace-md --mace-source mp --mace-variant small-0b2
--seed 0 --check-every 10 --dump-every 10 --checkpoint-every 10`.
The reaction-free run used rate mode with forward and reverse rates zero.
The reactive runs additionally used `--sigma-mid -100 --sigma-width 0.2
--max-reactions-per-check 1 --relaxation-dt-ps 0.00025
--relaxation-time-ps 0.00025`. Rate MACE used `--reaction-mode rate
--reaction-rate-ps 100 --reverse-rate-ps 0`; MC MACE used
`--reaction-mode metropolis --use-mace-mc` with reverse MC disabled.
The permissive sigma gate and rate are software stress-test settings,
not validated physical kinetics.

For each comparison, run the uninterrupted target and a separate run to the
aligned split boundary, using distinct output and checkpoint paths. Restart
the split run with `--restart-from` pointing to its checkpoint and `--steps`
set to the full target (not the number of additional steps). Keep the split
output paths, original inputs and simulation options unchanged. Compare the
resulting dumps and CSV logs. Checkpoint intervals must be positive multiples
of the reaction-check interval; a terminal partial interval cannot be extended.

| Scenario | Full steps | Split steps | Accepted events | Trajectory dump |
| --- | ---: | --- | ---: | --- |
| Reaction-free MACE-MD | 400 | 200 + 200 | 0 | Byte-identical (reported) |
| Rate MACE | 100 | 50 + 50 | 10 | Byte-identical (reported) |
| MC MACE | 100 | 50 + 50 | 4 | Byte-identical (reported) |

Rate MACE accepted one event at each check. MC MACE accepted events at steps
20, 30, 40 and 60; six other checks produced rejected proposals in the
uninterrupted run. The step-50 split therefore exercises continuation after
three accepted reactions and subsequent acceptance after restart.

## Numerical comparison

Candidate-log diffs were checked row by row. All discrete fields (including
step, rank, atom indices and acceptance flag) agreed. The 62 Rate MACE and
68 MC MACE differing row pairs differed only in geometry values.

| Maximum absolute candidate difference | Rate MACE (angstrom) | MC MACE (angstrom) |
| --- | ---: | ---: |
| Li-F distance | 1.87e-14 | 1.42e-14 |
| P-F distance | 8.22e-15 | 2.31e-14 |
| Sigma | 2.49e-14 | 2.66e-14 |

Accepted-event logs agreed in event sequence, atom identities, direction and
acceptance probabilities. The MC MACE energy comparison was:

| Step | Uninterrupted delta E (eV) | Restarted delta E (eV) | Absolute difference (eV) |
| ---: | ---: | ---: | ---: |
| 20 | -0.922132219246 | -0.922132219232 | 1.46e-11 |
| 30 | -3.024234666551 | -3.024234666536 | 1.46e-11 |
| 40 | -2.427725719048 | -2.427725719030 | 1.82e-11 |
| 60 | -0.375835798157 | -0.375835798168 | 1.09e-11 |

Differences were already present before the split boundary (candidate logs:
step 30 for Rate MACE, step 20 for MC MACE). Their magnitude is consistent
with floating-point roundoff. Proposed absolute tolerances for repeating
these comparisons are 1e-12 angstrom for geometry and 1e-9 eV for reaction
energy, with exact discrete fields and acceptance decisions. These are
comparison tolerances, not changes to the simulation or universal error bounds.

## GPU memory and limitations

An earlier Rate MACE run failed after its first accepted reaction with a
7.71 GiB allocation request. The memory optimization compiles the full MACE
force calculation (including negation), NVT initialization, Langevin
initialization and overflow-recovery force. The MACE NVT force callable is
retained across topology changes. Both reactive 100-step reference runs and
their split continuations completed without a reported OOM after this change.
The original allocation failure's precise cause and peak GPU usage were not
measured; long-duration memory stability remains untested.

The results establish numerical equivalence of recorded outputs for these
specific runs, not bitwise equality of complete internal state, equivalence
of all unlogged rejected-proposal energies, physical model accuracy,
production kinetics, or detailed balance of the geometry-changing MC scheme.
Reverse MC and other foundation-model variants were not exercised by these
actual restart runs. Repeat integration validation for other configurations.
