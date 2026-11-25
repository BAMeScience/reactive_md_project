# reactive-md

Reactive molecular dynamics with topology-changing reactions, built on [JAX-MD](https://github.com/jax-md/jax-md). The development repository supports classical OPLS-AA molecular dynamics, rate-based and Metropolis reaction schemes, and optional [MACE-JAX](https://github.com/ACEsuit/mace-jax) energy and force evaluation.

> **Development status (November 2026):** Short LiPF6 benchmarks have demonstrated MACE-JAX NVT integration and one accepted reaction in each of the rate and MACE-Metropolis modes. These are integration smoke tests, **not** production validation. Reactive runs have shown GPU memory-allocation warnings, and longer reactive runs and MD continuation after accepted reactions still need validation. See [Validation status](#validation-status-and-known-limitations).

## Features

- Classical OPLS-AA force-field energy and Nose-Hoover NVT dynamics.
- Topology-changing reactions with a chemistry-neutral MD driver and reaction handlers.
- LiPF6 decomposition example, with rate-based or sigma-gated Metropolis proposals.
- Optional activation-energy-derived reaction rates.
- Optional MACE-JAX **MD forces** (`--use-mace-md`) and **Metropolis energies** (`--use-mace-mc`), selectable in the five supported combinations below.
- Post-reaction Langevin relaxation and neighbor-list/force synchronization. The MACE-backed relaxation path is under active integration testing.
- Experimental A2 reaction handlers and compatibility entry points.

## Installation

Requires Python 3.12 or later and a JAX/JAX-MD-compatible environment. From the repository root:

```bash
python -m pip install -e .
```

For development and MACE support:

```bash
python -m pip install -e ".[dev,mace]"
```

The repository's `pyproject.toml` specifies the development dependencies. On a GPU cluster, install a CUDA-enabled JAX build compatible with the cluster's drivers and CUDA runtime; follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Do not assume a CPU JAX installation provides GPU support.

## Running LiPF6 simulations

The LiPF6 application accepts a LAMMPS data file and a corresponding settings file. The application CLI can be invoked directly:

```bash
python -m reactive_md.apps.lipf6 --help
```

The package also defines a `reactive-md` console entry point. Check `reactive-md --help` for the options exposed by the installed checkout.

### Classical NVT with rate-based reactions

```bash
python -m reactive_md.apps.lipf6 \
  --data system.data \
  --settings system.in.settings \
  --steps 1000 \
  --check-every 10 \
  --reaction-mode rate \
  --reaction-rate-ps 0.01 \
  --sigma-mid 0.0 \
  --sigma-width 0.2
```

### Classical MD with MACE Metropolis energies

```bash
python -m reactive_md.apps.lipf6 \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode metropolis \
  --use-mace-mc \
  --mace-source mp \
  --mace-variant small-0b2
```

Here the MD forces remain classical; MACE-JAX is used for the Metropolis energy evaluations.

### MACE MD with rate-based reactions

```bash
python -m reactive_md.apps.lipf6 \
  --data system.data \
  --settings system.in.settings \
  --steps 100 \
  --check-every 10 \
  --reaction-mode rate \
  --reaction-rate-ps 0.01 \
  --use-mace-md \
  --mace-source mp \
  --mace-variant small-0b2
```

### MACE MD with MACE Metropolis energies

```bash
python -m reactive_md.apps.lipf6 \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode metropolis \
  --use-mace-md \
  --use-mace-mc \
  --mace-source mp \
  --mace-variant small-0b2
```

The five supported combinations are:

| Reaction mode | `--use-mace-md` | `--use-mace-mc` | NVT forces | Metropolis energy |
| --- | --- | --- | --- | --- |
| metropolis | Off | Off | Classical | Classical |
| metropolis | Off | On | Classical | MACE-JAX |
| metropolis | On | On | MACE-JAX | MACE-JAX |
| rate | Off | Off | Classical | Not used |
| rate | On | Off | MACE-JAX | Not used |

MACE-MC is not used in rate mode. Metropolis with MACE-MD requires MACE-MC; the classical trial-energy path does not support MACE relaxation neighbor lists. Unsupported combinations are rejected at application entry. Post-reaction relaxation uses the selected MD energy and neighbor-list backend.

## LiPF6 reaction model

The example models

```text
LiPF6 <-> LiF + PF5
```

using the reaction coordinate

```text
sigma = d(P-F) - d(Li-F)
```

where F is the selected leaving fluorine. Negative sigma is reactant-like; positive sigma is product-like. `--sigma-mid` sets the midpoint of the sigmoid geometric factor, and `--sigma-width` controls its transition width. Candidates are ranked by sigma. Reverse reactions are disabled by default; reverse proposals use the recorded leaving F and Li partner rather than arbitrary new partners.

### Rate mode

In rate mode, an eligible candidate is subjected to geometric and rate-probability gates. The current algorithm uses the first-order probability

```text
p = f(sigma) * k_base * dt_reactive
```

where `f(sigma)` is the geometric factor and `dt_reactive` is the time between reaction checks. It is not the exact Poisson expression `1 - exp(-k_eff * dt_reactive)`. The base-rate gate must not exceed one. `k_base` can be specified with `--reaction-rate-ps` or derived from an activation energy with `--activation-energy-eV`.

For an activation barrier, the transition-state-theory base rate is

```text
k(T) = (k_B * T / h) * exp(-E_a / (k_B * T))
```

The optional reverse rate is configured separately using `--reverse-rate-ps`; it is not inferred from the forward barrier.

### Metropolis mode

Metropolis mode first uses the sigma-dependent geometric factor to decide whether to propose a reaction. The trial topology and geometry are then prepared and Langevin-relaxed. Energies of the original and trial configurations are evaluated using the selected classical or MACE model, and the energy difference enters the Metropolis acceptance test. Non-finite trial energy differences are rejected.

This sigma-gated, geometry-changing procedure is **not** a proven equilibrium Metropolis-Hastings sampler; proposal-density and reversibility considerations require separate validation. Experimental reverse proposals can be enabled with `--allow-reverse-mc`.

### Relaxation

Accepted rate-mode reactions and Metropolis trial proposals use Langevin relaxation. The principal settings are `--relaxation-dt-ps`, `--relaxation-time-ps`, and `--relaxation-tau-ps`. The resulting positions and velocities are passed to the MD driver; accepted topology changes require a refreshed force and a valid MD neighbor list.

## Output and tests

The LiPF6 application supports trajectory dumps, event logs, and candidate logs. For example:

```bash
python -m reactive_md.apps.lipf6 \
  --data system.data \
  --settings system.in.settings \
  --dump-file traj.dump \
  --event-log-file reactions.csv \
  --candidate-log-file candidates.csv \
  --candidate-log-top-n 20
```

Checkpoint/restart is implemented for all five LiPF6 simulation modes, including MACE-MD. Use `--checkpoint-file state.pkl --checkpoint-every 1000` (the interval must be a positive multiple of `--check-every`). To continue, use `--restart-from state.pkl --steps TOTAL_STEPS` with the same original input files, model, simulation settings and output paths. The target step count includes the completed steps. Checkpointed output files must be uncompressed; output beyond the saved boundary is rolled back on restart. A terminal partial reaction interval cannot be extended. Load only trusted checkpoints (pickle format).

MACE-MD checkpoints validate a backend fingerprint and reconstruct both neighbor lists. Regression tests use a differentiable surrogate rather than foundation-model weights. Actual MACE-MP `small-0b2` restart checks have also passed on the LiPF6 benchmark described below. Verify equivalence for your own model, system and software stack before relying on this feature for production.

Run the repository tests with:

```bash
python -m pytest -q tests/
```

Tests are grouped by subsystem under `tests/`:

| Directory | Coverage |
| --- | --- |
| `apps/` | Configuration, application dispatch and CLI |
| `engine/` | MD integration, units and relaxation |
| `forcefield/` | Classical force field and topology |
| `reactions/` | Shared reaction selection, acceptance and rates |
| `lipf6/` | LiPF6 reaction and topology workflows |
| `mace/` | MACE energy, model and MD interfaces |
| `restart/` | Checkpoint, restart and output consistency |
| `a2/` | A2 demonstration and reaction workflows |
| `a2/reference/` | Independent A2 numerical and statistical references |

Pytest discovers these directories recursively. For example,
`python -m pytest -q tests/engine/` runs the engine tests only.
The shared marker hook remains in `tests/conftest.py`.

The default suite includes all active tests, including A2 reference checks.
For focused CPU runs in the project environment:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -m a2
JAX_PLATFORMS=cpu python -m pytest -q -m a2_reference
JAX_PLATFORMS=cpu python -m pytest -q -m "a2 and not a2_reference"
JAX_PLATFORMS=cpu python -m pytest -q -m mace_interface
```

The markers classify dedicated test modules; they do not describe every
backend exercised by mixed application tests. MACE interface tests use
controlled models or surrogates and some require optional MACE packages.
They do not establish physical model accuracy. The independent A2 reference
checks include fixed-coordinate MH and moving-coordinate equilibrium
references, MD controls and calibration mathematics. They do not establish
detailed balance for the current geometry-changing rs@md[MC] workflow.
Archived tests are excluded from default discovery; see
[archived_tests/README.md](archived_tests/README.md).

## Validation status and known limitations

The Phase 6 CPU regression suite passed all 309 tests with strict markers.
Actual MACE-MP `small-0b2` GPU restart validation on a 5,264-atom
LiPF6-containing system produced the following results:

| Simulation | Uninterrupted / split steps | Accepted events | Result |
| --- | --- | --- | --- |
| MACE-MD, reactions disabled | 400 / 200 + 200 | 0 | Identical trajectory dumps |
| Rate MACE | 100 / 50 + 50 | 10 | Identical dumps; numerically equivalent reaction logs |
| MC MACE | 100 / 50 + 50 | 4 | Identical dumps; numerically equivalent reaction logs |

The reactive runs completed after compiling MACE force evaluation and
thermostat initialization paths to reduce GPU memory pressure. An earlier
reactive run failed with a 7.71 GiB allocation request. The successful short
runs do not establish peak memory requirements or long-duration stability.

See [MACE checkpoint/restart validation](validation/MACE_CHECKPOINT_RESTART.md)
for the protocol, observed numerical differences and scope of these results.
The comparisons establish recorded-output equivalence, not bitwise equality
of all internal checkpoint state or physical model accuracy. The artificial
reaction parameters used to exercise code paths are **not physical production
parameters**. Equilibrium sampling correctness remains a separate question.

## Repository architecture

| Component | Responsibility |
| --- | --- |
| `reactive_md/engine/contracts.py` | Chemistry-neutral reaction and observer contracts |
| `reactive_md/engine/nvt.py` | Nose-Hoover NVT stepping and neighbor-list recovery |
| `reactive_md/md_driver.py` | Generic MD loop, reaction dispatch, synchronization, observers |
| `reactive_md/reactions/` | LiPF6 and A2 reaction handlers and output adapters |
| `reactive_md/reaction.py` | LiPF6 reaction proposals and acceptance logic |
| `reactive_md/relaxation.py` | Post-reaction Langevin relaxation |
| `reactive_md/mace_setup.py`, `mace_energy.py`, `mace_mc_energy.py` | MACE-JAX model setup, energy functions, MC evaluator |
| `reactive_md/apps/lipf6.py` | LiPF6 application and CLI in the current development layout |

Compatibility imports such as `run_md_nvt_a2` and `run_md_nvt_with_reactions` are retained. The A2 reference simulation and its historical validation results should not be treated as validation of the newer LiPF6/MACE workflow.

## References

- Schoenholz, S. S. and Cubuk, E. D. (2020), *JAX M.D.: A Framework for Differentiable Physics*, NeurIPS. [Paper](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf).
- Batatia, I. et al. (2022), *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS. [Paper](https://openreview.net/forum?id=YPpSngE-ZU).
- [MACE-JAX repository](https://github.com/ACEsuit/mace-jax).
- Heuer and co-workers, *Journal of Chemical Theory and Computation* **17** (2021), 1074-1085 (rs@MD methodology).
- *Journal of The Electrochemical Society* **171** (2024), 080505 (LiPF6 reaction-coordinate motivation).

## License

See [LICENSE](LICENSE).

