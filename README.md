> **Phase 7:** Fixed-configuration MC energy audit: see [PHASE7_FIXED_CONFIGURATION_ENERGY_AUDIT.md](PHASE7_FIXED_CONFIGURATION_ENERGY_AUDIT.md). This is an optional diagnostic; the simulation algorithms are unchanged.

> **rs@md paper reproduction status (2026-10-29):** The A2 `metropolis` path has been replaced by the paper's Figure 3a trial-relaxation/Metropolis workflow. The historical fixed-coordinate MH validation is **not** a validation of this algorithm. For scope, density, energy-mode audit, and HPC test commands see [PAPER_REPRODUCTION_STATUS.md](PAPER_REPRODUCTION_STATUS.md). PEL II and NPT are postponed. Phase 5 energy-source audit: [PAPER_SOURCE_CODE_AUDIT_PHASE5.md](PAPER_SOURCE_CODE_AUDIT_PHASE5.md).

# reactive_md_project

# reactive-md

## Current development architecture

This is the **single development repository**. The original 100 ns A2
simulation is a read-only scientific reference; do not overwrite its code or
Python environment. The current structure separates:

- `reactive_md/engine/contracts.py`: chemistry-neutral reaction and observer contracts
- `reactive_md/engine/nvt.py`: Nose-Hoover NVT integration and neighbor-list recovery
- `reactive_md/md_driver.py`: generic MD loop, reaction dispatch, and observer notifications
- `reactive_md/reactions/`: A2 rate, A2 Metropolis and LiPF6 handlers/output adapters
- `tools/audit_a2_trajectory.py`: dependency-free 100 ns CSV consistency checker

Legacy `run_md_nvt_a2` and `run_md_nvt_with_reactions` imports remain supported.
The original small/reversible A2 example scripts are restored in `scripts/`.
See `DEVELOPMENT_STATUS.md` for verified tests and outstanding JAX-MD validation.

---

Reactive molecular dynamics framework for topology-changing simulations using JAX-MD.

The package supports:

- Classical OPLS-AA force fields
- Stochastic topology-changing reactions
- Metropolis-based reaction acceptance
- Rate-based reaction kinetics
- Activation-energy-driven reaction kinetics (rs@MD / Heuer-style)
- Optional MACE-JAX / MLIP energy evaluation for the Metropolis reaction acceptance step

The current implementation includes a LiPF6 decomposition example:

```text
LiPF6 -> LiF + PF5
```

The LiPF6 reaction is controlled by a single reaction coordinate

```text
sigma = d(P-F) - d(Li-F)
```

where the selected F atom is the leaving fluorine. Negative sigma values are reactant-like, while positive sigma values are product-like. The reaction code no longer uses independent hard Li-F or P-F distance gates to decide reactivity; the individual distances are used only to compute sigma and for diagnostic logging.

---

# Installation

## Basic installation

Clone the repository:

```bash
git clone https://github.com/BAMeScience/reactive_md_project.git
cd reactive_md_project
```

Install in editable mode:

```bash
pip install -e .
```

or with uv:

```bash
uv pip install -e .
```

## Running tests on HPC

Run from the repository root in the development environment:

```bash
python -m pytest -q tests/
# Or through SLURM:
srun --gres=gpu:1 --time=00:10:00 python -m pytest -q tests/
```

Pytest is configured with `pythonpath = ["."]` so imports from the
repository-local `scripts/` package work with either `pytest` or
`python -m pytest`. If test collection refers to `tests/test_reaction.py` or
`tests/test_topology_opls.py`, the checkout contains tests **not shipped in
this development archive**. Back up those old tests before migrating them;
they refer to interfaces absent from the current implementation. A clean
checkout is preferable to extracting the archive over an older source tree.

## Development installation

Install the package together with the test/development tools:

```bash
uv pip install -e .
uv pip install pytest ruff
```

Alternatively, install the included development extras:

```bash
uv pip install -e ".[dev]"
```

---

## Installation with MACE support

For MACE-assisted reaction acceptance:

```bash
uv pip install -e ".[mace]"
```

The currently tested dependency stack is:

```text
jax     0.10.1
jaxlib  0.10.1
flax    0.12.2
haiku   0.0.16
```

For GPU support, install the appropriate CUDA-enabled JAX version, for example:

```bash
uv pip install -U "jax[cuda13]"
```

---

# Running a Simulation

The package provides the command-line executable:

```bash
reactive-md
```

Display all available options:

```bash
reactive-md --help
```

---

# Input Files

The code requires:

1. A LAMMPS data file
2. A LAMMPS settings file

Example:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings
```

---

# Reaction Coordinate

The LiPF6 decomposition reaction uses the Fattebert-inspired coordinate:

```text
sigma = d(P-F) - d(Li-F)
```

Interpretation:

- `sigma < 0`: the leaving F is closer to P than to Li, so the geometry is PF6-like.
- `sigma = 0`: the leaving F is equally distant from P and Li.
- `sigma > 0`: the leaving F is closer to Li than to P, so the geometry is product-like.

Candidate reactions are ranked by sigma. The reaction probability is derived from sigma using a smooth sigmoid-like factor controlled by:

```text
--sigma-mid
--sigma-width
```

`--sigma-mid` is the sigma value where the geometric reaction factor is 0.5. `--sigma-width` controls how sharply the probability changes around `sigma_mid`.

Example:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode rate \
  --reaction-rate-ps 0.01 \
  --sigma-mid 0.0 \
  --sigma-width 0.2
```


---

# Reaction Modes

## 1. Metropolis Mode

Topology changes are proposed using the reaction coordinate σ and accepted according to a hybrid kinetic/thermodynamic Metropolis scheme.

The reaction coordinate σ is first evaluated to determine whether a reaction trial should be attempted. Only if this kinetic condition is satisfied is the expensive trial topology generated, FIRE-relaxed, and evaluated with the selected energy model (classical force field or MACE). The final thermodynamic acceptance is then determined using the Metropolis criterion based on the energy difference ΔE. Trial reactions yielding non-finite energy differences are automatically rejected.

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode metropolis \
  --sigma-mid 0.0 \
  --sigma-width 0.2
```

In classical mode, the Metropolis energy difference is computed with the classical force field. If MACE support is enabled, the Metropolis energy difference is evaluated with MACE-JAX instead.

## 2. Rate-Based Mode

Topology changes are accepted according to a prescribed base reaction rate modulated by the sigma-dependent geometric factor:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode rate \
  --reaction-rate-ps 0.01 \
  --sigma-mid 0.0 \
  --sigma-width 0.2
```

The effective rate is

```text
k_eff = k_base * f(sigma)
```

and the reaction probability during one reactive interval is

```text
p = 1 - exp(-k_eff * dt)
```

where:

- `k_base` is the prescribed or activation-energy-derived base rate in ps^-1.
- `f(sigma)` is the sigma-dependent geometric factor.
- `dt` is the reactive check interval.

## 3. Activation-Energy Mode

Instead of specifying a reaction rate directly, provide an activation barrier:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode rate \
  --activation-energy-eV 0.20 \
  --sigma-mid 0.0 \
  --sigma-width 0.2
```

The base reaction rate is computed using transition-state theory:

```text
k(T) = (kB * T / h) * exp(-Ea / (kB * T))
```

and then modulated by the sigma-dependent geometric factor.

---

# MACE-Assisted Reaction Acceptance

To evaluate reaction energies with MACE-JAX:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode metropolis \
  --use-mace-mc
```

The molecular dynamics trajectory still uses the classical force field. MACE is used only for the Metropolis reaction acceptance energy evaluation. The FIRE relaxation after the topology proposal currently uses the product-side classical force field.

---

# Trajectory Output

Write a LAMMPS-style trajectory:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --dump-file traj.dump
```

Compressed output:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --dump-file traj.dump.gz
```

---

# Event and Candidate Logging

Record accepted reactions:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --event-log-file reactions.csv
```

Record sigma-ranked reaction candidates:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --candidate-log-file candidates.csv \
  --candidate-log-top-n 20
```

The candidate log contains sigma-ranked candidates. It no longer contains `passes_lif`, `passes_pf`, or `passes_all`, because those belonged to the old hard-distance-gated model.

Useful diagnostic columns include:

```text
step, mode, rank, pf6_index, li_idx, leave_F, d_lif, d_pf, sigma, accepted
```

---

# Development

Run tests:

```bash
pytest
```

Run only the reaction-coordinate tests:

```bash
pytest tests/test_reaction_sigma.py
```

Run with verbose output:

```bash
pytest -v
```

---

# Project Structure

```text
reactive_md/
├── config.py
├── forcefield.py
├── reaction.py
├── md_driver.py
├── main.py
├── lammps_io.py
├── extract_params_oplsaa.py
├── reactions/
│   ├── lipf6.py
│   └── templates_pf5.py

tests/
├── test_candidate_selection.py
├── test_config.py
├── test_forcefield.py
├── test_lipf6.py
├── test_main.py
├── test_main_cli.py
├── test_md_driver.py
├── test_metropolis.py
├── test_reaction_coordinate.py
├── test_template_pf5.py
└── test_topology_change.py

| Test                          | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| `test_candidate_selection.py` | Reaction candidate identification and ranking.  |
| `test_config.py`              | Configuration defaults and validation.          |
| `test_forcefield.py`          | Force-field construction and evaluation.        |
| `test_lipf6.py`               | LiPF₆ topology and chemistry utilities.         |
| `test_main.py`                | Core simulation workflow.                       |
| `test_main_cli.py`            | Command-line interface.                         |
| `test_md_driver.py`           | MD driver and reaction integration.             |
| `test_metropolis.py`          | Metropolis acceptance criterion.                |
| `test_reaction_coordinate.py` | Reaction coordinate (σ) evaluation.             |
| `test_template_pf5.py`        | PF₅ template generation.                        |
| `test_topology_change.py`     | Topology modification after accepted reactions. |


```

---

# Citation

If you use this software in academic work, please cite:

- JAX-MD
```
 @inproceedings{jaxmd2020,
  author = {Schoenholz, Samuel S. and Cubuk, Ekin D.},
  booktitle = {Advances in Neural Information Processing Systems},
  publisher = {Curran Associates, Inc.},
  title = {JAX M.D. A Framework for Differentiable Physics},
  url = {https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf},
  volume = {33},
  year = {2020}
 }
```

- MACE / MACE-JAX, if used
 https://github.com/ACEsuit/mace-jax
 ```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 } 
```
  
- The original rs@MD methodology by Heuer and co-workers, J. Chem. Theory Comput. 2021, 17, 1074-1085
- The ab initio work motivating the LiPF6 reaction coordinate, Journal of The Electrochemical Society, 2024 171 080505

---

# License

See LICENSE file.

---

# Reaction plugins and the generic MD engine (experimental refactor)

`reactive_md.md_driver` now owns **only** the generic Nose–Hoover NVT
integrator, transactional neighbor-list overflow recovery, reaction-handler
protocol, and observer dispatch. It contains no A2 or LiPF6 candidate logic.

- `reactive_md.reactions.a2_handlers.A2RateHandler` wraps the existing A2
  kinetic cycle, including its optional Langevin relaxation.
- `reactive_md.reactions.a2_handlers.A2MetropolisHandler` wraps the
  fixed-coordinate Metropolis–Hastings cycle.
- `reactive_md.reactions.lipf6_handler.LiPF6Handler` adapts the existing
  LiPF6 callback to the generic interface.
- `reactive_md.observers.EnergyCSVObserver` and `EventJSONLObserver` implement
  chemistry-neutral output formats.

A generic simulation can be launched with the following pattern (using an
already constructed force field, system state, box, and A2 reaction model):

```python
from reactive_md.md_driver import run_md_nvt
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from reactive_md.observers import EnergyCSVObserver, EventJSONLObserver

handler = A2MetropolisHandler(reaction=reaction, rng=rng)
with open('energy.csv', 'w') as energy_file, open('events.jsonl', 'w') as event_file:
    result = run_md_nvt(
        key, cfg=cfg, init_positions=R, masses=masses, shift_fn=shift_fn,
        ff=ff, sys=system, box=box, reaction_handler=handler,
        observers=(EnergyCSVObserver(energy_file), EventJSONLObserver(event_file)),
    )
```

Existing entry points remain available:

- `reactive_md.md_a2_driver.run_md_nvt_a2` preserves A2-specific RDF,
  population history, LAMMPS dump, and CSV formats.
- `reactive_md.md_driver.run_md_nvt_with_reactions` remains a lazy
  compatibility alias for the LiPF6 driver.
- `reactive_md.md_driver.run_md_nvt_a2` remains a lazy compatibility alias.

Both compatibility drivers now use the same generic `run_md_nvt` engine.
LiPF6-specific CSVs and trajectory dumps live in `reactions/lipf6_output.py`;
the optional `on_before_reaction` hook captures pre-reaction energies and
frames without adding chemistry-specific branches to the MD engine.

The LiPF6 driver
also benefits from overflow-checked MD chunks and a refreshed cached force
following topology changes. That can change a trajectory compared with the
previous implementation, so run numerical regression checks before production.

The current A2 MC implementation is a *topology-only* sampler. Its attempt
frequency is not a physical reaction rate. The target distribution uses the
force-field energy (including the per-dimer energy offset); there is no
independent chemical-potential or standard-state correction. Coordinate-changing
MC proposals require their own reversible proposal-density correction.

This archive **includes** the restored `scripts/` package imported by
the small A2 tests. The original `jax-md` dependency is still required for
integration tests. Do not deploy this experimental refactor into an ongoing
production job without testing it in a separate environment.

## Development refactor status

See [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for the single-repository
plugin refactor, compatibility guarantees, test results, and outstanding
validation before production use.

### Phase 6: Paired NVT pilot

See [PHASE6_PAIRED_NVT_PILOT.md](PHASE6_PAIRED_NVT_PILOT.md) for a short, matched-initial-condition MC/rate diagnostic with explicit rate/cutoff inputs, MC attempt energies, and a machine-readable report. It is **not** a published NPT benchmark.
