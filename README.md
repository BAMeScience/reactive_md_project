# reactive_md_project

# reactive-md

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

## Development installation

Install the package together with the test/development tools:

```bash
uv pip install -e .
uv pip install pytest ruff
```

If you prefer optional dependency groups, move the `dev` dependency list into `[project.optional-dependencies]` in `pyproject.toml`, then use:

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

`r_pf_probe` is not a reaction gate. It is only used after a reaction has been accepted to move the leaving F away from P before product-side FIRE relaxation.

---

# Reaction Modes

## 1. Metropolis Mode

Topology changes are proposed according to the sigma-based geometric probability and accepted according to a Metropolis criterion:

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
├── test_config.py
├── test_forcefield.py
├── test_main.py
├── test_md_driver.py
├── test_reaction.py
├── test_reaction_sigma.py
├── test_template_pf5.py
└── test_topology_opls.py
```

---

# Citation

If you use this software in academic work, please cite:

- JAX-MD
- MACE / MACE-JAX, if used
- The original rs@MD methodology by Heuer and co-workers, J. Chem. Theory Comput. 2021, 17, 1074-1085
- The ab initio work motivating the LiPF6 reaction coordinate, if used in your study

---

# License

See LICENSE file.
