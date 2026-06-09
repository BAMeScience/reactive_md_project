# reactive_md_project
# reactive-md

Reactive molecular dynamics framework for topology-changing simulations using JAX-MD.

The package supports:

* Classical OPLS-AA force fields
* Stochastic topology-changing reactions
* Metropolis-based reaction acceptance
* Rate-based reaction kinetics
* Activation-energy-driven reaction kinetics (Heuer-style)
* Optional MACE-JAX energy evaluation for reaction acceptance

The current implementation includes a LiPF₆ decomposition example:

```
LiPF6 -> LiF + PF5
```

but the framework is designed to be extended to additional reactive processes.

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

---

## Installation with MACE support

For MACE-assisted reaction acceptance:

```bash
uv pip install -e ".[mace]"
```

The currently tested dependency stack is:

```
jax     0.10.1
jaxlib  0.10.1
flax    0.12.2
haiku   0.0.16
```

For GPU support, install the appropriate CUDA-enabled JAX version, e.g.

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

1. LAMMPS data file
2. LAMMPS settings file

Example:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings
```

---

# Reaction Modes

## 1. Metropolis Mode

Topology changes are accepted according to a Metropolis criterion.

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode metropolis
```

---

## 2. Rate-Based Mode

Topology changes are accepted according to a prescribed reaction rate:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode rate \
  --reaction-rate-ps 0.01
```

The reaction probability is

```
p = 1 - exp(-k * dt)
```

where

* `k` is the reaction rate in ps⁻¹
* `dt` is the reactive check interval

---

## 3. Activation-Energy Mode

Instead of specifying a reaction rate directly, provide an activation barrier:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --reaction-mode rate \
  --activation-energy-eV 0.20
```

The reaction rate is computed using transition-state theory:

```
k(T) = (kB * T / h) * exp(-Ea / (kB * T))
```

following the rs@md methodology of Heuer and co-workers.

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

The molecular dynamics itself still uses the classical force field. MACE is used only for reaction acceptance.

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

# Event Logging

Record accepted reactions:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --event-log-file reactions.csv
```

Candidate and near-miss logging:

```bash
reactive-md \
  --data system.data \
  --settings system.in.settings \
  --candidate-log-file candidates.csv
```

---

# Development

Install development tools:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
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
```

---

# Citation

If you use this software in academic work, please cite:

* JAX-MD
* MACE / MACE-JAX (if used)
* The original rs@md methodology by Heuer and co-workers

---

# License

See LICENSE file.

