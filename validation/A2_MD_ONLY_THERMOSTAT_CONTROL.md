# Two-atom MD-only thermostat and coordinate-mixing control

**Production baseline:** `develop`, commit `1d35ef8`, plus the previously installed moving-coordinate validation helpers. **No production code is modified.** The two-atom model, force field, periodic box, timestep and mass match the earlier NVT–MH diagnostic.

## Motivation

The four short coupled NVT–MH chains remained strongly dependent on their initial coordinates. Near chains repeatedly changed topology but stayed around 2 Å; far chains never entered the reactive region in 0.25 ps. We need to determine how much of that behavior occurs **without reactions**, and whether a stochastic thermostat provides a useful coordinate-mixing control.

This experiment holds topology fixed and runs:

| Integrator | Topology | Start |
|---|---|---|
| JAX-MD Nose–Hoover (production `NVTStepper`) | Unbound | Near (2.0 Å), far (3.81 Å) |
| JAX-MD Nose–Hoover (production `NVTStepper`) | Bonded | Near, far |
| JAX-MD Langevin (same force field and overflow-safe chunk logic) | Unbound | Near, far |
| JAX-MD Langevin (same force field and overflow-safe chunk logic) | Bonded | Near, far |

The stochastic Langevin integrator uses the same JAX-MD API and force field as the existing relaxation code; it is used **only in this validation script**. `--gamma 10` is a diagnostic friction setting in inverse simulation-time units, not a fitted physical friction constant. Both integrators use the same mass and timestep. Different random seeds are used for each case; results are not pathwise comparable.

## Reference and observables

For each fixed topology, the appropriate configurational reference is **P(r | topology)**, calculated by independent quadrature over the periodic cube. The full mixed-topology reference P(bonded)=0.28433 is *not* a target for these fixed-topology trajectories.

Every sample records minimum-image distance, signed displacement, momenta-derived kinetic-energy ratios, COM momentum norm and force norm. The JSON reports mean/min/max distance, early/late mean distance, radial histogram total variation versus the conditional reference, and heuristic integrated autocorrelation. CSV traces are written per case.

Kinetic ratios are `K_total/(3 kT)` for six unconstrained momentum components and `K_relative/(1.5 kT)` for the three relative components. These ratios **must not be interpreted as formal temperature tests** in a two-particle Nose–Hoover system, especially if COM motion is constrained or conserved. No COM momentum is artificially removed from the trajectories.

## Prerequisites

The following existing validation files must already be present:

- `validation/a2_mh_moving_reference.py`
- `validation/a2_mh_mixing_reference.py`
- `validation/run_a2_mh_moving_pilot.py`

These were installed in the earlier moving-coordinate validation stages.

## Install on HPC

Unzip this additive patch into a temporary directory; copy only its four files:

```bash
mkdir -p /tmp/a2_md_only_control
unzip -q reactive_md_a2_md_only_thermostat_control_20261027.zip -d /tmp/a2_md_only_control
cd ~/programs/reactive_md_project
cp /tmp/a2_md_only_control/validation/a2_md_only_reference.py validation/
cp /tmp/a2_md_only_control/validation/run_a2_md_only_thermostat_control.py validation/
cp /tmp/a2_md_only_control/validation/A2_MD_ONLY_THERMOSTAT_CONTROL.md validation/
cp /tmp/a2_md_only_control/tests/test_a2_md_only_reference.py tests/
```

## Run

```bash
python -m pytest -q tests/test_a2_md_only_reference.py

# Short diagnostic: 1000 MD steps = 0.25 ps per case, 8 cases total
python -m validation.run_a2_md_only_thermostat_control \
  --steps 1000 --sample-every 20 --burn-in-samples 10 \
  --chains-per-condition 1 --gamma 10 --tau-T 0.1 \
  --output validation/a2_md_only_thermostat_control.json
```

The script performs real force-field energy spot checks before any dynamics. Eight CSV traces and one JSON report are produced. Use a GPU allocation consistent with the existing JAX-MD environment. **The short run is for stability and performance, not for equilibrium acceptance/rejection.** Longer runs can be planned after inspecting runtime, coordinate ranges and kinetic diagnostics. To assess stochastic variability, increase `--chains-per-condition` rather than treating adjacent samples as independent.

## Interpretation

- If both thermostats remain near their initial distances, 0.25 ps is too short to diagnose an integrator defect. The same two-atom force field and mass naturally limit how far particles move in that time.
- If Langevin mixes coordinates faster than Nose–Hoover under longer, matched runs, thermostat ergodicity or COM constraints warrant closer inspection. This is not, by itself, proof of a Nose–Hoover bug.
- If neither integrator matches the conditional radial reference after demonstrably sufficient sampling, investigate force-field/integrator consistency, time units, timestep bias and ergodicity before making production changes.
- The bonded far start may remain outside the reaction cutoff because reactions are disabled and the harmonic potential can bring the atoms inward only through coordinate dynamics.
- The unbound potential is unshifted truncated LJ, and the bonded potential is harmonic with a constant energy offset. A constant offset affects mixed-topology probabilities but **not fixed-topology forces or conditional radial distributions**.

**Important limitations:** The radial reference histogram has finite quadrature/bin error. The sample trajectories are strongly correlated and start far from equilibrium. Total variation, autocorrelation and kinetic ratios are descriptive only. This experiment cannot certify canonical sampling or the full MD–MH equilibrium ensemble.

## What to share

Please share the eight per-case summary lines and the total runtime, especially the distance ranges, conditional radial TV and relative kinetic ratios. Do not commit the patch until the HPC checks pass.
