# A₂ thermostat calibration — one-shot, decisive control

**Baseline:** `develop` at `1d35ef8` plus previously installed validation helpers. **Scope:** two additive validation modules, one NumPy test module, this README. No production-code or reaction-physics changes.

## Why this test (rather than more 2.5 ps trajectories)

Existing two-particle MD-only results show Nose–Hoover COM momentum exactly zero and relative kinetic ratios near 0.5; Langevin COM and relative ratios are near 0.2–0.3. Coordinate histograms remain far from the fixed-topology reference. The immediate issue is whether the **momentum marginal of the thermostat** is calibrated at all. Repeating the coupled trajectory cannot answer this cleanly.

This test removes all forces, bonds, neighbor lists, and reactions. It calls JAX-MD's *same* `simulate.nvt_langevin` and `simulate.nvt_nose_hoover` APIs used by the existing NVT control, with exactly the same `dt`, `gamma`, `tau`, `mass`, and `kT` conventions.

One run covers **two mass interpretations × two temperatures**. For each case it measures:

1. Langevin momentum distribution **immediately after initialization**.
2. Langevin momentum distribution **after 1 ps burn-in**, averaged over 64 independently seeded trajectories; reports replica-based standard errors and early/late means.
3. Nose–Hoover **initialization only**, including the COM constraint. It does **not** assert that free-particle Nose–Hoover is ergodic.

For equal masses and a canonical unconstrained ensemble, the analytical expectations are

\[ \mathbb E[K_\mathrm{total}/(3kT)] = \mathbb E[K_\mathrm{relative}/(1.5kT)] = \mathbb E[K_\mathrm{COM}/(1.5kT)] = 1. \]

The exact identity `2 * K_total_ratio = K_relative_ratio + K_COM_ratio` is tested. If COM is constrained to zero, the canonical target for `K_total/(3kT)` is **0.5**, *provided* the three relative momentum components are thermalized; we do not apply a six-DOF target to such a constrained system.

## Mass and unit audit

If the force-field energies are kcal/mol, lengths Å and simulation time ps, then

`1 amu = 0.00239005736 (kcal/mol) * ps² / Å²`.

Thus **4 amu = 0.00956022945 internal mass units**, whereas the previous controls supplied `mass=4.0` directly. The effective mass is **418.4 times** the mass of 4 amu; this changes characteristic free velocities by a factor of **20.46**. This is a *conditional unit concern*, not a declaration that production mass=4 is wrong: verify the project's intended mass and time conventions. A mass conversion alone cannot explain a normalized equilibrium kinetic ratio of 0.25.

## Install on HPC

```bash
unzip -q reactive_md_a2_thermostat_one_shot_20261027.zip -d /tmp/a2_thermostat_one_shot
cd ~/programs/reactive_md_project
cp /tmp/a2_thermostat_one_shot/validation/a2_thermostat_calibration_math.py validation/
cp /tmp/a2_thermostat_one_shot/validation/run_a2_thermostat_calibration.py validation/
cp /tmp/a2_thermostat_one_shot/validation/A2_THERMOSTAT_ONE_SHOT_CALIBRATION.md validation/
cp /tmp/a2_thermostat_one_shot/tests/test_a2_thermostat_calibration_math.py tests/
```

## One command, four cases

```bash
python -m pytest -q tests/test_a2_thermostat_calibration_math.py
python -m validation.run_a2_thermostat_calibration \
  --replicas 64 --steps 4000 --sample-every 100 --burn-in-steps 1000 \
  --gamma 10 --tau-T 0.1 --dt 0.00025 \
  --mass-amu 4 --effective-mass 4 \
  --output validation/a2_thermostat_calibration.json
```

**Interpretation and decision (no further incremental pilot required):**

- **Langevin ratios near 1 for both masses and temperatures:** JAX-MD stochastic thermostat is calibrated in the zero-force setting. The prior 0.2–0.3 ratios then point toward how the force-field control initializes or advances the Langevin state, or toward sampling differences. Inspect that wrapper in one targeted code review before longer dynamics.
- **Langevin ratios near 0.25 in the zero-force control too:** a temperature/momentum scaling or API issue is implicated. Audit `kT`, `mass`, and JAX-MD initialization/update paths directly; do not run longer coupled simulations.
- **Langevin correct for one mass but not the other:** investigate mass-dependent integrator usage or timestep stability.
- **Langevin initialization near 1, later ratios drift away:** integration/noise/friction implementation or numerical error is implicated, not just initialization.
- **Nose–Hoover COM zero:** expected given the observed constrained initialization; it does not by itself certify the NH temperature or its configurational ergodicity.

The script prints four summary lines and writes the detailed JSON. Share **those four lines**, especially `init K`, `K_total`, `K_rel`, `K_COM`, and the status. This single control is designed to determine the *next corrective action*, not to declare A₂ equilibrium validated.

## Limitations

This is a momentum-marginal calibration, not a proof of configurational canonical sampling, and it cannot certify the full moving-coordinate MH ensemble. The assessment thresholds (`max(0.15, 5 × replica SE)`) are exploratory, not formal hypothesis tests. The mass conversion assumes kcal/mol–Å–ps units. Nose–Hoover free-particle dynamics is intentionally not used as a decisive equilibrium test. JAX-MD execution requires the HPC environment; only the independent NumPy tests can be run locally.
