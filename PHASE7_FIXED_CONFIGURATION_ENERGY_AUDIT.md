# Phase 7 — fixed-configuration MC energy audit

**Purpose.** Diagnose why the Phase 6 full 1 ps MC pilot accepted association and dissociation with negative local-plus-solvation energy differences. This adds a standalone **diagnostic only**; it does not change the paper-aligned MC/rate algorithms or the energy formula used for acceptance.

## Experiment

Two independent starting configurations use the same 16 A + 16 B constructed NVT lattice at 188 K:

- **Association:** A0/A1 unbonded, initially separated by 2.65 Å. Compare an unmodified-topology control relaxation with a topology-change + 0.25 Å-per-atom association translation + relaxation.
- **Dissociation:** A0/A1 bonded, initially separated by 1.50 Å. Compare an unmodified-topology control relaxation with a topology-change + 0.25 Å-per-atom dissociation translation + relaxation.

Each control/trial pair starts with identical coordinates, **zero** velocities, JAX PRNG key, temperature, and integrator settings. The control has no topology change or translation. These are **not** the actual Phase 6 step-10/20 states, and the two reaction directions do **not** share a starting configuration.

For each branch, log full-system potential, local-plus-solvation potential, independently calculated spectator-only shifted-LJ energy, and topology offset. Log the energy before the trial, immediately after topology change/translation, after trial relaxation, and after control relaxation. An additional force-field evaluation with the offset set to zero checks that V0 enters exactly once at fixed coordinates.

The difference between relaxed trial and relaxed control is a **diagnostic comparison only**. The actual rs@md MC criterion remains relaxed trial minus **unrelaxed** initial energy (plus the topology correction already carried in the force field). Do not substitute the control-adjusted value into MC acceptance.

## Run from your separate working directory

After synchronizing the Phase 7 source, run from your usual simulation folder. Make sure the `scripts` and `reactive_md` packages are both Phase 7 versions (the new script is `scripts/audit_a2_mc_energy.py`).

```bash
export JAX_ENABLE_X64=True
python -m scripts.audit_a2_mc_energy \
  --density-number-a3 0.0045 \
  --association-cutoff-a 3.0 \
  --relaxation-time-ps 0.005 \
  --output-dir results/phase7_short
```

The 0.005 ps run is only a software smoke test. If it succeeds, run the 1 ps test:

```bash
python -m scripts.audit_a2_mc_energy \
  --density-number-a3 0.0045 \
  --association-cutoff-a 3.0 \
  --relaxation-time-ps 1.0 \
  --output-dir results/phase7_full
```

**Compute cost:** Four independent relaxations, each with 4,000 steps at 1 ps. This is longer than the two-trial Phase 6 MC run. If 64-bit mode changes the results or performance, retain the setting in the output metadata.

**Outputs:** `energy_audit.json` contains full energy breakdowns; `energy_audit.md` summarizes MC, control-only, and control-adjusted differences and numerical reference checks.

**Pass criteria:** Finite energies; independent spectator LJ reference agrees with the spectator partition (within reported numerical tolerance); offset-once residual near zero. This is a numerical diagnostic, not proof of correct equilibrium sampling. If the reference fails, inspect `energy_audit.json` before any new long runs. The reference is valid only for this uncharged monatomic spectator setup.

**Known limitations:** The constructed starting states are not equilibrated. Relaxing the control and trial with the same PRNG key does not imply their trajectories remain coupled after topology change. The audit does not establish detailed balance or equivalence to the GROMACS energy-group settings in the original rs@md implementation.
