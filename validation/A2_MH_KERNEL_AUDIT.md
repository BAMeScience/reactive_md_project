# Two-atom moving-coordinate reference and MH kernel audit

**Baseline:** validated `develop` commit `1d35ef8`, plus the additive moving-coordinate pilot files. **Production code is unchanged.**

## Scientific distinction

The short pilot started at an A–A separation of 2.0 Å. At this fixed separation, the independent model gives `P(bonded | r=2 Å) ≈ 0.68547`. The full periodic-box canonical reference is `P(bonded) ≈ 0.28433`. The two values describe different distributions: the latter requires spatial equilibration. Thus the observed 80% bonded fraction in a 0.05 ps pilot does **not** by itself establish a sampling defect.

The independent two-atom proposal kernel has direction probabilities `p_assoc = p_diss = 1/2`. For `r <= 3 Å`, there is exactly one candidate in either direction and the proposal ratio is 1:

- `A(unbound -> bonded | r) = min(1, exp(-(U_b-U_u)/kT))`
- `A(bonded -> unbound | r) = min(1, exp(+(U_b-U_u)/kT))`

For `r > 3 Å`, association has no candidate. Dissociation is also **rejected** because its reverse association would be unavailable. This makes both topology transitions zero-probability at those fixed coordinates. It **does not** truncate the bonded equilibrium reference: a bonded pair may move beyond 3 Å through MD, and topology exchange can resume after it returns inside the cutoff. Both kernels preserve the intended Boltzmann measure locally if the coordinate-update kernel does so.

## Audit scope

The production audit tests the actual `a2_metropolis_reactive_cycle` and actual JAX-MD energies, using deterministic RNG draws. It checks:

1. Both topology energies and their differences against independent analytic potentials.
2. Correct candidate availability and accepted transitions at 2.0 and 2.95 Å.
3. Null association and reverse-blocked dissociation at 3.05 Å.
4. Minimum-image behavior across the periodic boundary at 2.0 Å.
5. Exact forward/reverse proposal ratios, acceptance probabilities, and local detailed-balance fluxes.
6. Preservation of coordinates and velocities by topology proposals.

The audit does **not** prove Nose–Hoover ergodicity, finite-timestep canonical sampling, or moving-coordinate equilibrium. It is a prerequisite to a longer convergence study, not a substitute.

## Installation

Extract the additive ZIP and copy its `validation/` and `tests/` files into the existing HPC checkout. Keep the existing `validation/a2_mh_moving_reference.py` and `validation/run_a2_mh_moving_pilot.py` from the previous moving-coordinate pilot. Do not replace production modules.

## HPC commands

```bash
cd ~/programs/reactive_md_project
python -m pytest -q tests/test_a2_mh_kernel_reference.py
python -m pytest -q tests/test_a2_mh_kernel_production.py
python -m validation.run_a2_mh_kernel_audit --output validation/a2_mh_kernel_audit.json
```

The full production audit builds multiple real JAX-MD force fields and may take several minutes. If a test fails, share the relevant traceback; do not adjust the reaction algorithm until we know whether the mismatch is in energy, proposal eligibility, or numerical tolerance.

**Interpretation:** A PASS means that the two-atom fixed-coordinate reaction kernel and independent reference are consistent at the audited coordinates. The next stage is a longer, independently initialized moving-coordinate experiment with coordinate-mixing diagnostics, autocorrelation-aware uncertainties, and a check of Nose–Hoover ergodicity.
