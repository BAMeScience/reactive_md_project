# A2 MH moving-coordinate pilot (two A atoms)

**Reference Git checkpoint:** `1d35ef8` on `develop`, as reported by the user.
This is an **additive-only validation patch**. It does not alter production
reaction handlers, the generic MD engine, or the validated rate-based A2 model.

## Why start with two atoms rather than four?

For two labeled A atoms, there are exactly two topologies (unbound and one
A2 dimer), and the *full moving-coordinate* canonical configurational
probability can be calculated independently by integrating the minimum-image
relative displacement over the entire periodic box. The previous four-atom
fixed-coordinate probabilities cannot serve as this reference.

Model: periodic cubic box of side **7 Å**, 300 K, two A particles with
sigma=1.2 Å, epsilon=0.2 kcal/mol, LJ cutoff=3 Å **unshifted**;
A2 bond potential `1.5*(r-2 Å)^2 - 0.5 kcal/mol`, replacing the bonded LJ
pair. Association cutoff=3 Å. There is no additional standard-state or
chemical-potential factor. For two *labeled* atoms, no extra combinatorial
factor is inserted. Identical masses in both topologies cause momentum
partition factors to cancel in the bonded/unbound ratio.

The independent reference integrates

`Z_T = integral_{[-L/2,L/2]^3} exp[-U_T(|r|)/kT] d^3r`

with tensor-product Gauss-Legendre quadrature; `P(bonded)=Z_b/(Z_u+Z_b)`.
The expected probability is approximately **0.284330** for these parameters
(96-point quadrature). The pilot checks actual JAX-MD force-field energies
against the independent analytic potentials at eight points **before**
performing any equilibrium comparison. If this check fails, do not use the
reference until the energy convention has been reconciled.

The small box is intentionally different from the earlier 30 Å four-atom
fixed-coordinate experiment. The particle mass is set to 4.0 amu to speed
coordinate exploration; this changes kinetics but not the canonical
coordinate distribution.

## Install in your existing `develop` checkout

Extract the additive ZIP into a temporary directory, then copy the four files:

```bash
mkdir -p /tmp/a2_moving_patch
unzip -q reactive_md_a2_moving_pilot_patch_20261027.zip -d /tmp/a2_moving_patch
cd ~/programs/reactive_md_project
cp /tmp/a2_moving_patch/validation/a2_mh_moving_reference.py validation/
cp /tmp/a2_moving_patch/validation/run_a2_mh_moving_pilot.py validation/
cp /tmp/a2_moving_patch/validation/A2_MH_MOVING_PILOT.md validation/
cp /tmp/a2_moving_patch/tests/test_a2_mh_moving_reference.py tests/
```

Run the fast independent-reference tests:

```bash
python -m pytest -q tests/test_a2_mh_moving_reference.py
```

Run the **short integration pilot** (real NVT Nose-Hoover + production MH):

```bash
python -m validation.run_a2_mh_moving_pilot \
  --steps 200 --check-every 10 --burn-in-checks 5 --chains 2 \
  --output validation/a2_mh_moving_pilot.json
```

If the pilot works, share the two per-chain lines, pooled bonded fraction,
energy spot-check result, and any warnings or traceback. The report includes
per-chain topology populations, acceptance counts and mean bond distances.

## Interpretation and limitations

**A successful pilot is not an equilibrium PASS.** At 200 MD steps with
`dt=0.00025 ps`, each chain spans only 0.05 ps; this is insufficient for
coordinate mixing. Consecutive topology observations are highly correlated,
so a binomial standard error would be misleading. Nose-Hoover dynamics in a
very small system can also be nonergodic. A mismatch with 0.284330 at this
stage is expected and is *not* evidence that MH violates detailed balance.

After a successful pilot we should (a) assess the cost of repeated topology
force-field rebuilding and NVT recompilation, (b) select longer independent
chains with adequate coordinate mixing and an uncertainty estimator based on
correlation times / block means, and (c) independently cross-check a stochastic
NVT integrator if deterministic Nose-Hoover sampling is nonergodic. Any
performance changes must preserve production physics and the A2 rate path.

The initial offset for a pre-bonded system is set explicitly and the energy
spot-checks verify both topologies. A periodic cutoff strictly below half the
box length avoids minimum-image cutoff ambiguity.
