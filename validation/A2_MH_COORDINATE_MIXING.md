# A2 MH: moving-coordinate mixing diagnostic

**Production baseline:** `develop` commit `1d35ef8`, with the previously installed moving-coordinate pilot and kernel-audit validation files. This patch is **additive only**. It does not alter the generic engine, MH handler, A2 rate model, or force field.

## Question

The 200-step two-atom NVT–MH pilot observed 80% bonded versus the full-box canonical reference 28.433%. The independent production-kernel audit passed at 2.00, 2.95, 3.05 Å and across a periodic boundary. The next question is whether coordinates explore the periodic box and whether topology/coordinate populations converge across different starts. This experiment is **diagnostic, not an equilibrium certification**.

## What it measures

- Four deliberately distinct starts: unbound at 2.0 Å, bonded at 2.0 Å, unbound at 3.81 Å, and bonded at 3.81 Å. Subsequent chains repeat this cycle with different seeds. The far starts are outside the 3.0 Å reaction cutoff, where topology moves are blocked until MD brings the atoms back into range.
- Each chain records the minimum-image distance, signed displacement components, topology, reaction outcome, and whether the pair lies outside the association cutoff at **every reaction check**.
- An independent tensor-product quadrature calculates the full-box bonded fraction, topology-resolved radial shell distributions, and reference mean distances. The script verifies eight real JAX-MD energy points before launching MD.
- Each chain reports the first/second-half bonded fraction, distance range, signed-displacement standard deviations, and a *heuristic* autocorrelation time/effective sample size for bonded state and radius.
- The pooled report includes radial and joint topology–radial total variation, and a *heuristic* batch-means standard error. CSV traces allow plotting time evolution without extra dependencies.

## Installation

The four files in this ZIP require these **already-installed** files from the previous pilot:
`validation/a2_mh_moving_reference.py` and `validation/run_a2_mh_moving_pilot.py`.

```bash
mkdir -p /tmp/a2_mixing_patch
unzip -q reactive_md_a2_mh_coordinate_mixing_patch_20261027.zip -d /tmp/a2_mixing_patch
cd ~/programs/reactive_md_project
cp /tmp/a2_mixing_patch/validation/a2_mh_mixing_reference.py validation/
cp /tmp/a2_mixing_patch/validation/run_a2_mh_coordinate_mixing.py validation/
cp /tmp/a2_mixing_patch/validation/A2_MH_COORDINATE_MIXING.md validation/
cp /tmp/a2_mixing_patch/tests/test_a2_mh_mixing_reference.py tests/
```

## Run on HPC

```bash
python -m pytest -q tests/test_a2_mh_mixing_reference.py

# First: short performance and trace pilot (0.25 ps per chain)
python -m validation.run_a2_mh_coordinate_mixing \
  --steps 1000 --check-every 20 --burn-in-checks 10 \
  --chains 4 --block-size 10 \
  --output validation/a2_mh_coordinate_mixing.json
```

Each chain produces `validation/a2_mh_coordinate_mixing_chainN.csv`, plus a combined JSON report. Start with this **short run** to establish stability, runtime, and whether the far/near starts explore different regions. **Do not interpret 1000 steps (0.25 ps) as an equilibrated simulation.** Do not increase the length until we inspect this diagnostic and HPC cost.

## Interpretation and limitations

The reference is the canonical configurational integral for the stated unshifted LJ and harmonic bond with -0.5 kcal/mol offset in a 7 Å periodic cube. It assumes that the MD coordinate kernel samples the correct canonical distribution and that the overall chain is ergodic; the fixed-coordinate MH audit only establishes local detailed-balance consistency. Nose–Hoover dynamics for two atoms may fail to be ergodic. Finite integration timestep can also bias sampling.

**Radial shell quadrature:** The shell boundaries are discontinuous, so 96-point Gauss quadrature converges less smoothly for shell probabilities than for the total partition functions. Use the radial histograms as a *diagnostic*, not a stringent acceptance test. We should check shell convergence with a higher order before final quantitative inference.

**Autocorrelation/SE:** Short trajectories, nonstationarity, and rare topology transitions can make estimated autocorrelation times, ESS, and batch standard errors misleading. A constant-topology chain deliberately returns `null` ESS rather than implying independence. The pooled batch SE assumes approximately independent batches; it must not be used to claim equilibrium until mixing and stationarity are established. Radial and joint TV distances are descriptive, not formal p-values.

**Important:** No code path silently changes the thermostat, proposal kernel, reaction cutoff, mass, or force field. The mass is 4 amu solely to accelerate the diagnostic and is not a physical argon kinetics model. The four starts are *not* drawn from the canonical ensemble; compare first and second halves and across starts for convergence before considering the pooled estimate.

## What to share

Share the four per-chain lines, the pooled summary, the block SE, and total runtime. If any chain never reaches the reactive region or all chains remain strongly start-dependent, we'll examine coordinate dynamics and Nose–Hoover behavior before spending GPU hours on longer runs.
