# Phase 6 — paired NVT pilot (MC vs rate)

## What this validates

This is a **short algorithmic and stability pilot** for the existing 16 A + 16 B, PEL I, 188 K NVT model. It does **not** reproduce the paper's 200 A + 824 B, 1 bar NPT numerical results. The starting configuration is a constructed lattice with one deliberately close A–A pair; it has not been equilibrated. Density, candidate cutoff, and rate constants are intentionally **not** inferred from the paper.

Both branches run from the same initial coordinates, box, temperature, timestep, thermostat configuration, seed, V0, and candidate cutoff. The run metadata includes an initial-position checksum. The methods are not expected to follow the same trajectory after reactions.

## Commands

Run from the extracted project root with JAX-MD installed.

**A. Inspect the commands first** (does not import JAX-MD):

```bash
python -m scripts.run_a2_pilot_pair \
  --density-number-a3 0.0045 --association-cutoff-a 3.0 \
  --association-rate-ps 2.0 --dissociation-rate-ps 2.0 --dry-run
```

The numbers `0.0045`, `3.0`, and `2.0` are **illustrative diagnostic choices only**. In particular, `2.0 ps^-1` is NOT a measured reaction rate from PEL II. With `check-every=10`, `dt=0.001 ps`, the example gives `Gamma*tau=0.02` per eligible rate candidate. The initial configuration is not an equilibrated solvent.

**B. Cheap software smoke test** (shortened relaxation, intentionally *not* the paper setting):

```bash
python -m scripts.run_a2_pilot_pair \
  --density-number-a3 0.0045 --association-cutoff-a 3.0 \
  --association-rate-ps 2.0 --dissociation-rate-ps 2.0 \
  --steps 20 --check-every 10 --relaxation-time-ps 0.005 \
  --output-dir results/paper_phase6_smoke
```

This only checks that both programs complete and produce finite energies and consistent output. No events in such a short run is normal.

**C. Paper-duration relaxation pilot** (1 ps per MC *attempt*, 0.25 fs relaxation step, so potentially expensive even for 100 ordinary MD steps):

```bash
python -m scripts.run_a2_pilot_pair \
  --density-number-a3 0.0045 --association-cutoff-a 3.0 \
  --association-rate-ps 2.0 --dissociation-rate-ps 2.0 \
  --steps 100 --check-every 10 \
  --output-dir results/paper_phase6_pilot
```

Replace all illustrative parameters with values justified for your intended scientific experiment before interpreting kinetics. To use independently measured PEL II rates, pass them explicitly. The paired runner rejects `Gamma*tau > 0.1` to avoid a clearly unsuitable pilot regime; `Gamma*tau << 1` is the actual theoretical requirement, and 0.1 is **not** an endorsed accuracy threshold.

## Output

- `mc/samples.csv`, `rate/samples.csv`: dimer counts, cumulative accepted associations/dissociations, potential and kinetic energy, candidate counts.
- `mc/events.csv`, `rate/events.csv`: accepted reactions.
- `mc/mc_attempts.csv`: all MC attempts, accepted/rejected status, acceptance probability and the local+solvation `delta_energy_kcal_mol`; rate branch produces an empty header-only file.
- `mc/metadata.json`, `rate/metadata.json`: configuration, checksum, counters, and rate probabilities.
- `pilot_protocol.json`: exact commands and settings.
- `pilot_summary.json` and `pilot_report.md`: consistency-checked comparison and limitations.

Re-analyze an existing paired run with:

```bash
python -m scripts.analyze_a2_pilot --root results/paper_phase6_pilot
```

## Interpretation and next gate

1. **Software:** Both runs finish; no nonfinite energy, no neighbor-list overflow, consistent event/attempt counters and same initial-position checksum.
2. **MC:** Check candidate counts, attempted and rejected trials, Metropolis probabilities, and local+solvation energy differences. The initial and final total PE are not the Metropolis energy difference; the thermostat and relaxation alter coordinates.
3. **Rate:** Check the supplied `Gamma*tau` and accepted event counts. No event in a short pilot is not a failure. There is no MC-like acceptance fraction for the rate method.
4. **Science:** A short, un-equilibrated 32-particle trajectory cannot establish equilibrium dimer concentrations, RDF agreement, true reaction rates, or numerical parity with the published NPT system. The NVT density and numerical mass/thermostat calibration remain open.

Only after this gate should we plan an equilibrated, multi-seed NVT parameter sweep and compare its equilibrium structure to the paper, with the ensemble discrepancy made explicit.
