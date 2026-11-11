# LiPF6 reversible topology + Langevin prototype

This is a **development patch**, not a validated production simulation.

## What changed

- Replaced LiPF6 FIRE relaxation with the existing shared JAX-MD Langevin routine.
- Returned relaxed velocities to the generic NVT handler; reinitialize the Nose-Hoover thermostat on accepted events.
- Recorded the leaving fluorine and associated lithium per reacted PF6 unit.
- Added reverse-candidate selection, a P-F proximity gate, and exact restoration of the original PF6 bonded terms, atom parameters, and molecule IDs from an initial-state snapshot.
- Enabled reverse events as opt-ins: `--allow-reverse-mc` or `--reverse-rate-ps`.
- Kept the forward rate law `p_sigma * k * dt`, with reverse rate off by default.
- Added `direction` to accepted-event CSV rows, retaining existing column order before the appended field.

## Testing

- `python -m compileall -q reactive_md`
- `pytest -q tests/test_lipf6_reverse_roundtrip.py` (3 tests passed locally)
- Stubbed forward/reverse rate and Metropolis orchestration smoke checks passed locally.
- The actual JAX-MD numerical simulation and energy/relaxation stability are **not tested** here: `jax_md` is unavailable in this environment.
- A full parser/round-trip check on the 5,264-atom input was attempted but exceeded the container's memory limit; the synthetic topology test passed.

## Suggested first SLURM pilot

Run a very short job with `--reaction-mode rate` and `--reverse-rate-ps 0` initially, enabling event logs. Use the activation-energy-derived forward rate or a test rate of your choice. Check energies, neighbor-list stability, velocities, and charge after each accepted reaction. Then explicitly test one reverse event using a justified test-only reverse rate and inspect the PF6 bond/angle restoration. Do not interpret reverse rate results as physical without independent parameterization.

## Important limitations

- Reverse recombination uses the *recorded* leaving F and Li partner; LiF is nonbonded.
- The optional hybrid Metropolis reverse mode is an exploratory sigma-gated proposal; it does **not** include a Metropolis-Hastings proposal-ratio correction, so it should not be interpreted as an exact equilibrium sampler.
- Reverse events are allowed only while the recorded F is within 4.0 A of P and Li. These are implementation guardrails, not calibrated kinetic capture radii.
- Default Langevin relaxation is 5 ps at 0.25 fs and tau=0.01 ps; this is inherited from A2 and must be tested for LiPF6 stability and performance.
