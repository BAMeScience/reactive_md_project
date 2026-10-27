# A2 Metropolis-Hastings: coupled NVT-MH integration validation

**Reference:** User-provided `reactive_md_project-develop.zip`, extended only
with the previously reviewed MH validation and sampling tools. **No production
Python code was modified** for this stage.

## Scientific scope

The fixed-coordinate synthetic and real-energy MH sampling experiments passed
on HPC (four chains each). The next question is whether the *production generic
NVT driver* correctly interleaves Nose-Hoover MD with topology proposals.

This first-stage audit checks integration invariants, **not** the equilibrium
stationary distribution for evolving coordinates. The MH target remains the
specified model `exp(-U(R, topology) / kT)`, including one energy offset per A2
bond; no standard-state or chemical-potential factor is included.

## Run from your HPC `develop` checkout

Copy only the following new files from the archive into your checkout:

- `validation/run_a2_mh_coupled.py`
- `validation/A2_MH_COUPLED_RUN.md`
- `tests/test_a2_mh_coupled_integration.py`

The script also imports `validation/run_a2_mh_sampling.py` from the previously
validated sampling package. Keep that existing file in your checkout.

```bash
python -m pytest -q tests/test_a2_mh_coupled_integration.py

python -m validation.run_a2_mh_coupled \
  --steps 60 --check-every 1 --replay \
  --output validation/a2_mh_coupled_integration.json

python -m pytest -q tests/
```

## Checks and interpretation

The experiment starts separately with zero and two dimers and runs the **actual
production** `run_md_nvt` with `A2MetropolisHandler`. It uses real JAX-MD
force fields, uncached energy evaluations, periodic displacement and Nose-Hoover
NVT dynamics. No production function is monkeypatched.

The audit asserts:

1. All positions, momenta, velocities, forces and energies remain finite.
2. Topology remains a valid disjoint matching of four A atoms.
3. Total force-field offset equals `number_of_dimers * bond_energy_offset`,
   including an initially two-dimer topology.
4. Each accepted MH move leaves coordinates, momenta and velocities unchanged
   at the instant of topology change; no relaxation or thermostat reset occurs.
5. Reported MH energy difference agrees with independently evaluated pre/post
   force-field energies at the **same** coordinates (2e-3 kcal/mol tolerance).
6. Cached force after each accepted move agrees with a fresh JAX gradient.
7. Neighbor lists do not report overflow at accepted checkpoints.
8. Rejected and null proposals preserve the original force field and topology.
9. Reaction checks and accepted-event counts are consistent.
10. Optional same-seed replay reproduces the same trajectory on the same device.

`--steps 60` is a smoke test, not a sampling duration. No minimum accepted-event
count is imposed; if a run happens to have zero accepted moves, the accepted
move-specific invariants were **not exercised**. In that case increase
`--steps` and/or try a different `--seed` before considering the integration
validated. Also note that a short same-seed replay is a reproducibility check,
not proof of equilibrium.

## Report to share

Only the two printed lines for `Initial dimers=...`, the `Coupled MD-MH
integration: PASS/FAIL` line, and any traceback if a check fails. The JSON
contains detailed trajectories and can remain local on HPC.

## After this stage

If both starting topologies exercise accepted moves and all checks pass, move
to **coordinate-dependent equilibrium validation**. That requires an independent
configurational reference (e.g. carefully controlled reference Monte Carlo),
multiple long chains and uncertainty accounting. The fixed-coordinate ten-state
Boltzmann probabilities cannot be reused as the expected frequencies for
moving coordinates.
