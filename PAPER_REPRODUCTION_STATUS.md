# rs@md algorithmic reproduction: NVT / PEL I phase

Source: Biedermann, Diddens and Heuer, *rs@md: Introducing Reactive Steps at the Molecular Dynamics Simulation Level*, Sections 2.1, 3.2, 3.2.1, Fig. 3, Eqs. (10)-(11).

## Implemented in this change

- Replaced A2 `metropolis` cycle with paper Figure 3a ordering: enumerate eligible association and dissociation candidates, choose **one uniformly from their union**, tentatively change topology, translate pair, Langevin relax, evaluate before/after potential energies, apply Eq. (10), then commit or fully roll back physical state.
- The A2 MC handler now passes masses, shift function, and published relaxation parameters, and informs the NVT engine of accepted post-relaxation coordinates and velocities. Rejection returns original positions, velocities, topology and force field. RNG advancement is not rolled back.
- Removed the direction-first proposal probability and fixed-coordinate proposal-ratio acceptance from the reproduction path.
- Added deterministic trial/rollback/selection tests (`tests/test_a2_paper_mc.py`).
- Added an NVT PEL I driver (`scripts/run_a2_paper_nvt.py`) requiring an explicit **number density**, with shifted LJ, the paper's A2 bond stiffness and three V0 choices, and published reaction-relaxation parameters. The candidate cutoff is explicit and provisional.
- The existing A2 rate cycle already implements Eq. (11) and the paper's basic multi-candidate/relaxation sequence. It was not modified in this patch; its edge cases still require verification against author source.
- Retired the active production MH detailed-balance test and three legacy runners/tests that target the now-replaced fixed-coordinate algorithm. Independent numerical reference utilities are not acceptance tests for paper fidelity.

## Not yet established

1. **MC energy bookkeeping (Phase 5 source audit).** The authors' original C++ supports full-system, local-only, and local-plus-solvation energy modes. Our A2 MC now defaults to local-plus-solvation (full minus spectator-only), with the V0 correction included once in the force field. Full-system remains available as `energy_mode="full"`. See `PAPER_SOURCE_CODE_AUDIT_PHASE5.md` for source-level evidence and remaining GROMACS/averaging limitations.
2. **Reaction candidate cutoff.** The paper recommends nearest-neighbor selection based on RDF; 2.7 A is a provisional user-visible setting, not a demonstrated paper candidate cutoff.
3. **Rate source.** The NVT runner accepts external association/dissociation rates; PEL II derivation is postponed.
4. **Density and ensemble.** This runner uses user-specified NVT number density, not the paper's NPT conditions. No claim of exact numerical reproduction of published figures.
5. **Mass/thermostat calibration.** The runner's `--mass` is a numerical JAX-MD integrator mass; it must be calibrated for physically quantitative MD. The ordinary MD thermostat remains NVT Nose-Hoover rather than the paper's NPT velocity-rescale thermostat.
6. **No real JAX-MD execution here.** This environment lacks `jax_md`. The four new deterministic algorithm tests were executed with dependency stubs; the production JAX-MD tests must run in the project's HPC environment.

## Commands (project environment)

```bash
python -m pytest -q tests/test_a2_paper_mc.py tests/test_a2_rate.py
python -m scripts.run_a2_paper_nvt --mode mc --density-number-a3 0.0045 --steps 100 --check-every 10 --output-dir results/paper_mc_smoke
python -m scripts.run_a2_paper_nvt --mode rate --density-number-a3 0.0045 --association-rate-ps 0.00015 --dissociation-rate-ps 0.00006 --steps 100 --check-every 10 --output-dir results/paper_rate_smoke
```

The sample density and rates are **illustrative**, not paper-derived. Smoke runs only; do not run production simulations until all paper-algorithm tests and force-field checks pass.
