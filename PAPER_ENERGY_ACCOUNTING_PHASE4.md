> **Historical Phase 4 snapshot.** Superseded by [Phase 5 original-source audit](PAPER_SOURCE_CODE_AUDIT_PHASE5.md); the MC energy default has since changed.

# Phase 4: MC potential-energy bookkeeping audit

## Primary source

Biedermann, Diddens and Heuer, *rs@md: Introducing Reactive Steps at the Molecular Dynamics Simulation Level*, section 3.2, Equation (10), pp. 14-15 (manuscript pagination).

The paper explicitly writes `Delta V_ij = Delta V_MD_ij + V_corr`. It calls `Delta V_MD` a **naive difference in local potential energies** and notes that prior implementations considered either (a) reacting molecules alone or (b) reacting molecules plus solute-solvent interactions. The authors state that including reactant-solvent interactions was **essential to recover the equilibrium product concentration**. `V_corr` aligns the potential-energy zero points of the different force fields and may include separately determined reaction energies.

The paper does not give an exact per-particle/pair decomposition or the actual correction constants used by the C++ code. The author's public GitHub repository is `https://github.com/myrabiedermann/rsmd`, but its `src` tree was not accessible in this environment. **The original implementation's energy convention remains unverified.**

## Current JAX-MD code

`reactive_md/reactions/a2_metropolis.py`, `a2_metropolis_reactive_cycle`:

- Before the trial, computes `U_before = ff.energy_fn(R, nlist)['total']` with the original topology and original coordinates.
- Applies trial topology, translation and Langevin relaxation.
- After relaxation, computes `U_after = trial_ff.energy_fn(R_relaxed, nl_relaxed)['total']`.
- Uses `Delta U = U_after - U_before` in Equation (10).
- `_finalize_trial` sets `energy_offset = n_A2 * bond_energy_offset`, and the force-field wrapper includes that offset in `total`.

Thus the implementation uses a **full-system** energy difference across **different coordinates**. The offset is included once, but full-system energy also counts any solvent-solvent energy change during relaxation. This is a reasonable *explicit interpretation*, not proven identical to the authors' *local* energy difference. No speculative correction has been applied.

## Additional test

`tests/test_a2_paper_mc_jaxmd.py::test_real_mc_delta_energy_accounting_with_relaxation` checks both association and dissociation using real JAX-MD energies. It verifies that:

1. The production-reported `delta_energy` equals the actual full-system before/after difference.
2. The per-dimer energy offset changes exactly once, in the correct direction.
3. The total energy difference decomposes into the non-offset contribution plus the offset change.

**This test validates internal bookkeeping, not author-code parity.** It does not resolve the local-versus-full energy definition. Its trial relaxation is intentionally two MD steps, not the production 1 ps relaxation.

## Next scientific decision

Before changing the MC acceptance energy, obtain the C++ source's local-energy selection and `V_corr` convention (or an explicit statement from the authors). In particular, verify whether the C++ code evaluates reactant/solvent energy groups before and after relaxation, and whether it excludes changes involving only solvent atoms. Do not infer that from Equation (10) alone.

No change to the production MC acceptance formula, candidate cutoff, rate algorithm, NVT setup, PEL II or NPT was made in Phase 4.

## Verification

Run in the user's JAX-MD environment:

```bash
python -m pytest -q tests/test_a2_paper_mc_jaxmd.py
python -m pytest -q
```

Only Python syntax checks were run in this environment; JAX-MD is unavailable here.
