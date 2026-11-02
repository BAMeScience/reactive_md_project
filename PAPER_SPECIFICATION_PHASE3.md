# Phase 3: MC energy convention and candidate-distance specification

## Scope

Biedermann, Diddens, Heuer, *rs@md: Introducing Reactive Steps at the Molecular Dynamics Simulation Level*, JCTC 2021, DOI 10.1021/acs.jctc.0c01189. We reproduce PEL I algorithms in NVT with user-selected density. PEL II and NPT remain deferred.

## 1. Candidate distance: numerical value is not supplied in the paper

- Section 3.2 (p. 14 of the manuscript): reaction candidates must satisfy at least a distance criterion; the authors recommend selecting nearest neighbors and estimating a distance cutoff from an A-A radial distribution analysis.
- Section 2.1 and Section 4.2: `r2 = 2.7 Angstrom` is the PEL II *barrier position* and a dividing boundary for measuring transitions in PEL II. It is **not** identified as the numerical cutoff used by the reactive-step candidate search.
- Section 4.2: a second, larger cutoff at the first minimum after the nearest-neighbor peak defines the neighboring-monomer state for rate extraction. That too is not an explicit published candidate-search cutoff.

**Decision:** The NVT production runner and `A2Reaction` now require an explicit association-candidate cutoff; no published value is silently inferred. The examples require the user to supply a cutoff chosen from an A-A RDF; no numerical value is presented as published. The input and provenance are recorded in metadata. When a representative A-A RDF becomes available for our selected NVT density, choose a cutoff from its first-neighbor shell and document that choice.

## 2. MC potential-energy difference: known and unknown

- Section 3.2 (p. 15): the paper's Equation (10) is `p=min(1, exp(-beta * (Delta V_MD + V_corr)))`.
- The paper says `Delta V_MD` is a naive difference in *local* potential energies, and that inclusion of reactant-solvent interactions is essential. `V_corr` aligns potential-energy zero points of different force fields and may contain reaction energies.
- The main text does **not** give a precise atom/pair decomposition for the local energy or a code-level formula for `V_corr`. The listed Supporting Information concerns the LJ potential, rate method, relaxation, and supplementary results; it does not, from its description, establish this missing MC bookkeeping.
- The present implementation computes `U_after(full system, relaxed coordinates, product topology) - U_before(full system, original coordinates, reactant topology)` and includes the per-dimer `V0` offset through the force field. This includes solvent-solvent changes during relaxation and is an **explicit interpretation, not a verified match to the original C++ program**.

**Decision:** Do not replace this with a guessed local energy expression or add an additional `V_corr` that could double-count `V0`. Do not declare MC numerical fidelity established. The authors' public C++ repository is `https://github.com/myrabiedermann/rsmd`; the precise energy terms and reaction input files still need direct source verification. This environment could access the public repository landing page but not its source tree.

## 3. Implementation changes

- `reactive_md/reactions/a2.py`: require explicit `association_cutoff` and explain why `2.7` cannot be assumed.
- `scripts/run_a2_paper_nvt.py`: make `--association-cutoff-a` required and validate finite/positive; record cutoff provenance.
- `reactive_md/reactions/a2_metropolis.py`: clarify the known limitation of full-system energy difference.
- Tests that previously relied on the implicit cutoff now specify the value explicitly; a new test checks that omission raises an error.
- No change to Metropolis acceptance formula, trial relaxation, rate algorithm, force-field physics, or NVT density.

## 4. Verification

- Python syntax compilation completed.
- Full pytest and JAX-MD integration **not executed here** because `jax_md` is not installed. The user's previous Phase 2 suite passed in their environment, but the new cutoff validation must be checked again.

Run:

```bash
python -m pytest -q tests/test_a2_reaction.py tests/test_a2_candidate_vectorization.py
python -m pytest -q tests/test_a2_paper_mc.py tests/test_a2_paper_mc_jaxmd.py tests/test_a2_paper_rate_order.py
python -m pytest -q
```

**Next scientific gate:** inspect the authors' source implementation of energy accounting and obtain an RDF-based candidate cutoff for our selected density, rather than claiming either was specified numerically in the paper.
