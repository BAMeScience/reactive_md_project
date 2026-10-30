# rs@md NVT / PEL I — integration audit, phase 2

**Scope:** Biedermann et al., Figure 3a (rs@md[MC]) and Figure 3b (rs@md[rate]); PEL I, NVT at explicit density. PEL II and NPT remain postponed.

## Changes in this phase

1. Added `tests/test_a2_paper_mc_jaxmd.py`: **real JAX-MD** checks for topology-dependent energy offsets, association/dissociation energy round-trip, accepted relaxed-state handoff, rejected-state rollback, and NVT engine synchronization/cached-force consistency. Integration tests use **two relaxation steps** for speed; this is test-only, not a change to the published 1 ps production relaxation.
2. Added `tests/test_a2_paper_rate_order.py`: deterministic Figure 3b checks that all initial candidates are randomized, overlapping candidates are skipped after an atom reacts, disjoint events can occur in one reactive cycle, topology is built once, relaxation runs once after all accepted events, and zero accepted events leave state untouched.
3. In `reactive_md/reactions/a2_rate.py`, made non-finite rate inputs fail explicitly and reject a relaxed neighbor list that reports overflow, instead of committing an invalid state.
4. Corrected outdated fixed-coordinate-MH descriptions in the generic NVT engine contracts/docstrings. No MC acceptance rule was changed.

## What was actually executed here

- **7/7 deterministic Figure 3b tests passed** with dependency stubs for the missing `jax_md` package.
- Python syntax compilation of changed files passed.
- **Real JAX-MD tests were NOT run here**, because this environment does not contain `jax_md`. Run them in the project's `reactive_jaxmd` environment. Do not infer their success from the stub tests.

## HPC validation commands

From the updated repository root:

```bash
python -m pytest -q tests/test_a2_paper_rate_order.py
python -m pytest -q tests/test_a2_paper_mc_jaxmd.py
python -m pytest -q tests/test_a2_paper_mc.py tests/test_a2_rate.py
python -m pytest -q
```

The new real-JAX-MD suite may expose integration bugs that the dependency-stub tests could not detect. If it fails, **investigate the production code and the test assumptions against the paper**; do not change the published algorithm merely to make the test pass.

## Still unresolved before production runs

- The paper's precise MC local-energy bookkeeping and correction term `V_corr` remain unverified against the authors' source/SI. The code currently uses the full-system potential difference including the topology-dependent A2 offset. This is an explicit **approximation/interpretation**, not a verified author-code match.
- The association-candidate distance cutoff remains provisional; it must be justified separately from the rate-measurement distance thresholds.
- The numerical mass/thermostat calibration for quantitative dynamics remains open.
- The rate inputs are externally supplied, not derived from PEL II.
- NVT density is user-specified; no claim is made of numerical reproduction of the paper's NPT results.

**Next gate:** real-JAX-MD tests must pass, then resolve MC energy bookkeeping and candidate cutoff before long simulations.
