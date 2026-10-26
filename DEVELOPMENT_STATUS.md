# Single development repository — core-engine separation milestone

This archive is the **only development repository**. The uploaded running
repository is a scientific reference, not an installation or merge target.
No files in the original running environment have been changed.

## Architecture

```text
reactive_md/
  engine/
    contracts.py          # ReactionContext/Outcome, handler and observer protocols
    nvt.py                # Nose-Hoover NVT stepping, force refresh, nlist retries
  md_driver.py            # Generic reaction/observer orchestration, stable API
  md_a2_driver.py         # Legacy import compatibility
  reactions/
    a2.py, a2_rate.py     # Unmodified reference A2 chemistry and kinetic algorithm
    a2_metropolis.py      # A2 fixed-coordinate MH reaction algorithm
    a2_handlers.py        # A2 strategy adapters
    a2_driver.py          # A2 compatibility runner and output observer
    lipf6_handler.py      # LiPF6 adapter
    lipf6_driver.py       # LiPF6 compatibility runner
    lipf6_output.py       # LiPF6 logging/trajectory observer
  forcefield.py           # Unmodified reference forcefield
  relaxation.py           # Unmodified reference relaxation integrator
scripts/
  run_a2_small.py         # Restored from running reference
  run_a2_reversible.py    # Restored from running reference
tools/
  audit_a2_trajectory.py  # Standalone CSV consistency/summary checker
validation/
  a2_100ns_reference.json # Observed reference statistics, not pass/fail tolerances
```

The generic MD driver has no A2/LiPF6 chemistry or output branches. Existing
`run_md_nvt`, `run_md_nvt_a2`, and `run_md_nvt_with_reactions` entry points
remain available. The NVT stepper is independently importable. Importing the
core contracts does not require JAX-MD.

## Scientific preservation

The following files are byte-identical to the supplied running-version archive:

- `reactive_md/reactions/a2.py`
- `reactive_md/reactions/a2_rate.py`
- `reactive_md/forcefield.py`
- `reactive_md/relaxation.py`

This milestone changes code organization, not the A2 kinetic reaction
probabilities, potential, or relaxation method. Thermostat reset after an
accepted kinetic reaction is preserved as an explicit legacy behavior.

## Tests run locally

- Python syntax compilation: passed.
- 11 focused tests: passed (generic driver, NVT recovery/force synchronization,
  CSV audit); these do not exercise actual JAX-MD integration.
- Original 100 ns trajectory + event CSV audit: passed; reported mean A2
  population 68.7539, potential energy -2206.5158 kcal/mol, temperature
  188.025 K for 10–100 ns, with 403 associations and 337 dissociations.
- Real JAX-MD numerical regression and full suite: **not run** (`jax_md`
  is not installed in the authoring environment).

## Using the CSV audit

```bash
python tools/audit_a2_trajectory.py path/to/a2_trajectory.csv \
  --events path/to/a2_events.csv --equilibration-ps 10000
```

It verifies event/trajectory consistency and returns descriptive statistics.
The JSON reference file records the observed 100 ns run, but does not enforce
exact reproduction of stochastic event counts or trajectories.

## Next scientific validation in a separate JAX-MD environment

1. Install this single repository into a **separate** test environment.
2. Run `python -m pytest -q tests/test_generic_md_driver.py tests/test_engine_stepper_contract.py tests/test_a2_plugin_adapter.py tests/test_lipf6_plugin_adapter.py`.
3. Run the JAX-MD A2 and LiPF6 integration tests and a short baseline-vs-refactor
   comparison using matching parameters and software versions.
4. Validate outputs, event bookkeeping, energies, temperature and RDFs.
5. Treat A2 MH equilibrium correctness and LiPF6's legacy Metropolis-labelled
   reaction procedure as separate scientific validation tasks.

The production 1024-particle runner and its online RDF accumulator were not
included in the original repository archive. The two restored small A2 scripts
are complete; the production runner is **not** claimed as restored/runnable.

## HPC pytest collection troubleshooting (2026-10-26)

The HPC traceback shows three imports of `scripts.run_a2_small` /
`scripts.run_a2_reversible` failing under the `pytest` console command even
though both scripts exist. `pyproject.toml` now sets pytest's `pythonpath` to
`.` so the repository root is available to the test runner. The equivalent
explicit command is `python -m pytest -q tests/` from the repository root.

Two further failing files, `tests/test_reaction.py` and
`tests/test_topology_opls.py`, **do not exist in this development archive**.
They were presumably retained in the local checkout after extracting over an
older tree. They reference missing legacy `Candidate` and
`discover_pf6_and_li` interfaces. Do not silently ignore these tests or
reintroduce old production APIs without reviewing their intended behavior.
Preserve them in a separate archive for later test migration, then test
a clean extraction of the current development archive.

The authoring environment lacks `jax_md`, so the HPC JAX-MD test suite
still needs to run on the user's machine.

## HPC full-suite follow-up (2026-10-26)

After moving the two legacy tests out of the suite, the user reported
**76 passed, 4 failed**. All four failures came from a stale test-only
`monkeypatch.setattr(a2_rate, "debug_forcefield_at_positions", ...)`
statement in `tests/test_a2_multiple_events.py`. The production
`a2_rate_reactive_cycle` no longer imports or calls that diagnostic helper.
The obsolete monkeypatch was removed; the test still mocks only the
Langevin relaxation and exercises real candidate selection, topology
updates, force-field rebuild, exclusion masks and pair translations.

This revision changes **one test file and this status note only**;
no production modules or physics were modified. The HPC full suite
needs to be rerun to verify the four test cases pass beyond setup.

## HPC full-suite runtime follow-up (2026-10-26, second run)

The next HPC run reported 5 failing tests after successful collection:
four A2 multiple-event geometry tests and one legacy dummy-MD-driver test.
The A2 tests expected 3.40 A after dissociation and no movement after
association, but the original, unchanged A2Reaction defaults to a 0.25 A
translation **per reacting atom** in both directions. From 2.65 A initial
separation, the correct pre-relaxation expectations are 3.15 A after
dissociation and 2.15 A after association. The test now checks those
values and verifies that rejected candidates leave their coordinates
unchanged. This corrects outdated assertions, not the production physics.

The old `tests/test_md_driver.py` was not included in the previous
development ZIP and was left in the user's checkout. Its fake neighbor
list returned `None` and its `allocate` method did not accept the
`extra_capacity` keyword required by the new overflow-safe integrator.
The current ZIP now includes a migrated test with a JAX-compatible
neighbor-list stand-in (`idx`, `did_buffer_overflow`, `allocate`, `update`).
The engine's overflow checks were not weakened to accommodate the stub.

Only test files and documentation changed in this revision. The original
A2 kinetic reaction, force field, and relaxation files remain byte-identical
to the running reference archive. The full suite and the migrated real
JAX-MD dummy-driver test require the HPC environment.
