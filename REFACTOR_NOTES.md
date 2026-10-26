# Reaction-plugin refactor: LiPF6 plugin migration

## What changed

- `reactive_md/md_driver.py` is now a generic NVT engine. It contains the
  `ReactionHandler` protocol, `ReactionContext` / `ReactionOutcome`, observer
  interface, `run_md_nvt`, and reusable `NVTStepper`.
- The `NVTStepper` retries entire overflowing chunks and validates the final
  coordinates. It refreshes the cached force whenever topology changes.
- `reactive_md/reactions/a2_driver.py` preserves the existing A2-specific
  sampling and LAMMPS/RDF outputs, but delegates integration to `NVTStepper`.
- `reactive_md/reactions/lipf6_driver.py` is now a small compatibility wrapper
  that delegates all integration and reaction dispatch to `run_md_nvt`.
- `reactive_md/reactions/lipf6_output.py` owns the existing LiPF6 CSV formats,
  diagnostics, and trajectory dumps, using the optional `on_before_reaction`
  observer hook and the standard post-check observation hook.
- `reactive_md/md_a2_driver.py` remains a backwards-compatible import wrapper.
- `reactive_md/main.py` uses the LiPF6 adapter directly.
- `reactive_md/reactions/a2_handlers.py` supplies `A2RateHandler` and
  `A2MetropolisHandler` for the generic `run_md_nvt` interface.
- `reactive_md/reactions/lipf6_handler.py` adapts the legacy LiPF6 callback.
- `reactive_md/observers.py` provides generic energy CSV and event JSONL
  observers. Chemistry-specific observers can be added independently.
- `tests/test_generic_md_driver.py` exercises generic reaction dispatch and
  observation boundaries with a fake integrator.

## Compatibility and validation

- Legacy imports `reactive_md.md_driver.run_md_nvt_a2` and
  `reactive_md.md_driver.run_md_nvt_with_reactions` are available lazily.
- The existing `run_md_nvt_a2` and LiPF6 runner signatures are retained.
- `python -m compileall -q reactive_md`: PASS.
- Nine focused plugin and observer tests using a fake integrator: PASS.
- Real JAX-MD LiPF6 and A2 numerical regression: NOT RUN.
- Full `pytest` and numerical JAX-MD validation: NOT RUN. `jax_md` is not
  installed in this environment, and the provided archive lacks `scripts/`
  referenced by existing tests.

## Scientific cautions

- The existing LiPF6 Metropolis-labelled pathway uses an asymmetric candidate
  choice, sigmoid gate, coordinate preparation and FIRE relaxation. It should
  **not** be interpreted as a rigorously detailed-balanced equilibrium MC
  method. This refactor does not change that scientific algorithm.
- The A2 Metropolis handler uses the existing fixed-coordinate MH method;
  its efficiency may be poor at the selected LJ and bond-length scales.
- The LiPF6 runner now checks neighbor-list overflow and refreshes the cached
  force after topology changes. This intentional correctness change can alter
  numerical trajectories relative to older versions.
- Both A2 and LiPF6 compatibility runners now delegate to the same generic
  engine. Their chemistry-specific output is isolated in observers.
- A2 and LiPF6 kinetic chemistry has not been modified in this milestone.

## Suggested verification on the target JAX-MD environment

1. Install the repository and its test dependencies in a separate environment.
2. Restore the `scripts/` directory used by the original A2 tests.
3. Run `python -m pytest -q tests/test_generic_md_driver.py tests/test_a2_plugin_adapter.py tests/test_lipf6_plugin_adapter.py`.
4. Run `python -m pytest -q tests/test_md_a2_driver.py tests/test_a2_rate.py tests/test_a2_reaction.py`.
5. Run the full test suite, then compare a short kinetic A2 trajectory,
   event counts, population history, and RDF outputs with the baseline.
6. Only after validation, run short A2 MC and LiPF6 checks.
7. Do not change files or the Python environment of the ongoing 100 ns job.

## Core-engine separation (subsequent milestone)

- Moved public handler/observer contracts to `reactive_md/engine/contracts.py`.
- Moved the existing NVT integrator to `reactive_md/engine/nvt.py` without
  changing its numerical algorithm; the JAX-MD import is now deferred until
  the integrator is initialized.
- Kept `reactive_md/md_driver.py` as the stable generic orchestration facade,
  re-exporting the previous public names and preserving lazy legacy imports.
- Restored the original small/reversible A2 demonstration scripts from the
  running reference archive; did not recreate missing production RDF code.
- Added a dependency-free A2 trajectory/event CSV auditor and the observed
  100 ns reference statistics.
- Corrected the `pyproject.toml` development dependency group so `.[dev]`
  works as documented.
- Added focused tests for overflow retry, nonfinite chunk rejection,
  thermostat-chain preservation, and CSV/event consistency.
