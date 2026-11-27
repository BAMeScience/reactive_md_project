# Performance profiling

## Scope

Package A (Phase 7.2) provides opt-in host-side wall-clock instrumentation for the
common NVT driver and the LiPF6 reaction, relaxation, MACE energy and output
paths. The generic driver also reports coarse-grained operations for other
chemistries using that driver; detailed A2-specific instrumentation is outside
this package. It does not alter force fields, model weights, precision, random
number draws, reaction probabilities, relaxation settings, neighbor-list growth
policies, checkpoint formats or scientific output schemas.

The baseline for this package is development snapshot `c29f4b4`. The instrumentation
supersedes the earlier, driver-only timing patch; do not apply both patches.
No performance results are asserted until actual measurements have been collected.

## Enable profiling

Set `REACTIVE_MD_PERFORMANCE=1` before starting an ordinary simulation or restart.
Unset it, or set it to `0`, for ordinary uninstrumented execution. The existing
MACE NVT capacity and GPU allocator diagnostics can be enabled independently:

```bash
export REACTIVE_MD_PERFORMANCE=1
export REACTIVE_MD_NLIST_DIAGNOSTICS=1
export JAX_ENABLE_X64=True
```

Run the usual `python -u -m reactive_md.main lipf6` command with your existing
physical parameters. Capture stdout and stderr in a new terminal log. For example,
when using a Bash array `ARGS` containing your complete simulation or restart
arguments:

```bash
set -o pipefail
time python -u -m reactive_md.main lipf6 "${ARGS[@]}" 2>&1 | tee profile.log
```

Use a fresh process for each benchmark, an unoccupied GPU, and an aligned,
backed-up checkpoint if restarting. Keep the model, precision, integration
settings and other physical parameters fixed. A saved target step is an absolute
step, not the number of additional steps to run. Do not overwrite a running
simulation's output files.

Each timing line begins with `[REACTIVE_MD PERF] ` followed by a JSON object.
The fields are:

- `run_id`: process-local identifier, distinct for separately started Python
  processes; `span_id`: unique span number within that process.
- `parent_id`: immediately enclosing span number, or null for a top-level span.
- `step`: absolute MD step, inherited by nested stages; null when no MD step
  is known (e.g. initialization/restoration or isolated direct calls).
- `stage`, `elapsed_s`, `status`: operation name, inclusive wall time and whether
  the operation returned normally (`ok`) or raised an exception (`error`). An
  ordinary rejection or recoverable overflow is not an exception.
- `chunk_steps`, `steps_per_s`: accepted chunk work and throughput for the
  `md_chunk` and `langevin_chunk` stages. Rates include retry work and compilation
  overhead. Attempt records may have `chunk_steps` but are never counted as
  completed integration steps.
- Stage-specific fields such as `attempt` (zero based), `extra_capacity`,
  `relaxation_step` (local Langevin step) and `requested` (whether dump output
  was configured). Retry/allocation records describe attempted operations, not
  proof that the allocation or attempted chunk was accepted.

Records are written on scope exit, so child records normally precede their
parent records. The timer waits on relevant returned JAX array leaves only when
enabled. It does not transfer them to the host just to measure time. Operations
that already synchronize retain their existing synchronization. No new waits or
performance lines are added when the instrumentation is disabled.

## Stage reference

| Stage | Scope |
| --- | --- |
| `md_initialize_or_restore` | Driver initialization or checkpoint restoration, including neighbor reconstruction; excludes pre-driver application setup and model loading |
| `initial_observers` | Initial observation for a fresh simulation |
| `md_chunk` | Entire NVT integration chunk, including failed attempts, recovery, and validation |
| `nvt_chunk_attempt` | A single compiled integration call, including any first-use compilation |
| `nvt_overflow_recovery` | Reallocation and force refresh before replaying a failed chunk |
| `nvt_neighbor_allocate` | One NVT list allocation and overflow check |
| `classical_neighbor_list` | Classical reaction neighbor-list refresh in MACE-MD |
| `before_reaction_observers` | Pre-reaction observer callbacks |
| `observer_unwrap` | Coordinate unwrapping for LiPF6 output |
| `observer_potential_energy`, `observer_kinetic_energy` | Energy reporting |
| `observer_mace_neighbor_allocate` | MACE reporting neighbor-list allocation/update |
| `trajectory_dump` | Trajectory frame output (may be a no-op if no dump is configured) |
| `reaction_attempt` | Whole reaction handler invocation, including rejected proposals |
| `candidate_search` | Forward/reverse candidate search, filtering and record preparation |
| `reaction_energy_before`, `reaction_energy_after` | Initial and relaxed Metropolis energies (classical or MACE) |
| `mace_mc_neighbor_allocate`, `mace_mc_energy_kernel` | MACE MC list allocation/update and compiled scalar energy call |
| `trial_topology`, `trial_forcefield`, `trial_probe` | Trial topology construction, force-field build and trial geometry placement |
| `trial_relaxation` | Entire LiPF6 trial Langevin relaxation, including rejected MC proposals that reached this stage |
| `langevin_initialize` | Relaxation list allocation and initial Langevin state |
| `langevin_chunk` | Complete relaxation chunk, including retry work and checks |
| `langevin_chunk_attempt` | A single compiled relaxation integration call |
| `langevin_overflow_recovery`, `langevin_neighbor_allocate` | Recovery/force refresh and individual allocation attempts |
| `reaction_sync` | Commit accepted reaction to active MD state and refresh forces/list |
| `after_reaction_observers` | Post-reaction callbacks, including event/candidate logging |
| `checkpoint` | Checkpoint serialization and writing, including observer checkpoint state |

All stages are measured on the host, outside JIT-transformed function bodies.

## Analyze logs

The analysis script needs only Python's standard library. Give it full terminal
logs or extracted JSONL files from this consolidated package:

```bash
python validation/analyze_performance.py profile.log --output profile_summary.json
```

Optional extraction (not necessary for the analyzer):

```bash
grep '^\[REACTIVE_MD PERF\]' profile.log \
    | sed 's/^\[REACTIVE_MD PERF\] //' > performance.jsonl
```

Multiple non-overlapping logs can be combined:

```bash
python validation/analyze_performance.py profile_part1.log profile_part2.log \
    --output combined_summary.json
```

Do not supply the full log and its extracted JSONL together: duplicate spans are
rejected. Logs produced by the earlier, driver-only patch do not have the nested
span identifiers expected here.

The analyzer reports counts, errors, inclusive and exclusive wall time, median
successful-call time and aggregate integration throughput. The JSON summary also
includes first and maximum successful-call times. It subtracts each recorded
span's immediate children when computing exclusive time. **Do not sum inclusive
stage totals across levels.** For example, `trial_relaxation` is already inside
`reaction_attempt`, and `langevin_chunk` is inside `trial_relaxation`.
`root_measured_s` sums completed root spans (including roots that exited with an
exception), not total elapsed program time. `orphan_spans` identifies spans whose
parent is missing, such as when a job is terminated before all scopes exit. A
partial log can undercount measured time. Malformed JSON is reported rather than
silently discarded.

Successful integration throughput is computed as the sum of accepted chunk
steps divided by the sum of successful whole-chunk times; internal attempt records
are not counted. Regular MD and trial relaxation rates are reported separately.
Do not interpret their sum as physical simulation time: rejected MC trials are
included in computational work but not in committed relaxation time.

## Interpreting results and limitations

- First-use timings include compilation, dispatch and execution; these are not
  independent XLA compilation times or GPU kernel profiler measurements. For
  targeted investigation, use a separate run with `JAX_LOG_COMPILES=1` and, if
  necessary, JAX/XLA profiling tools. Compilation logs may be voluminous.
- Additional synchronization and logging affect measured execution time. Use
  profiling to locate bottlenecks and repeat uninstrumented runs for unbiased
  end-to-end performance comparisons. Short stages can be dominated by timing
  and I/O overhead; the nested logging overhead may appear in parent exclusive
  time. Run first-use and steady-state analyses separately when needed.
- Some host-side work is not assigned an individual span, and root measured time
  excludes application parsing/model loading and work outside driver scopes.
  Observer and force-field timing is only as precise as the synchronization on
  available result arrays. Output timing includes actual writes, not necessarily
  durable storage latency beyond the application's existing flush behavior.
- GPU memory statistics and neighbor dimensions are provided by the existing
  `[MACE NLIST]` diagnostics, not by the timing analyzer. Correlate the logs to
  distinguish overflow-driven shape changes from compilation or allocation costs.
- Profiling does not fix OOM failures or accelerate integration kernels. Those
  remain separate optimization and validation tasks. The persistent observer JIT,
  vectorized candidate search and conservative NVT overflow recovery are retained.

## Validation

The package adds tests for disabled/enabled timing, nested context and exception
handling, driver result/PRNG/checkpoint-schedule equivalence, MC acceptance/gate
rejection, rate-mode batching, Langevin rollback and MACE surrogate energy and
observer timing, plus analysis of nested, multi-process and incomplete logs.
The project's normal regression suite remains the authoritative validation.
Use both the usual CPU suite and explicit X64 targeted tests before committing.
Actual MACE GPU measurements and long-run memory stability must be assessed on
hardware with the project's installed JAX, JAX-MD and MACE dependencies.
