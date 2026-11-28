# Package B: MACE computational optimization and controlled kernel benchmark

## Scope and baseline

Baseline: `73b03d11ddfaeb01344f0982e6b60fdc2b6efb0f` (Package A profiling).
The supplied X64 MACE-MP `small-0b2` profile contains 320 MD chunks (320,000
regular steps), seven trial relaxations, four accepted-reaction synchronization
calls and one successful NVT overflow recovery. There were 9,287 timing spans,
zero orphan spans and no reported errors. The source default relaxation is 5 ps
at 0.00025 ps/step, consistent with 2,800 50-step relaxation chunk attempts
(140,000 trial steps). No relaxation parameter is shortened in this package.
The default temperature in this source is 400 K unless explicitly overridden.

| Operation | Inclusive measured time (s) | Share of measured root time |
|---|---:|---:|
| NVT integration | 29,715.387 | 66.47% |
| Trial relaxation | 12,678.461 | 28.36% |
| Pre-reaction observers | 2,089.408 | 4.67% |
| All other measured root work | 223.670 | 0.50% |
| Total | 44,706.926 | 100% |

NVT and Langevin integration kernels together dominate the profile. Their
inclusive timings include neighbor updates and integrator operations, so they
are not independent measurements of MACE force evaluation. Candidate search
(13.004 s) and checkpoint writing (0.140 s) are no longer priority bottlenecks.
Profiling is synchronized; absolute timings may differ with profiling disabled.

## Default optimizations

- The driver supplies the MACE neighbor list validated at the observation's
  current positions, together with its backend provenance. The LiPF6 energy
  observer reuses it only if the producer is the same neighbor function and the
  list has no overflow. Standalone observers, classical paths and mismatched or
  invalid lists retain the original fallback. No MACE list is placed on a
  classical force-field bundle. Reusing the list changes only energy reporting;
  summation roundoff can differ from a freshly allocated list.
- A simulation-owned, single-entry `LangevinKernelCache` retains the compiled
  initializer, force function and chunk kernel across MACE trial reactions.
  Backend identities, dtype, X64 status and integrator scalars determine reuse.
  Each trial supplies fresh coordinates, velocities, masses, neighbors and RNG.
  No trial force field or state is retained. Different shapes and neighbor-list
  static metadata may still cause compilation inside JAX. Classical relaxation
  continues constructing its own kernels.
- Dense MACE Langevin overflow recovery now uses approximately 10% capacity
  growth (at least eight slots), with an explicit minimum-width check. Failed
  trial chunks including RNG state are discarded before replay, as before.
  Classical and other list formats retain their previous recovery policy.

The existing MD force algorithm, MACE weights, physical parameters, PRNG
progression and checkpoint format remain unchanged on the default path. Kernel
reuse does not imply that JAX has only one compiled executable. The previous
profile attributes 492.612 s (about 1.1% of measured root time) to observer MACE
neighbor allocation; this is an opportunity, not a measured Package B speedup.
The 107.154 s of Langevin initialization and first-chunk compilation costs may
also be reduced. Neither change by itself is expected to halve overall runtime.

## Experimental energy-only adapter and benchmark

The local adapter originally calls the MACE model with `compute_stress=False`
and extracts energy; MD then differentiates that scalar externally. Upstream
JAX-MD's scalar MACE wrapper explicitly also requests `compute_force=False`:
<https://jax-md.readthedocs.io/en/main/_modules/jax_md/energy.html> (see
`mace_neighbor_list`). The optional energy-only adapter uses that same keyword.
It may avoid unnecessary internal work, but the installed MACE-JAX model must
support it, and XLA may already eliminate some unused outputs. No speedup or
numerical equivalence for actual model weights is claimed before measurement.
The original default call signature is preserved for older model adapters.

Run the two variants in **separate processes** on the same idle GPU and X64
input. The benchmark does not run MD or write simulation files. It compiles and
times the scalar energy and its negative gradient separately, with synchronization,
three warm-up calls and ten timed calls by default. It records compiler lowering
and compilation times, first execution, samples, medians, output dtypes and
compiler memory estimates when available. Compiler memory estimates are not
whole-process allocator peaks. It verifies model, device kind, geometry, species
and neighbor-index fingerprints before comparing energy and all force components.

```bash
BENCH_DIR=$(mktemp -d ./benchmark_mace_kernels.XXXXXX)
COMMON=(--data "$HOME/react_LiPF6/init_test/mixture_classical_after_300K_NPT.data"
        --settings "$HOME/react_LiPF6/init_test/mixture.in.settings"
        --source mp --variant small-0b2 --capacity-multiplier 1.25)
JAX_ENABLE_X64=True JAX_PLATFORMS=cuda \
python -u -m validation.benchmark_mace_kernels "${COMMON[@]}" \
    --mode legacy --output "$BENCH_DIR/legacy.json"
JAX_ENABLE_X64=True JAX_PLATFORMS=cuda \
python -u -m validation.benchmark_mace_kernels "${COMMON[@]}" \
    --mode energy-only --compare-to "$BENCH_DIR/legacy.json" \
    --output "$BENCH_DIR/energy_only.json"
```

An optional `--checkpoint /path/to/immutable-copy.pkl` in **both** commands
uses coordinates from a trusted local checkpoint (pickle is not safe for
untrusted inputs). Supply the same original data/settings to preserve atom
ordering, species and box. Never use a checkpoint actively being replaced by a
running process as the two benchmark inputs. Input hashes are checked during
comparison. The default tolerances are 1e-8 eV absolute for the total energy,
1e-8 eV/Å absolute and 1e-10 relative for each force component; a mismatch fails
rather than silently widening tolerances. Inspect numerical errors, timing and
memory results on several representative configurations before enabling the
experimental production option.

Only after successful actual-model comparisons, test `--mace-energy-only` in
**fresh, separate** short MD/MC simulations with matched seeds and parameters.
This flag applies to MACE-MD and MACE-MC if used. It does not change dtype,
weights or the outer force differentiation. It changes the backend identity and
checkpoint metadata explicitly, and cannot be switched on or off when restarting
an existing checkpoint. Keep it absent for production continuation from legacy
checkpoints. Compare trajectory, reaction energy and event results before any
production adoption. Incompatible model versions should fail visibly; there is
no silent retry with the legacy adapter. The stand-alone benchmark is not a
substitute for actual integration throughput (the outer integrator JIT can
optimize operations differently).

## Regression and profiling

```bash
JAX_ENABLE_X64=False JAX_PLATFORMS=cpu python -m pytest -q --strict-markers
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu python -m pytest -q \
    tests/engine/ tests/mace/ tests/apps/ tests/restart/ tests/lipf6/ tests/reactions/
```

The added contracts cover backend-cache identity/invalidation, different trial
seeds, X64 and float32 preservation, dense overflow rollback, valid-list reuse
and fallback, external differentiation with both adapter call conventions,
benchmark comparison errors, and the explicit experimental CLI mode. Existing
restart tests also check that switching to the experimental adapter rejects the
legacy checkpoint without changing any output files.

For default-path validation, resume the backed-up legacy checkpoint without
`--mace-energy-only` or run a fresh matched benchmark if already at the final
step. Keep `REACTIVE_MD_PERFORMANCE=1` and `REACTIVE_MD_NLIST_DIAGNOSTICS=1` for
comparable diagnostics. Analyze the new log with
`python validation/analyze_performance.py profile.log --output profile.json`.
Compare observer allocation counts, initialization/first-chunk times, actual
integration throughput, reaction statistics and GPU memory peaks. The default
observer allocation count should be zero for valid supplied MACE lists.

## Deliberately deferred changes

The exact MACE-JAX featurizer/model internals and JAX-MD implementations are
external dependencies, not included in this repository. Graph compaction,
custom force kernels, energy caching inside the integrator and global
neighbor-static-metadata replacement are not introduced without inspecting
those versions and establishing actual energy/force/trajectory equivalence.
The contributor's dynamic classical kernels are likewise not ported here:
the supplied profile is MACE-specific and the old contribution contains
forced float32 parameter conversion and topology padding needing a separate
classical baseline and gradient validation. No model, relaxation, timestep,
physical reaction parameters or precision are changed to obtain a speedup.
MD/MC performance and real GPU stability remain HPC validation tasks.
