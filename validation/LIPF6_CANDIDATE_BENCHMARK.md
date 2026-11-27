# LiPF6 forward candidate-search performance validation

## Scope and attribution

Phase 7.3 ports the batched displacement idea from the contributor’s `dev_imp` archive into the current forward candidate search. The port retains scalar NumPy norms, stable ordering and a fallback for non-vectorizable displacement callbacks. The contributor’s name was not supplied; confirm attribution before publication. This benchmark does not assess MD integration, reverse search, occupied-Li filtering, MACE energy evaluation or relaxation.

## Method

- 5,264 atoms, 33 PF6 units and 33 Li ions; cubic periodic box of side 38.96383266137923 Å.
- JAX 0.10.1, float64 coordinates, CPU (`cpu:0`) and GPU (`cuda:0`); exact hardware models were not recorded.
- Actual initial input coordinates, with synthetic masks of 0, 5 and 15 reacted PF6 units; no topology changes or MD.
- Scalar reference first, three warm-up calls and 20 timed calls per implementation and case; alternating timed order.
- Candidate count, identities and ranking checked exactly; finite geometries compared with absolute tolerance 1e-12 Å and zero relative tolerance.
- Both searches return host-side scalar distances, so timing includes transfers necessary to return results.

The installed search function SHA-256 reported by both runs is:

`3104ea6531c445659405b62e51315a1ae096db37bc49146ca47381d4d1d64e46`

## Steady-state results

| Backend | Masked PF6 | Candidates | Scalar median (s) | Vectorized median (ms) | Median speedup | Scalar mean ± SD (s) | Vectorized mean ± SD (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0 | 6534 | 5.0034 | 27.607 | 181.23x | 5.0052 ± 0.0103 | 33.393 ± 26.459 |
| CPU | 5 | 5544 | 4.2644 | 24.904 | 171.23x | 4.2656 ± 0.0079 | 29.503 ± 21.315 |
| CPU | 15 | 3564 | 2.7420 | 18.300 | 149.84x | 2.7422 ± 0.0069 | 18.245 ± 0.635 |
| GPU | 0 | 6534 | 27.4293 | 34.797 | 788.28x | 27.4231 ± 0.0462 | 40.900 ± 25.630 |
| GPU | 5 | 5544 | 23.2716 | 31.320 | 743.03x | 23.2704 ± 0.0218 | 31.488 ± 0.853 |
| GPU | 15 | 3564 | 14.9789 | 24.855 | 602.66x | 14.9750 ± 0.0170 | 24.689 ± 0.717 |

## First-call timings

| Backend | Masked PF6 | Scalar (s) | Vectorized (s) |
|---|---:|---:|---:|
| CPU | 0 | 5.1474 | 0.7004 |
| CPU | 5 | 4.2619 | 0.6804 |
| CPU | 15 | 2.7421 | 0.7097 |
| GPU | 0 | 27.6305 | 0.8476 |
| GPU | 5 | 23.2210 | 0.8419 |
| GPU | 15 | 14.8823 | 0.9205 |

First-call timings include execution and compilation overhead; they do not isolate compilation time, and caches may be shared. The vectorized search has occasional timing outliers; medians are the primary comparison.

## Numerical results and limitations

All six cases report exact candidate identity and ranking agreement on every timed call. First-call maximum absolute differences in Li–F distance, P–F distance and sigma are zero in every case. Each timed call passed the 1e-12 Å absolute comparison; maxima across timed calls are not recorded. These results do not establish bitwise trajectory equivalence for arbitrary geometries.

GPU median speedup ranges from 602.66x to 788.28x; CPU from 149.84x to 181.23x. At 500 checks, holding the candidate count fixed at one of the tested levels, the GPU median timings imply approximately 2.08–3.80 hours saved in search alone, excluding compilation. This is an illustrative estimate, not an end-to-end measured speedup. No conclusion about long-duration GPU memory stability follows from this benchmark.

## Reproduction

Run from the repository root in the installed project environment, on an idle device. Set DATA and SETTINGS to the original input files (not distributed here). Use fresh output paths.

```bash
DATA="$HOME/react_LiPF6/init_test/mixture_classical_after_300K_NPT.data"
SETTINGS="$HOME/react_LiPF6/init_test/mixture.in.settings"
BENCH_DIR=$(mktemp -d ./benchmark_candidates.XXXXXX)
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu \
python -u validation/benchmark_lipf6_candidates.py \
    --data "$DATA" --settings "$SETTINGS" --output "$BENCH_DIR/cpu.json"
JAX_ENABLE_X64=True JAX_PLATFORMS=cuda \
python -u validation/benchmark_lipf6_candidates.py \
    --data "$DATA" --settings "$SETTINGS" --output "$BENCH_DIR/gpu.json"
```

Raw results: [CPU](lipf6_candidate_benchmark/cpu.json), [GPU](lipf6_candidate_benchmark/gpu.json). The supplied JSON is preserved unchanged, including absolute local paths. Exact GPU/CPU models, dependency versions beyond JAX, input hashes and Git commit were not captured; source fingerprint is available.

## Conclusion

Retain the vectorized forward search. Further search optimization is lower priority than profiling MD and relaxation and validating conservative MACE overflow recovery.
