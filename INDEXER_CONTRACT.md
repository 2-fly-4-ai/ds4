# Metal indexer contract correction — 2026-09-07

This is a correctness correction, not a throughput optimization.

## Cause and scope

Scalar Metal decode previously delayed indexer selection until more than
1,024 compressed entries existed. The model defines top-k as 512, and the
batch verifier already selects above 512. In the intervening range, scalar
decode attended to entries that the model's indexer would exclude.

The helper now uses the model-defined cutoff. Legacy larger overrides are
bounded by it. Smaller overrides still cannot select prematurely because the
caller also checks the model top-k. This affects DS4 scalar and shared-batch
Metal dispatch; SSD Metal uses the same contract. Non-Apple policy is left
unchanged pending hardware-specific validation. No GLM/Qwen kernels, model
files, quantization, cache format, or speculative defaults are changed.

The root cause predates this experiment (`9ba160ae`). Its earlier comment
incorrectly described the threshold as implementation-only.

## Validation

- Build succeeds for server, benchmark and targeted regression binaries.
- Threshold unit regression passes default/64/128/256/512/1024/2048/4096 and
  invalid configuration, checking every entry count through 4097.
- Live whole-model Metal and 8 GiB SSD streaming both pass 16 scalar steps
  after a 2052-token prompt. Every step verifies finite full logits, the
  corrected cutoff, and 512 distinct in-range selected entries while the
  compressed count is between 513 and 517. Both runs emit the same tokens;
  this smoke test does not claim cross-backend full-logit equivalence.
- Eight fixed-build API requests complete: coding/editing at 2K, DSpark off
  and on, each cold and cached. All four cold/cached output pairs match.
- Server parser/cache and agent unit tests pass.
- Existing session-snapshot and Metal short-prefill tests pass on the actual
  Vision-Exp model. GLM short-matmul parity passes all tested shapes and the
  prefix-pool regression passes 528 cases. These are targeted tests, not a
  claim that every optional model/golden-fixture suite was run.
- CUDA/ROCm/distributed hardware was not tested; non-Apple behavior is not
  changed by this patch.

## Cost, without output-length confounding

Same long saved editing prompt, 128 teacher-forced tokens per frontier,
old → corrected → corrected → old order, M5 Max, existing Vision-Exp late-Q4.
No speculation in these fixed-sequence measurements.

| Context | Old runs | Corrected runs | Mean change |
|---|---:|---:|---:|
| 2K | 39.54, 38.35 t/s | 36.05, 36.04 t/s | −7.4% |
| 8K | 35.29, 35.06 t/s | 35.08, 35.14 t/s | −0.2% |

The 2K slowdown is real: the old route skipped required sparse selection.
At 8K both already select, and observed rates are effectively unchanged.
Only two runs per arm were taken; no statistical confidence interval is claimed.
Prefill code is unchanged; sequential warm-up/thermal variation must not be
reported as a prefill gain/loss caused by this patch.

Cached real editing API: corrected fast path without DSpark **48.87 t/s**,
versus old **51.14 t/s**, same 178-token output. Corrected DSpark-enabled
editing is **37.51 t/s**, old **37.49 t/s**, also the same output. Coding
outputs change after the semantic correction, so their wall times are not
an identical-work speed comparison.

## Separate rejected handoff experiment

`experiment/lookup-handoff-20260906` contains diagnostic scalar-row paths
that make the tested cross-policy logits and state exact through 32K. Those
paths are too slow to promote. They are **not** part of this focused patch.
This patch does not claim universal byte-equivalence across all optimized
speculative batch schedules; ordinary scalar/batch arithmetic differences
remain a separate limitation. The unvalidated backoff optimization stays off.

Local raw evidence:
`speed-bench/lookup-handoff-20260906/indexer-*`, `final-canonical-*`,
`canonical-api-*`, and `CANONICAL-PROGRESS.md` in the main Desktop checkout.
