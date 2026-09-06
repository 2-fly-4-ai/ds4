# M5 indexer barrier reduction — 2026-09-07

Baseline: corrected main `7418583`. These kernels preserve the model's top-512
selection; they do not restore the incorrect old 1024-entry dense shortcut.

## Changes

- **Score kernel:** compute the 64 independent head contributions, then add
  them in the original head order. This removes 31 threadgroup barriers:
  33 barriers become two, with unchanged dot products and accumulation order.
- **Short top-k:** retain the existing 1024-wide bitonic comparison network
  and its non-stable tie order. SIMD shuffles replace within-group shared
  memory exchanges. Cross-group exchanges retain both publication and
  consumption barriers. This applies only to one row, top-k 512, and
  513–1024 candidates on M5, with a pipeline thread-limit guard.

Both defaults are M5-only. Existing kernels remain available for other
hardware/shapes and diagnostic controls:
`DS4_METAL_DISABLE_INDEXER_STAGED_SCORE=1` and
`DS4_METAL_DISABLE_SHORT_TOPK_SHUFFLE=1`.
No CPU/CUDA/ROCm code, model files, quantization, cache format, top-k value,
sampling policy or speculative-routing defaults change.

## Measurements

The corrected stage profiler places a real command-buffer boundary before
scoring; its former null label did not split the GPU work and misleadingly
included preceding work in the score interval. The profiling-only source
change is not included in this release patch. Profiled GPU spans are not
unprofiled generation timing.

| Context | Score per indexed layer | Top-k | Indexed attention |
|---|---:|---:|---:|
| 2K | 11.62 µs | 18.61 µs | 47.25 µs |
| 4K | 18.70 µs | 24.83 µs | 47.59 µs |

Warm isolated measurements around 532–1024 candidates reduced short top-k
from roughly 18–21 µs to 10–11 µs, and reduced scoring by several µs.
The whole model does much more work, so its gain is substantially smaller.

Controlled whole-model scalar decode uses 128 identical tokens from the
same restored prompt frontier, four runs per arm in ABBA/BAAB order, no
speculation, no profiler or full-logit oracle readbacks inside the timing.

| Context | Corrected baseline | New kernels | Throughput gain |
|---|---:|---:|---:|
| 2K | 42.22 t/s | 42.53 t/s | +0.74% |
| 3K | 41.73 t/s | 42.11 t/s | +0.89% |
| 4K | 41.12 t/s | 41.30 t/s | +0.44% |
| 8K | 40.64 t/s | 40.85 t/s | +0.51% |

These are a small gain, not recovery of the prior 7.4% correction cost.
No statistical confidence interval or prefill improvement is claimed.
macOS power mode was read as Automatic; no power/fan settings were changed.
An independent repeat with the final default-enabled dispatch measured
**+0.78% at 2K and +0.95% at 3K**, again four observations per arm.

## Correctness and integration

- Kernel oracle compares every score bit and every selected index against
  the existing kernels, including all-zero/equal-score and padded-row cases.
  The final release oracle passes **96 cases**, covering 512, 513, 514, 532,
  767, 768, 769, 1023, 1024, 1025, 2048 and 8192 candidates.
- Whole-model oracle checks all 128 full-logit vectors and complete final
  snapshots on five 2K tasks (code, copy, edit, story, JSON), plus editing
  frontiers at 3K, 4K and 8K. All pass.
- Whole-model SSD streaming with an 8 GiB cache passes the same 128-step
  full-logit and final-snapshot comparison at 2K.
- Final default-enabled release checks repeat SSD, 2K and 3K and add **32K**
  editing: all 128 full-logit vectors and the 476,776,332-byte final snapshot
  match the baseline at 32K. GLM short-matmul parity and all 528 prefix-pool
  regression cases also pass.
- API coding/editing, DSpark off/on, cold/cached: all eight candidate requests
  match the eight corresponding baseline output hashes and token counts.
  Visible API hashes supplement, not replace, the full-logit/state oracle.
- This does not claim universal scalar-versus-speculative-route equivalence;
  it establishes candidate-versus-baseline equivalence for these kernel changes.
- Non-M5, CUDA/ROCm and distributed hardware were not tested. Their existing
  dispatch remains unchanged; larger-batch top-k also retains its old path.

The first API candidate run was invalid: the harness launched the candidate
binary from main's directory, loading old Metal sources without the new kernel.
It also incorrectly accepted error-terminated partial streams. Both harness
defects were fixed and all eight requests rerun under a fresh `-v2` tag.
The failed evidence is preserved and excluded from the results above.

Local evidence and sequential launchers:
`speed-bench/indexer-barriers-20260907/` in the main Desktop checkout.
`make indexer-kernel-test` runs the isolated numeric regression. Build
`tests/test_indexer_live_parity` for the optional model/SSD oracle; its
`HOT_TASK`, `HOT_CTX`, `HOT_SSD`, and `HOT_TIMING` inputs are supplied by
the saved `run.py` and `release_check.py` launchers.
