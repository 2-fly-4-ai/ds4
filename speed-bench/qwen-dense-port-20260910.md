# Dense Qwen focused port — 2026-09-10

## Scope

This branch is an isolated focused port on top of current `main`; production
`main` is unchanged.  It binds Qwen 3.8's bundled `blk.64.nextn.*` head,
restores deterministic API session state, and reuses the existing exact
multi-row target path for dense-Qwen prefill.  No DeepSeek or GLM dispatch is
changed.

Runtime escape hatches:

- `DS4_QWEN_NEXTN_DRAFT=0` disables the bundled NextN drafter.
- `DS4_QWEN_PREFILL_BATCH=0` restores one-token dense-Qwen prefill.
- `DS4_MTP_SPEC_DISABLE=1` disables neural speculation while retaining
  batched prefill.
- `DS4_QWEN_MTP_PREFILL=1` opt-in teacher-forces the bundled NextN attention
  cache from prompt hidden states.  It is intentionally not a default.
- `DS4_QWEN_MTP_PROFILE=1` prints aggregate draft acceptance at shutdown.

## Correctness

- 15/15 matched greedy API cells were byte-identical across scalar prefill,
  batched prefill, and batched prefill plus bundled NextN.
- Fixed-seed temperature-0.7 output was byte-identical across all three paths.
- An additional 896-token control was byte-identical between scalar and
  batched prefill.
- Two interleaved API sessions matched their corresponding sequential hashes.
- The Qwen session rewind/replay test passed on the 27B Q8 model.
- Builds, kernel checks, server tests, and Metal tensor-equivalence checks pass.
  The full suite reports exactly the three pre-existing Vision-Exp versus
  repository-golden mismatches: two `short_code_completion` vector checks and
  `long_story_4096` top-5 overlap 3/5.

## Measured M5 Max results

Model: `Qwen3.8-27B-Q8_0.gguf`; macOS Automatic.  Every configuration starts a
fresh server.  The order is scalar A, batch A, batch+NextN, batch B, scalar B,
so the averages include the observed thermal drift.

| Prompt tokens | Scalar prefill A/B | Batched prefill A/B | Mean prefill speedup |
| ---: | ---: | ---: | ---: |
| 22 | 1.360 / 1.462 s | 0.382 / 0.376 s | 3.75x |
| 349 | 20.778 / 26.208 s | 4.980 / 6.167 s | 4.22x |
| 1,005 | 63.895 / 72.221 s | 15.021 / 19.440 s | 3.96x |

The earlier independent 896-token control measured 57.508 s at 15.58 t/s
versus 12.547 s at 71.41 t/s: 4.58x faster, byte-identical.

| Prompt/output tokens | Scalar A/B wall | Batch A/B wall | Batch+NextN wall | Batch+NextN vs scalar mean |
| ---: | ---: | ---: | ---: | ---: |
| 22 / 48 | 4.330 / 5.437 s | 3.328 / 3.460 s | 1.805 s | 2.71x |
| 349 / 48 | 23.876 / 29.851 s | 8.159 / 9.567 s | 7.529 s | 3.57x |
| 1,005 / 32 | 66.170 / 74.731 s | 17.338 / 22.022 s | 17.947 s | 3.93x |

The combined run drafted 175 tokens and accepted 82 (46.9%).  Sampled decode
intentionally stayed on the exact plain path; batched prefill reduced its TTFT
from 1.478 s to 0.443–0.446 s without changing output.

## Reproduction

`speed-bench/qwen_dense_api_matrix.py BUILD MODEL OUTPUT_DIR` writes raw server
logs, response text, and `results.json`.  `QWEN_MATRIX_CASES` and
`QWEN_MATRIX_CONFIGS` accept comma-separated filters for quick focused reruns.

## Remaining limitations

- The dense-Qwen Metal pool is capped at 4,096 context tokens.  This focused
  port does not claim long-context support beyond that boundary.
- The target recurrent/KV pool is global.  Owner tracking plus exact replay
  makes interleaved sessions deterministic, but switching sessions has a
  performance cost; fully per-session target state remains future work.
- The NextN attention cache is reset for a fresh API prompt rather than being
  prefilled from every prompt hidden state by default.  The opt-in teacher-forced
  experiment raised acceptance from 48.4% to 64.8% in the 48-output-token pair,
  and from 48.8% to 57.1% in the 256-output-token matrix.  It improved ABBA mean
  wall time by 7% for 30-in/256-out coding and 14% for 347-in/256-out systems
  prose, but made the 349-in/48-out case about 5% slower.  Requested output caps
  do not predict when a real response will stop, so automatically enabling this
  would create a production regression risk.  A batched NextN-prefill kernel or
  a reliable per-request profitability signal is the remaining route to making
  it a safe default.
