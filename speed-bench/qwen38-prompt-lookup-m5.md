# Qwen3.8 prompt lookup on M5 Max

Measured on an Apple M5 Max with 128 GiB RAM in Automatic power mode. The
model was `Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf`; generation was greedy
and all comparisons used identical prompts and output SHA-256 hashes.

The retained implementation verifies one anchor plus up to seven free drafts
from the session's own token history. It uses Qwen's native recurrent-state
snapshot after the anchor, commits full matches directly, and restores then
replays the agreeing prefix after a partial match. No draft model or extra
model file is required. Embedded MTP remains the fallback when enabled and no
prompt-derived continuation exists.

## Results

| Workload | Control | Prompt lookup | Change | Draft acceptance |
|---|---:|---:|---:|---:|
| Repeated passage, plain fallback | 48.11 t/s | 117.86 t/s | **+145.0% / 2.45x** | 439/439 (100%) |
| Repeated passage, MTP available | 63.93-68.88 t/s | 114.71 t/s | **+66.5% to +79.4%** | 439/439 (100%) |
| Code edit with partial rollback | 46.75 t/s | 87.69 t/s | **+87.6%** | 132/140 (94.3%) |
| Novel technical response, MTP | 57.84 t/s | 58.32 t/s | **+0.8%** | no lookup pass |
| OpenAI-compatible API code edit | n/a | 91.21 t/s | production-path check | 127/133 (95.5%) |
| API repeated passage at 13,040-token prompt | n/a | 112.60 t/s | long-context check | 439/439 (100%) |

The repeated-passage absolute rates moved with thermal state. Matched hashes,
route counters, and direction of the gain were stable. The novel-text A/B was
repeated after warming the CPU-mapped PLE rows; the initial cold-side result
was discarded rather than attributed to prompt lookup. The 13K API prompt
prefilled at 647.87 t/s and completed 512 output tokens without a rollback or
state error.

## Depth sweep

The same 512-token repeated continuation produced identical output at every
depth:

| Draft depth | Throughput | Verifier passes | Tokens committed/pass |
|---:|---:|---:|---:|
| 1 | 74.83 t/s | 242 | 1.00 |
| 3 | 96.80 t/s | 121 | 3.00 |
| 7 | **117.86 t/s** | 63 | 6.97 |

Seven is the default and the current native verifier ceiling. Use
`DS4_PROMPT_LOOKUP_MAX` for controlled A/B tests and
`DS4_PROMPT_LOOKUP_DISABLE=1` for complete rollback.

## Correctness and regression checks

- Repeated copy and edited-code outputs were byte-identical to prompt lookup
  disabled.
- The edit test exercised two partial verifier blocks and restored the exact
  expected C output.
- Qwen Metal kernel tests passed.
- DeepSeek and GLM real-model smoke outputs matched current `main` byte for
  byte; decode was 47.23 versus 47.37 t/s for DeepSeek and 39.46 versus
  39.41 t/s for GLM.
- The broad `ds4_test` run passed session snapshot/rewind, long context,
  tool-call recovery, Metal kernels, and tensor equivalence. Its three known
  golden-vector failures remain because the local `ds4flash.gguf` points to
  the newer Vision-Exp model rather than the repository's golden fixture.
