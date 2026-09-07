# GLM eight-expert SSD cache — 7 September 2026

Base: b372ac4. Isolated experiment; existing quant and resident defaults unchanged.

**Outcome:** validated and promoted as an optional GLM SSD-streaming improvement.
At an 8 GB cache budget, matched 256-token runs improve from 1.00 to 12.20 t/s
mean decode. Single 16/32 GB samples reach 13.24/14.08 t/s. No quant conversion,
no resident-mode switch, no Qwen expert-streaming implementation, and no push.

## Implementation

Our GLM hybrid has IQ2_XXS routed gate/up and Q2_K down with eight selected
experts. Existing Metal selected-slot IQ2/Q2 support was restricted to six.
The change reuses the existing eight-expert IQ2 cached-address infrastructure,
adding a Q2_K address-table down kernel. That kernel calls the same per-expert
dot-product implementation as resident inference; the existing separate expert
reduction retains its order. Six-expert fused and masked paths are untouched.

The cache remains bounded and owns weight buffers. Existing in-flight resource
tracking protects entries while GPU commands consume their addresses. No model
conversion, quant changes, new downloads, CPU inference or power/fan changes.
Non-Apple support guards and CUDA/ROCm code are unchanged; no remote hardware
or distributed runs are claimed.

## Initial check

The 2K/64-token pilot matches all 64 resident full-logit files byte-for-byte.
Its 11.43 t/s includes logit writes and diagnostic profiling, so it is not the
clean timing result. Logs confirm the new static decode map and actual expert
cache activity rather than the old per-layer fallback.

## Validation plan and artifacts

`run.py parity`: resident reference vs 8/32 GB streaming at 2K, old streaming
reference at 8K, plus disabled-path control and DeepSeek streaming regression,
all full-logit hashes. `run.py aligned` checks resident equality at matched caps.
The initial `parity.out` preserves the failed cross-policy comparison; the
harness was subsequently corrected using the diagnosis described below.
`run.py timing`: no-dump ABBA streaming and resident controls, 256 fixed tokens;
additional 16/32 GB budgets. `run.py state`: streaming snapshot/rewind, MTP off/on.

All runs are serial on the M5 Max 128 GiB. Cache budgets include prefill reserves
and are not total application memory. The new global decode map adds resident
non-expert weights: at 8 GB budget the reported planned total grows from about
15.85 to 20.21 GiB. This tradeoff is explicit, not an equal-total-RAM comparison.
macOS file caching remains active; logical file reads are not physical cold-SSD
bandwidth. Streaming still aims to reduce memory use, not beat resident speed.

## Attention/work-limit diagnostic

The original 8K comparison at 10,240 allocated context failed resident-versus-
streaming logit equality. This is also present on unmodified b372ac4: old fallback
and new cached execution match all 64 full-logit files exactly. Existing policy
uses a 4K attention/work cap for resident mode and 8K for streaming at that
allocation. The disabled candidate reproduces the old fallback as well.

An additional comparison keeps the actual prompt at 8K but allocates 65,536,
where both existing policies choose 4K. The 8 GB cached run then matches all
64 resident files exactly. This experiment does not change that pre-existing
mode policy, and does not claim unconditional cross-mode equality.

## MTP integration correction

The first MTP streaming snapshot check failed while the old fallback passed.
The two-row verifier can finish with only its last layer mapped, then defer its
output projection to a separate batched head. That head originally assumed its
weights remained mapped. Both separate output-head helpers now restore the
runtime static set when necessary (including subsequent draft weights). This
is streaming-only; resident behavior is unchanged.

After the fix, ordinary snapshot and the 16-cycle MTP snapshot/context-reuse
test pass. MTP matches the old fallback's nine single / seven double cycles and
23 accepted tokens; context reuse has three single / one double cycles. The
generic rewind selector skips on GLM because it is Qwen-specific: do not count
that skip as an additional GLM rewind test. Snapshot testing covers restoration.

## Clean streaming timings

2K prompt, 4K allocation, 256 teacher-forced tokens, no logit dumps or MTP.
All values are measured on this machine, not promises across workloads/devices.

| Run | Budget | Prefill t/s | Decode t/s | Final 64 decode t/s |
|---|---:|---:|---:|---:|
| Old A | 8 GB | 425.54 | 1.00 | 1.00 |
| New B | 8 GB | 422.62 | 12.07 | 11.37 |
| New B | 8 GB | 422.40 | 12.33 | 11.41 |
| Old A | 8 GB | 300.25 | 1.00 | 1.00 |
| New, extra budget sample | 16 GB | 347.20 | 13.24 | 13.23 |
| New, extra budget sample | 32 GB | 381.53 | 14.08 | 15.41 |

Matched 8 GB mean decode rises from 1.00 to 12.20 t/s (12.2x); final-block
mean is 11.39 t/s. Larger-budget rows are single samples, not repeated ABBA.
Prefill timing is variable; this is a decode improvement, not a claimed prefill
win. The first old/new prefills closely match while the last old run is slower.

## Resident preservation checks

Initial resident ABBA, 2K prompt and 256 fixed tokens:

| Model / metric | Old mean | New mean |
|---|---:|---:|
| GLM complete decode | 24.165 | 23.515 |
| GLM final 64 decode | 23.510 | 23.455 |
| DeepSeek complete decode | 26.170 | 26.820 |
| DeepSeek final 64 decode | 26.120 | 26.715 |

All entries t/s. GLM sustained final-block difference is -0.23%; DeepSeek shows
no slowdown. Whole-run/prefill figures show run-order sensitivity, especially
the first run after a mode/model switch. A reverse-order BAAB prefill/decode
guard follows rather than treating all that variation as a kernel effect.
These are this session's rates, not directly comparable to earlier cool runs.
`pmset -g therm` reported no recorded warnings; it does not establish why
individual GPU rates varied, and no thermal-throttling diagnosis is claimed.

Reverse-order BAAB, 128 fixed tokens (new / old / old / new):

| Model / build | Prefill t/s | Decode t/s | Final 64 decode t/s |
|---|---:|---:|---:|
| GLM new | 380.54 | 24.22 | 24.01 |
| GLM old | 411.51 | 25.11 | 24.76 |
| GLM old | 434.17 | 25.33 | 24.89 |
| GLM new | 434.96 | 25.16 | 24.76 |
| DeepSeek new | 761.29 | 31.83 | 28.20 |
| DeepSeek old | 467.85 | 27.97 | 26.95 |
| DeepSeek old | 496.55 | 28.59 | 27.69 |
| DeepSeek new | 523.48 | 28.90 | 27.87 |

The large initial prefill differences are not reproduced in the final warm
GLM pair (434.17 vs 434.96). Do not call this a resident speedup or claim tight
confidence intervals from these few variable runs. Resident kernel arithmetic
and dispatch are unchanged; no meaningful sustained decode regression observed.

Full golden-model `make test` on final Apple-scoped source passed (exit 0).
Qwen kernel suite also passed. Optional unavailable tests retain their existing
skips. A new explicit live regression alternates another session's independent
prefill with a hot GLM session's decode: **16/16 full-logit arrays byte-identical**
to uninterrupted generation. It builds with `make tests/test_glm_stream_interleave`;
run with `make test-glm-stream-interleave DS4_TEST_MODEL=/absolute/model.gguf`.

Source work was saved in 830cf44 and ed2d1f9, with regression coverage b2363d1.
Focused promotion on main is **315a330, 15c2fe3, 293c03e**. Main builds all five
frontends and passes the live MTP snapshot/context-reuse and 16-step interleaved
session regression. Final main 2K streaming output matches **64/64 full-logit
files** from the pre-change resident reference. No model/default changes and
nothing pushed. No inference process remains running after final verification.

## Scope and optional use

This adds an efficient ordinary-decode SSD mode for the existing GLM hybrid;
it does not alter resident inference, quant files, Qwen's PLE setup or defaults.
MTP integration is correctness-tested separately; the fixed-token throughput
measurements do not use MTP or prompt lookup. GLM prompt lookup remains disabled
in SSD mode by its existing capability guard.

After promotion, optional CLI invocation from the repository:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 8GB --ctx 4096
```

Omit the two SSD options for the existing faster resident mode on this 128 GiB
machine. This is not a recommendation to switch the normal server to streaming.

For reruns after promotion, reconstruct b372ac4 as the baseline and change the
harness's `main` path to that baseline build. Do not compare a promoted main
binary to itself and label it old-versus-new.
