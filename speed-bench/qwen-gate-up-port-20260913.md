# Focused Qwen gate/up fusion — 2026-09-13

Parent main: `e0f811e`. Production candidate contains four source/build/test files only, plus this evidence. It does not merge the profiling branch or the failed wider-Q8-output-tile experiment. No model or quant changes; no power/fan changes.

## Why this target

The two-token verifier MoE-stage diagnostic showed roughly 51% gate/up, 33–34% down, 10% router and 5% reduction. These are instrumented command-buffer GPU spans, not exact fractions of unperturbed request latency. Gate/up and down include the shared expert slot.

The new Q4_K kernel computes gate and up together, reusing each activation load. Each accumulator retains the existing K traversal, dequantization expressions, SIMD reduction, and SiLU multiplication. The original kernel is retained unchanged. A separate Metal pipeline avoids adding the experimental runtime branch to the existing kernel.

Selection: Qwen's resident MoE-mid wrapper, Q4_K expert weights, exactly two input tokens (principally MTP verification, also a two-token small prefill). No extra enable flag is needed. `DS4_QWEN_GU_DISABLE=1` is an independent baseline diagnostic. Shared-expert row-dot arithmetic is unchanged. Other token widths retain the existing kernel. SSD uses its existing direct kernel dispatch; CUDA/ROCm and GLM/DeepSeek paths are unchanged. Kernel argument layout is unchanged.

## Final clean-candidate comparison

M5 Max 128 GiB, installed Qwen3.8 Flash Next Q4_K/MXFP4, external PLE. Same copy/LRU-code/lighthouse-story prompts used in the preceding reports; actual input tokens 279/50/51, output caps 512/256/256, allocated context 4096, temperature 0, no thinking, MTP enabled in both arms, NAX prefill flag enabled in both. One process at a time. Main/candidate/candidate/main order for each task.

| Workload | Main t/s runs | Candidate t/s runs | Main mean | Candidate mean | Change |
|---|---|---|---:|---:|---:|
| Fresh code | 56.16, 55.33 | 56.64, 56.24 | 55.75 | 56.44 | +1.2% |
| Original story | 47.80, 47.67 | 49.37, 49.08 | 47.74 | 49.23 | +3.1% |
| Copied passage | 99.76, 86.52 | 95.64, 91.01 | 93.14 | 93.33 | +0.2% |

All 12 outputs match byte-for-byte within their task and match previous baseline hashes. These are generation speeds, excluding load and prefill. Copying is unchanged within substantial variation; do not claim a new copy gain. Coding improvement is small and especially sensitive to noise. Earlier independent separate-kernel comparisons showed +1.7% code and +2.8% story; together these support a modest improvement, not a universal percentage.

The first runtime-branch implementation looked +3–4% faster against its own baseline but was inconclusive against main and was not promoted. The separate-kernel version reduced instrumented gate/up spans by approximately 12–18%; profiling perturbs dispatch, and other stages also varied, so those figures are diagnostic rather than claimed application speedups. Exploratory code/evidence is preserved separately at `0937653` on `experiment/qwen-mtp-cost-20260913`.

## Validation on this clean candidate

- All default Metal binaries built successfully.
- `make test-qwen4-kernels` passed the existing suite, 63 exact Q8 comparisons, and 48 gate/up intermediate-output byte comparisons (12 directly exercise fused T=2; others are unchanged-width controls). Shapes up to E=2560/F=640, tail rows, 3/10 expert slots, shared/no-shared cases, and repeated expert IDs are covered. The make target now runs the gate/up checks automatically.
- Copy/code at 2048 and 8192 actual prompt tokens: all four full-model exactness tests passed, including target logits, target cache state, token stream, rewind, and restored eight-token continuation. Copy generates 256 tokens; code 128.
- Temperature 0.7 coding at 2048 input tokens, 64 generated tokens: same exact target/logit/cache/rewind/continuation checks passed. This is bounded sampled-path coverage, not a universal sampling proof.
- GLM verifier kernel regressions: `SHORT_MATMUL_COMPLETE failures=0`, `POOL_COMPLETE cases=528 failures=0`.
- Supported IQ2_XXS/MXFP4 Qwen SSD smoke: output matched main, using 4 GiB expert cache, context 2048, 36 input tokens/32 output tokens and MTP enabled. This checks the untouched SSD dispatch; Q4_K SSD support was not added. Warm OS-cache timings are not an SSD speed claim.
- `git diff --check` passed.

The exactness harness permits disposable neural-proposal-state differences, never target-logit/state drift. Full golden-model suite, API transport benchmarks, very long contexts, and CUDA/distributed hardware were not rerun or certified. This change is Metal-only and does not alter non-Metal source or interfaces.

## Reproduction and decision

Use the existing `tests/run_qwen_q4k_wide_oracle.sh`, `tests/run_qwen_q8_main_comparison.sh`, and `tests/run_qwen_q8_stream_smoke.sh`. Main-comparison script requires an old-main checkout for a valid future A/B. Resident model and PLE paths are documented in `qwen-q8-port-20260913.md`; they were not modified. Raw final logs/outputs are in the adjacent directory.

Retain this small, exact Qwen-specific gain. It adds to the prior Q8 verifier work, not a replacement for it. MTP still uses its existing enable setting; no router policy changes are included.

## Main promotion

Focused commit `d6d0594` was fast-forwarded onto main and all default Metal binaries rebuilt successfully. A post-merge 256-token lighthouse-story smoke matched the candidate output byte-for-byte. It reported 62.55 generation t/s, with 107/149 drafts accepted (71.8%); this single run is a correctness/deployment smoke, not a replacement for the paired comparison above or evidence of an additional speed gain. Build and smoke artifacts are archived alongside the candidate evidence. No server was started.
