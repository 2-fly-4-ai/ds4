# M5 FP32 small-Qwen attention optimization

Baseline: main `13517e9`. Implementation merged to main as `1093763`.
Hardware: M5 Max 128 GB, existing Automatic power
settings. No quant, model weights, KV precision, sampling policy, or MTP depth
was changed. The production change is confined to the small-Qwen Metal
attention wrappers used by 27B/35B target and native-MTP execution.

## What won

The old attention scan uses two threadgroup barriers for every cached position.
The new kernel assigns one position to each of eight SIMD groups, computes
eight scores, and amortizes the barriers across those positions. It retains
FP32 KV, the eight original 32-wide reductions, and chronological online
softmax. Explicitly disabling reassociation for the final partial-sum addition
is essential: the first register-sum prototype was faster but differed by a
few float ULPs, and was rejected. This is synchronization/dataflow work, not
FP16 KV conversion or a new speculative-decoding algorithm.

Default selection: M5 only, position >= 1024. Short contexts and other devices
retain their old kernels because no reproducible short-context application
gain was established. The implementation sits below CLI/API/MTP selection.

## Fixed-token forward measurements, MTP off

Identical prompt and continuation token IDs, 32 timed forward steps. These
timings exclude logit-file I/O, model loading, and prefill; they are not API
end-to-end throughput. Synthetic repeated-token prompts test controlled state,
not task quality. The API suite below supplies natural workloads.

| Model | Actual input tokens | Main t/s | Candidate t/s | Gain |
|---|---:|---:|---:|---:|
| 35B Q8 | 2048 | 43.680 | 61.690 | +41.2% |
| 35B Q8 | 3840 | 28.860 | 44.958 | +55.8% |
| 27B Q8 | 2048 | 13.790 | 16.020 | +16.2% |
| 27B Q8 | 3840 | 11.048 | 13.919 | +26.0% |

The 2K samples used the initial all-context candidate; the 3840-token samples
used the final gated dispatch. At these decode positions both select the same
new kernel. Single samples, not confidence intervals.

Prefill times (main -> candidate seconds): 35B 2K 7.871 -> 7.761,
35B 3840 15.604 -> 15.118; 27B 2K 4.482 -> 4.086,
27B 3840 10.574 -> 9.735. These are secondary observations, not a separately
validated prefill optimization claim.

## Natural API workloads, existing MTP enabled

[Final gated, reverse-order A/B](FINAL-API.md) uses the original archived
coding/story/JSON prompts with actual input lengths 1368/2034/2037 and a
256-token output cap, temperature 0. Candidate ran before main. The final
gated pass improved 35B by 16.7–22.1% and 27B by 12.4–15.7%.

[Initial A/B](INITIAL-API.md) ran main before candidate, also covering short
prompts and cold/repeat pairs. Longer-context gains were 16.5–22.5% for 35B
and 6.4–10.4% for 27B. Both orders show a positive effect; variation between
passes means the precise percentage is not a universal guarantee. Short
results ranged -3.4% to +2.3%, so the final release keeps the original dispatch
there. Do not present the short-context t/s as a new optimization gain.

Raw requests/results/configuration and server traces are in the eight sibling
`../api-health/qwen27-q8-attention-*` directories. The initial shell driver was
edited while it was running and reported a parse error after all 48 responses
were written. Those responses were checked separately; the final, unmodified
reverse-order driver completed cleanly with all 12 requests.

## Correctness evidence

- Four fixed-token comparisons: **31,784,960 / 31,784,960 exact float logits**,
  all 128 row argmaxes identical, at 2048 and 3840 input tokens on both Q8 models.
- Kernel tests: 112 cases across 16/2 and 24/4 query/KV heads, contexts
  1/7/8/9/128/2048/3840, rows 1/2/5/8, gated and ungated, nonzero layer offset,
  periodic and pseudorandom inputs. Exact reference output throughout, plus an
  independent double-precision attention oracle (2e-5 absolute bound).
- Whole-model runtime tests passed on 35B Q8, 27B Q8, and 27B Q4 at short
  context, plus both Q8 models at 3840: 96-token serial/MTP equality, exact
  hot-cache restoration after another session, repeated continuation, cache
  capacity checks, stale-mark rejection, and engine teardown.
- Initial API: 24/24 exact main/candidate response pairs and 24/24 exact,
  positively reused cold/repeat cache pairs. Final API: 6/6 additional exact
  main/candidate pairs (60 requests total across the two API passes).
- `make` and `ds4_test --server` passed. No large CPU inference or simultaneous
  model processes were used; no watchdog crashes occurred.

Boundary validation also passed on both Q8 models with 1000-token prompts
and 96-token continuations, crossing the 1024-position dispatch threshold in
serial and MTP paths. Main was rebuilt with `make` after fast-forward merge.
The rebuilt-main smoke then passed 8/8 exact API responses and 4/4 positive
cache reuse pairs, on both Q8 models at short and 2037-token context with
native MTP enabled. Raw evidence is in `../api-health/qwen27-q8-attention-main35`
and `../api-health/qwen27-q8-attention-main27`. All test servers were stopped.
These eight requests bring the complete API count to 68.

No runtime claims are made here for CUDA, other Apple generations, DeepSeek,
GLM, Qwen Next, or SSD streaming. Their implementations were not changed by
this port; the new dispatch is limited to the resident small-Qwen path.

## Reproduction

Keep baseline `13517e9` checked out separately: runtime Metal sources must
belong to the executable under test. Set `QWEN_AB_BASELINE_ROOT` to that built
checkout for the A/B and long-check drivers. They refuse a different revision
to prevent accidentally comparing updated main against itself.

Build `tests/qwen_attn_tile_bench`; run normally and with `QWEN_TEST_RANDOM=1`
(optionally `SAFE=1` for strict Metal math). Build `tests/qwen_fixed_token_probe`
against each revision, use `PROBE_CONTEXT` for the writer, and share its token
file with the reader. Compare dumps with `tests/compare_qwen_fixed_logits.py`.
The long-check driver expects `/tmp/qwen-attn-baseline-probe` linked against
the baseline objects; full logit dumps are intentionally not committed.

`QWEN_TEST_CONTEXT=3840 tests/test_qwen_small_runtime MODEL` exercises long
cache/MTP state. Use the installed 27B sidecar via `DS4_QWEN_MTP_HEAD` for 27B.
`tests/qwen_attn_ab.sh` supports QWEN_AB_ORDER, QWEN_AB_CONTEXTS,
QWEN_AB_REPEAT, and QWEN_AB_SUFFIX; the final pass uses `candidate base`,
`2048`, `1`, and `-gated`. Reports use the same suffix/repeat environment.
