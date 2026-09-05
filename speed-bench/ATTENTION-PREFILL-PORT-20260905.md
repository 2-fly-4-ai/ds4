# Focused attention / prefill port — 2026-09-05

## Outcome

Two focused changes are retained, separately committed:

- `132af26`: protect the softmax maximum before threadgroup scratch reuse in
  GLM indexed attention, for both FP16 and FP32 caches.
- `9709a94`: select full-64/tail-32 expert tiles for the validated M5 Max
  DeepSeek Flash 1,024-row prefill case.

No upstream or experimental branch was merged wholesale. GLM copy-free
preparation, compact/fused verifier prototypes and other experimental flags
were not ported. `ds4.c`, server routing, MTP policy, cache policy and the
installed quants are unchanged. The repair applies wherever the shared
attention kernel is used, not only to the API.

Hardware: M5 Max, 128 GiB, macOS Automatic. No power/fan setting changed.
Models: existing Vision-Exp late-Q4 DeepSeek and hybrid Q2/Q4_K GLM-5.3 Flash.
Tests ran serially, with only one GPU workload active.

## Paired DeepSeek prefill results

Each corpus used 1,024 warm-up tokens per variant and four alternating
ABBA/BAAB repetitions: eight measured runs per side, sixteen per corpus.
Rates are total tokens / total time. These are warmed prefill results, not
first-load/JIT latency or whole-request speedups.

| 1,024-token corpus | Old | New default | Gain | Saved per prefill |
|---|---:|---:|---:|---:|
| Code | 666.70 t/s | 674.63 t/s | +1.19% | 18.1 ms |
| Structured text | 681.99 t/s | 686.82 t/s | +0.71% | 10.6 ms |
| Prose | 668.45 t/s | 679.64 t/s | +1.67% | 25.2 ms |

All 48 timed prefills had identical full logits. The new default also matched
all 257 continuation-logit checkpoints after a 1K prompt.

Scope is deliberately exact: M5 Max, resident non-GLM execution, 256 experts,
top-6 routing, the tested 4096↔2048 projections, matching IQ2_XXS/Q2_K pipelines,
and exactly 1,024 actual matrix token rows. Activation auditing observed 258
eligible calls at 1K and none at 512, 1,025 or 2,048 rows. This is **not** an
optimization for every prompt below 1K. Decode, other sizes, streaming and
other model/device shapes retain their existing routes.

Same-binary rollback switch:
`DS4_METAL_DISABLE_DS4_SHORT_PREFILL_TAIL32=1`.
In raw prefill logs, `control` is the **new default** and `candidate` is the
**old rollback**. The CSV and table reverse those labels correctly.

## GLM barrier timing and correctness

To avoid comparing different generated texts, an additional timing-only
test forced identical corpus token IDs through both implementations. It used
the same host binary and all the same shaders except `dsv4_misc.metal` from
base `3a9ebe6` versus the repaired file. Runs were old/new/new/old, with 64
untimed decode tokens followed by 256 timed tokens at each context.

| Prefix | Old faulty kernel | Repaired kernel | Measured timing change |
|---|---:|---:|---:|
| 2,048 tokens | 31.20 t/s | 30.98 t/s | −0.73% |
| 8,192 tokens | 29.64 t/s | 30.36 t/s | +2.43% |

These small movements are **not a new GLM generation-speedup claim**. At 2K,
the measured cost was about 0.24 ms/token. The old path has a known numerical
fault, so its internal states need not be identical even under the same input
token trace. This test measures fixed-input execution time, not quality parity;
the separate new-default correctness checks are authoritative.

- New-default GLM at 2K and 8K: 257 exact logit checkpoints each.
- Existing GLM KDA, Qwen kernels (including MTP helpers), and sampling unit
  suites: PASS.
- 64 timed natural-generation runs across both models: identical repeated
  token IDs and final logits within each case/settings group.
- GLM temperature 0.7 and native-MTP-on cases passed as well.

## Current workload generation rates

Established prompts, two warm-up runs then four timed repeats per case;
allocated context 4,096, actual prompt lengths shown. Output cap 256 tokens.
DeepSeek edit ended naturally at 178 tokens; other rows here used the cap.
These engine-route timings exclude prefill. They are current observed rates,
**not before/after gains from the 1K prefill selector**.

| Workload | DS prompt | DS generation | GLM prompt | GLM generation |
|---|---:|---:|---:|---:|
| Repeated copy | 266 | 74.67 t/s | 259 | 54.33 t/s |
| Code edit | 421 | 56.30 t/s | 382 | 42.08 t/s |
| Fresh coding | 32 | 40.25 t/s | 34 | 33.93 t/s |
| Story | 32 | 39.91 t/s | 34 | 38.74 t/s |
| JSON | 47 | 39.59 t/s | 47 | 38.23 t/s |

Neural MTP was off in this table; prompt lookup remained enabled and actually
committed tokens on copy/edit. Fresh code/story used plain decode.

Separate GLM runs:

| Settings | Copy | Code | Story | JSON | Edit |
|---|---:|---:|---:|---:|---:|
| Temperature 0.7, MTP off | 60.50 | 38.34 | 38.33 | 38.72 | — |
| Temperature 0, native MTP enabled | 59.92 | — | — | — | 46.27 |

All values are t/s. Both lookup and neural speculation executed in the MTP
suite. These separate settings suites were **not paired MTP-on/off tests**;
do not infer a causal MTP or temperature advantage from them. Machine-state
variation is visible across the run periods. No router default changed and
no DSpark sidecar was loaded. The API can also use its existing greedy-chain
path; these per-route engine numbers are not interchangeable with API timings.

## API regression and remaining caveat

Local-only temporary servers were tested with `/v1/models` and streamed
`/v1/chat/completions`: coding, repeated coding, sampled story, and JSON.
All eight requests on the port returned content and usage successfully.
API requests were capped at 128 tokens; truncated code/JSON here is intentional
and these smoke checks do not assert complete-program/JSON validity.

The stricter cold-versus-cached check discovered an **existing DeepSeek cache
rewind difference**:

- DeepSeek cold code and its cached repeat differed at temperature zero.
- The untouched main build reproduced the same difference.
- All six deterministic API outputs (code, cached code and JSON for both
  models) matched byte-for-byte between untouched main and this port.
- GLM cold/cached code matched within both builds.
- Diagnostic `DS4_KV_REWIND_REUSE=0` made DeepSeek's repeated code identical;
  all four requests in that diagnostic passed.

Consequently the default-cache API diagnostic correctly reports
`FAIL_REPEAT_PARITY`, not a fabricated universal PASS. This is not introduced
by these kernel changes: server/cache code is unchanged, the 22-token DeepSeek
prompt cannot activate the 1K selector, and baseline/port outputs agree.
The diagnostic disables rewind only for its own process; production cache
defaults remain untouched. Cold/warm rewind parity is a separate follow-up
target. Do not claim universal cold-versus-cached byte equivalence.

## Reproduction and artifacts

- `make all speed-bench/route_repeat_bench speed-bench/fixed_teacher_bench`
- `sh speed-bench/attention-prefill-validate.sh`
- `sh speed-bench/attention-barrier-timing.sh /path/to/base-3a9ebe6/dsv4_misc.metal`
  (script verifies that the old source matches the base blob; do not pass the
  now-updated main shader).
- `python3 speed-bench/attention-prefill-api-smoke.py`
  (strict default intentionally detects the documented DeepSeek cache issue).
- `DS4_SMOKE_DIAGNOSTIC=1` records both models before returning a parity failure;
  `DS4_SMOKE_SERVER_ROOT`, `DS4_SMOKE_RESULT_DIR`, and `DS4_SMOKE_MODEL` select
  the baseline executable, output directory, and optional single model.
- `python3 speed-bench/attention-prefill-summary.py` validates the completed
  engine gates and regenerates `prefill.csv`, `generation.csv`, and
  `barrier-timing.csv` under `attention-prefill-results/`.

The scripts reference the existing local model files and prior prompt/corpus
artifacts. Source/binary/prompt fingerprints are recorded alongside the logs.
The GLM unit-test Makefile gained its missing `ds4_image.o` link dependency,
exposed by a clean build. This does not change inference.

This is a targeted port/regression campaign, not a new 100K/256K benchmark or
the repository's full golden-model quality suite. Test servers are stopped
after their requests. Experimental branches and the user's prior benchmark
artifacts are preserved; nothing is pushed remotely by this task.
