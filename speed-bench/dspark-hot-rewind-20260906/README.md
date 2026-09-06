# Complete DeepSeek/DSpark hot rewind — 2026-09-06

This follows the target-only candidate `618e969` on main base `7322056`.
The separate GLM cost policy (`4232000`) is not included.

## Cause and repair

The old engine trimmed tokens without restoring the target compressors. The
target-only candidate fixed that and retained exact prefill logits, but did not
restore all speculative state. Three successive isolation tests established:

1. DSpark's acceptance-window/pause policy survived from the future continuation.
   Restoring this policy fixed the story repeat, but JSON still differed.
2. Captured prompt features and DSpark's draft KV ring/metadata were overwritten
   by later inference. Restoring them made all five API repeat pairs pass.
3. A stronger per-block oracle found prompt lookup's adaptive depth and gate
   backoff were also missing: its first repeat block returned 11 tokens instead
   of the original 8. Those policy fields now rewind with the prompt too.

The snapshot now owns target SWA/compressor state, exact logits, draft capture
and ring state, and decode policy. Completed compressed target KV stays shared.
New draft proposals are invalidated on rewind; their restored input history is
retained. The DSpark scheduler remains enabled, with the same thresholds and
verification math. No quant or kernel changes, no relaxed comparison tolerance.

Failed stages are retained in `trace.jsonl`, `schedule-only.jsonl` and
`missing-lookup-policy.log`. Detailed corresponding server traces are in
`../cache-router-20260906/{scheduler-trace,schedule-restored,full-draft-restored}`.

## Engine correctness evidence

| Test | Result |
|---|---|
| Prompt lengths 17, 127, 128, 129, 259, 2048 | Six pass |
| Sampled 2K, temperature 0.7 | Pass |
| Original coding prompt, greedy neural route | 42 calls / 128 tokens; pass |
| Original story prompt, greedy neural route | 81 calls / 128 tokens; pass |
| JSON prompt, sampled neural route | 38 calls / 128 tokens; pass |
| Actual 8192-token prompt, 16384 allocation | Pass (lookup workload) |

Each of these eleven cases generates 128 tokens, restores and repeats them,
overwrites the entire physical raw ring through further prefill, restores, and
repeats all 128 tokens again: 4,224 generated tokens in total. At every returned
route block the oracle requires identical route, returned count, token IDs and
all 129,280 target logits. Restored serialized target state is byte-identical.
Unmarked rewind/rebuild and snapshot-load invalidation are checked too.

These comparisons establish cold-versus-restored execution parity. They do
not establish universal scalar-versus-batched verifier equivalence, statistical
sampling quality, or performance across every context/model. The 8K case tests
longer cache lifetime with lookup; it is not an 8K MTP throughput benchmark.

## Cost and scope

With DSpark loaded, observed snapshot storage is 241,883,136–251,658,240 bytes
(about 231–240 MiB), including full allocated draft feature buffers. The 8K
case restores in 2.879 ms; short neural cases about 2.5 ms. These are individual
observations, not confidence intervals. Ordinary DeepSeek retains the smaller
roughly 22.4 MiB target-only snapshot at the tested window size.

Copying happens at prompt mark/rewind, not per generated token. The snapshot is
lazy, per-session and freed with the session. Allocation size depends on graph
prefill/raw capacity; 240 MiB is not a universal maximum for arbitrary settings.

The new hot state path is single-device Apple Metal, including SSD streaming.
Legacy MTP, CPU, CUDA, distributed and placement paths do not use it. Legacy MTP
has different history and remains on its existing path, not newly certified.
Multimodal API requests do not publish this text hot frontier. No checkpoint
file-format change. Unmarked backwards rewinds on the supported path rebuild
instead of silently reusing stale KV; edited-history prefill can therefore cost
more. No GPU model tests were run on CPU or CUDA/distributed hardware.

## Final API and regression validation

| Check | Result |
|---|---|
| DSpark scheduler on: five tasks, cold/repeat, temperatures 0 and 0.7 | 20 requests pass |
| Ordinary DeepSeek: same task/temperature matrix | 20 requests pass |
| Supported SSD streaming, 4 GiB expert budget | Two requests pass |
| DeepSeek baseline/candidate API, including stop handling | 14 requests pass |
| GLM candidate scalar versus native-MTP/lookup router API | 14 requests pass |
| Qwen baseline/candidate API | 14 requests pass |
| CLI before/after hashes, DeepSeek/GLM/Qwen | All three pairs match |
| Metal CLI/server build, CPU-only object compile, server unit suite | Pass |

Total final supported API requests: **84**. The DeepSeek regression compares
the repaired cached response against the original cold response, not against
the baseline's known broken cached result. No hash/logit tolerance was loosened.

An attempted **SSD + DSpark sidecar** combination was refused by the existing
engine restriction (`--ssd-streaming is not compatible with --mtp-model yet`).
That is retained in `api-ssd/` and is NOT counted as passed coverage. The
supported ordinary SSD run is `api-ssd-plain.jsonl`. This patch does not add
SSD/sidecar compatibility.

The final build includes an explicit exclusion for legacy MTP. The preceding
DSpark/ordinary oracle and repeat matrices exercise the same included paths;
the final cross-model CLI/API regressions use the final guarded build.

Single short CLI speed sanity readings (128-token limit, unchanged output):

| Model | Before t/s | After t/s |
|---|---:|---:|
| DeepSeek | 42.96 | 43.02 |
| GLM | 35.48 | 35.48 |
| Qwen | 40.49 | 44.20 |

These are not repeated performance estimates. In particular no Qwen speedup
is attributed to this DeepSeek cache fix; its execution path is unchanged.
The repair preserves the decoding implementation, rather than claiming new
generation throughput. Tiny-prompt prefill rates include fixed startup costs
and should not be compared with long-prompt prefill benchmarks.

## Disposition

Promote the complete cache repair to local main after these checks. The GLM
cost-policy experiment remains isolated at `4232000`; no new quant, kernel,
router threshold or sampling default is promoted. No remote push is performed.
The full golden-model suite and CUDA/distributed hardware execution are not
claimed by this focused validation.
