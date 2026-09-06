# DeepSeek hot-prompt rewind investigation — 2026-09-06

Historical target-only investigation. The subsequent complete-frontier repair
and its stronger validation are documented in
`../dspark-hot-rewind-20260906/README.md`; the unresolved-DSpark disposition
below describes the earlier candidate, not that follow-up.

Base: main `7322056`. Candidate branch: `experiment/cache-parity-cost-router-20260906`.
This is a focused local experiment, not an upstream merge. No quant/model files,
Metal kernels, or production router defaults have been changed.

## Proven baseline defect

`baseline-diagnosis.log` reproduces two problems on the unchanged engine:

1. Rewinding one token and replaying it changes all 129,280 final logits
   relative to the original batched prefill (maximum absolute difference 2.16085).
2. After 128 generated tokens, token-only rewind also leaves advanced compressor
   counters/state behind (maximum logit difference 6.37160).

Loading the full original snapshot restores exactly identical logits. This is
not evidence that the quant is damaged: the baseline cache rewind is incomplete.

## Candidate repair

The engine saves the live sliding-window rows, mutable attention/indexer
compressors, compressed-row counters, and exact prompt logits at a hot prompt
frontier. Completed compressed KV rows remain shared. The server restores the
entire marked prompt, rather than replaying its final token through a different
arithmetic path.

Unmarked backwards rewinds invalidate the checkpoint, forcing a safe rebuild.
That can cost extra prefill for edited history, but avoids stale state reuse.
Prefix extensions, snapshot loads, and invalidation have explicit mark-lifetime
handling. CPU/CUDA/distributed/placement execution does not use the new Metal
snapshot path. The serialized checkpoint format is unchanged.

The maximum observed extra GPU snapshot storage is 23,478,272 bytes (22.4 MiB),
plus about 0.5 MiB of host logits. The final 2K case measured 2.028 ms to mark and
0.700 ms to restore. These are single measurements, not a latency guarantee.

## Passed checks

| Check | Evidence |
|---|---|
| Six prompt lengths: 17, 127, 128, 129, 259, 2048 | `hot-rewind-six.log` |
| Exact serialized target state/logits after generation, repeated rewind, complete raw-ring overwrite, and unmarked rebuild | Same six-case log |
| 128 regenerated tokens match for every length | Same six-case log |
| Additional 2K test regenerates 16 tokens after physical ring overwrite | `hot-rewind-wrap-generation.log` |
| Same 2K target-state/scalar-regeneration oracle with DSpark sidecar loaded | `hot-target-with-dspark.log`, pass |
| Snapshot load invalidates old hot mark | State tests |
| CLI baseline/candidate identity, seven API cases | `api.jsonl` and `api/` |
| Five API tasks, cold/repeat, temperature 0 and 0.7, fixed seed | All 20 requests pass: `repeat.jsonl` |
| SSD streaming, 4 GiB expert budget, cold/repeat 16-token smoke | Both requests match: `ssd.jsonl` |
| Metal build, CPU-only object compilation, server unit suite | Build/unit logs in this directory |

API tasks are coding, stories, JSON, copy and editing. Sampled tests establish
same-seed repeat identity for these requests, not a distributional quality claim.
These tests use the existing Vision-Exp late-Q4 quant. They do not substitute
for the repository's different golden checkpoint or CUDA/distributed hardware tests.

## Remaining DSpark exception — do not hide this

The opt-in DSpark API test passed coding but failed story cold/repeat identity.
Both response hashes and the failure are retained in `dspark.jsonl`. The ordinary
20-request test and SSD test do not certify this draft-side path. No universal
cache equivalence or complete DSpark repair is claimed. Keep the candidate off
main until this exception is addressed or an explicitly narrower port is chosen.

The 2K state oracle with the DSpark sidecar loaded passes exact target payload,
logits and scalar regeneration, including raw-ring wrap. This localizes the
remaining observable mismatch to speculative execution/history interaction;
it does not yet identify its first incorrect intermediate tensor.

Two final isolation runs:

- Untouched main with DSpark fails even the coding cold/repeat pair:
  `dspark-baseline.jsonl`. This establishes a pre-existing DSpark repeat issue,
  not that its exact story divergence was reproduced on baseline (that run
  stopped at coding).
- Candidate with `DS4_DSPARK_SCHEDULER=0` passes all ten requests/five pairs:
  `dspark-no-scheduler.jsonl`. The flag was process-local, diagnostic only;
  production defaults are unchanged. This demonstrates scheduler-dependent
  speculative behavior in this corpus. It does not establish scalar equivalence
  of every DSpark verifier path.

The next repair needs to address scheduler/draft-history lifetime at the hot
frontier and verify route-independent target advancement. Do not permanently
disable scheduling or relax exact comparison to call this finished.

## Final disposition

Saved as a local experimental commit, **not merged or pushed**. Main remains
`7322056`; no server is left running by these tests. The ordinary DeepSeek cache
defect has a validated candidate repair, but DSpark-enabled repeat correctness
is unfinished. GLM/Qwen full regression and any production promotion should
follow resolution or an explicitly scoped port, not precede it.

## Cost-router experiment

This is separate, on `experiment/glm-cost-router-20260906` in
`/Users/brianfarley/Desktop/ds4-router-cost`. Its diagnostic controls are NOT in
this cache-fix branch or main. See that branch's
`speed-bench/cost-router-20260906/README.md` for the timing and attribution results.
