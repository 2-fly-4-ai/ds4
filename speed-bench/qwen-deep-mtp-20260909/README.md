# Qwen deep-MTP experiment ledger

Date: 2026-09-09

This directory records the Qwen3.8 Flash-Next verifier investigation.  The
failed multi-depth controller remains archived on
`experiment/qwen-deep-mtp-20260909`; the promotion branch deliberately keeps
only the exact short-row kernels, verifier wrapper, rewind fix, and tests used
by the proven depth-one MTP and prompt-lookup routes.

## External ceiling reproduced on this M5 Max

Using the exact MLX-Serve 26.9.2 Flash-Next pack:

| Workload | Serial | Adaptive MTP | Forced best seen |
|---|---:|---:|---:|
| Ordinary coding | 57.5 t/s | 96.4 t/s | n/a |
| Repetitive code edit | 57.5 t/s | 118.3 t/s | 153.8 t/s at depth 8 |
| Novel prose | 57.8 t/s | 54.9 t/s | MTP is a loss |

This is the performance target, not a claim about the current DS4 prototype.

## Confirmed findings

1. The prototype can accept multiple neural drafts.  One depth-three run
   committed 40 draft tokens over 22 verifier cycles (1.82 drafts/cycle).
2. A same-checkpoint scalar-versus-`T=4` harness produces the same four token
   argmaxes in the controlled probe.
3. The first small-row full-logit differences come from F16/Q8/F32 projection
   schedules. Diagnostic scalar-row oracles make layers 0--2 bit-identical.
4. The first remaining independent target-forward difference is layer 3's
   full-attention output. Attention preparation, Q, and gate are exact; the
   attention output is not. Disabling `ATTN_MM` did not fix it and worsened
   final-logit drift.
5. HC-pair (variant 2), GDN-R4 (variant 3), and serial predictor (variant 7)
   each alter the proposal route at the same near-tie.  Variant 6 and the full
   variant 8 do too. Proposal-route identity is therefore not a valid
   correctness requirement by itself.
6. With relaxed proposal grouping, variant 8 emitted the same first eight
   committed target tokens, but the final payload differed. The first payload
   difference is the final logits at byte 244; the first post-logit difference
   is layer 0's GDN recurrent state at byte 993528. Deep mode is not safe for
   promotion yet.
7. The scalar authoritative-verifier oracle removes all target-state drift.
   Target tokens/logits, every trunk layer, PLE/position state, and eight tokens
   of continuation after restoring each payload are byte-identical. The only
   different payload span is the MTP proposal layer's value cache; restore
   invalidates the pending proposal, and the continuation test proves it cannot
   affect the target output.
8. Existing one-token Metal matvec pipelines can dispatch Q8, F16, and F32
   verifier rows in a two-dimensional grid while retaining the exact one-token
   reduction order.  On the 64-token coding probe this progression measured:

   | Verifier implementation | Deep MTP | Matching plain | Result |
   |---|---:|---:|---:|
   | fully scalar correctness oracle | 14.52 t/s | 47.69 t/s | correct, unusable |
   | first exact hybrid | 42.81 t/s | 47.52 t/s | correct, slower |
   | exact Q8 row grid | 47.93 t/s | 47.52 t/s | about +0.9% |
   | exact Q8/F16/F32 row grids | 49.93 t/s | 47.69 t/s | about +4.7% |
   | plus exact attention row grid | 50.87 t/s | 47.84 t/s | about +6.3% |
   | unsafe ordinary batched ceiling | 53.94 t/s | 47.74 t/s | +13%, divergent |

   The exact path has recovered most, but not all, of the ordinary-batched
   ceiling without relaxing target correctness.
9. Attention drift came from using one maximum-position softmax split plan for
   every verifier row.  The exact row-grid kernel computes the same split count
   and boundary each row would use during one-token decode.  At sparse context,
   the index-score matrix kernel was a second source because it rounds query
   rows to half.  The sparse verifier now retains the FP32 one-token scorer in
   a row grid.  Dense short-context and sparse 2K-context probes both pass full
   target logits/state and restored-continuation checks.
10. One- and two-token rewind now pass exact target snapshot comparison by
    retaining verified anchor logits alongside the recurrent-state snapshot.
11. Exact verifier selection is now scoped by the Qwen graph rather than four
    diagnostic environment variables.  Depth-one MTP, partial replay, and
    prompt lookup all enter the same exact-row wrapper.
12. The 16-row prompt-lookup lane had one additional boundary bug: at more than
    eight rows the HC mixer silently selected its prefill schedule.  A layer-0
    trace showed 963--1448 changed values per row (maximum about 1.7e-6).  The
    verifier now retains the decode-equivalent HC schedule through 16 rows.
    The trace became bit-identical in all 16 rows, and the complete 256-token
    copy oracle passed target logits, target state, and restored continuation.
13. Deeper neural drafting is not the best production default.  On fresh
    128-token measurements, exact depth-one MTP substantially beat both plain
    decode and adaptive depth three.  Production therefore stays at depth one;
    the depth 2--5 prototype is not part of the promotion.

## Current matched economics

All entries below use exact target logits/state and interleaved timing arms.
The structured case was thermally noisy and is retained as directional only.

| Workload/context | Plain median | Deep MTP median | Change | Accepted/cycles |
|---|---:|---:|---:|---:|
| coding, short prompt, depth 3 | 47.84 t/s | 50.87 t/s | +6.3% | 41/22 timed |
| reasoning, short prompt, depth 3 | 47.64 t/s | 49.32 t/s | +3.5% | 41/22 timed |
| novel prose, short prompt, depth 3 | 47.51 t/s | 34.65 t/s | -27.1% | 33/30 timed |
| structured JSON, short prompt, depth 3 | 37.57 t/s | 38.27 t/s | +1.9% noisy | 45/17 timed |
| coding, padded 2K, depth 3 | 45.17 t/s | 40.20 t/s | -11.0% | 39/24 |
| coding, padded 2K, depth 5 | 45.47 t/s | 32.30 t/s | -29.0% | 42/20 |

These historical results rejected deep MTP for the release path.  Deeper
proposals do not amortize the current long-context verifier, so the clean
promotion carries none of the multi-depth controller or its runtime state.

## Default-router validation after exact 16-row repair

These are single matched same-process checks after the machine had already run
an extended kernel campaign.  Absolute rates are thermally depressed, so the
paired ratios—not comparison with older cool-machine peaks—are the evidence.
Every greedy pair passed position-aligned full logits, target-only serialized
state, and an eight-token continuation after restoring both final snapshots.

| Workload/context | Plain | Default route | Change | Route detail |
|---|---:|---:|---:|---|
| novel coding, 43-token prompt | 44.63 t/s | 62.45 t/s | **+39.9%** | exact depth-one MTP |
| novel prose, 40-token prompt | 42.41 t/s | 52.11 t/s | **+22.9%** | exact depth-one MTP |
| padded coding, 2,048-token prompt | 42.02 t/s | 56.50 t/s | **+34.4%** | exact depth-one MTP |
| repeated copy, 268-token prompt | 40.85 t/s | 85.01 t/s | **+108.1% / 2.08x** | 7 rows, then automatic 15-row lookup |
| code edit, 421-token prompt | 26.84 t/s | 39.53 t/s | **+47.2%** | lookup plus depth-one MTP fallback |

The repeated-copy default made 20 lookup passes, drafted/committed 230/230
tokens, and automatically promoted from seven to fifteen drafts after eight
full matches.  The code-edit check committed 126/139 lookup drafts (90.6%) and
used exact depth-one MTP elsewhere.  A forced 15-draft oracle also passed all
256 tokens with 100% lookup acceptance.

For comparison, adaptive depth three on the same coding shape finished at
43.17 versus 43.74 t/s plain (slightly worse overall), even though its first
eight-cycle window looked profitable.  Novel prose rejected depth three by a
wide margin.  This falsifies depth three as a universal default and justifies
the retained lookup -> depth-one MTP -> plain ordering.

Builds, the Qwen Metal kernel suite, and all model-independent tests pass.
DeepSeek Vision-Exp and GLM smoke outputs are byte-identical to production.
The explicit DeepSeek `ds4_test` run passes snapshot, rewind, 30K context,
tool-call, SSD-cache-pressure, local-golden, kernel, and tensor-equivalence
checks; its two official long-vector failures are the known fixture/model
version mismatch, not introduced by this Qwen-only change.

The real OpenAI-compatible server was also started with `--mtp`.  A 64-token
coding request decoded at **61.39 t/s** and a 128-token repeated-copy request at
**87.25 t/s**.  The latter made 14 prompt-lookup passes, committed 92/93 drafts
(98.9%), then shut down cleanly.  This proves the shared router works in the
API path, not only in the internal harness.

The archived depth-three fallback was retested after preventing discarded MTP
proposals from being computed during its plain backoff.  On novel prose it
recognized the loss after one measured window, used only 13 neural calls, and
ran plain for the rest.  It still finished 45.22 versus 47.63 t/s plain; this
is why that controller was excluded rather than hidden behind a production
environment flag.

## Next gates, in order

1. Keep prompt lookup first, exact depth-one MTP second, and plain decode as the
   fallback; retain a route only when matched wall time improves.
2. Re-run byte-exact full/partial acceptance, rejection, rewind, reload, and
   long-run parity whenever the verifier kernels change.
3. Treat deeper neural drafting as a separate research branch until proposal
   and verification cost improve enough to beat depth one across workloads.
4. Explore a quantized proposal-only vocabulary head and compiled verifier
   lane without changing target arithmetic.

## Separate dense-Qwen track retained

The earlier Qwen3.8 27B findings are not discarded.  They need their own matched
quant baseline, then isolated experiments for affine Q4/group-64 weights,
split-K-style rows 2--7, and NAX/cooperative rows 8--16.  Proposal-head,
shortlist, rollback, instrumentation, and adaptive-router work may be shared;
Flash-Next recurrent/attention kernels may not be assumed to transfer.

## Promotion rule

No merge until the target correctness contract passes and median matched wall
time improves. Failed switches and divergent runs stay documented rather than
being hidden from the benchmark.
