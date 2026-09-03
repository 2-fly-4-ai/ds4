# M5 Max adaptive prompt-verifier campaign

Measured 2026-09-03 on an Apple M5 Max with 128 GiB RAM in macOS Automatic
power mode.  The DeepSeek measurements used the late-Q4 model
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AOutQ4K-L27-42-chat-v2-imatrix-0731.gguf`
fully resident on Metal.  Native MTP, SSD streaming, and multi-session
batching were disabled.

This campaign started from the existing seven-draft prompt-lookup path and
asked whether longer verifier blocks could reduce the number of memory-bound
model passes without hurting mixed or low-acceptance workloads.

## Result shipped in the working tree

DeepSeek prompt lookup still starts with seven drafts.  On M5 Metal it now
promotes to 15 drafts only after eight consecutive verifier passes accept their
entire block.  Any no-match or partial acceptance resets it immediately to
seven.  The 16-row verifier uses the existing eight-row F16 matvec kernel over
two row tiles plus the classic Q4 path; the wider dispatch is scoped only to an
active DeepSeek prompt-verification pass.

That scope is important.  Ordinary prefill, decode, GLM, and unrelated 9--16
row graphs retain their previous dispatch.  The portable default on non-M5
hardware also remains seven drafts.

## End-to-end A/B

The rollback side used `DS4_PROMPT_LOOKUP_ADAPTIVE=0`.  The candidate side
used the final automatic defaults.  Rates below exclude prefill and compare
the same generated token streams.  Four-run tests used ABBA order.

| Workload | Fixed depth 7 | Adaptive 7 -> 15 | Change | Output check |
|---|---:|---:|---:|---|
| 2K repeated continuation, 512 tokens | 84.32 t/s | 97.76 t/s | **+15.9%** | SHA-256 identical |
| Agent/source reproduction, 256 tokens | 67.21 t/s | 76.05 t/s | **+13.2%** | SHA-256 identical |
| Code-edit continuation, 256 tokens | 47.06 t/s | 48.25 t/s | **+2.5%** | SHA-256 identical |
| 16K repeated continuation, 512 tokens | 62.62 t/s | 83.77 t/s | **+33.8%** | SHA-256 identical |
| 99,200-token continuation, 256 tokens | 51.95 t/s | 59.69 t/s | **+14.9%** | SHA-256 identical |

The 16K run had substantial thermal/run-order movement, so its exact absolute
gain should not be generalized.  Its ABBA direction agreed with the less noisy
2K, agent, code-edit, and 100K checks.  The 100K prefill was 269.058 seconds on
rollback and 266.498 seconds on candidate; that difference is treated as noise
because this optimization changes decode only.

The final 100K output hash was
`0d53bf449909a5600d343da30bdc0d91ceb3fac72aeae90606829f4e025a2027`
on both sides.  At 2K, the adaptive schedule reduced 64 verifier passes at
seven accepted drafts per pass to 36 passes at 13.22 accepted drafts per pass.
At 100K it reduced 32 passes to 20.

## Production API validation

A real `/v1/chat/completions` request with `thinking:false`, temperature zero,
and a copy-style prompt switched from the chained greedy decoder into prompt
lookup after 23 generated tokens.  It completed 512 tokens at 84.20 t/s in the
warm state.  Shutdown telemetry reported 35 verifier passes, 454/454 drafted
tokens accepted, 100% acceptance, and 12.97 accepted drafts per pass.  This
confirms the adaptive path is active through the API, not only through the raw
CLI harness.

The same shared evaluator is also wired into `ds4-agent` ordinary and raw
routes.  An unrelated model/tool-routing outlier in one agent run selected a
tool on its first output token, before prompt lookup could execute; it was
discarded and replaced rather than counted as an optimization result.

## Adversarial and no-match controls

A repeating-record prompt designed to diverge late never reached the promotion
threshold.  Candidate and rollback both recorded 183 attempts, 170 no-matches,
13 verifier passes, 91 drafted tokens, 73 committed tokens, and 80.2%
acceptance.  Both produced hash
`597cfbc6d5c2f9befa5921c17dc937ba2833f052915c180cc3623a618532d115`.

A novel-code control recorded 128/128 no-matches, zero verifier passes, and
only 0.60 ms total prompt scan time.  This confirms the deeper verifier is not
entered when the prompt cannot supply useful drafts.

## GLM-5.3 follow-up

The same conservative state machine now applies to resident single-GPU GLM
5.3 on M5 Metal, starting at three drafts and promoting to 15 after eight
consecutive full accepts.  Unlike DeepSeek, GLM uses its native batched
DSA/KDA verifier and does not need the F16 bridge.  Any no-match or partial
accept resets the streak.  Explicit `DS4_PROMPT_LOOKUP_MAX` continues to mean
a fixed depth rather than an adaptive ceiling.

Seeded sampled A/Bs produced identical bytes:

| Workload | Fixed depth 3 | Adaptive 3 -> 15 | Change |
|---|---:|---:|---:|
| 1.35K repeated continuation, 256 tokens | 57.77 t/s | 62.25 t/s | **+7.8%** |
| 1.75K C-source reproduction, 256 tokens | 58.96 t/s | 61.88 t/s | **+5.0%** |
| 93K repeated continuation, 256 tokens | 32.28 t/s | 40.38 t/s | **+25.1%** |

At 93K the adaptive policy reduced 49 verifier passes to 29, accepted 162 of
165 drafts (98.2%), and replayed the one partial acceptance exactly.  Prefill
was 239.6 versus 248.2 seconds and is treated as thermal noise because the
policy is inactive during prefill.

A 14.4K four-run sequence drifted monotonically from 50.6 to 42.2 t/s as the
machine heated, so its raw ABBA average is not attributed to the code.  The
adaptive runs nevertheless produced the same bytes and counters, reduced 53
useful verifier passes to 38, and did not expose a correctness failure.

For greedy verification, selecting the intermediate row argmaxes on the GPU
and reading back only the final full logits row added 0.6% in a controlled
fixed-depth-15 A/B.  The path is M5-only by default and has a dedicated
`DS4_GLM_PROMPT_LOOKUP_GPU_TOP1=0` rollback.

### GLM routed-MoE verifier follow-up

The generic routed-MoE batch path previously used its fused IQ2 gate+up+SwiGLU
kernel only through five rows. GLM's promoted verifier commonly sends 15--16
rows, so those passes fell back to separate gate and up expert-weight reads
plus a separate activation kernel. The M5 resident IQ2/Q2 top-8 shape now
keeps the fused pair kernel through 16 rows. The dispatch is limited to GLM,
M5, one resident GPU, top-8 routing, IQ2_XXS gate/up and Q2_K down tensors;
prefill, SSD streaming, tensor parallelism, other quants, and other hardware
retain their former path.

Rates exclude prefill. Tests used mirrored order and identical output hashes:

| Workload | Previous ceiling | Optimized fused path | Change |
|---|---:|---:|---:|
| Hot default-policy repeated continuation, 256 tokens | 58.80 t/s | 60.46 t/s | **+2.81%** |
| Seeded sampled prose, 128 tokens | 39.29 t/s | 39.88 t/s | **+1.49%** |
| Raw C-source reproduction, 256 tokens | 45.93 t/s | 46.15 t/s | **+0.48%** |
| Fixed-depth-15 repeated continuation, 256 tokens | 70.95 t/s | 72.64 t/s | **+2.38%** |
| Full 15-draft/16-row continuation, 256 tokens | 72.97 t/s (15-row ceiling) | 74.71 t/s | **+2.39%** |

On the hot default-policy repeat test, median promoted-verifier latency fell
from 249.73 to 240.45 ms (**-3.72%**). All four transcripts shared SHA-256
`b586e5dd0df3b8725bc118db2975f2b63f14ec0b3ab62ac6ab04806c4b994b9e`.
The sampled test used temperature 0.7 and seed 314159; all four outputs shared
SHA-256 `ce9876058fa0ebb3a104cfa4e980c943dbf9dbbaf53ab40d391948e100f6cdc8`.

A final no-env `/v1/chat/completions` smoke test completed 138 prompt tokens at
104.38 t/s and 128 generated tokens at 58.97 t/s, returned valid usage and
model metadata, and shut down cleanly. Its ambiguous continuation stayed at
the three-draft shallow policy, so it validates API integration and fallback,
not the wide-kernel speedup measured above.

A mirrored 14,400-token prompt check measured 44.62 versus 44.23 t/s
(-0.88%, thermal/order noise) and unchanged prefill within 0.09%. That output
found only 70 reusable drafts across 18 passes (3.89 drafts/pass), so the
promoted kernel rarely engaged. All four long-context outputs were identical.
This is the intended neutral case rather than evidence of a long-context
speedup: the change accelerates useful wide verifier passes, not attention or
prefill.

Mixed sampled controls remained conservative. Novel prose and structured/code
prompts either never promoted or never found a match; their seeded outputs
were identical and throughput was neutral within run-order noise.  An
ambiguous repeated prompt also stayed at depth three because interspersed
no-matches reset the streak.

A production `/v1/chat/completions` C-source reproduction request confirmed
the default, no-env route. It promoted after eight full accepts, completed 256
sampled tokens at 64.07 t/s, and reported 209/209 drafts accepted across 21
verifier passes (9.95 drafts per pass). A deliberately ambiguous API prompt
stayed shallow, confirming that the same reset policy is active in the server.

## Kernel correctness

- The Q4 extension uses the pre-existing classic kernel and was exact against
  the former dispatch in isolated full-logit checks.
- The F16 bridge reuses the validated eight-row specialization.  CPU-oracle
  tests now cover 9, 12, and 16 rows and pass, along with the existing 128-row
  prefill case.
- Switching the F16 reduction schedule is numerically valid but not bitwise
  identical across every vocabulary logit.  This is floating reduction-order
  drift, not a wrong matrix product.  Therefore the release criterion was
  oracle correctness plus byte-identical end-to-end outputs across favorable,
  mixed, long-context, and adversarial tests.

## Exact-16 Q4_K attention-output TensorOps tile

GLM-5.3's 15-draft prompt-lookup pass evaluates 16 rows. Its Q4_K attention
output is a single 8192-to-4096 KDA or 16384-to-4096 DSA projection. The
generic Metal matrix kernel uses a 64x32 tile at this size: two SIMDgroups
compute the live first 16 rows and two compute a clamped duplicate that the
boundary store discards.

The retained M5 specialization instantiates the existing direct-RHS,
double-buffered TensorOps kernel at exactly 16 rows. It reuses each staged
dequantized Q4_K tile across the verifier rows, performs no unused second-half
work, and leaves all other shapes on their existing paths.

Clean-tree mirrored A/B, fixed 15 drafts, 512 generated tokens:

| Route | End-to-end decode | Median verifier pass |
|---|---:|---:|
| Generic rollback | 78.44 t/s | 203.86 ms |
| Exact-N16 TensorOps | 80.55 t/s | 198.50 ms |
| Change | **+2.69%** | **-2.63%** |

At 4,807 prompt tokens, after GLM crossed into compact indexed attention, the
same route measured 75.61 versus 73.86 t/s (+2.36%) and 211.02 versus
215.91 ms per verifier pass. Greedy, temperature-0.7 sampled, short-context,
long-code, and indexed-context comparisons produced identical seeded output
bytes. A real-model layer-0 tensor oracle measured max absolute drift
2.93e-5 and RMS drift 4.00e-6, the expected cooperative-matmul reduction-order
difference rather than a wrong projection. As expected for routed MoE, that
small perturbation can amplify through the network (the layer-44 hidden dump
measured max 3.59 and RMS 0.110), so seeded end-to-end output parity—not hidden
bit identity—is the release criterion, and the generic route remains available
as an immediate rollback.

## Rejected experiments

- Lowering the 24-token activation gate occasionally changed near-tie output
  and did not deliver a repeatable overall win.  The gate remains unchanged.
- Unconditionally using depth 15 was about 8% slower on the hostile
  late-divergence prompt.  This led to the eight-full-pass promotion rule.
- Extending Q8 and F32 small-batch dispatch was removed.  The Q8 change was
  slower in isolation, and the broader dispatch changed far more of the graph
  than needed.
- A new true 16-row F16 tile passed the numeric oracle but was about 17% slower
  end to end.  Reusing the eight-row kernel over two tiles was faster.
- Promotion thresholds of two and four full passes were too eager on mixed
  traces.  Eight preserved the old schedule on the adversarial case while
  still promoting early in copy-heavy spans.
- GLM per-layer command-buffer flush suppression was neutral at both 9 and 15
  rows and was removed.
- Collapsing all GLM mHC output rows in one dispatch produced identical output
  but no steady-state gain and was removed.
- A staged GLM 3 -> 7 -> 15 policy saved one verifier pass in several replay
  traces, but the clean repeated-text A/B was only +0.54% and sampled results
  were order-sensitive. It was removed instead of adding policy complexity.
- Extending the DeepSeek 9--16-row Q4 verifier bridge to GLM initially looked
  positive while cool, but hot default-policy A/B was decisively negative:
  61.35 fell to 58.37 t/s (-4.87%) and verifier latency rose 7.10%. GLM stays
  excluded from that bridge.
- Directly accumulating all eight Q2 down experts removed the scratch/sum
  pass but serialized too much expert work (75.03 -> 72.44 t/s). A separate
  fused sum8+shared-output tail was also slower by 0.85%. Both were removed.
- Avoiding dead gate/up activation writes changed verifier median by only
  -0.45% and lost 0.33% end to end. Changing the Q4 bridge from two to four
  SIMDgroups reversed from +0.38% in the short sweep to -0.49% in the longer
  A/B. Neither marginal variant shipped.
- Raising GLM to a 32-row/31-draft verifier crossed into a bad graph regime on
  the C-source fixture: about 61.6 t/s fell to 28.0 t/s, acceptance collapsed,
  and seeded output changed. The experimental storage expansion was removed;
  16 verifier rows remains the validated ceiling.
- A custom four-SIMDgroup Q4_K row-sharing matvec was byte-identical but 1--2%
  slower because its extra staging/barriers could not beat the existing matrix
  kernel. A 128x16 variant and a bit-identical 64-thread 64x16 variant measured
  -1.4% and about +0.4%, respectively; both were removed in favor of the
  consistently faster TensorOps tile.
- The same direct-RHS TensorOps approach does not cross over at smaller row
  counts. Apple TensorOps rejects a true N=4 tile; padding it to N=8 raised
  steady verifier latency from about 50.8 to 58.8 ms. A true N=8 tile measured
  83.12 versus 85.21 t/s end to end (-2.45%) and 95.76 versus 93.33 ms per
  pass. Both experiments produced identical output and were removed. The
  classic Q4 matvec remains correct for N=4/N=8; TensorOps begins at N=16.

## Validation completed

The final tree passes:

```text
make -j8 ds4-agent ds4 ds4-server ds4_test
make -j8 ds4_cpu.o
./ds4_agent_test
./ds4_test --server
./ds4_test --long-context
./ds4_test --metal-kernels
python3 -m py_compile speed-bench/prompt_lookup_policy_replay.py
git diff --check
```

The long-context regression completed all 30,474 prompt tokens.  The full
Metal-kernel group passed, including the new F16 9/12/16-row oracle coverage.

## Controls

- `DS4_PROMPT_LOOKUP_ADAPTIVE=0`: keep the former fixed seven-draft DeepSeek
  schedule.
- `DS4_GLM_PROMPT_LOOKUP_ADAPTIVE=0`: keep the former fixed three-draft GLM
  schedule.
- `DS4_GLM_PROMPT_LOOKUP_GPU_TOP1=0`: restore full intermediate-logits
  readback for greedy GLM verification.
- `DS4_METAL_DISABLE_M5_GLM_TINY_PAIR_16=1`: restore the former five-row
  ceiling for the GLM IQ2/Q2 fused gate+up+SwiGLU kernel.
- `DS4_METAL_GLM_TINY_PAIR_MAX_TOKENS=N`: diagnostic override for that fused
  small-batch threshold (1--31).
- `DS4_METAL_DISABLE_M5_GLM_ATTN_Q4_MPP_N16=1`: restore the generic 64x32
  Q4_K matrix tile for GLM's 16-row attention-output projections.
- `DS4_METAL_DISABLE_M5_PL_MV16=1`: disable only the M5 verifier bridge.
- `DS4_PROMPT_LOOKUP_MAX=N`: explicitly override the draft depth from 1 to 15.
- `DS4_PROMPT_LOOKUP_DISABLE=1`: disable prompt lookup globally.

## Conclusion

The next-experiment order was correct: policy replay identified the depth
opportunity, kernel isolation found the narrow F16 bottleneck, adversarial A/B
forced an adaptive rather than unconditional rollout, and production-route
tests proved the gain is reachable by users. The shipped policy and GLM
routed-pair kernel are real single-session improvements on copy-heavy and
agent/code continuations while remaining effectively neutral when prompt
lookup is not useful.
