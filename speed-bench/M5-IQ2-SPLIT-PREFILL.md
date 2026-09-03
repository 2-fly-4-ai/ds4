# M5 IQ2 split-prefill campaign

Measured 2026-09-03 on a MacBook Pro M5 Max with 128 GiB RAM in macOS
Automatic power mode. This is a focused adaptation of the useful kernel idea
from antirez/ds4 PR #864; no upstream branch was merged.

## Retained implementation

The new resident routed-MoE specialization divides a 32-row expert tile across
two independent two-SIMDgroup cooperative matrix pairs. It double-buffers the
weight and activation tiles in 12 KiB of threadgroup memory and uses an exact
binary16 IQ2 dequantization lookup table only inside this specialization.

The previous N32 and GLM N64 kernels remain compiled alongside it. The new path
is automatic only for M5, resident single-GPU, non-quality IQ2_XXS gate/up with
Q2_K or IQ2_XXS down projection, at 32 or more input rows. SSD streaming,
tensor parallelism, graph dumps, other quant layouts, and other hardware keep
their existing routes.

`DS4_METAL_DISABLE_M5_IQ2_SPLIT_MPP=1` is a complete same-binary rollback. It
restores both the prior kernel and prior IQ2 arithmetic. On GLM at 4096 or more
rows it restores the former N64 default; below that it restores legacy N32.
`DS4_METAL_ENABLE_M5_GLM_MOE_MPP_N64=1` forces N64 for crossover diagnosis.

## DeepSeek-V4-Flash Vision-Exp late-Q4

Fixed-context trials used A/B/B/A order with a 45-second idle period before
every process. The control and candidate are the same executable.

| Context | Old N32 trials | Split trials | Old average | Split average | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 701.03, 805.58 | 769.08, 860.53 | 753.31 t/s | 814.81 t/s | +8.16% |
| 16,384 | 685.35, 683.11 | 710.48, 715.21 | 684.23 t/s | 712.85 t/s | +4.18% |

A single-order long-context screen measured +3.58% at 32K and -1.49% for the
next 32K at a 64K frontier. At that point long-context attention dominates and
thermal order is comparable to the remaining kernel delta, so neither result
is promoted as a universal 64K claim.

Both full 100K paths completed. Split measured 459.02 t/s and the subsequent
heat-soaked rollback measured 356.85 t/s. That pair is retained only as a
completion/stability check: the second four-minute run was thermally
confounded, so the apparent +28.6% is not attributed to this kernel.

Correctness was bit-exact:

- eight full-vocabulary frontier dumps from 2K through 16K, or 1,034,240
  logits, were byte-identical;
- 32 greedy decode steps, or 4,136,960 logits, were byte-identical.

The decode rate remained effectively unchanged, as expected: the new path is
used during multi-row prefill, not one-token decode.

## GLM-5.3-Flash Q2/Q4K

At 2K, GLM cannot use its former 4K-gated N64 specialization. The balanced
legacy-N32 versus split-N32 test measured 530.42 versus 568.16 t/s, a +7.12%
gain.

At full 4K waves, split-N32 was compared with both the former N64 default and
legacy N32 in forward/reverse order:

| Frontier | N64 average | Split-N32 average | Gain vs N64 | Legacy-N32 average | Gain vs legacy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 511.49 t/s | 518.66 t/s | +1.40% | 414.68 t/s | +25.07% |
| 8,192 | 438.79 t/s | 443.08 t/s | +0.98% | 367.83 t/s | +20.46% |
| 12,288 | 428.61 t/s | 433.72 t/s | +1.19% | 370.11 t/s | +17.19% |
| 16,384 | 413.19 t/s | 415.73 t/s | +0.61% | 367.77 t/s | +13.04% |

Split-N32 therefore becomes the M5 default. N64 had already recovered most of
the old N32 deficit, which is why the net 4K+ gain over the shipped default is
only about 0.6-1.4%, not the much larger gain visible against legacy N32.

Correctness was again bit-exact: four 4K-through-16K full-vocabulary dumps
(619,520 logits) matched N64 byte-for-byte, and eight post-2K decode dumps
(1,239,040 logits) matched legacy N32 byte-for-byte.

## Why the earlier GLM prefill attempt did not ship

PR #953's batched-router idea measured roughly +4-8%, but it failed the model
output gate badly on the current GLM quant: its first comparison changed the
argmax from token 2195 to 45152, matched 0 of 154,880 logits exactly, and had
0.8589 RMS error. It remains rejected.

The exact split kernel was a different issue. A lookup-table-only screen was
neutral, and the complete split kernel was initially hidden at 4K+ by the
existing N64 routing policy. Three-way profiling and full-model A/B showed that
the complete split dataflow works on GLM, while preserving exact outputs. The
small advantage over N64, rather than the invalid router result, is what is
retained.

## Validation

- Release build completed without warnings from these changes.
- Metal kernel suite passed.
- DeepSeek and GLM runtime route diagnostics selected the intended split path.
- The rollback selected old N32 for DeepSeek and former N64/old N32 for GLM.
- All output comparisons listed above were byte-identical.

The complete `make test` run passed the session snapshot, 30K long-context,
tool-call, Metal short-prefill, Metal kernel, tensor-equivalence, server, and
other non-golden checks. It ended with the same three model-dependent golden
failures present before this port: `ds4flash.gguf` currently points to the
Vision-Exp late-Q4 checkpoint rather than the repository's reference golden
fixture, so the logprob vector, SSD-streaming vector, and local top-5 golden
checks do not match that fixture.
