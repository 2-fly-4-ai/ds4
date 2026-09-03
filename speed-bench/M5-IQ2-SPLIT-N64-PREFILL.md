# M5 IQ2 split-N64 prefill campaign

Measured 2026-09-03 on a MacBook Pro M5 Max with 128 GiB RAM. macOS remained
in Automatic power mode; the user forced maximum fan speed for the balanced
performance runs to reduce thermal-order noise. This branch starts from the
already-proven split-N32 implementation in `009ed43` and does not merge an
upstream branch.

## Retained implementation

The split-N64 Metal kernel keeps two independent 16-row cooperative
accumulators per two-SIMDgroup pair. The pair consumes two 32-row banks while
sharing one dequantized expert-weight tile between them. A complete 64-row
activation tile and the weight tile are double-buffered in 16 KiB of
threadgroup memory. Compared with issuing two split-N32 work items, this halves
the work-item count and IQ2/Q2 weight dequantization work for a full tile.

The specialization is selected only for resident, single-GPU, non-quality M5
prefill using IQ2_XXS gate/up experts, a Q2_K or IQ2_XXS down projection, top-6
or top-8 routing, and at least 1024 input rows. Split-N32 remains selected for
32--1023 rows. SSD streaming, tensor parallelism, graph dumps, other hardware,
and other quant layouts retain their prior routes.

`DS4_METAL_DISABLE_M5_IQ2_SPLIT_MPP_N64=1` is the N64-only same-binary
rollback. `DS4_METAL_ENABLE_M5_IQ2_SPLIT_MPP_N64=1` forces N64 down to 64 rows
for crossover diagnosis. The existing
`DS4_METAL_DISABLE_M5_IQ2_SPLIT_MPP=1` remains the aggregate rollback for both
split specializations.

## Kernel-stage screen

At a 4096-row wave, layer-7 routed-MoE profiling measured:

| Model and stage | Split-N32 | Split-N64 | Change |
| --- | ---: | ---: | ---: |
| DeepSeek gate | 12.664 ms | 11.138 ms | -12.05% |
| DeepSeek up | 12.743 ms | 11.193 ms | -12.16% |
| DeepSeek down | 11.396 ms | 9.865 ms | -13.43% |
| GLM gate | 17.889 ms | 15.270 ms | -14.64% |
| GLM up | 18.050 ms | 15.773 ms | -12.62% |
| GLM down | 16.521 ms | 13.461 ms | -18.52% |

Map, activation, and sum stages are not accelerated; the whole-model gain is
therefore smaller than the matrix-stage gain.

## Balanced full-model results

The 2K, 4K, and 4K-to-16K comparisons used A/B/B/A ordering with identical
binaries and equal cooldowns. A is split-N32; B is split-N64.

| Model | Prefill interval | Split-N32 trials | Split-N64 trials | A average | B average | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Vision-Exp late-Q4 | 2,048 | 756.91, 758.13 | 784.87, 783.46 | 757.52 t/s | 784.17 t/s | +3.52% |
| DeepSeek Vision-Exp late-Q4 | 4,096 | 701.37, 700.42 | 728.31, 730.48 | 700.90 t/s | 729.40 t/s | +4.07% |
| DeepSeek Vision-Exp late-Q4 | 4K to 16K | 620.34, 617.00 | 642.29, 641.64 | 618.67 t/s | 641.97 t/s | +3.77% |
| GLM-5.3-Flash Q2/Q4K | 2,048 | 535.77, 534.07 | 554.83, 553.83 | 534.92 t/s | 554.33 t/s | +3.63% |
| GLM-5.3-Flash Q2/Q4K | 4,096 | 500.19, 500.90 | 522.40, 520.66 | 500.55 t/s | 521.53 t/s | +4.19% |
| GLM-5.3-Flash Q2/Q4K | 4K to 16K | 410.31, 408.72 | 424.08, 424.39 | 409.52 t/s | 424.24 t/s | +3.59% |

The low-row crossover sweep used the same mirrored ordering:

| Model | Actual prefill rows | Split-N32 average | Split-N64 average | Change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek | 512 at position 0 | 595.31 t/s | 594.13 t/s | -0.20% | keep N32 |
| DeepSeek | 512 at position 512 | 597.87 t/s | 593.70 t/s | -0.70% | keep N32 |
| DeepSeek | 1,024 | 684.45 t/s | 695.83 t/s | +1.66% | select N64 |
| GLM | 512 at position 0 | 445.75 t/s | 446.65 t/s | +0.20% | neutral; keep N32 |
| GLM | 512 at position 512 | 458.41 t/s | 457.70 t/s | -0.15% | keep N32 |
| GLM | 1,024 | 490.48 t/s | 500.01 t/s | +1.94% | select N64 |

This establishes a shared 1024-row crossover without model-name special cases.

## Correctness and validation

The automatic route and N64-only rollback produced byte-identical outputs for
both models:

- five full-vocabulary frontier dumps per model at 1K, 2K, 4K, 8K, and 16K;
- 646,400 compared DeepSeek frontier logits and 774,400 compared GLM frontier
  logits;
- eight post-2K decode steps per model, covering 1,034,240 DeepSeek logits and
  1,239,040 GLM logits.

The release build completed cleanly. Automatic route diagnostics selected
split-N32 at 512 and split-N64 at 1024 for both top-6 and top-8 routing. The
N64-only rollback selected split-N32 at 1024. The full test suite passed its
Metal kernel, Metal tensor-equivalence, 30K long-context, session snapshot,
tool-call, agent, server, and utility checks. It ended with the same three
model-fixture golden failures present before this change because
`ds4flash.gguf` points to the newer Vision-Exp late-Q4 checkpoint rather than
the repository's reference golden model.
