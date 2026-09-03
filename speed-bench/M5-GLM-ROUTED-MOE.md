# M5 GLM-5.3 routed-MoE campaign

> Update (2026-09-03): the N64 path documented below remains available, but
> the later exact split-N32 specialization is now the M5 default. It measured
> 0.61-1.40% faster than N64 across balanced 4K-through-16K frontiers and
> 7.12% faster than legacy N32 at 2K. See
> [M5-IQ2-SPLIT-PREFILL.md](M5-IQ2-SPLIT-PREFILL.md).

## Prior N64 optimization

The resident GLM-5.3 hybrid quant routes eight rows per token across 256
experts. The original grouped TensorOps path always built 32-row expert work
items. At a full 4096-token prefill wave, each expert receives about 128 rows
on average, so a 64-row work item reduces map/dispatch overhead and reuses each
staged expert weight tile across twice as many routed rows.

The accepted path adds:

- a 64-row top-8 expert work-map specialization;
- matching IQ2_XXS gate/up and Q2_K-down Metal4 cooperative kernels;
- exact 64-row dispatch sizing instead of launching the old 32-row upper bound;
- an M5-only resident GLM guard at `n_tokens >= 4096`;
- rollback with `DS4_METAL_DISABLE_M5_GLM_MOE_MPP_N64`.

SSD streaming, tensor parallelism, debug graph dumps, other quant layouts, and
shorter prefill waves retain their existing kernels.

## Balanced measurements

Hardware: MacBook Pro M5 Max, 128 GB, Automatic power mode.

| Workload | 64-row result versus 32-row | Exact runs |
| --- | ---: | ---: |
| 512 code, forced crossover | -15.31% | 4 |
| 2048 code, forced crossover | +0.11% | 8 |
| 4096 code | +2.72% | 16 |
| 4096 prose | +2.74% | 16 |
| 8192 code, two 4096 waves | +3.42% | 8 |
| 4096 code, default versus rollback | +2.87% | 16 |

All 76 measured outputs, covering 11,770,880 full-vocabulary logits, were
bit-identical. The short-prompt regression establishes that this is a
large-wave specialization rather than a universal replacement.

## Rejected nearby experiments

- Fusing the two independent gate/up TensorOps operations serialized their
  cooperative matrix work and was roughly 20% slower.
- Fusing expert reduction, shared add, and HC expansion was exact but ranged
  from +1.79% at 512 code to -0.33% at 1K prose and was flat at 4K.
- Double-buffering the 32-row TensorOps kernel increased threadgroup-memory
  pressure and reduced sustained 2K performance by 0.61%.
- Explicit Metal resource-use hints reduced 1K performance by 1.58%.

The useful boundary was therefore expert-row occupancy and work-tile width,
not launch-tail fusion or extra staging.
