# DeepSeek V4.1 scalar fusions and encoder residency — 2026-09-12

## Decision

- Keep the three single-row V4.1 Metal fusions. They preserve the established
  operation order, are byte-exact, and provide a small repeatable decode win.
- Keep encoder residency opt-in while the CED layer-major experiment remains
  opt-in. It pays at 8K with a constrained SSD expert cache, but not at 512.
- Keep the certified hybrid limit at 8,192 rows. A 16,384-row routed-FFN
  command was rejected cleanly by Metal for insufficient memory. No 32K or
  64K command was submitted after that capacity boundary was established.
- Never use the rejected multi-row V4.1 Q projection. This experiment changes
  only single-row kernels and model residency/page placement.

## Scalar decode fusions

The retained kernels combine these adjacent, single-row operations without
moving them across the Q projection or another model operation:

1. Q/KV RMS normalization plus V4.1 BF16 rounding.
2. Q RoPE plus its final V4.1 BF16 rounding.
3. KV initial BF16 rounding, RoPE, window E4M3 simulation, final BF16 rounding,
   and raw-cache FP16 storage.

The existing scalar attention-output/HC fusion was instrumented and confirmed
selected on the M5 Max for the V4.1 Q8/Q8 attention-output matrices.

An all-in-one Q/KV finalizer was exact in an isolated arithmetic oracle but
changed full-model logits because it moved KV finalization before Q projection.
Production does not call it. The repaired KV finalizer stays at the original
post-Q location.

### Performance

M5 Max 128 GiB, Metal, V4.1 IQ2XXS-w2Q2K, 512-token security prompt, 64 greedy
tokens, cold SSD streaming, 16 GiB cache target. Ordering was baseline,
candidate, candidate, baseline. Baseline disables only the three new scalar
fusions.

| Metric | Baseline mean | Candidate mean | Change |
| --- | ---: | ---: | ---: |
| Prefill | 25.335 t/s | 25.560 t/s | +0.89% |
| Generation | 11.015 t/s | 11.140 t/s | +1.13% |
| Steady generation | 11.160 t/s | 11.295 t/s | +1.21% |
| First-token latency | 163.569 ms | 163.681 ms | neutral |

The per-run values are in `results.csv`.

## Full resident encoder A/B

The opt-in path maps only model spans needed by encoder layers 0..19, requests
Metal residency, and densely faults one address per VM page. After the encoder
pass completes, it restores the ordinary static decoder map before layer 20's
global-cache projection. That restore also accounts for decoder map recovery in
the measured request.

M5 Max 128 GiB, cold SSD streaming, 8 GiB cache target:

| Prompt | Layer-major | Resident encoder | Change | Resident transition |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 20.98 t/s | 20.74 t/s | -1.14% | 6.57 s |
| 8,192 | 36.69 t/s | 41.12 t/s | +12.07% | 11.99 s |

The 8K request fell from 223.304 s to 199.245 s end to end, a 10.78% latency
reduction including residency and decoder-map recovery. Encoder residency used
75.32 GiB across 64 model spans and peaked at about 97 GiB resident during the
8K encoder.

This does not approach the reported 800 t/s. Profiling showed why: the existing
hybrid still advances attention token-by-token inside each encoder layer. SSD
read time was only a few seconds; attention scheduling and compute dominate the
roughly three-minute 8K run. Reaching hundreds of prompt tokens per second
requires a new exact multi-query/shared-latent attention dataflow, not more
page-placement tuning.

## Correctness and safety

- Metal kernel regression suite: pass.
- New scalar kernel arithmetic: zero mismatches for Q norm, KV norm, Q RoPE,
  and KV finalization fixtures.
- Full V4.1 model: frontier and four greedy decode logit arrays were byte-exact
  between scalar baseline and candidate.
- Residency: frontier and all four decode logit arrays were byte-exact at both
  512 and 8K.
- The 16K capacity probe returned Metal `Insufficient Memory`; macOS remained
  healthy. It did not use the previously rejected multi-row projection path.

## Controls

- `DS4_METAL_DISABLE_V41_QKV_NORM_ROUND_FUSE=1`
- `DS4_METAL_DISABLE_V41_Q_ROPE_ROUND_FUSE=1`
- `DS4_METAL_DISABLE_V41_KV_FINALIZE_FUSE=1`
- `DS4_METAL_TRACE_V41_FUSIONS=1`
- `DS4_METAL_V41_CED_RESIDENT_ENCODER=1` (opt-in)
