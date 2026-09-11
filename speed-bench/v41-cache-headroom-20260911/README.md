# DeepSeek V4.1 phase-aware expert-cache headroom — 2026-09-11

## Decision

Reject increasing the Metal streaming expert cache from 8,318 to 9,086 slots
after prefill.  The larger cache reduced misses and logical expert reads during
a 512-token decode, but did not improve balanced wall time.  No runtime code
change was retained.

The experiment used the existing expert-count override from process start.  A
V4.1 CED encoder has 20 routed layers × 384 experts = 7,680 possible encoder
entries, below the ordinary 8,318-slot budget.  Consequently the candidate can
retain the ordinary prefill working set and lazily occupy the additional 768
slots during decoder replay/generation without requiring a cache flush.

## What happened

At 32 generated tokens both configurations remained below capacity and had
identical cache state: 7,678 entries, 76,802 hits, 7,678 misses, zero evictions,
and 71.17 GiB live cache.  At 128 tokens, the 9,086-slot candidate still had
only 8,281 live entries.  The larger budget therefore cannot benefit ordinary
short responses.

At 512 generated tokens the capacity difference became active:

| Metric | 8,318 slots | 9,086 slots | Change |
|---|---:|---:|---:|
| Cache hits | 189,783 | 190,069 | +286 |
| Cache misses | 9,897 | 9,611 | −286 |
| Evictions | 1,579 | 525 | −1,054 |
| Logical expert pread | 91.74 GiB | 89.09 GiB | −2.65 GiB |

Those cleaner counters did not convert into faster inference:

| 512-token generation | Baseline runs | Candidate runs | Baseline mean | Candidate mean | Change |
|---|---:|---:|---:|---:|---:|
| Overall generation | 15.32, 15.72 t/s | 14.76, 15.81 t/s | 15.520 t/s | 15.285 t/s | −1.51% |
| Steady decode | 15.73, 16.32 t/s | 15.31, 16.43 t/s | 16.025 t/s | 15.870 t/s | −0.97% |
| Final 64-token block | 16.28, 17.16 t/s | 16.14, 17.30 t/s | 16.720 t/s | 16.720 t/s | 0.00% |

Runs were ordered baseline → candidate → candidate → baseline on the same M5
Max, using the same 512-token prompt, staged-Engram CED hybrid, SSD streaming,
and 512 generated tokens.  The cache counters are deterministic across the two
runs of each configuration.

## Interpretation

The avoided logical reads are not the exposed bottleneck in this workload;
macOS file caching and existing overlap hide enough of them that another 7.12
GiB of expert residency has no measurable payoff.  This is exactly why cache
statistics alone are not an optimization result.  Retain the smaller default
budget and spend the memory on context/runtime headroom.

