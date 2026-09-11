# DeepSeek V4.1 parallel Engram gather

Measured on an M5 Max with the V4.1 IQ2XXS-w2Q2K model, its FP8 Engram
sidecar, Metal, SSD expert streaming, CED prefill, hybrid CED execution, and
staged Engram injection enabled. Each timed cell is a fresh `ds4-bench`
process with zero generated tokens. Runs use ABBA ordering.

## Result

| Prompt | Serial mean | Four-worker mean | Gain |
| ---: | ---: | ---: | ---: |
| 512 | 19.93 t/s | 22.31 t/s | +11.92% |
| 2,048 | 29.30 t/s | 34.83 t/s | +18.87% |
| 4,096 | 31.88 t/s | 37.96 t/s | +19.09% |

At 4K the profiler measured the two serial mapped-table gathers at 14.654 and
14.070 seconds (28.730 seconds total). Four workers reduced them to 4.116 and
3.979 seconds (8.101 seconds total), a 3.55x reduction in the targeted work.
Upload remained negligible. FFN time was unchanged within noise (43.608 versus
43.676 seconds), which confirms the improvement is localized to host Engram
gathering.

## Correctness

The complete frontier logit vector and every measured teacher-forced decode
vector were byte-identical between serial and parallel gathering for:

- 512-token security prompt plus four decode steps, with four global threads;
- 2,048-token security prompt plus four decode steps, with four global threads;
- 2,048-token story prompt plus two decode steps, with four global threads;
- 512-token security prompt plus two decode steps, with the default 12-thread
  process and the Engram operation capped internally at four workers.

This is safe because each worker reads immutable token history and mapped FP8
tables, then writes a disjoint output row. It does not change floating-point
operation order within a row. A separate experiment that batched Engram WKV
projection dispatches on the GPU was rejected: its tiny numerical differences
eventually changed routed-expert selection.

## Controls

Parallel gathering is the default whenever staged V4.1 Engram prefill is used
on Apple platforms. Set
`DS4_METAL_V41_CED_PARALLEL_ENGRAM_GATHER=0` to restore serial gathering.
Only the gather uses at most four active pool threads; the configured global
thread count remains unchanged for all other work and all other model paths.

The stage profiler is opt-in with
`DS4_METAL_V41_CED_HYBRID_STAGE_PROFILE=1`.
