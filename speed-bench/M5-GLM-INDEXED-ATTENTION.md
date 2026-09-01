# M5 GLM 5.3 indexed-attention campaign

All measurements below used the GLM-5.3 Q2/Q4K attention checkpoint on an
M5 Max in Automatic power mode. Candidate and control runs alternated in
ABBA/BAAB order, and every accepted timing run required bit-identical
full-vocabulary logits.

## Result

No production runtime change was retained. The existing 256 MiB score tile,
8-query by 32-key score kernel, and 2048-row Q8 stripe are at the measured M5
optimum. The useful retained change is a resumed-tail benchmark mode that can
isolate a long-context suffix without recomputing its prefix for every run.

## Long-context tail measurements

The benchmark evaluated a 95,000-token prefix once, snapshotted the complete
session state, and restored it before each timed 4,096-token suffix.

| Candidate | Control | Candidate | Delta | Exactness | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| 512 MiB score scratch | 310.54 tok/s | 304.39 tok/s | -1.98% | 1,239,040 logits exact | reject |
| 128 MiB score scratch | 281.94 tok/s | 281.64 tok/s | -0.10% | 1,239,040 logits exact | reject |

The larger buffer removed the second score slice but was slower, showing that
the 256 MiB cap is useful cache/occupancy tiling rather than an accidental
limit. Halving it added launches without improving the sustained operating
point.

## Other exact experiments

| Candidate | Context region | Delta | Exactness | Decision |
| --- | --- | ---: | --- | --- |
| Dynamic use of unused score elements | 12K to 16K | +0.76% | 2,478,080 logits exact | reject: reverses at long context |
| Dynamic use of unused score elements | 60K to 64K | -0.34% | 1,239,040 logits exact | reject |
| 64-key score tile | 12K to 16K | +0.06% | 1,239,040 logits exact | reject: noise |
| 16-query score tile | 12K to 16K | -2.04% | 1,239,040 logits exact | reject |
| 4096-row Q8 stripe | 0K to 4K | +0.18% | 2,478,080 logits exact | reject: noise |

A GLM TensorOps/NAX score prototype failed the correctness gate: all 154,880
final logits differed on its first candidate run. It was removed.

## What this establishes

The long-context score path performs one score calculation per query/key pair;
the scratch slices divide query rows, not key columns. Increasing the scratch
cap therefore saves only a small number of launches and does not eliminate a
second pass over the same scores. At 95K, the wider dispatch is slower.

Further GLM prefill work should target a different algorithmic boundary, such
as a correctness-preserving fused score/partial-top-k design that avoids the
large score write, rather than retuning the existing score-row tile. The next
higher-upside M5 project is routed MoE, which dominates both GLM and DS4 layer
profiles.
