# M5 Max prefill follow-up campaign

Measured 2026-09-03 on an Apple M5 Max with 128 GiB RAM in macOS Automatic
power mode. GLM tests used `GLM-5.3-Flash-Q2-Q4K-Attention.gguf`; DeepSeek
tests used the resident late-Q4 Vision-Exp model selected by `ds4flash.gguf`.
Timing comparisons used alternating ABBA/BAAB order. Candidate output had to
match the control's full-vocabulary logits bit-for-bit.

## Outcome

No speculative prefill kernel from this campaign was retained. The current
defaults were faster after sustained, order-balanced testing. The benchmark
harness now reports base-prefill, snapshot-save, and snapshot-restore timing;
this exposes a large already-shipped win from exact shared-prefix reuse.

## GLM QK-low weight reuse

Experimental two-, four-, and eight-row Q4 kernels tried to amortize each
dequantized weight tile across more query rows.

| Candidate | Control | Candidate | Delta | Exactness | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| 4-row tile | 517.70 t/s | 394.69 t/s | -23.76% | exact | reject |
| 8-row tile | 485.61 t/s | 316.71 t/s | -34.78% | exact | reject |
| 2-row tile, sustained | 480.42 t/s | 479.44 t/s | -0.20% | 1,239,040 floats exact | reject |

The two-row screen's initial +0.25% disappeared in the larger reversed-order
run. Wider kernels increased register/threadgroup pressure enough to overwhelm
the dequantization reuse. All experimental code was removed.

## DeepSeek routed-MoE row width

An opt-in 64-row map/TensorOps route initially appeared +2.72% in a short
screen, but the reversed sustained run measured 672.34 versus 659.38 t/s
(-1.93%) with 1,034,240 floats exact. A decomposition run measured the existing
32-row fused gate+up kernel 1.44% faster than separate gate/up dispatches.

The existing routed kernel operates in 32-row banks. N48 has no natural exact
mapping, N64 loses once the existing pair fusion is included, and N96 adds more
padding and pressure. This closes the N48/N64/N96 family without retaining a
runtime change.

## GLM attention-LORA reuse

The proposed multi-query K/V reuse was already present. GLM's causal prefill
calls the non-vector FlashAttention kernel with an eight-query by 64-key tile;
the K/V tile is staged once and shared across those eight query rows, and no
full attention matrix is materialized.

Keeping the eight-query tile but reducing the workgroup from eight to four SIMD
groups measured 448.55 versus 445.09 t/s (-0.77%), with all 1,239,040 logits
exact. The experiment was removed. A non-Flash fallback could not be used as a
valid performance control because its first output comparison differed in all
154,880 vocabulary logits.

## Long-context indexer score/top-k

A synchronized profile of layer 7 on a resumed 12K-to-16K, 4,096-token suffix
measured the following four-run averages:

| Stage | Average |
| --- | ---: |
| Indexer score | 25.41 ms |
| Indexer top-k | 7.08 ms |
| QK-low | 70.54 ms |
| Routed MoE | 68.31 ms |

Score plus top-k is about 14% of this profiled DSA layer, but only 11 of 45 GLM
layers use the DSA indexer. Even eliminating both stages entirely would cap the
whole-prefill improvement near 3-4% at this context; a real kernel can recover
only part of that.

The current score kernel shares every 128-wide key tile across eight queries.
An exact top-512 state for eight queries does not fit alongside its score tiles
in M5 threadgroup memory. Reducing to two queries fits but rereads the key cache
about four times, sacrificing more bandwidth than the avoided score write is
expected to save. Approximate local top-k was rejected because it changes model
selection. Previous exact 128/256/512 MiB score-scratch and score-tile tests
also established the current 256 MiB, 8-query by 32-key configuration as the
measured optimum.

## Exact shared-prefix reuse

The API server already supports live token/text prefix hits, in-memory rewind,
GLM prompt-boundary rewind state, and disk KV snapshots. The benchmark's
resumed-tail path uses the same exact session snapshot API.

| Model and prefix | Cold prefill | Snapshot save | Snapshot restore | Snapshot size | Avoided-prefill speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM 5.3, 4,096 tokens | 8,751.8 ms | 35.4 ms | 19.2 ms | 251.4 MB | 455x |
| DeepSeek, 4,096 tokens | 6,479.6 ms | 11.2 ms | 7.6 ms | 80.4 MB | 854x |
| GLM 5.3, 12,000 tokens | not isolated in this run | 63.7 ms | 41.2 ms | 440.6 MB | n/a |

All four restored runs per 4K model matched exactly: 619,520 GLM logits and
517,120 DeepSeek logits. The control/candidate no-op differences (-0.25% GLM,
-1.02% DeepSeek) are ordinary timing noise and do not affect snapshot timing.

This is the highest-value practical prefill optimization for repeated system
prompts, long agent histories, regeneration, and edited branches: do not
recompute a prefix that the server can restore exactly in milliseconds.

## Next priority

For one-off cold prompts, the profile still points to QK-low and routed MoE,
but simple wider row tiles have now failed. A credible next kernel project needs
a different dataflow, not another tile-size sweep. For real API workloads,
improving prefix-cache hit rate, retention, routing, and observability has much
higher expected value than the remaining low-single-digit indexer ceiling.
