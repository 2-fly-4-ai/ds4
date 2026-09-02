# GLM-5.3 exact sampled prompt lookup

Measured on an M5 Max with 128 GiB RAM in Automatic power mode using
`gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf`, resident Metal,
native API/CLI sampling, MTP disabled, and thinking disabled.

Prompt lookup uses a deterministic continuation from the session token history,
but samples every verifier row with the target model's ordinary temperature,
top-k, top-p, min-p, and RNG. A sampled disagreement restores the committed
KDA state, replays the accepted prefix, and evaluates the sampled replacement.
The feature is therefore distribution-preserving; the matched seeded A/B tests
below also produced identical output bytes.

## Results

| Workload | Temperature | Baseline | Sampled lookup | Change | Seeded output |
|---|---:|---:|---:|---:|---|
| Repeated line, API | 0.2 | 38.61 t/s | 61.74 t/s | +59.9% | identical |
| Repeated line, API | 0.4 | 38.82 t/s | 61.42 t/s | +58.2% | identical |
| Repeated line, API | 0.7 | 38.64 t/s | 58.99 t/s | +52.7% | identical |
| Repeated line, API | 1.0 | 38.63 t/s | 61.69 t/s | +59.7% | identical |
| C source reproduction, API | 0.7 | 38.84 t/s | 63.88 t/s | +64.5% | identical |
| Repeated line after 8,793-token prompt, API | 0.7 | 32.22 t/s | 47.91 t/s | +48.7% | identical |
| Repeated line, short CLI run | 0.7 | 39.30 t/s | 54.02 t/s | +37.5% | identical |
| Novel story/no match, API | 0.7 | 38.77 t/s | 39.02 t/s | +0.6% | identical |

The long-context test crosses GLM's 4,096-token full-attention work cap and
therefore exercises the compact indexed DSA/KDA verification path. Across a
mixed sampled run, the verifier also exercised partial accepts and a first-row
miss: 564 tokens were drafted, 556 matched (98.6%), with five partial passes.

Prefill is unchanged; all reported values are decode throughput. The lookup
scan remained sub-millisecond in aggregate, including the 8,793-token prompt.

## Routing and rollback

Sampled prompt lookup is automatic on supported resident single-GPU GLM-5.3
Metal sessions through temperature 1.0. It remains disabled for thinking,
tools, textual stop strings, `ignore_eos`, SSD streaming, tensor parallelism,
native MTP, and native session batching. A first-row miss backs off for 64
tokens; partial acceptance backs off for eight.

```sh
# Disable only positive-temperature prompt lookup.
DS4_GLM_PROMPT_LOOKUP_SAMPLING=0 ./ds4-server ...

# Explicitly experiment above temperature 1.0.
DS4_GLM_PROMPT_LOOKUP_SAMPLING=1 ./ds4-server ...

# Disable all prompt lookup globally.
DS4_PROMPT_LOOKUP_DISABLE=1 ./ds4-server ...
```

## Validation

- Temperatures 0.2, 0.4, 0.7, and 1.0 matched baseline content hashes.
- API and CLI paths matched their corresponding seeded baselines.
- Full acceptance, partial acceptance, sampled replacement, no-match fallback,
  and compact indexed long-context verification were exercised.
- `make -j8 ds4 ds4-server ds4_test`, `./ds4_test --server`, and
  `git diff --check` pass.
- The full test suite retains three known custom-quant golden mismatches that
  predate this feature: one logprob selection fixture, its SSD-streaming
  counterpart, and one local top-5-overlap fixture. No new failure appeared.
