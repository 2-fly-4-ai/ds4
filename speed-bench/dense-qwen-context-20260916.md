# Dense Qwen context-capacity fix — M5 Max 128 GB

Date: 2026-09-16  
Platform: Apple M5 Max, 128 GB, macOS Automatic power mode

## Defect and fix

The dense Qwen Metal runner allocated its base and MTP KV pools with a fixed
4,096-token capacity even when the frontend accepted a larger `--ctx`. An
8,192-token Qwen 27B prompt therefore failed with:

```
Qwen prompt length 8192 exceeds dense Metal cache 4096
```

The pools now take their capacity from the engine's requested context before
the first session allocates them. Pool offsets remain immutable while an
engine is live and are reset when the engine closes. The MTP reset count and
fallback capacity were also made safe for the larger allocation.

Startup memory reporting now includes the process-global dense-Qwen base and
MTP KV pools, which were previously absent from the reported total.

## Correctness

At the original 4K context, the pre-change and candidate binaries produced
byte-identical frontier logits and byte-identical logits for every generated
token on both installed dense Qwen families:

| Model | Baseline decode | Candidate decode | Logit result |
|---|---:|---:|---|
| Qwen 3.8 27B Q4_64A | 24.59 t/s | 24.71 t/s | byte-identical |
| Qwen 3.6 35B-A3B Q8 | 75.85 t/s | 75.48 t/s | byte-identical |

The timing changes are ordinary run-to-run noise; the context fix does not
alter the established 4K numerical route.

The whole-model runtime test passed for both models, including session
ownership, rewind/restore, repeated continuation, engine-close cleanup, and
MTP draft/accept parity. The dense Qwen runtime regression now additionally
asserts that the Metal pool capacity follows the engine context.

Kernel/shape validation also passed:

- Qwen 35B GGUF contract (`41` layers, `30` GDN, `10` full-attention)
- Qwen 27B/35B GDN double-reference suite
- Qwen 35B routed-MoE oracle suite
- model-supervisor profile/command/artifact tests

## Long-context measurements

These are fresh, uncached API/benchmark runs after the fix:

| Model | Prompt | Prefill | Decode | Result |
|---|---:|---:|---:|---|
| Qwen 3.8 27B Q4_64A | 8,192 | 65.11 t/s | 11.82 t/s | completed |
| Qwen 3.8 27B Q4_64A | 12,487 | 58.71 t/s | 10.75 t/s | completed in 216.98 s |
| Qwen 3.6 35B-A3B Q8 | 7,520 | 228.53 t/s | 32.74 t/s | completed in 33.92 s |

The decode figures are context-dependent and are not comparable to short,
warm “vanity” decode results. Their purpose here is to prove correct operation
beyond the former 4K boundary.

## Capacity selected for the Pi GUI supervisor

Both model files declare a native 262,144-token context in GGUF metadata.
Production settings are constrained by measured unified-memory use:

| Profile | Configured context | Measured/planned memory evidence |
|---|---:|---|
| Qwen 3.8 27B Q4_64A | 131,072 | same 65 GiB KV pool as Q8, with smaller 14.32 GiB weights |
| Qwen 3.8 27B Q8 | 131,072 | 64 GiB base KV + 1 GiB MTP KV; 92.31 GiB total plan; zero swap |
| Qwen 3.6 35B-A3B Q8 | 262,144 | 40 GiB base KV + 1 GiB MTP KV + 35.19 GiB weights; zero swap |

The 27B models deliberately stop at 128K in the default picker because their
FP32 per-layer KV layout would require about 130 GiB of KV alone at 262K. The
35B MoE model has a much smaller KV layout and safely allocated its full native
262K window on this machine.

