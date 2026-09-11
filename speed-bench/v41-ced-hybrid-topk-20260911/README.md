# DeepSeek V4.1 CED hybrid indexed-attention extension — 2026-09-11

## Decision

Keep the fix and raise the opt-in Metal CED hybrid ceiling from 1,024 to 4,096
prompt tokens.  The widened layer-major encoder must preserve one top-512
indexer selection per prompt token.  The source index layer now writes each
selection directly into the already-allocated `top_k × prefill_capacity`
matrix, and its consumer layers bind the matching row through a zero-copy
tensor view.

Enable the complete path with:

```sh
DS4_METAL_V41_CED_PREFILL=1 \
DS4_METAL_V41_CED_HYBRID_PREFILL=1 \
DS4_METAL_V41_CED_STAGE_ENGRAM=1 \
./ds4-bench --metal --ssd-streaming \
  -m gguf/DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf \
  --engram gguf/DeepSeek-V4.1-Flash-Engram.gguf \
  --prompt-file tests/long_context_security_prompt.txt \
  --ctx-start 4096 --ctx-max 4096 --gen-tokens 0
```

`DS4_METAL_V41_CED_HYBRID_MAX=N` remains an experiment override for 129 through
8,192 rows.  The default stops at the largest frontier certified here: 4,096.

## Correctness bug found and fixed

The original 1,024-row limit ended immediately before indexed attention became
active.  With compression ratio two, 1,025 prompt tokens leave 512 compressed
rows and remain exact.  At token 1,025 of a longer prompt, the source layer
creates its first top-512 selection.

Token-major execution computes a selection in the shared source layer and its
consumer layers use that same token's selection.  The first widened
layer-major implementation let the source layer run across the whole prompt
while retaining only its final selection.  Earlier consumer rows therefore
observed the wrong top-k.  A layer-local dump proved layers 0 through 3 were
byte-identical and layer 4 was the first mismatch.

An initial repair copied 2 KiB of selected indices for every indexed consumer
row.  It restored exact output but erased the 2K speedup.  The retained repair
has no GPU copy: top-k writes directly to the prompt row and attention consumes
a view of that row.  It adds no persistent Metal allocation.

The final implementation matched the serial CED oracle over 36 complete
129,280-logit arrays (4,654,080 floats):

- 1,025 and 1,088 security tokens: frontier plus eight teacher-forced steps;
- 1,536 security tokens: full frontier;
- 2,048 security tokens: frontier plus eight teacher-forced steps;
- 2,048 story/prose tokens: frontier plus four teacher-forced steps;
- 4,096 security tokens: frontier plus two teacher-forced steps.

Every compared float was byte-identical.  Selected-token agreement was not
used as a substitute for full-logit parity.

## Performance

Machine: Apple M5 Max, 128 GiB, Metal, SSD expert streaming.  The 2K security
result is a clean baseline/candidate/candidate/baseline sequence without logit
dumps or generation.  Other rows are directional timings from the strict
correctness runs.

| Corpus / prompt | Serial CED | Hybrid | Gain |
|---|---:|---:|---:|
| Security, 1,088 | 22.14 t/s | 25.39 t/s | +14.68% |
| Security, 1,536 | 23.45 t/s | 27.49 t/s | +17.23% |
| Security, 2,048 balanced mean | 25.225 t/s | 29.025 t/s | +15.06% |
| Story/prose, 2,048 | 27.40 t/s | 32.23 t/s | +17.63% |
| Security, 4,096 | 28.26 t/s | 31.55 t/s | +11.64% |

The balanced 2K individual runs were 25.36 and 25.09 t/s for serial CED, and
29.11 and 28.94 t/s for the hybrid.  The percentage gain narrows by 4K as
indexed long-context attention becomes a larger fraction of wall time, but it
remains a substantial improvement.

Decode rates from the two-to-eight-token correctness runs are intentionally
not claimed: full-logit serialization and first-token/cache effects dominate
such tiny samples.  This change targets prompt processing and leaves normal
decode outside prefill unchanged.

