# M5 Max prompt-lookup context matrix

Measured 2026-09-03 on an Apple M5 Max with 128 GiB RAM in macOS Automatic
power mode, branch `integration/deepseek-v4-vision-exp`, commit `9fc6f85`.
Both models were fully resident on Metal. Native MTP, SSD streaming, batching,
tools, and thinking were disabled.

The baseline set `DS4_PROMPT_LOOKUP_DISABLE=1`; the optimized set used the
current automatic defaults. Each matched pair used the same prompt, seed,
sampling controls, 256-token output limit, and 101,000-token server allocation.
The long-context workload continued a deterministic repeated sequence so that
prompt lookup had useful, verifiable work. The reported context is the API's
actual token count, not a byte-based estimate.

## DeepSeek V4 Flash late-Q4

Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AOutQ4K-L27-42-chat-v2-imatrix-0731.gguf`

Greedy, temperature 0:

| Prompt tokens | Lookup disabled | Current | Change | Output |
|---:|---:|---:|---:|:---|
| 2,051 | 47.75 t/s | 64.78 t/s | +35.7% | byte-identical |
| 15,708 | 42.64 t/s | 60.51 t/s | +41.9% | byte-identical |
| 51,001 | 32.12 t/s | 55.00 t/s | +71.2% | byte-identical |
| 99,597 | 28.12 t/s | 52.92 t/s | +88.2% | byte-identical |

Geometric-mean decode improvement across the four contexts: **1.578x
(+57.8%)**.

Fresh-server novel-text control, 39-token prompt: 49.20 versus 49.21 t/s
(effectively neutral), byte-identical.

Across the four long-context optimized requests the server recorded 143 useful
verification passes after one initial miss, 998 drafted/committed tokens,
100% acceptance, 6.98 committed tokens per pass, and 5.84 ms total lookup scan
time.

## GLM 5.3 Flash optimized Q2/Q4K

Model: `GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf`

Greedy, temperature 0, EOS ignored so every matched run emits all 256 tokens:

| Prompt tokens | Lookup disabled | Current | Change | Output |
|---:|---:|---:|---:|:---|
| 2,065 | 33.14 t/s | 47.83 t/s | +44.3% | byte-identical |
| 16,017 | 32.40 t/s | 48.26 t/s | +49.0% | byte-identical |
| 50,017 | 27.78 t/s | 38.86 t/s | +39.9% | byte-identical |
| 99,217 | 26.49 t/s | 34.84 t/s | +31.5% | byte-identical |

Geometric-mean decode improvement across the four contexts: **1.410x
(+41.0%)**.

Fresh-server novel-text control, 41-token prompt: 40.03 versus 40.04 t/s
(effectively neutral), byte-identical.

The temperature-0.7 sampled-lookup check used top-p 0.95, top-k 40, and seed
424242. It improved from 39.85 to 68.22 t/s (**+71.2%**) with byte-identical
sampled output. The optimized server recorded 54 three-token verification
passes, 162/162 drafted tokens committed, and 100% acceptance for this check.

## Interpretation

- The optimization changes decode, not prefill. Prefill remained on the same
  kernels and varied with run order and thermals, so it is not attributed to
  prompt lookup.
- DeepSeek benefits increasingly at long context on this copy-heavy workload,
  reaching 1.88x at about 100K tokens.
- GLM's gain remains substantial through about 100K but declines as compact DSA
  verification itself becomes more expensive with context.
- Novel output receives no meaningful speedup and no meaningful overhead.
- These are favorable repetition/code-agent-style workloads, not an all-prompt
  average. The optimization safely falls back on novel prose.
