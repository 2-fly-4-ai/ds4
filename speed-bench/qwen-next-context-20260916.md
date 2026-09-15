# Qwen context-capacity validation — 2026-09-16

Hardware: Apple M5 Max, 128 GiB unified memory, macOS 26.6.2. Runtime was the
current `main` build. The installed quality Q4 Qwen3.8 Flash Next model and its
external PLE sidecar were used. No model files or quantization were changed.

## Qwen3.8 Flash Next Q4

One growing `ds4-bench` session used teacher-forced 64-token decode at each
frontier. Teacher forcing deliberately excludes MTP and prompt lookup so the
decode result is the base-kernel floor, not a draftability-dependent peak.

| Context | Newly prefilled | Prefill | Base decode |
|---:|---:|---:|---:|
| 65,536 | 65,536 | 841.31 t/s | 39.56 t/s |
| 131,072 | 65,536 | 692.21 t/s | 36.85 t/s |
| 262,144 | 131,072 | 652.29 t/s | 35.07 t/s |

The 266,240-token diagnostic allocation planned 89.17 GiB total: 69.73 GiB
resident model, 8.46 GiB KV and 10.98 GiB buffers. The completed run recorded
zero swap-ins and zero swap-outs. Qwen reports 262,144 as its native context;
larger contexts require explicit YaRN scaling and were not enabled here.

Decision: the supervisor's Qwen Next and Qwen Next Vision capacities are
262,144. The live 262,144-token server plans 88.97 GiB.

## Dense Qwen 27B and 35B

An 8K prompt was attempted with a larger allocation for both installed dense
Qwen profiles. Both fail before prefill with:

```
Qwen prompt length 8192 exceeds dense Metal cache 4096
```

This is not a model-memory limit. The 27B Q4 probe planned 14.59 GiB at a
69,632-token allocation; the 35B Q8 probe planned 35.36 GiB at a 16,384-token
allocation. The current dense-Qwen runner hard-codes `g_qwen_pool.max_ctx` to
4096 and sizes its Metal K/V tensors from that constant.

Decision: keep the dense 27B/35B supervisor values at 4096 until the global
Metal pool becomes context-sized and passes long-prefill, decode, rewind, MTP,
and repeated-session correctness tests. Merely changing the profile number is
known to create a broken endpoint.

## Existing DeepSeek V4 and GLM evidence

The same machine previously completed fully resident DeepSeek V4 and GLM 5.3
sweeps through 262,144 tokens across code, prose and structured corpora. At the
262K frontier, base decode was 20.47 t/s median for DeepSeek and 27.79 t/s for
GLM; cumulative cold prefill was about 12.2 and 12.3 minutes respectively.
Those measured capacities now replace the supervisor's former arbitrary 100K,
40K and 32K values. DeepSeek V4.1 remains at its separately validated 16K SSD
streaming configuration.
