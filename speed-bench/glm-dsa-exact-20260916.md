# M5 GLM-5.3 exact DSA attention port, 2026-09-16

Source: [antirez/ds4 PR #964](https://github.com/antirez/ds4/pull/964),
phased exact indexed-attention kernels only. This is not a wholesale PR merge.
The fork retains its existing prefill, MTP, prompt lookup, quantization, and
other decode optimizations. The new path is selected only for GLM-5.3 on M5
Metal, F16 compact KV, Q8_0 V, no RoPE tail, at least 128 selected rows. Set
`DS4_METAL_DISABLE_M5_GLM53_DSA_EXACT=1` to use the old path.

Hardware/model: M5 Max 128 GiB, Automatic power, local
`GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf`. Isolated server per arm,
`--metal --ctx 16384`; same executable, model, requests, and host. Each API
request produced 256 tokens, no thinking, seed 1234. “2K” and “8K” below
refer to prompt fixtures with 1,317 and 7,641 actual prompt tokens.

| Mode | Prompt | Old decode t/s | New decode t/s | Gain | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| MTP off, temperature 0 | 1,317 | 31.24 | 33.83 | +8.3% | same SHA-256 |
| MTP off, temperature 0 | 7,641 | 28.88 | 31.59 | +9.4% | same SHA-256 |
| MTP on, temperature 0, ABBA arm means | 1,317 | 33.93 | 35.90 | +5.8% | same SHA-256, MTP schedule |
| MTP on, temperature 0, ABBA arm means | 7,641 | 29.17 | 30.93 | +6.0% | same SHA-256, MTP schedule |
| MTP on, temperature 0.7, one seeded pair | 1,317 | 33.32 | 34.16 | +2.5% | same SHA-256, MTP schedule |

The ABBA sequence was old/new/new/old. Heat noticeably lowered throughput
throughout the run (for example old 8K fell 31.59 to 26.74 t/s), so the arm
means are a bounded screen, not a universal expected gain. Prefill was not
improved by this decode-only change, and its rates also drifted with heat.
Wall time includes prefill and therefore may move differently from decode t/s.

Validation: `make test-glm53-kda` passed. A new GPU test compares exact versus
generic outputs byte-for-byte at 128, 513, and 2,051 selected rows, with
invalid/sentinel selections. Full-model outputs matched byte-for-byte in every
reported greedy and sampled arm; MTP accept/reject/skip counts matched too.
`make test` reached `ds4_test` but cannot complete in this checkout because
its default golden `ds4flash.gguf` is absent; a direct `ds4_test` attempt
reported the same missing file. No result is claimed for that suite.

Other upstream candidates checked: PR #920's final MTP snapshot fusion is
mostly superseded by the fork's out-of-place KDA shadow/no-replay transaction,
and its full branch was not suitable to import. PR #892's wider MTP depths
were slower than width two in the author's M5 data. The standalone top-eight
router specialization and BF16 HC fusion from #964 did not improve this Q2/Q4
model in local isolation; the latter is not selected for its Q4_K HC weights.
