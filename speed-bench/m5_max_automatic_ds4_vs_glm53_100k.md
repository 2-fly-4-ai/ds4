# DS4 vs GLM 5.3 Flash through 100K context

Measured on an Apple M5 Max with 128 GB unified memory in macOS Automatic
power mode. The unrelated MTPLX server was stopped. Both models were fully
resident; SSD expert streaming was disabled.

Models:

- DS4: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB mapped)
- GLM: `GLM-5.3-Flash-Q2-Q4K-Attention.gguf` (88.27 GiB mapped)

## Method

The parity suite used the same three source corpora for both models: Italian
prose, real DS4 C/Metal source, and varied structured JSON/SQL records. At
2,048, 4,096, 8,192, 16,384, 32,768, 65,536, and 100,000 token frontiers,
`ds4-bench` consumed 128 teacher-forced continuation tokens. This removes the
large bias caused when one model happens to generate an easy repetitive token
stream while retaining content-dependent tokenizer and expert-routing effects.

The headline decode number is the final 64-token block, not the opening burst
or whole-run average. Prefill wall time is cumulative from zero to each
frontier. Models used their own tokenizer, so a frontier is the same exact
token count and same source corpus, but not necessarily the same final byte.

## Sustained decode (final 64 tokens, t/s)

| Context | DS4 prose | GLM prose | DS4 code | GLM code | DS4 structured | GLM structured |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 45.69 | 28.98 | 40.26 | 29.85 | 37.27 | 31.18 |
| 4,096 | 41.51 | 28.60 | 31.03 | 29.34 | 34.58 | 30.45 |
| 8,192 | 41.38 | 28.70 | 35.43 | 29.03 | 34.47 | 32.12 |
| 16,384 | 39.99 | 28.22 | 31.85 | 28.63 | 34.18 | 26.54 |
| 32,768 | 38.56 | 27.53 | 27.75 | 23.49 | 32.11 | 28.41 |
| 65,536 | 30.39 | 25.31 | 28.71 | 25.16 | 27.34 | 25.77 |
| 100,000 | 26.67 | 21.90 | 25.05 | 21.91 | 25.28 | 25.02 |

DS4 won all 21 matched base-decoder cells. The cross-workload median was:

| Context | DS4 median | GLM median | DS4 advantage (ratio of medians) |
|---:|---:|---:|---:|
| 2,048 | 40.26 | 29.85 | +34.9% |
| 4,096 | 34.58 | 29.34 | +17.9% |
| 8,192 | 35.43 | 29.03 | +22.0% |
| 16,384 | 34.18 | 28.22 | +21.1% |
| 32,768 | 32.11 | 27.53 | +16.6% |
| 65,536 | 28.71 | 25.31 | +13.4% |
| 100,000 | 25.28 | 21.91 | +15.4% |

## Cumulative prefill wall time

| Context | DS4 median time | GLM median time | DS4 effective advantage |
|---:|---:|---:|---:|
| 2,048 | 3.31 s | 4.69 s | 1.42x |
| 4,096 | 7.18 s | 10.11 s | 1.41x |
| 8,192 | 14.54 s | 21.09 s | 1.45x |
| 16,384 | 29.28 s | 43.08 s | 1.47x |
| 32,768 | 62.42 s | 90.23 s | 1.45x |
| 65,536 | 140.92 s | 192.71 s | 1.37x |
| 100,000 | 242.70 s | 326.84 s | 1.35x |

At 100K, the matched per-workload results were:

| Workload | DS4 | GLM | Effective prefill rate DS4 / GLM |
|---|---:|---:|---:|
| Prose | 216.53 s | 328.29 s | 461.8 / 304.6 t/s |
| Code | 264.53 s | 326.84 s | 378.0 / 306.0 t/s |
| Structured | 242.70 s | 306.79 s | 412.0 / 326.0 t/s |

Thus DS4 saved roughly 64-112 seconds of prefill at 100K in these controlled
runs. That is the largest practical advantage in the comparison.

## GLM real-output API MTP check

The same raw continuations were replayed through `ds4-server`, temperature 0,
128 output tokens, first plain and then with embedded MTP depth 2. These are
real generated outputs rather than teacher-forced tokens.

| Context | Workload | Plain | MTP | Change | Byte-identical |
|---:|---|---:|---:|---:|---|
| 2,812 | Prose | 31.41 | 27.24 | -13.3% | yes |
| 2,447 | Code | 31.60 | 28.10 | -11.1% | yes |
| 2,340 | Structured | 32.11 | 28.24 | -12.1% | no |
| 100,507 | Prose | 21.59 | 25.86 | +19.8% | yes |
| 100,737 | Code | 21.56 | 25.20 | +16.9% | yes |
| 100,769 | Structured | 21.42 | 25.26 | +17.9% | yes |

All six raw continuations entered a thinking state. The short-context loss is
consistent with earlier findings: low MTP acceptance cannot repay verifier
overhead. At ~100K, MTP consistently recovered about 17-20% and made GLM decode
roughly competitive with DS4's 25-27 t/s base results. It did not improve
prefill; apparent API prefill differences between the plain and MTP passes are
not a controlled A/B because Automatic mode moved between thermal/fan operating
points during the hour-long sustained campaign.

## Memory and thermal behavior

- DS4 planned 82.65 GiB total and 1.89 GiB context buffers at 100,257 allocated tokens.
- GLM planned 92.49 GiB total and 4.22 GiB context buffers at the same allocation.
- GLM's median first-64 to final-64 decode drop was about 7.9% at 16K and 13.1% at 64K; DS4's was about 2.3% and 3.7%. GLM was more sensitive to sustained thermals in this suite.
- Both remained comfortably inside 128 GB and neither paged experts from SSD.

## Conclusion

For this M5 Max, DS4 is the faster base engine at every tested context and
workload. Its lead is largest at short context, narrows at long context, and it
also prefills 100K about 1.24-1.52x faster depending on content. GLM's MTP is
not an unconditional speed switch: it loses on the tested short thinking
continuations but provides a repeatable 17-20% decode gain around 100K, bringing
long-context generation close to DS4 while retaining GLM 5.3's different model
capabilities.

Raw parity CSVs are the six `m5_max_automatic_context100k_*.csv` files. The API
MTP results are in `m5_max_automatic_glm53_api_mtp_100k.csv`.
