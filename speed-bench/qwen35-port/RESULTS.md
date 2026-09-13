| Model / task | Prompt tokens | Off decode t/s | On decode t/s | Exact output | Repeat exact | Cached TTFT ms |
|---|---:|---:|---:|---|---|---:|
| 35B Q8 short-code | 69 | 83.62 | 95.00 | True | True | 42.7 |
| 35B Q8 short-story | 70 | 83.56 | 63.70 | True | True | 41.7 |
| 35B Q8 short-reasoning | 87 | 82.56 | 94.16 | True | True | 41.6 |
| 35B Q8 short-json | 73 | 86.01 | 105.29 | True | True | 41.0 |
| 35B Q8 short-edit | 450 | 68.92 | 84.19 | True | True | 47.1 |
| 35B Q8 2048-code | 1368 | 47.58 | 71.14 | True | True | 58.6 |
| 35B Q8 2048-story | 2034 | 39.30 | 41.87 | True | True | 67.6 |
| 35B Q8 2048-reasoning | 2016 | 39.30 | 63.26 | True | True | 65.0 |
| 35B Q8 2048-json | 2037 | 40.37 | 78.51 | True | True | 66.5 |
| 35B Q8 2048-edit | 2169 | 38.02 | 62.09 | True | True | 67.4 |
| 27B Q8 short-code | 69 | 16.39 | 36.92 | True | True | 104.5 |
| 27B Q8 short-story | 70 | 16.48 | 21.28 | True | True | 104.1 |
| 27B Q8 short-reasoning | 87 | 16.41 | 31.71 | True | True | 106.9 |
| 27B Q8 short-json | 73 | 16.65 | 31.18 | True | True | 105.0 |
| 27B Q8 short-edit | 450 | 15.47 | 33.88 | True | True | 113.9 |
| 27B Q8 2048-code | 1368 | 13.52 | 29.62 | True | True | 134.9 |
| 27B Q8 2048-story | 2034 | 12.24 | 16.92 | True | True | 147.8 |
| 27B Q8 2048-reasoning | 2016 | 12.22 | 28.42 | True | True | 143.8 |
| 27B Q8 2048-json | 2037 | 12.38 | 30.17 | True | True | 140.7 |
| 27B Q8 2048-edit | 2169 | 12.11 | 25.53 | True | True | 144.5 |
| 27B Q4 short-code | 73 | 23.51 | 41.03 | True | True | 90.1 |
| 27B Q4 short-story | 74 | 23.29 | 24.86 | True | True | 86.7 |
| 27B Q4 short-reasoning | 91 | 23.21 | 39.75 | True | True | 89.4 |
| 27B Q4 short-json | 77 | 24.15 | 43.36 | True | True | 91.2 |
| 27B Q4 short-edit | 454 | 22.12 | 39.05 | True | True | 97.4 |
| 27B Q4 2048-code | 1372 | 18.03 | 36.31 | True | True | 118.2 |
| 27B Q4 2048-story | 2038 | 15.94 | 19.19 | True | True | 131.2 |
| 27B Q4 2048-reasoning | 2020 | 15.93 | 31.90 | True | True | 125.6 |
| 27B Q4 2048-json | 2041 | 16.33 | 35.05 | True | True | 124.1 |
| 27B Q4 2048-edit | 2173 | 15.70 | 29.29 | True | True | 127.8 |

Decode here is client-observed (completion tokens minus one)/(wall time minus TTFT). Single-run measurements, not confidence intervals. Output is capped at 256 tokens. Cold and repeat requests are paired; cache gains are not decode gains.

Cold/repeat output equality with positive cache reuse: 60/60 pairs.

## Regression output checks

| Run | Completed requests | Exact against archived baseline | Median decode t/s |
|---|---:|---:|---:|
| qwen27-q8 | 20/20 | 10/10 | 14.46 |
| qwen27-q4-64a | 20/20 | 10/10 | 19.68 |
| qwen-next-q4 | 5/5 | 5/5 | 62.20 |
| glm53 | 5/5 | 5/5 | 40.12 |
| ds4-0731 | 5/5 | 5/5 | 41.94 |
| ds4-vision-exp | 5/5 | 5/5 | 41.87 |
| ds41-ssd-safe | 5/5 | 5/5 | 12.79 |

## Rebuilt main smoke checks

- qwen27-q8 main35-verified: exact candidate/cold/repeat output; positive cache reuse; PASS.
- qwen27-q8 main27-verified: exact candidate/cold/repeat output; positive cache reuse; PASS.
- qwen27-q4-64a main27-verified: exact candidate/cold/repeat output; positive cache reuse; PASS.
