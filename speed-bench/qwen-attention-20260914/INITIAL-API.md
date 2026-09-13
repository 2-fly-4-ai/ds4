| Model | Workload | Prompt tokens | Main t/s | Candidate t/s | Gain | Exact |
|---|---|---:|---:|---:|---:|---|
| 35B Q8 | short-code | 69 | 107.17 | 108.15 | +0.9% | yes |
| 35B Q8 | short-story | 70 | 72.45 | 74.13 | +2.3% | yes |
| 35B Q8 | short-json | 73 | 118.23 | 119.68 | +1.2% | yes |
| 35B Q8 | 2048-code | 1368 | 78.13 | 90.99 | +16.5% | yes |
| 35B Q8 | 2048-story | 2034 | 46.06 | 56.41 | +22.5% | yes |
| 35B Q8 | 2048-json | 2037 | 86.87 | 102.51 | +18.0% | yes |
| 27B Q8 | short-code | 69 | 37.03 | 35.76 | -3.4% | yes |
| 27B Q8 | short-story | 70 | 21.50 | 21.43 | -0.3% | yes |
| 27B Q8 | short-json | 73 | 32.90 | 32.82 | -0.2% | yes |
| 27B Q8 | 2048-code | 1368 | 30.74 | 32.71 | +6.4% | yes |
| 27B Q8 | 2048-story | 2034 | 17.40 | 19.05 | +9.5% | yes |
| 27B Q8 | 2048-json | 2037 | 31.42 | 34.69 | +10.4% | yes |

Exact main/candidate pairs: 24; exact, positively cached repeat pairs: 24.
MTP enabled; 256-token cap. Client decode excludes TTFT. Single cold request per cell, not confidence intervals.
