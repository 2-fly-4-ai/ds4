| Model | Workload | Prompt tokens | Main t/s | Candidate t/s | Gain | Exact |
|---|---|---:|---:|---:|---:|---|
| 35B Q8 | 2048-code | 1368 | 78.94 | 92.16 | +16.7% | yes |
| 35B Q8 | 2048-story | 2034 | 46.72 | 57.03 | +22.1% | yes |
| 35B Q8 | 2048-json | 2037 | 89.33 | 108.56 | +21.5% | yes |
| 27B Q8 | 2048-code | 1368 | 31.99 | 35.95 | +12.4% | yes |
| 27B Q8 | 2048-story | 2034 | 17.84 | 20.64 | +15.7% | yes |
| 27B Q8 | 2048-json | 2037 | 31.64 | 36.26 | +14.6% | yes |

Exact main/candidate pairs: 6; exact, positively cached repeat pairs: 0.
MTP enabled; 256-token cap. Client decode excludes TTFT. Single cold request per cell, not confidence intervals.
