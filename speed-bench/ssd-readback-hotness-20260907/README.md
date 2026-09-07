# SSD experiment — rejected for production

Base 6da89cf; focused bounded preload-hotness adaptation from upstream 91dda0b.
The earlier correctness sweep also included edcafe4 router-readback reuse, which
was removed before final timings because our GLM hybrid cannot use that path.

Full-logit comparisons: 64/64 identical for each model/configuration (results.json).
Those correctness timings include logit writes and must not be used as clean TPS.
No-dump timings are in timing-results.json. Longer ABBA measurements are in
long-hotness-results.json. See the Qwen wide RESULTS.md report for combined tables.

DS4 logical expert reads fell 3.3%, but wall-clock generation did not improve.
Do not promote the source experiment. GLM's mixed IQ2/Q2_K eight-expert layout
falls back to per-layer streaming; its selected-cache guard supports six experts
for this recipe. Increasing the budget does not supply the missing kernel.

These are internal APFS SSD engine tests with OS caching intact, not cold physical
SSD bandwidth measurements or a physical 64 GB machine test. Source experiment
and results are committed separately so only documentation can be archived on main.
