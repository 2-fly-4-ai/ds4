# Focused GLM verifier port — 2026-09-06

## Decision and scope

Promote the validated short-verifier arithmetic and lookup rollback repair from
`experiment/glm-mtp-contract-20260905` (source tip `1753a39`) onto main
baseline `4f4107c`. This is a selected code port, **not an upstream or whole
experimental-branch merge**.

- Keep scalar-order BF16/Q4_K/Q8/F32 reductions through 16 verifier rows,
  decode-equivalent KDA and indexed attention, fused latent-KV cache rounding,
  and scalar-order pooled-index ranking.
- Save the live pre-block partial index-pool tail, restoring it before rejected
  lookup prefixes are replayed. Scratch is graph-owned, lazily allocated and freed.
- Remove the GLM N16 Q4 TensorOps dispatch that changed verifier arithmetic.
  Larger prefill matrix dispatch remains intact.
- Restrict the newly selected C scheduling paths to Apple Metal. Review found
  the Q8 shared scalar-row helper is a CUDA stub, so blindly copying the
  experimental choices across backends would have been unsafe.
- No GGUF/quant, CLI/API defaults, router gates, model support, or checkpoint
  serialization changes. Existing prompt lookup and opt-in native MTP remain.
- Bounded/full draft history stays **opt-in in the experimental worktree**.
  It is not added to main by this port; neither its experimental checkpoint
  trailer nor diagnostic capture hooks are promoted. Its measured editing win
  remains saved on that branch. Cost-aware routing is separate future work.

## Fresh correctness validation

| Check | Result |
|---|---|
| Short BF16/Q4 matmul scalar parity | 21/21, zero differing outputs |
| F16/F32 pool-prefix rollback matrix | 528/528 |
| Original five GLM tasks, greedy | 1,280 tokens; all pass |
| GLM copy/edit starting at 4,080 tokens, crossing 4,096 | 417 tokens; both pass |
| GLM code/copy/edit, temperature 0.7 | 673 tokens; all pass |
| Combined live GLM differential checks | **2,370 tokens, 1,138 route calls, 10 dirty checkpoint restores** |
| CLI before/after, DeepSeek/Qwen/GLM | All three output hashes identical |
| Streaming API, seven cases on each model/configuration | **42 requests; all 21 matching pairs identical** |
| Extra DeepSeek API ABBA | 28 requests; every corresponding output identical |
| GLM SSD streaming, 4 GiB expert budget, 16 generated tokens | Before/after output identical |
| GLM KDA standalone GPU tests | Pass |
| Server unit suite, including final rebuild | Pass |
| CLI GPU argument suite | 78 pass, zero failures |
| GPU argument, prompt prefix, layer pack, placement units | Pass |
| Metal CLI/server/test builds and CPU-only object build | Pass |

The live oracle checks full logits after every returned route block, completed
index-pool rows and live raw pool tails at every block, and latent KV/KDA state
after dirty restores and at the end. Greedy additionally checks token identity.
For sampled MTP, returned tokens are teacher-forced through scalar; native
accept/reject sampling consumes a different RNG stream, so same-seed text
identity is **not** asserted. This is not a sampling-distribution statistical
test or a model-quality evaluation.

The short and boundary runs were made before the final Apple-only dispatch
guard; that guard leaves their Apple behavior unchanged. Sampled/API/SSD
checks and final builds use the guarded port. No history mode was enabled.

### Known baseline issue, kept visible

The first smoke run stopped because DeepSeek's cold code response differed
from its cached repeat **on untouched main**. Raw evidence remains in
`smoke.jsonl` and `smoke/`.

The corrected regression criterion compares each same request/cache state
before/after, and separately reports cold/cached identity. DeepSeek's
cold/cached difference reproduces on both builds and in the ABBA rerun; Qwen
and GLM repeated cases match. This port does **not** fix or certify universal
cold-prefill versus cached-replay equivalence. No tolerance was relaxed in
the GLM verifier logits/state oracle.

## Performance sanity checks

Machine: M5 Max, 128 GiB; macOS Automatic power mode. Settings are recorded in
`hardware.log`, `os-version.log`, and `power-settings.log`. No power/fan
settings were changed. Models run sequentially, never concurrently.

These are short regression checks, not a new comprehensive context benchmark.
The initial sequential DeepSeek API pass varied substantially, so it was
followed by **baseline → candidate → candidate → baseline**, seven identical
requests per server, two observations per build/case:

| DeepSeek API case | Baseline mean seconds | Port mean seconds | Wall-time change |
|---|---:|---:|---:|
| code | 2.973 | 2.981 | 0.29% |
| code-repeat | 2.683 | 2.688 | 0.18% |
| story | 2.935 | 2.977 | 1.42% |
| json | 2.983 | 3.030 | 1.58% |
| copy | 3.752 | 3.674 | -2.09% |
| edit | 3.668 | 3.687 | 0.54% |
| stop | 0.817 | 0.814 | -0.43% |

The initial large slowdown did not reproduce. This does not establish
sub-percent equivalence, but shows no meaningful regression in these checks.

CLI decode readings (one 128-token-limited prompt per build, same output):

| Model | Before t/s | Port t/s |
|---|---:|---:|
| DeepSeek Vision-Exp late-Q4 | 47.84 | 47.69 |
| Qwen Next compatible Q4 | 45.10 | 49.02 |
| GLM hybrid, scalar | 39.46 | 39.18 |

Do not credit Qwen's one-pass difference to this GLM repair. The runs are
sequential and timing varies with the machine's operating point. GLM SSD
readings were 17.99 → 18.23 t/s; likewise a smoke check, not a proven speedup.

### Corrected GLM API router versus scalar

Same candidate build, scalar lookup-disabled versus existing router with
`--mtp --mtp-exact-sampling`; history off. All paired response hashes match.
These numbers measure the **whole existing router benefit**, not an
incremental gain caused by this repair. One request per cell, streaming,
4096 capacity; actual prompt lengths 20/20/22/26/259/382.

| Task | Output tokens | Scalar wall seconds | Router wall seconds | Wall-time change |
|---|---:|---:|---:|---:|
| code | 128 | 3.596 | 3.135 | -12.8% |
| code-repeat | 128 | 3.257 | 2.838 | -12.9% |
| story | 128 | 3.552 | 3.635 | 2.3% |
| json | 128 | 3.564 | 3.192 | -10.4% |
| copy | 256 | 7.629 | 5.623 | -26.3% |
| edit | 256 | 7.974 | 7.165 | -10.2% |

Wall time includes prompt processing and local HTTP streaming. It is not
decode-only t/s. Story remains slightly slower: the router is useful, not
an oracle that always chooses the cheapest route.

## Reproduction and limitations

- `make glm-verifier-kernel-test`; these no-checkpoint Metal regressions are
  also dependencies of macOS `make test`.
- `make tests/test_glm_router_parity`, then
  `ROUTER_POOL_EACH=1 ROUTER_RESTORE=1 ./tests/test_glm_router_parity MODEL PROMPT_FILE...`.
  Use `ROUTER_LENGTH=4080` for the boundary, `ROUTER_TEMP=0.7` for sampling.
- `python3 tests/verifier_port_smoke.py BASELINE_BUILD CANDIDATE_BUILD MODEL_ROOT OUTPUT_DIR`.
  Extra DeepSeek run used `VERIFIER_SMOKE_FAMILY=ds4
  VERIFIER_SMOKE_API_ONLY=1 VERIFIER_SMOKE_ABBA=1`.
- Raw logs and response text are retained beside this report.
- CPU compilation was tested, not huge CPU inference. CUDA source paths were
  protected/reviewed, not compiled or run on an NVIDIA device. No distributed,
  other-Apple-chip, or long-context performance certification is claimed.
- The official golden checkpoint is not installed, so this is not a claim that
  the full official golden-model suite passed. Installed hybrid/experimental
  checkpoints are not silently substituted as the golden oracle.
