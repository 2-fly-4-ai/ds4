# Qwen optional SSD expert streaming — 2026-09-07

Base: `057a234`. Isolated worktree: `ds4-qwen-expert-stream`.
Hardware: M5 Max, 128 GiB; macOS Automatic power setting left unchanged.

Promoted by focused cherry-pick to main as `8557740` (experiment `58957b8`).
All five frontends were rebuilt on main. The actual-main 80-slot interleaved
snapshot test and MTP snapshot/rewind tests passed again. No push was performed,
no inference server was left running, and the default DeepSeek model symlink
was preserved. The experiment worktree retains the large raw binary traces.

## Scope

An optional lower-memory mode for the installed
`Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf` and its external
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf`. No quant conversion or model replacement.
Resident defaults remain unchanged. The existing Q4 model is not supported by
this first streaming implementation and is rejected explicitly when requested.

The shared Metal expert cache now accommodates 512 experts per layer and ten
selected experts, independently of DeepSeek/GLM compute-kernel limits. Qwen's
decode kernels take GPU addresses of cached IQ2 gate/up and MXFP4 down weights;
their row-dot arithmetic, shared-expert calculation, and reduction are retained.
Large prefill batches retain the existing tiled kernels and map one layer at a
time. Small batches use the selected-expert cache. Cache entries are protected
until their GPU work completes.

MTP is opt-in. Its different-size-class predictor expert weights stay resident
as one extra layer (about 1.30 GiB), rather than entering the trunk cache slab.
Prompt lookup is disabled only for this new SSD mode pending dedicated tests;
the existing resident routers and defaults are unchanged.
Cold, explicit preload, and full-layer-prefix options are rejected for this
first Qwen implementation rather than silently pretending to implement them.

## Plain decode timing

2,048-token prompt, 3,072-token allocation, 256 teacher-forced tokens, MTP off.
No full-logit tracing in these timing runs. Same fixed corpus and token sequence.

| Total expert budget | Decode t/s | Final 64 t/s | Planned engine memory |
|---|---:|---:|---:|
| 8 GiB, run A | 15.88 | 17.25 | 17.03 GiB |
| 16 GiB | 21.12 | 24.44 | 25.03 GiB |
| 32 GiB | 21.66 | 26.58 | 41.03 GiB |
| 8 GiB, run B | 16.26 | 17.73 | 17.03 GiB |

The repeated 8 GiB mean is **16.07 t/s**. The 16/32 GiB entries are single
samples, not statistical confidence intervals. Cache warming explains why
end-of-run throughput can exceed the whole-run average; don't substitute the
final block for the full-run result. Prefill was 671–757 t/s in this small sweep;
no prefill speedup is claimed.

The budget includes a 1.64 GiB prefill reserve: an 8 GiB request leaves about
6.36 GiB for 3,984 dynamic expert slots. Static weights, graph scratch, and KV
are additional. Planned engine memory excludes CPU PLE pages, macOS, other apps,
and reclaimable filesystem pages. This is **not** a physical 64 GiB machine
certification or a cold-SSD throughput measurement: the macOS file cache was
left intact throughout.

A separate 8K/256-token memory-instrumented run reported 21.67 GiB planned engine
memory, 42.17 GiB maximum RSS, and 12.74 GiB peak memory footprint from macOS
`time -l`; these are different measurements and must not be conflated. That run
decoded at 14.60 t/s. In particular, the cache budget is **not** an RSS limit.

## Correctness evidence

- 128-token prompt: all 8 complete decode-logit arrays equal resident baseline.
- 2K prompt: all 64 complete arrays equal resident baseline with an 8 GiB budget.
- 8K prompt: all 64 complete arrays equal resident baseline at both 8 and 32 GiB.
- 32K prompt: all 32 complete arrays equal resident baseline with an 8 GiB budget.
- An intentionally tiny 80-slot cache forces eviction across layers. Eight
  full-logit continuation steps after snapshot restore, interleaved with another
  session's prefills, were byte-identical to uninterrupted execution.
- MTP snapshot and rewind tests passed: 16 snapshot cycles, 5 single/11 double
  cycles, 27 accepted tokens; context reuse also passed.
- Coding, story, and JSON: 48 MTP cycles each. Streamed MTP matched resident MTP
  accepted-token counts, token sequences, and full final-logit arrays per cycle.
  The traces contained 82, 83, and 90 accepted tokens respectively. These are
  comparisons against **resident MTP**, not a universal MTP-versus-plain claim.
- Existing Qwen kernel suite passed. Full `make test` passed with the repository's
  0731 golden model; model-specific tests retain their normal self-skips.
- DeepSeek and GLM's existing SSD paths each matched their pre-change baseline
  on all 64 complete decode-logit arrays in separate 2K checks.
- CLI returned `READY`. A local-only SSD+MTP API server returned `READY`, then
  `DONE`; the second turn reported 20 cached prefix tokens. The owned server was
  stopped after the test. These one-token replies are smoke tests, not TPS tests.
- Unsupported Q4, a too-small cache, and the unimplemented cold option were
  rejected with the expected errors; the final 80-slot interleave test passed.

## Existing-speed preservation

Same 2K corpus, 4K allocation, teacher-forced tokens; baseline/candidate/candidate/
baseline order. These are per-mode controls, not comparisons between models.

| Mode | Tokens per run | Before mean t/s | After mean t/s |
|---|---:|---:|---:|
| Resident Qwen Q4 | 128 | 38.09 | 38.23 |
| Resident GLM hybrid | 128 | 25.48 | 25.72 |
| GLM SSD, 8 GiB | 128 | 11.02 | 11.17 |
| DeepSeek SSD, 8 GiB | 128 | 9.58 | 9.49 |
| Resident DeepSeek, longer control | 512 | 38.28 | 38.30 |

The initial 128-token DeepSeek control was inconclusive: 31.24 before versus
28.72 after, with the first baseline starting at 36.95 t/s and ending at 30.15.
The reverse-order follow-up also showed substantial run-order movement. Rather
than hiding those results or asserting a thermal cause, the final 512-token ABBA
control was added: 38.28 versus 38.295 t/s overall, and 37.865 versus 37.775 t/s
in the final 64-token blocks. All raw runs are retained. These checks found no
material sustained decode regression; tiny differences are not speedup claims.

All major model runs were serialized. No CUDA/distributed hardware inference was
run: the new compute path is Apple-only, and existing Qwen inference is already
Metal-only. CPU whole-model inference was not used for this experiment.

## Optional invocation

From the project directory after promotion, ordinary streaming:

```sh
./ds4 \
  -m gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf \
  --ple gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --ctx 4096 --ssd-streaming --ssd-streaming-cache-experts 8GB
```

Add `--mtp` to enable the resident predictor plus streamed trunk verifier.
The server uses the same engine/session implementation and model flags.

## Reproducibility

`run.py` captures fixed-token runs and full-logit hashes. `generation.py` compares
resident/streamed MTP traces using `tests/test_qwen_stream_generation.c` linked
separately against baseline/candidate core objects. `guard.py` runs ABBA controls.
Raw binary traces remain local; JSON hashes and small CSV/log outputs are suitable
for version control. Preserve a `057a234` baseline before rerunning after promotion:
the scripts' `main` path must not silently become the candidate reference.
