# Qwen M5 focused port — 2026-09-07

Baseline: `3209636`. Source: Ivan's [PR #991](https://github.com/antirez/ds4/pull/991),
especially commits `632f34c` and `236cb2a`; inspected branch tip `c5ae2eb`.
This is a focused adaptation, not an upstream merge or a quant replacement.

Hardware: Apple M5 Max, 128 GiB, Automatic (`powermode 0`). One model process
at a time. Existing `Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf`, including its
embedded MTP and CPU-mapped PLE. No whole-file/model conversion was performed.

## Candidate disposition

| Change | Disposition |
| --- | --- |
| Two-token HC mixing | Adapted to share weights while preserving each scalar expression/reduction; exact in tested F16/F32/Q8 kernels and live runs |
| One/two-row GDN R4 | Reuse existing four-column kernel on M5; preserve the existing 3–16-row lookup verifier route |
| Accepted MTP predictor pair | Our serial encoding in one command batch; stages two separate embeddings, removes one wait without batching arithmetic |
| Large-prefill HC norm reuse | Reuse RMS across eight chunks; preserve all chunk reduction boundaries; M5, E=2560, HC=4, batches >=2048 only |
| Upstream batched predictor | Rejected: first accepted draft changes 244,928 logits, max absolute difference 2.82526e-5, despite matching argmax |
| Split EH projection diagnostic | Rejected: same draft-logit mismatch; EH projection splitting does not fix it |
| Upstream shared-activation HC pair | Rejected as written: failed exact F16 and live target-logit comparison; scalar-preserving adaptation above passes |
| M3 launch geometry / Q4_K specializations / IQ2 recipe | Not imported; different hardware or quant layout, no demonstrated win for this existing Q4_0-trunk model |

`experimental-source.patch` preserves the superseded diagnostic implementation
against the baseline. It is evidence, not a patch to apply to production.
Failed paths are removed from the candidate source, not hidden behind switches.

The retained routes are M5-gated and disabled in quality mode. The predictor
pair additionally retains the existing Q4_0-trunk pipeline eligibility gate.
There is no change to MTP enablement, sampling, draft acceptance, or the router's
lookup/MTP/fallback policy. Shared engine changes cover CLI and API callers.

## Sustained decode measurements

`serial-wide/`: 128 output tokens, eight excluded warmup sequences, then eight
timed sequences in ABBA/BAAB order, four per arm. Same initial snapshot and
sampler seed (123); identical route counts and outputs required. Throughput is
total tokens / total measured wall time, not the mean of best rates. Oracle
readbacks and snapshot restores are outside the timed interval.

These runs test the retained arithmetic/scheduling bundle before removal of
unused HC threadgroup arguments and conversion to default-on M5 dispatch.
`final-wide/` rechecks all seven workloads after that cleanup; `final-decode/`
is a fresh sustained coding timing check of the final candidate.

| Workload | Actual input tokens | Route policy | Baseline t/s | Candidate t/s | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| Original story | 40 | MTP, temperature 0 | 49.65 | 50.88 | +2.48% |
| Structured JSON | 54 | MTP, temperature 0.7 | 55.64 | 58.00 | +4.25% |
| Copy | 269 | Automatic lookup/MTP | 73.16 | 73.26 | +0.13%, neutral |
| Code edit | 422 | Automatic lookup/MTP | 64.78 | 66.43 | +2.54% |
| Coding | 8,192 | MTP, temperature 0 | 47.78 | 50.35 | +5.38% |
| Code edit | 8,192 | Automatic lookup/MTP | 62.80 | 62.58 | -0.36%, neutral |
| Coding | 32,768 | MTP, temperature 0 | 45.44 | 47.39 | +4.29% |
| Coding, MTP off | 40 | Plain, temperature 0 | 36.06 | 36.29 | +0.64%, small/neutral |

The filenames' `2048` denotes requested test context, not a 2K-token prompt:
those short prompts really contain 40–422 tokens. Longer cases are explicitly
padded to the stated token count. Actual allocated capacity is context +1024.
Fixed 128-token continuations are speed/parity probes, not task-quality scores.
There is no claim of a universal speedup, or that copying gains generalize to
novel prose. The +/- sub-percent cases are not counted as solid wins.

Earlier `wide/` lacked sustained warmups and had an unstable short-edit result
(initial baseline 77 t/s, later ~58 t/s). It is retained but superseded by the
warmed balanced run, not silently discarded as a successful result.

## Prefill

The original six-arm prefill probes (`prefill-8192.log`, `prefill-32768.log`)
passed complete state/logit comparison after prefill and a 16-token continuation.
Their timed ordering was not fully balanced against sustained drift, so they
are preliminary timing evidence only. `final-prefill.log` uses two excluded
warmups followed by balanced ABBA/BAAB order for the final candidate.

Synthetic HC normalization (`norm.log`, `final-norm-kernel.log`) is exact in
F32/F16/Q8 at 2K and 8K batches. It is roughly 1.5–2.3x faster for this kernel;
that is **not** a whole-model speedup claim.

Final balanced 8K prefill: **13,144.03 → 12,713.81 ms**, or
**623.25 → 644.34 input tokens/s (+3.38%)**. All ten prefill/state/continuation
comparisons pass. The preliminary 32K probe also passes exact state and
continuation checks; its timing is not used as the headline result.

The final sustained short-coding rerun is **41.59 → 43.65 t/s (+4.96%)**.
Absolute rates differed across this long session; do not compare a fresh run's
peak against a later baseline. The same-run balanced comparison is the claim.

## Correctness and reproduction

`tests/test_qwen_mtp_port.c` compares accepted token IDs, route and count at each
cycle, every target logit, every pending draft logit, draft/parent validity, and
the complete serialized final state against the reference route. It exercises
acceptance/rejection, sampled replacement, lookup, plain fallback, and restored
sessions. The 32K state payload comparison covers about 1.24 GB. This is strong
finite-test evidence, not a mathematical guarantee for every possible prompt.

```sh
make all tests/test_qwen_mtp_port tests/test_qwen_prefill_port tests/test_qwen_norm_port
make test-qwen4-kernels
python3 tests/run_qwen_port.py --wide 8 --label fresh-parity
python3 tests/run_qwen_port.py --variants 8 --timing --steady --label fresh-timing
HOT_CTX=8192 ./tests/test_qwen_prefill_port /absolute/path/to/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf
```

Diagnostic reference controls, used only for A/B testing:
`DS4_QWEN_HC_PAIR_DISABLE`, `DS4_QWEN_GDN_R4_DISABLE`,
`DS4_QWEN_MTP_SERIAL_DISABLE`, `DS4_QWEN_NORM_REUSE_DISABLE`.
Presence disables that new optimization; normal users do not need these flags.

## Final validation and hold on promotion

**The candidate is saved on the experiment branch, not merged into main.**
The new paths demonstrate gains and pass matched-reference checks, but the
user requested every test green before promotion. An additional existing Qwen
cold-prefill-versus-replay test is still red on both baseline and candidate.
Its threshold has not been relaxed and the test has not been disabled.

| Check | Result |
| --- | --- |
| All five frontends + CPU core compilation | Pass; one pre-existing format warning in `tests/ds4_test.c` |
| Expanded Qwen kernel suite | Pass, including exact HC and GDN state/output/snapshot checks |
| Final seven-workload full-logit/draft/state suite | 7/7 pass, short/8K/32K, sampled and greedy, lookup and neural routes |
| Final balanced 8K prefill + 16-token continuation | 10/10 exact comparisons pass |
| Qwen MTP session snapshot test | Pass, including single- and double-token cycles |
| Exact port-vs-reference forced-MTP rewind test | Pass, complete state/logits after one- and two-token rewinds |
| Streaming API, stop, repeated/subsequent requests | Pass, baseline/candidate output hashes match |
| Live cached API follow-up | Pass, same output hash, 549 cached prompt tokens and 24 new tokens on both builds |
| DS4 Vision-Exp and GLM current-quant CLI comparison | Pass, output hashes match production |
| DS4 SSD streaming, 8GB cache | Pass, output hash matches production |
| GLM continued prefill, two 2K extensions | Pass using the installed hybrid quant |
| `make -k test` with the installed dated `ds4f-q2` 0731 file | Exit 0; official vectors and local golden pass (top-20 max absolute delta 0) |
| Extra stock Qwen rewind test, MTP forced on | **Fails: same three assertions on production, candidate, and candidate with all new routes disabled** |
| Extra stock Qwen rewind test, MTP off | **Fails: same two assertions on production and candidate** |

The failing stock test compares a fresh batched prefill of the complete replay
with a rewound/re-evaluated session (`tests/ds4_test.c:202,212`). Both paths keep
the same top token, but some top-eight log probabilities exceed its 0.002
tolerance. This is not sufficient evidence to label either route correct or to
call it merely a harmless rounding issue. Its underlying discrepancy remains
unresolved. The new matched-route oracle uses the same prompt, 1024 capacity,
five generated tokens, forced acceptance, and both rewinds; it proves that this
port does not change that scenario's serialized states/logits relative to the
reference route (`rewind-port-exact.log`).

The standard `make test` run uses a DeepSeek model, so its Qwen rewind and GLM
continued-prefill entries skip; those were explicitly run above. The optional
DeepSeek MTP/DSpark depth tests and extended streaming decode/prefill oracle
were not enabled in that invocation. SSD streaming was tested independently.
CUDA/distributed hardware execution and other Apple chips are not certified;
their source paths are unchanged. No model files, power setting, production
defaults, or serving configuration were changed, and test servers are stopped.

API/cross-model reproduction needs a separate baseline build. Set
`QWEN_PORT_BASELINE=/absolute/path/to/baseline` and optionally
`QWEN_PORT_MODEL_ROOT=/absolute/path/to/model/repository`. These harnesses refuse
to compare a build with itself and preserve prior logs instead of overwriting
them. The portable C probes take the model path directly.
