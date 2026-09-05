# Qwen M5 pipeline promotion — 2026-09-05

Baseline: main `2cd6b15`. Research checkpoint: `99dac52`.
Model: unchanged `Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf`.
Hardware: M5 Max, 128 GiB; macOS Automatic.

## Audit of today's wins

Already on main: `b983e4e` (Q4 fast-pack + embedded MTP), `b85470c`
(guarded Qwen lookup), `2cd6b15` (fused verifier widths). Earlier DeepSeek
and GLM work remains in main's ancestry. No other measured unmerged win
was found in today's local branches/reports.

The rectangular 16x64 expert tile lost 11.48% and stays rejected. The GPU
top-1 experiment lost about 0.5–1% and is not promoted. Full checkpoint
restore was measured as useful, but that mechanism already existed; an
automatic checkpoint-placement policy has not been implemented.

## Promoted mechanisms

1. Adapt seven → fifteen lookup drafts after eight consecutive full matches;
   reset on a partial match or no lookup candidate. Keep the small-batch
   numerical routes through sixteen rows, including partial replay.
2. Restore the anchor and replay an accepted suffix in one batch instead
   of repeated one-token forward calls.
3. Store the Q4_0 expert intermediate in half precision, matching the
   conversion the down projection already performed. Both dispatches now
   receive a single explicit format choice; mixed pairs keep FP32.

The wider/replay default is M5 + Q4_0 target-trunk gated. The embedded MTP
layer is not included in the trunk eligibility check. Other Qwen quants
retain the seven-draft route. Half intermediates are selected per supported
layer. These changes are in the shared engine, not only the API.

No model re-quantization, unrelated DeepSeek/GLM routing change, automatic
MTP enablement, or sampled prompt-lookup policy change is included.
`DS4_QWEN_PIPELINE_DISABLE=1` selects the old pipeline for A/B checks.
Normal use needs no new enabling flags.

## Original controlled measurements

Greedy, nonthinking; ABBA, two observations per side. Short-suite capacity
4096 with maximum 512 output tokens, allowing EOS. Longer prompts use a
32768 capacity and 256-token output limit. Generation rate is engine timing,
not complete request throughput.

| Workload | Before t/s | After t/s | Gain |
|---|---:|---:|---:|
| Repeated passage | 119.92 | 145.53 | +21.35% |
| Code edit | 92.92 | 95.64 | +2.92% |
| JSON cases | 60.27 | 61.57 | +2.15% |
| CSV edits | 60.50 | 65.41 | +8.12% |
| Fresh code | 50.20 | 50.23 | Neutral |
| Original story | 50.21 | 50.21 | Neutral |
| Copy, 8,988 input tokens | 111.71 | 136.82 | +22.48% |
| Edit, 9,141 input tokens | 85.08 | 89.69 | +5.42% |
| Copy, 30,279 input tokens | 108.94 | 131.44 | +20.66% |

Isolated half-intermediate prefill: 1021.39 → 1056.20 t/s (+3.41%).
Combined-candidate 30K prefill: 1020.61 → 1048.49 t/s (+2.73%).
The copying figures do not characterize original prose generation.

Separate native-MTP-enabled comparisons: copy 118.22 → 142.38 t/s;
edit 95.77 → 97.95; fresh code 66.03 → 66.03. MTP remains compatible,
but lookup gains must not be presented as neural-MTP gains.

Original final CLI comparisons all matched output bytes (26 pairs); API
stream/stop/next-request checks matched four pairs. This is tested parity,
not a universal byte-equivalence or model-quality claim.

## Promotion validation

The cleanup adds mixed Q4_0/Q8_0 tests in both directions, a shared-format
predicate test, and GDN widths 9–16 with the verifier route explicitly active.
The downstream half-intermediate comparison covers 947,200 exact float
values. The expanded Qwen kernel suite passed. All five Metal frontends built,
and the CPU-only core compiled successfully.

The promotion suite produced 26 matching CLI output comparison pairs across
seven short workloads, three longer prompts, and three MTP-enabled cases.
After narrowing dispatch scope to the supported Qwen session, eight further
CLI comparison pairs matched. Final API streaming/stop/next-request checks
matched all four pairs; DeepSeek Vision-Exp late-Q4 and GLM hybrid-Q4 32-token
cross-model smokes matched the pre-promotion main build. These are smokes,
not a new exhaustive DeepSeek/GLM quality or speed suite.

| Promotion retest | Before t/s | Default-on t/s | Gain |
|---|---:|---:|---:|
| Copy, final scoped build | 119.52 | 145.06 | +21.37% |
| JSON, final scoped build | 59.76 | 61.12 | +2.28% |
| Late edit, final scoped build | 111.75 | 126.90 | +13.56% |
| CSV edits, promotion suite | 52.82 | 57.42 | +8.70% |
| Copy / 8,988 input tokens | 111.74 | 133.58 | +19.55% |
| Edit / 9,141 input tokens | 87.49 | 90.02 | +2.89% |
| Copy / 30,279 input tokens | 110.63 | 130.71 | +18.15% |

Combined-default 30K prefill: 994.36 → 1030.64 t/s (+3.65%).
MTP-enabled ABBA: copy 117.64 → 141.94 t/s; edit 95.19 → 97.28;
fresh code 65.58 → 65.55 (neutral). Fresh story ABBA was also neutral.

Some JSON/fresh-code promotion timings drifted substantially between control
runs with concurrent desktop activity. Do not interpret their aggregate
differences as extra wins; the final scoped JSON repeat was stable, and no
new fresh-code speedup is claimed. The final single-pair MTP coding smoke
also had a cold/slow control; use the earlier ABBA for speed, not that pair.

An initial eligibility check mistakenly included the differently quantized
MTP layer and left the optimization disabled; that was corrected before
the complete promotion suite. No gains are claimed from that abandoned
partial run.

Raw promotion/final logs and JSON are archived locally at
`/Users/brianfarley/Documents/Codex/2026-08-18/let/qwen-pipeline-promotion-2026-09-05/`.
The local harnesses are under `speed-bench/qwen-pipeline/`; run from the
candidate repository root. They use this machine's model/fixture paths.
For cross-model checks, pass explicit baseline and candidate build directories
to `regression-smoke.sh`; comparing two copies of current main is not a
pre-change regression check.

Raw original research is preserved locally at:
`/Users/brianfarley/Documents/Codex/2026-08-18/let/qwen-pipeline-experiments-2026-09-05/`.
The experimental commit remains available independently of the production
cleanup; rejected code is not required to reproduce the promoted defaults.
