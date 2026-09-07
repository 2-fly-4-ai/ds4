# Focused Qwen / SSD follow-up — 7 September 2026

Base: `6da89cf`. Focused validated source changes are now on main as `b6c9ad0`
and `cafeeae`. Existing model and decoding defaults are unchanged. No blanket
upstream merge and no GitHub push.

## Verified results

- External Qwen PLE: published sidecar SHA-256 matches exactly. Using our current
  Q4 model with embedded versus external PLE produced **128/128 byte-identical
  full-logit files**, not merely matching sampled text. Snapshot and rewind
  passed both ordinary and MTP modes.
- CLI vision: current baseline failed both image turns with token ID -1. Focused
  model-aware message formatting fixes both image turns and both follow-up text
  turns, with MTP off and on. Answers correctly identify the INPUT/ENCODER/SUMMARY
  diagram and NORTH HARBOR ticket fields. This is a functional regression test,
  not broad vision-quality certification.
- JPEG decoder: baseline failed 31 subcases; candidate passes all 39 generated/
  fixture cases against Pillow RGB output. Includes grayscale, progressive,
  subsampling and odd image dimensions.

## Longer tool-cache tests

Existing Q4; temperature 0, thinking enabled; three fixture read_file calls and
final answer; streaming and non-streaming. These are tool *protocol* tests, not
real filesystem tool execution. Cache fix was already on main before this turn.

| Initial actual tokens | MTP | Old omitted-reasoning follow-up mean | Retained-history follow-up mean | Status |
|---:|:---:|---:|---:|---|
| 7,963 | off | 9.727 s | 1.440 s | Pass |
| 7,963 | on | — | — | Failed third tool contract; excluded |
| 33,443 | off | 38.518 s | 1.609 s | Pass |
| 33,443 | on | 40.979 s | 1.327 s | Pass |

Three completed A/B/control groups contain 96 responses. Within each retained-
history comparison, content, reasoning and tool functions match exactly against
the old server with explicit reasoning replay. Call IDs are intentionally ignored.
Only 30/25/21 new prompt tokens are processed in the three cached follow-ups.
This is avoided-prefill latency, **not a raw decode TPS multiplier**. One timing
pass, with some concurrent downloading/building: small timing deltas are not wins.

The 8K MTP failure also occurs on the old server with explicit reasoning replay:
it says it is reading README.md but emits no tool call. That excludes a clean
all-workload claim. Qwen already uses strict argmax verification here; the GLM
MTP margin option is not the cause. Model behavior versus numerical route effects
has not yet been isolated. No MTP default was changed.

## Source attribution

Focused adaptations from ivanfioravanti/ds4-metal:

- External PLE loader: `b020c9f` and subsequent Qwen branch integration.
- Native agent PLE option: `4010cc6`.
- Model-aware CLI vision turn: `948edcd`.
- JPEG decoder and regression coverage: `681b9ce`.

Separate SSD experiments: `edcafe4` (router readback reuse) and `91dda0b`
(bounded preload priorities). Unchanged small-prefill patch `d7ade9f` requires
all-IQ2 routed gate/up/down; our GLM uses Q2_K down, so it is not eligible.

Further diagnosis: our GLM hybrid's **IQ2 gate/up + Q2_K down, eight experts**
does not qualify for Metal's selected expert-cache path. The exact guard in
`glm_stream_selected_expert_cache_supported` requires six experts for IQ2/Q2_K.
The alternative address path supports all-Q2_K, all-Q4_K or IQ2/IQ2, not our
mixed eight-expert layout. Logs confirm per-layer fallback. Thus the initial
interpretation as merely an undersized/thrashing cache was incomplete: a 32GB
budget cannot enable a missing kernel. Generalizing that exact IQ2/Q2 down path
to eight experts is a concrete future streaming target; it does not speed up
our already-resident default. No such kernel was changed in this port.

## Existing-Q4 controls

Qwen kernel suite passes. Coding main-versus-port controls at ~2K/~8K/~32K
match all six hashes (MTP off/on). Across the twelve broader Q4 task/context
pairs, MTP on/off matches eleven hashes. The differing case is the 8K story,
not coding; its unchanged-main controls match the port in BOTH modes, confirming
that the on/off difference predates this port. IQ2 also matches eleven
hashes, with its difference in the 32K story. Do not claim universal MTP
byte-equivalence. Earlier commentary incorrectly attributed the Q4 difference
to coding; the exact manifest comparison corrected that attribution.

Bounded quality baseline: Q4 **7/12**, first twelve embedded GPQA/SuperGPQA/AIME
questions, temperature0, no MTP, 1536 total generation tokens with 384/192
soft/hard answer reserves. All five failed cases reached the token ceiling.
This measures that bounded budget, not the model's unrestricted ability.

Disk KV checkpoints are disabled for quant speed/quality comparisons. The native
agent now automatically isolates external-PLE configurations by resolved model,
PLE and vision file identities (path, device/inode, size, modification time).
Tests cover deterministic reuse, different model/PLE paths, changed files and
preservation of the legacy default directory. Two live IQ2 runs returned READY;
the second reused its own 1,114-token system checkpoint. Existing caches were not
deleted or moved. Server disk caches remain explicitly configured: give each
quant/PLE combination its own directory. This is not a full weights-content hash.

## Fixed-token speed comparison

M5 Max 128 GiB; existing power/fan settings left untouched. Two separate cold-
context runs per frontier, same corpus and 128 teacher-forced decode tokens,
MTP off, allocated context = frontier + 4,096. Timed runs do not dump logits.
These are means of a small sweep, with thermal/run-order variation—not precise
confidence intervals or isolated bit-width experiments.

| Actual context | Q4 prefill t/s | IQ2 prefill t/s | Q4 decode t/s | IQ2 decode t/s |
|---:|---:|---:|---:|---:|
| 2,048 | 891.94 | 849.50 | 45.54 | 42.64 |
| 8,192 | 904.39 | 850.34 | 44.44 | 42.36 |
| 32,768 | 877.16 | 829.87 | 42.22 | 41.69 |

The smaller recipe did not demonstrate a consistent speed upgrade. Q4 remains
the recommended existing option on this 128 GiB machine. For a tighter memory
budget, IQ2 is useful: resident model weights fall from **71.29 to 46.88 GiB**.
At allocated 4K, IQ2 reports **50.49 GiB planned model + graph + KV memory**.
CPU PLE working pages, macOS and other applications are additional.
This is **not a physical-64-GB fit certification**. The publisher also labels it
a [64 GB testing candidate](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2).

## Real-output API samples

Same four instructions, temperature 0/no thinking, streaming, up to 256 output
tokens, no disk KV cache, 36,864 allocated context. Actual prompts are roughly
1,850 / 7,987 / 33,487 tokens (one-token differences by task). Metric below is
`(completion_tokens - 1) / (wall_time - first_output_time)`: a client-observed
generation estimate, **not total request throughput including prefill**.
Most samples hit the 256-token cap; these are speed samples, not completed-task
quality scores. IQ2's 32K coding sample stops naturally at 204 tokens. Different
quants generate different continuations, so use the fixed-token table as well.

| Prompt target | Task | Q4 MTP off | Q4 MTP on | IQ2 MTP off | IQ2 MTP on |
|---|---|---:|---:|---:|---:|
| ~2K | Coding | 43.93 | 57.18 | 49.92 | 57.93 |
| ~2K | Story | 41.27 | 47.61 | 50.22 | 51.27 |
| ~2K | Structured JSON | 42.21 | 54.79 | 49.64 | 56.87 |
| ~2K | Reasoning | 43.45 | 55.99 | 50.36 | 59.21 |
| ~8K | Coding | 43.85 | 56.04 | 48.31 | 56.92 |
| ~8K | Story | 41.09 | 45.07* | 46.77 | 49.28 |
| ~8K | Structured JSON | 41.55 | 55.37 | 45.26 | 53.23 |
| ~8K | Reasoning | 42.90 | 55.31 | 41.87 | 55.81 |
| ~32K | Coding | 41.06 | 54.58 | 39.77 | 57.57 |
| ~32K | Story | 39.08 | 47.52 | 41.32 | 48.93* |
| ~32K | Structured JSON | 39.57 | 54.42 | 40.43 | 55.72 |
| ~32K | Reasoning | 40.71 | 56.50 | 41.12 | 56.15 |

All values t/s. *MTP-on/off content hashes differ; exclude these from strict
same-output speedup claims. Each quant matches 11/12 on/off hashes. Q4's actual
8K story controls reproduce both outputs on unchanged main. No MTP default was
changed and this report does not certify universal MTP numerical equivalence.

## Quality screen and preservation guard

Both recipes scored **7/12**, with the same seven cases passing under the stated
1,536-token budget. IQ2 took about 6 minutes; Q4 about 5 minutes. All five failed
cases hit the generation ceiling in both runs. This small budget-limited screen
is neither a full model-quality evaluation nor evidence of general equivalence.

Interleaved main/candidate/candidate/main throughput guard, current Q4 unchanged,
2K teacher-forced 512 tokens, 36,864 allocation:

| Metric | Main | Port |
|---|---:|---:|
| Mean complete decode | 44.65 t/s | 44.93 t/s |
| Mean final 64-token block | 43.58 t/s | 43.53 t/s |

No meaningful sustained slowdown observed; do not call the small difference a
new optimization. The PLE/vision changes do not modify the decode kernels.

## Integration status

Source commits on the experiment branch: `76f0f14` (PLE/vision/JPEG), `cb70ece`
(safe automatic native-agent cache namespace). Full golden-model regression,
Qwen kernel suite, all per-model state checks and native-agent integration pass.
IQ2 vision completes both image/follow-up pairs with MTP off/on. Main promotion
commits are `b6c9ad0` and `cafeeae`; the full main golden-model regression and
main JPEG regression both exit successfully. The actual main IQ2 build also
passes MTP snapshot (16 cycles), context reuse and rewind. No inference server
or benchmark remains running. Optional unavailable hardware/head
tests retain their documented skips; this is not CUDA or DSpark certification.
The SSD timing experiment is rejected, with no SSD source patch on main.

## SSD results and decision

Fixed 2K context / 64 teacher-forced tokens, no logit writes in timed runs,
downloads stopped/completed. These exercise the SSD-streaming engine on the
internal APFS SSD with the OS page cache intact; logical reads are **not a
physical cold-disk bandwidth measurement**, nor a simulated 64 GB machine.

| Model | Resident decode | Existing 8 GB streaming budget | Candidate 8 GB | Candidate 32 GB |
|---|---:|---:|---:|---:|
| GLM hybrid | 33.88 t/s | 1.30 t/s | 1.30 t/s | 1.30 t/s |
| DeepSeek late-Q4 | 44.65 t/s | 16.09 t/s | 16.19 t/s | 17.13 t/s |

The budget includes prefill reserves; it is not all live expert-cache capacity.
GLM's unsupported mixed eight-expert cache path explains its per-layer fallback.
Qwen currently lacks this explicit bounded expert-cache path; CPU PLE demand
paging is a separate mechanism and is not mislabeled as Qwen expert streaming.

The longer DeepSeek cache-priority ABBA experiment used 512 fixed tokens at 2K
and an 8 GB budget. Existing runs were 17.00 and 15.21 t/s; candidate 15.37 and 14.90 t/s.
Logical expert reads fell deterministically from 571.12 to 552.21 GiB (**3.3%**),
but wall-clock generation did not improve. **Not promoted.** Thermal/IO variation
means these runs do not establish a precise slowdown either.

The earlier correctness sweep compared all 64 full-logit files for resident,
existing 8 GB, candidate 8 GB and candidate 32 GB: all matched for both models. That
sweep included the inactive readback patch; final timing isolated hotness only.
`edcafe4` is left out because our GLM quant never reaches its intended path.

## Download and cleanup

IQ2 main: `50,343,093,376` bytes; SHA256
`31f1e193771a5f3fdaa7af5865e0417513e8e8cac37579d388052ca54c889d4b`.
PLE sidecar: `32,000,157,440` bytes; SHA256
`66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`.
Both match the pinned published files. The sidecar reuses identical installed
PLE tensor bytes plus downloaded metadata; the full-file hash verifies the
reconstruction. The rejected first reconstruction and unused PLE partial are
recoverable in `/Users/brianfarley/.Trash/ds4-iq2-test-20260907/`. No existing
DeepSeek, GLM or Qwen quant was deleted or overwritten.

Optional IQ2 launch from the main repository (existing defaults unchanged):

```sh
./ds4 -m gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf \
  --ple gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --ctx 32768
```

Add `--mtp` only if wanted; it remains opt-in. Add the installed Qwen vision
encoder with `--vision gguf/mmproj-Qwen3.8-Flash-Next-F16.gguf` for image turns.

## Model recipe inventory (local GGUF metadata)

Across 1,255 common tensor names, 217 have different storage types:

| Tensor group | Count | Our current Q4 | Downloaded IQ2 recipe |
|---|---:|---|---|
| Trunk expert gate | 48 | Q4_0 | IQ2_XXS |
| Trunk expert up | 48 | Q4_0 | IQ2_XXS |
| Trunk expert down | 48 | Q4_0 | MXFP4 |
| SSM alpha control projection | 36 | Q8_0 | F32 |
| SSM beta control projection | 36 | Q8_0 | F32 |
| MTP expert down | 1 | Q8_0 | MXFP4 |

The current Q4 also embeds the PLE table; the IQ2 file omits it and uses the
separate official sidecar. Therefore Q4-versus-IQ2 measurements compare complete
recipes, **not an isolated change to gate/up bit width**. Testing just the cheaper
MTP down projection on a copy of our Q4, with target weights unchanged, is a
plausible next experiment, not a demonstrated gain from this run.
