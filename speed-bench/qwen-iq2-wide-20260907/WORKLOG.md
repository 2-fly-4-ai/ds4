# Historical journal — wide follow-up (2026-09-07)

FINAL STATUS: Read RESULTS.md for the completed findings. Downloads and experiments
are complete. Focused PLE/vision/JPEG and agent-cache fixes are on main at b6c9ad0
and cafeeae. Existing defaults are retained, SSD source changes were rejected,
and nothing was pushed. Entries below preserve chronological intermediate states,
including hypotheses subsequently corrected; they are not outstanding tasks.

User authorized all four follow-ups: longer tool loops, IQ2 download/speed/quality,
SSD curiosity tests, and remaining GLM/agent/vision ports. Main must retain all
previous wins and existing default quant. Main is 6da89cf, not pushed.

## Worktrees and changes

- Main `/Users/brianfarley/Desktop/ds4`, unchanged this turn except downloads.
- This worktree: `experiment/qwen-iq2-wide-20260907`, based on 6da89cf.
  Uncommitted focused external PLE support: ds4.h option; ds4_model own sidecar
  pointer; engine owns mapping; binding and CPU gather use correct mapping;
  weights ple_external keeps offsets outside main Metal maps. All five
  frontends accept --ple, shared help and agent parser tests added. Existing
  inline PLE behavior unchanged. tests/ds4_test.c reads DS4_TEST_PLE.
  Built all five, server/agent units pass before final help/parser changes;
  latest build-vision.log and agent-unit.log/server-unit.log track latest.
- Focused 948edcd CLI model-aware image-turn formatter port applied.
- Focused 681b9ce JPEG decoder port applied. Imported test_jpeg_decode.py:
  baseline 31 failing subcases, candidate all 39 JPEG cases exact Pillow pass.
  CLI vision test imported/adapted --ple optional for our embedded Q4 model.
- Separate `/Users/brianfarley/Desktop/ds4-ssd-readback-hotness`, branch
  experiment/ssd-readback-hotness-20260907, also based on 6da89cf:
  edcafe4 (reuse router readback) and 91dda0b (bounded preload priorities)
  focused ports applied and built. Not tested on model yet. Do not merge blindly.

## Download/integrity

Target directory main/gguf/qwen38-iq2-test; existing ds4flash.gguf link untouched.

IQ2 revision 672c52bbea7865352c8f0aa766c43939acf0f0d5:
Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf
50,343,093,376 bytes; SHA256 31f1e193771a5f3fdaa7af5865e0417513e8e8cac37579d388052ca54c889d4b.
Original curl PID83728 stopped cleanly after saving 26,232,479,744 bytes.
At ~18:26 switched to finish_iq2_download.py, exec42778 / PID86859, four checked
HTTP ranges; contiguous prefix retained. File now has .gguf.partial suffix and
range checkpoints in iq2-ranges.json. Do not infer completion from sparse file
length. Script verifies published full SHA before restoring .gguf and writing
iq2-integrity.txt. HF/Xet alternative did not improve throughput;
its process 84425 was terminated, partial hf-cache retained, not a second active
download. Range helper is the only remaining transfer.

PLE is COMPLETE and EXACT official bytes, reconstructed using installed tensor
and downloaded header/trailing tensors, full published SHA matches:
Qwen3.8-Flash-Next-PLE-Q4_1.gguf, 32,000,157,440 bytes,
SHA256 66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1.
HF Q4 revision 59a55fb819c82be7b162948282b50bd1a1e290b7.
Installed Q4 inline table at 74603986880, 32,000,153,600 bytes, Q4_1.
Official sidecar table at 3552; three metadata tensors follow at 32000157152,
total 288 trailing bytes (downloaded ple-tail.bin). First attempt incorrectly
assumed 3840 header; rejected SHA recorded ple-integrity.log. Correct v2 has
matching hash in ple-integrity-v2.log. materialize_ple.py records method.
Stopped PLE curl PID 84854. Unused partial and rejected reconstruction moved
recoverably to `/Users/brianfarley/.Trash/ds4-iq2-test-20260907/`.

## Live benchmark running

long_tool_ab.py reuses prior API harness request code, baseline binary in
`/Users/brianfarley/Desktop/ds4-qwen-mtp-port` (d1f0354 runtime exact to 81eb42c;
verified git diff ds4.c/server/Metal empty), candidate this worktree.
Both use existing Q4 quant, temp0, thinking on, three sequential read_file calls
then a final summary with assertions version=2.7.4, entry=serve, tests=pytest.
No actual tool execution. Streaming and non-streaming, MTP on/off, 8K/32K.
Allocated context is requested frontier +4096. Actual 8K ~7963; 32K ~33443.

8K MTP off pilot session71781 completed, assertions PASS.
8K MTP on FAILED at third requested tool: returned text without a tool call.
Old baseline with explicit reasoning replay reproduces that failure. Not proved
to be a cache regression or a verifier bug; exclude from successful claims.
32K MTP off/on and explicit-reasoning controls completed PASS. All GPU runs serial.
First 32K MTP-off candidate followups reuse 33516/33605/33681 tokens and process
only 30/25/21 new; ~1.4–2.5s vs baseline ~35–40s. Preliminary, downloads/builds
ran concurrently, so do not treat tiny timing deltas as controlled wins.
Explicit-reasoning controls completed. verify_long.py passes three completed
groups (96 responses), excludes failed 8K MTP group. Content, reasoning and tool
functions match in retained-history controls; cold omitted histories are
different and are NOT expected byte-identical.

External PLE: 128 full-logit files exact vs embedded. State snapshot/rewind pass
MTP off/on. CLI Qwen vision passes all four turns off/on; GLM same passes;
DeepSeek shared CLI vision passes ordinary (no embedded MTP). Full make test
PASSED with installed0731golden, full-regression.log (session24591 exit0).
Current GPU queue90751: GLM/DS4 vision completed then Q4 API four tasks x three
contexts, off then on. Download runs separately. Do not start another model.

SSD correctness sweep completed (session35625 exit0): all64logitfiles equal
resident/base8/candidate8/candidate32 for bothmodels. Timing includes logit disk
writes; must rerun --timing for clean speed. Initial GLM resident33.63t/s vs8GB
1.24,32GB1.25; DS4resident44.44 vsbase8 16.57,candidate8 16.74,candidate32 17.05.
Short64token teacherforced 2K cases, not physical64GB/coldSSD certification.
No SSDpatch promoted; no significantwin yet. New parallel downloader PID86859
is not paused by old run_ssd.py PID83728 check: wait fordownload or explicitly
update pause check before further I/O-sensitive tests.

## Outstanding actions

1. Finish long loops and explicit-reasoning controls; summarize per-turn cached
   tokens, prefill, TTFT, wall time, preserve raw JSON. Can repeat balanced trials
   if needed, but no need endless 32K cold rebuilds once mechanism proven.
2. Test external PLE with current Q4: official sidecar payload equals inline
   table. Compare same fixed-token logits embedded vs --ple, plus MTP snapshots
   using DS4_TEST_PLE. Do not mistake successful loading for correctness.
3. CLI image tests old baseline vs candidate using existing
   gguf/mmproj-Qwen3.8-Flash-Next-F16.gguf (no need download Q8) and two existing
   tests/vision-fixtures/glm53 PNGs. Test MTP off/on, image changes, text followups.
   JPEG tests prove pixel parity; not a general vision-quality benchmark.
4. Actual GLM/DeepSeek SSD benchmarks (resident vs 8/32GiB cache; fixed token
   context/decode; ideally pause curl only for I/O-sensitive timing and resume).
   New d7ade9f small-GLM-prefill patch requires gate/up/down all IQ2, while our
   hybrid GGUF has 43 layers IQ2 gate/up and Q2_K down. Thus unchanged patch is
   ineligible; don't port it blindly. Existing Q2 GLM path already reads prefetch
   IDs at ds4_metal.m ~38029. edcafe4 may be ineligible as well; measure/inspect.
   91dda0b affects DeepSeek preload rank, not usual GLM demand-filled cache.
   Qwen graph has no explicit bounded SSD expert-cache orchestration; its PLE
   demand paging is distinct. Do not label Qwen --ssd-streaming as supported
   simply because CLI accepts the flag. Inspect/reject if unsupported.
5. Once IQ2 full verified: smoke/load, per-model state tests, paired Q4/IQ2
   prefill/decode at2K/8K/32K and API coding/story/structured with MTP on/off.
   Quality: same deterministic task suite and budgets; don't claim equal quality
   from speed or a handful of answers. ds4-eval offers --questions/--tokens etc.
6. Full regression suite with installed 0731 golden file, plus GLM and DS4
   snapshots/SSD, build all5, unit checks and no source drift to old quant path.
   Save tested wins only, report all rejected/inactive ideas, no main blanketmerge.

## Dependencies and conventions

.venv in THIS worktree (do not stage) has numpy, Pillow, huggingface_hub, hf_xet,
gguf. Python3.9 from Xcode tools. GGUFReader fails on current Q4's pre-existing
duplicate GGUF.version metadata; for inspection a subclass _push_field can
ignore duplicate names. Never rewrite current quant just to satisfy reader.
Use apply_patch for source/test edits. No agents authorized. No parallel giant
model processes. No giant CPU inference. Main's pre-existing untracked artifacts
must be preserved. User settings remain Automatic.
