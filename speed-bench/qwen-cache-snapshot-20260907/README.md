# Qwen visible tool KV and snapshot follow-up — 2026-09-07

Baseline: `81eb42c` (all previously promoted Qwen optimizations retained).
Hardware: M5 Max, 128 GiB; macOS Automatic, no power-setting changes.
Model: existing `Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf`; no new quant downloaded.

## Focused ports

- `antirez/ds4` `c0a6119`: reserve a separate fmemopen terminator byte without
  changing serialized payload length. Add grow/reuse byte comparisons and real
  model memory-versus-file comparisons. The old implementation PASSED the
  model-free byte test on this Mac: this is portability hardening, not evidence
  of reproduced macOS corruption or a speed improvement.
- `ivanfioravanti/ds4-metal` `658b2fc`: preserve the exact Qwen tool-turn live
  checkpoint when an OpenAI chat client omits reasoning. The visible key stops
  before `<|im_end|>`; the next suffix evaluates that boundary. Reject images,
  Responses/Anthropic, other models, truncation and unclosed thinking. Unit
  coverage includes thinking on/off and visible assistant content.
- `ffd85d4`, test-only rewind comparison: match the prefill/decode split;
  retain the original 0.002 log-probability tolerance. The corrected test also
  PASSES against the unchanged `81eb42c` inference object files with MTP on.
  This corrects the test oracle; it is not a newly fixed kernel defect.

No kernels, quantization, automatic router policy, or MTP defaults changed.
No wholesale upstream merge or unrelated new tool-parser implementation.

## Measured live API A/B

`tool_ab.py` starts exactly one server at a time on a random loopback port.
Same existing model, 8,192 allocated context, actual prompts 1,897–1,933 tokens,
temperature zero. Two balanced repetitions (base/candidate, candidate/base),
streaming and non-streaming, a read_file call and a fixture tool result. No tool
is actually executed. Four follow-up observations per mode and configuration.
Disk KV is not enabled. Each request and response is saved in JSON and traces.

| Tool follow-up | Baseline mean wall | Candidate mean wall | Interpretation |
|---|---:|---:|---|
| No thinking, MTP off | 456.7 ms | 452.1 ms | Neutral (~1%) |
| No thinking, MTP on | 428.9 ms | 433.3 ms | Neutral (~1%) |
| Thinking omitted by client, MTP off | 3,325.3 ms | 1,153.1 ms | 65.3% lower latency; 2.88x |
| Thinking omitted by client, MTP on | 3,994.2 ms | 1,023.4 ms | 74.4% lower latency; 3.90x |

For omitted-reasoning follow-ups, baseline rebuilt 1,990 tokens (zero cached).
Candidate retained 1,980 tokens and evaluated 30 new tokens. No-thinking already
retained 1,922 tokens and evaluated 32 on both versions.

| Omitted-reasoning follow-up | Baseline | Candidate |
|---|---:|---:|
| Mean prefill, MTP off | 2.444 s | 0.202 s |
| Mean streaming first-event latency, MTP off | 2.554 s | 0.273 s |
| Mean prefill, MTP on | 3.239 s | 0.228 s |
| Mean streaming first-event latency, MTP on | 3.391 s | 0.297 s |

These are reduced reprocessing/turn-latency measurements, NOT 2.9–3.9x faster
raw decode kernels. Modes were run sequentially: do not use their cold-prefill
timings to infer the speed effect of MTP itself. Output lengths also differ when
baseline discards reasoning. Cache-token counts and prefill logs independently
establish the avoided work. This is one coding-tool fixture, not a broad quality
or long-context benchmark.

## Correctness controls

80 API responses / 40 two-request conversations including explicit-reasoning
controls. All requested tool names/arguments and returned version checks pass.
No-thinking answer hashes match baseline/candidate, with MTP on and off.

Omitted-reasoning cold-baseline output is NOT claimed byte-equivalent to retained
history. Instead, explicit-reasoning replay controls were run on both versions:
candidate omitted-reasoning follow-ups match their answer AND reasoning exactly
for streaming/non-streaming, independently with MTP off and on. Do not infer
MTP-on versus MTP-off equivalence from this result.

Additional checks completed:

- All five frontends build; server and agent unit tests pass.
- Model-free snapshot test passes normally and with ASan/UBSan.
- Qwen snapshot and corrected rewind tests pass with MTP off and on.
- Qwen MTP snapshot: 16 cycles, eight single/eight double returns; context reuse.
- GLM hybrid-quant MTP snapshot: 16 cycles; context reuse and file-byte checks.
- DeepSeek Vision-Exp late-Q4 snapshots: resident and 8 GiB SSD cache, pass.
- Corrected rewind test linked against baseline inference: pass (MTP on).
- `git diff --check`: pass.

Full `make -k test` passed (exit 0), using the installed 0731 golden model:
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
See `full-test.log`. Model/flag-gated tests correctly skip: GLM continued prefill,
Qwen rewind under the DeepSeek model, optional SSD decode-prefill comparison,
and DeepSeek external MTP/DSpark depth tests. Separate Qwen/GLM/SSD snapshot and
Qwen rewind runs above cover the changes; this is not a claim every optional
hardware/model configuration ran.
CUDA/ROCm and distributed hardware were not exercised; no corresponding kernel
or distributed code changed. The snapshot allocation fix is backend-independent.

## Sources

- https://github.com/antirez/ds4/commit/c0a6119f363ef82125877142f13fb3fe491cba14
- https://github.com/ivanfioravanti/ds4-metal/commit/658b2fc1b345a759fb7691ab53fae6aea95ac9b0
- https://github.com/ivanfioravanti/ds4-metal/commit/ffd85d426313ace6dae805e9f2fb4424d5b427fd

## Reproduce

From an isolated worktree beside `/Users/brianfarley/Desktop/ds4`:

```sh
make -j6 all ds4_test ds4_agent_test tests/test_snapshot_bytes
./tests/test_snapshot_bytes
./ds4_test --server
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --rounds 2
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --rounds 2 --thinking
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --rounds 2 --mtp
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --rounds 2 --mtp --thinking
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --thinking --replay-reasoning
python3 speed-bench/qwen-cache-snapshot-20260907/tool_ab.py --thinking --replay-reasoning --mtp
python3 speed-bench/qwen-cache-snapshot-20260907/verify_results.py
```

The harness uses the Desktop main binary as baseline: after promotion, use an
unchanged `81eb42c` build for a meaningful new A/B, not the newly promoted binary.
