# Daily use on this M5 Max

Updated 2026-09-14. Production checkout: `/Users/brianfarley/Desktop/ds4`,
branch `main`. Validated implementation: `1093763`; subsequent evidence and
benchmark-baseline safeguards: `b91080f`. Optimization experiments are stopped.
Use this checkout, not a sibling experiment worktree. Existing quants are unchanged.

## Start a model

Run ONE model at a time, from a fresh Terminal without experimental environment
overrides. Keep the Terminal open while using its API. Ctrl-C stops the server
and releases its model; wait for it to exit before starting another.

### Qwen 27B Q8: API with its installed native-MTP sidecar

```sh
cd /Users/brianfarley/Desktop/ds4
DS4_QWEN_MTP_HEAD="$PWD/gguf/qwen-small-quality/mtp-Qwen3.8-27B-Q4_64A.gguf" \
./ds4-server --metal --ctx 4096 --host 127.0.0.1 --port 8000 \
  -m gguf/qwen-small-quality/Qwen3.8-27B-Q8_0.gguf
```

To use the validated smaller quant, change only the target filename to
`Qwen3.8-27B-Q4_64A.gguf`; retain the same sidecar.

### Qwen 35B-A3B Q8: API with bundled native MTP

```sh
cd /Users/brianfarley/Desktop/ds4
env -u DS4_QWEN_MTP_HEAD ./ds4-server --metal --ctx 4096 \
  --host 127.0.0.1 --port 8000 \
  -m gguf/qwen-small-quality/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf
```

Use the file inside `35b-mtp`, not the older similarly named file above it.
These small-Qwen models currently have a **4096-token total context limit**
(prompt plus answer). Live in-process cache reuse works; durable disk KV does
not. The 35B port is resident Metal only.

For terminal chat, use `./ds4` in place of `./ds4-server`, remove `--host` and
`--port`, and add `--temp 0`. Keep the same model, context and sidecar settings.
This is terminal chat, not a graphical desktop chat app.

### GLM 5.3 Flash: conservative, validated short-context API

```sh
cd /Users/brianfarley/Desktop/ds4
DS4_GLM53_PREFILL_CHUNK=256 ./ds4-server --metal --ctx 1024 \
  --host 127.0.0.1 --port 8000 --mtp-timing --mtp-exact-sampling \
  -m gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf
```

This reproduces the latest low-scratch regression configuration, not a claim
that GLM supports only 1024 tokens. Longer contexts require more memory.
The older `start_glm53_chat.command` uses a different quant/context and launches
the agent; it is not the configuration above.

### DeepSeek V4.1: conservative SSD configuration

```sh
cd /Users/brianfarley/Desktop/ds4
./ds4-server --metal --ctx 4096 --host 127.0.0.1 --port 8000 \
  -m gguf/DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf \
  --engram gguf/DeepSeek-V4.1-Flash-Engram.gguf \
  --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16GB
```

Only short prompts were validated for this conservative configuration. This is
text-only; vision is not part of that regression. The older
`start_deepseek41_server.command` selects vision, 16K context and an automatic
cache budget instead: **do not treat it as this tested configuration**.
Do not repeat full-encoder residency or large CPU inference experiments:
earlier experiments caused macOS watchdog crashes. Passing short tests is not
a guarantee against OS/driver failures under other memory loads.

## Connect and chat

Wait for the server's ready message. In another Terminal:

```sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Write a short Python function to remove duplicates while preserving order."}],"temperature":0,"max_tokens":256}'
```

`qwen` is the small-Qwen model ID; for other models use the ID returned by
`/v1/models`. An OpenAI-compatible chat client can use base URL
`http://127.0.0.1:8000/v1`. The API is available only while the server runs.
Loopback is local-only. For another computer, explicitly bind `--host` to this
Mac's current Tailscale IP and use that address in the client. Avoid `0.0.0.0`
unless you intend LAN exposure; do not expose an unauthenticated API publicly.

### Pi GUI on the Mac mini

The Mac mini's Pi model registry contains an additional provider named
`DwarfStar on M5 Max`. It points over Tailscale to
`http://100.109.208.12:8000/v1` and advertises the resident DeepSeek V4 Flash
0731 model with a 100,000-token context. It does not replace Pi's existing
default provider or model.

Double-click `start_pi_coding_server.command` on this MacBook, wait for the
server-ready message, then open Pi GUI's model picker on the Mac mini and choose
`DwarfStar — DeepSeek V4 Flash (100K)`. Pi reloads `models.json` when its model
picker opens; restart Pi GUI if an already-open picker does not refresh.

Pi's automatic compaction remains enabled. With the 100K advertised context,
its installed defaults reserve 16K tokens for the next response and retain
approximately 20K recent tokens while summarizing older work. The configured
384K `maxTokens` value is an output ceiling, not the active context allocation:
the server's 100K context is still the hard prompt-plus-output limit. This high
ceiling avoids an arbitrary client-side response cap; the server stops at the
available context boundary.

## What remains enabled

Production Metal kernels, quant-specific optimizations and supported live-cache
reuse remain in main. M5 small-Qwen FP32 attention automatically uses the exact
tiled kernel at positions >=1024; shorter contexts retain the previous kernel.
No precision reduction was added by that optimization.

Prompt lookup, native MTP and fallback eligibility are model- and request-specific.
There is no universal switch guaranteeing every draft method is used or faster.
Small-Qwen native MTP is enabled by default when its head is available; setting
`DS4_QWEN_NEXTN_DRAFT=0` disables it for a control run. Temperature zero is the
validated greedy MTP comparison setting, not a requirement for ordinary chat.
Nonzero-temperature requests must not be assumed to get the same acceleration.
Do not add DSpark/DFlash files or experimental flags just to make a launch 'auto'.

## Evidence and boundaries

- [Small-Qwen port and all-model regression](speed-bench/qwen35-port/README.md):
  120 small-Qwen API requests, 30 exact off/on pairs, 60 exact cache pairs,
  and 25 exact archived replies across DeepSeek 0731, Vision-Exp, V4.1 SSD,
  GLM and Qwen Next. Includes native-MTP/state/teardown tests.
- [Merged M5 attention win](speed-bench/qwen-attention-20260914/README.md):
  exact full-logit checks, 112 synthetic cases and 68 API requests. Final
  matched 27B Q8 code/story/JSON rates were 35.95/20.64/36.26 t/s with MTP,
  at actual prompts of 1368/2034/2037 tokens. These are workload measurements,
  not a guaranteed speed or prefill rate.
- Latest dense-FFN experiments are **rejected, not production**. Their report
  lives in sibling `ds4-qwen27-dense-profile/speed-bench/qwen27-dense-profile-20260914/README.md`
  at experiment commit `5d95624`. No missing performance win awaits merging.
- Daily-use handoff: binaries up to date and `./ds4_test --server` passed.
  No new model benchmark campaign was run for this documentation pass.
  CUDA/distributed and other Macs were not runtime-tested here.

Keep experiment worktrees as evidence, not launch locations. This handoff does
not claim the entire repository's historical test suite is clean or that every
context/model combination was retested today. Stop on abnormal memory pressure,
errors or hangs instead of launching another model alongside it.
