# Smaller-Qwen API protocol repair

Integrated onto main in `9638cfd`, then rebuilt in the normal Desktop checkout. Post-integration server unit tests, tokenizer checks for both quants, and two live Q8 chat JSON requests passed. `/v1/models` was explicitly checked to advertise `qwen`. No remote push was performed.

## What was wrong

The smaller Qwen backend was omitted from three integration points:

1. Server chat rendering selected DeepSeek syntax rather than ChatML.
2. Vocabulary initialization left ChatML marker IDs unset, and text splitting did not select the GGUF's declared `qwen35` pre-tokenizer. Merely supplying a ChatML string to `/v1/completions` did not repair that tokenizer dispatch.
3. `/v1/models` advertised the small Qwen model as DeepSeek.

The repair extends the existing Qwen protocol handling to the smaller backend, selects text splitting from its tokenizer metadata, and advertises the canonical `qwen` ID. Both installed 27B quants declare `tokenizer.ggml.pre=qwen35`. Stop handling now has both the ChatML end marker and document-end marker initialized. No weights, quant formats, kernels, MTP verifier, or cache layout were changed.

## Validation

See [CHECKS.md](CHECKS.md) for measured counts, exact-text comparisons, and speed sanity checks. The `final-chat` runs cover short and target-2K requests in five disciplines, an additional JSON request, and sampled coding/story requests at temperature 0.7. They use the normal chat endpoint, not a completion workaround. The separate `cache-repeat` runs issue consecutive identical JSON requests; consult usage and logs before claiming actual cache reuse. `thinking` exercises Q8's reasoning stream.

MTP-on/off comparisons use identical greedy requests. Existing-model regressions replay archived requests from the earlier production matrix and compare the returned text, not just whether HTTP succeeded. Each model runs alone. A GLM 40K-allocation attempt hit the 112 GiB reserve guard and was stopped; it is retained under `glm53-regression-guard-stop`. The completed GLM comparison uses the previously measured 1K-allocation / 256-token scratch configuration on both sides. No memory threshold was increased.

Local checks:

```sh
make -j4 ds4 ds4-server ds4_test tests/test_qwen_chat_tokens
./ds4_test --server
./tests/test_qwen_chat_tokens /absolute/path/to/Qwen3.8-27B-Q8_0.gguf
./tests/test_qwen_chat_tokens /absolute/path/to/Qwen3.8-27B-Q4_64A.gguf
```

The tokenizer check only reads GGUF metadata and token tables; it performs no CPU or GPU inference. It checks the declared pre-tokenizer, stop IDs, and equality of rendered versus directly assembled ChatML tokens. This is not a full logits/golden-fixture, vision, CUDA, or distributed-inference certification. Inference regression testing here is on the local M5 Max. Diagnostic replay scripts require the retained `production-matrix-20260914` artifacts and its helper script in the main checkout.

## What is not fixed by this patch

- **Qwen 35B-A3B:** main's small-Qwen execution path is fixed to the dense 27B shape; recognizing `qwen35moe` alone cannot make its MoE graph work. It now fails with an accurate unsupported-runtime message rather than a missing DeepSeek metadata key. Actual support is still a porting task. The older `experiment/qwen-small-router-fix-20260908` worktree contains a broader runtime and prior 35B tests; relevant history includes `b2396f6` and `1f47c8d`. The former spans 31 files and over 8,000 added lines, so merging it wholesale would endanger unrelated newer work. Port model configuration, tensor binding, routed expert execution, recurrent/KV state, and native MTP support selectively, then validate against that reference build. No evidence here calls for requantizing or relabeling the installed GGUF.
- **Instruction-following misses:** a successful stream is not a schema guarantee. Q4's target-2K JSON request returned one object instead of three array items. The earlier DeepSeek JSON misses remain. This repair does not enforce structured decoding, change prompts to hide misses, or certify quant quality.
- **Performance:** the new and old malformed prompts have different token streams. Their speeds are not a valid optimization A/B. The MTP-on/off comparisons are matched; the other-model speed checks are single-run sanity checks, not statistical regression guarantees.

The intended next step is the isolated 35B runtime port, not another change to the already-working DeepSeek or GLM kernels.

Consecutive identical 27B JSON requests returned identical text on both quants, but reported `cached_tokens=0` and `hot prompt rewind snapshot unavailable`. Functional replay passes; prompt-cache acceleration for that path remains separate work. Q8 thinking-mode testing produced a reasoning stream up to the 256-token cap, not a completed final answer.
