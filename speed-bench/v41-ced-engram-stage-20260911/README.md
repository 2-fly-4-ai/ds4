# DeepSeek V4.1 CED Engram staging — 2026-09-11

## Decision

Keep as an opt-in Metal experiment.  During the bounded text-only CED hybrid
prefill, gather all prompt Engram inputs before encoding each Engram layer and
encode that layer in one command buffer.  The Q8 projection and injection are
still emitted one row at a time, preserving the established arithmetic order.

Enable it together with the parent hybrid experiment:

```sh
DS4_METAL_V41_CED_PREFILL=1 \
DS4_METAL_V41_CED_HYBRID_PREFILL=1 \
DS4_METAL_V41_CED_STAGE_ENGRAM=1 \
./ds4-bench --metal --ssd-streaming \
  -m gguf/DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf \
  --engram gguf/DeepSeek-V4.1-Flash-Engram.gguf \
  --prompt-file tests/long_context_security_prompt.txt \
  --ctx-start 512 --ctx-max 512 --gen-tokens 0
```

The routed-FFN batch scratch is idle during attention and is reused for the
staged Engram inputs and outputs.  No persistent Metal allocation is added.
The bounded 1,024-row hybrid needs about 48 MiB of temporary host gather memory
while staging an Engram layer.

## Correctness

The candidate was compared against the same build with
`DS4_METAL_V41_CED_STAGE_ENGRAM` unset.  At 129, 512, and 1,024 prompt tokens,
the full frontier logit JSON plus eight raw F32 teacher-forced decode-logit
arrays were byte-identical (`diff -qr` produced no output).

The first implementation used one multi-row Q8 projection.  It was faster, but
changed logits because its reduction order differs from decode.  That version
was discarded.  The retained version only changes staging and synchronization;
the projection remains a sequence of one-row operations in one command buffer.

## Performance

Machine: Apple M5 Max, 128 GiB, Metal, SSD expert streaming.  Each frontier was
run in baseline → candidate → candidate → baseline order, with no logit dumps
or generated tokens during the timed pass.

| Prompt | Baseline runs | Candidate runs | Baseline mean | Candidate mean | Gain |
|---:|---:|---:|---:|---:|---:|
| 512 | 21.57, 21.58 t/s | 22.60, 22.11 t/s | 21.575 t/s | 22.355 t/s | +3.62% |
| 1,024 | 26.14, 24.82 t/s | 27.72, 27.74 t/s | 25.480 t/s | 27.730 t/s | +8.83% |

The 1,024-token baseline pair varied more than the candidate pair.  The table
reports the measured balanced means rather than hiding that variance.  This is
not a claim about prompts beyond the current 1,024-token hybrid boundary.

## Regression audit

- `tests/test_deepseek41_spec`: pass.
- Metal indexer parity suite: pass, including all tested boundary sizes.
- Eval extractor self-tests: pass.
- Agent tests: pass.
- `make test` reaches `ds4_test`, then stops because this checkout does not
  contain its expected `ds4flash.gguf` golden fixture.  No installed
  experimental model was substituted for that missing oracle.
- `git diff --check`: clean.

The ordinary CED path, non-Apple builds, non-V4.1 models, visual prompts, and
prompts outside the parent hybrid's bounds remain unchanged.

