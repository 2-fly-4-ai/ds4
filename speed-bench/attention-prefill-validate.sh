#!/bin/sh
# Run in the integration worktree. Tests are serial to avoid GPU contention.
set -u
main=/Users/brianfarley/Desktop/ds4
glm=$main/gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf
ds=$main/gguf/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AOutQ4K-L27-42.gguf
prompts=/Users/brianfarley/Documents/Codex/2026-08-18/let/qwen-pipeline-promotion-2026-09-05/qwen-promoted-short
out=speed-bench/attention-prefill-results
rollback=DS4_METAL_DISABLE_DS4_SHORT_PREFILL_TAIL32
for test in test_glm53_kda test_qwen4_kernels test_sampling; do
 ./tests/$test > "$out/$test.log" 2>&1
 rc=$?; echo "$test exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
done
for corpus in code structured prose; do
 if [ "$corpus" = code ]; then prompt=$main/ds4.c; else prompt=$main/speed-bench/context-corpora/$corpus.txt; fi
 # Harness 'control' is the new default; 'candidate' enables the OLD rollback.
 ./speed-bench/metal_prefill_variant_bench -m "$ds" --prompt-file "$prompt" --prefix-tokens 1024 --warmup-tokens 1024 --repeats 4 --candidate-env "$rollback" > "$out/ds-prefill-$corpus-1024.log" 2>&1
 rc=$?; echo "DS prefill $corpus exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
done
for n in 512 1024 1025 2048; do
 DS4_METAL_TRACE_DS4_SHORT_PREFILL_TAIL32=1 ./speed-bench/metal_prefill_variant_bench -m "$ds" --prompt-file "$main/ds4.c" --prefix-tokens "$n" --warmup-tokens 256 --repeats 1 --candidate-env "$rollback" > "$out/ds-activation-$n.log" 2>&1
 rc=$?; echo "DS activation $n exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
done
./speed-bench/metal_decode_schedule_bench -m "$ds" --prompt-file "$main/ds4.c" --prefix-tokens 1024 --warmup 0 --tokens 256 --candidate-env "$rollback" > "$out/ds-state-1024.log" 2>&1
rc=$?; echo "DS state exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
for n in 2048 8192; do
 ./speed-bench/metal_decode_schedule_bench -m "$glm" --prompt-file "$main/ds4.c" --prefix-tokens "$n" --ctx 9216 --warmup 0 --tokens 256 --candidate-env BENCH_NOOP > "$out/glm-state-$n.log" 2>&1
 rc=$?; echo "GLM state $n exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
done
for model in ds glm; do
 if [ "$model" = ds ]; then gguf=$ds; else gguf=$glm; fi
 ./speed-bench/route_repeat_bench "$gguf" BENCH_NOOP 0 256 4096 "$main/speed-bench/qwen_prompt_lookup_copy_chat.txt" "$main/speed-bench/qwen_prompt_lookup_edit_chat.txt" "$prompts/code.txt" "$prompts/story.txt" "$prompts/json.txt" > "$out/$model-greedy.log" 2>&1
 rc=$?; echo "$model greedy exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
done
./speed-bench/route_repeat_bench "$glm" BENCH_NOOP 0.7 256 4096 "$main/speed-bench/qwen_prompt_lookup_copy_chat.txt" "$prompts/code.txt" "$prompts/story.txt" "$prompts/json.txt" > "$out/glm-sampled.log" 2>&1
rc=$?; echo "GLM sampled exit=$rc"; [ "$rc" = 0 ] || exit "$rc"
WIDE_MTP=1 ./speed-bench/route_repeat_bench "$glm" BENCH_NOOP 0 256 4096 "$main/speed-bench/qwen_prompt_lookup_copy_chat.txt" "$main/speed-bench/qwen_prompt_lookup_edit_chat.txt" > "$out/glm-mtp.log" 2>&1
rc=$?; echo "GLM MTP exit=$rc"; exit "$rc"
