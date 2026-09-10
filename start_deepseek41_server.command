#!/bin/zsh

script_dir=${0:A:h}
cd "$script_dir" || exit 1

gguf_dir=${DS4_GGUF_DIR:-"$script_dir/gguf"}
model=${DS4_V41_MODEL:-"$gguf_dir/DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf"}
engram=${DS4_V41_ENGRAM:-"$gguf_dir/DeepSeek-V4.1-Flash-Engram.gguf"}
vision=${DS4_V41_VISION:-"$gguf_dir/DeepSeek-V4.1-Flash-Vision-Encoder.gguf"}

for required in "$model" "$engram" "$vision"; do
  if [[ ! -s "$required" ]]; then
    print -u2 "Missing DeepSeek V4.1 artifact: $required"
    print -u2 "See gguf-tools/README.md#deepseek-v41-flash for conversion instructions."
    exit 1
  fi
done

exec ./ds4-server --metal \
  -m "$model" \
  --engram "$engram" \
  --vision "$vision" \
  --ssd-streaming \
  --ctx "${DS4_CTX:-16384}" \
  --host "${DS4_HOST:-127.0.0.1}" \
  --port "${DS4_PORT:-8000}"
