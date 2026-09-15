#!/bin/zsh

script_dir=${0:A:h}
cd "$script_dir" || exit 1

model=${DS4_PI_MODEL:-"$script_dir/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AOutQ4K-L27-42-chat-v2-imatrix-0731.gguf"}
cache_dir=${DS4_PI_KV_DIR:-"/Users/brianfarley/Library/Caches/ds4-pi-kv"}

if [[ ! -s "$model" ]]; then
  print -u2 "Missing DwarfStar coding model: $model"
  exit 1
fi

mkdir -p "$cache_dir" || exit 1

print "Starting DwarfStar for Pi GUI"
print "  API: http://${DS4_HOST:-100.109.208.12}:${DS4_PORT:-8000}/v1"
print "  Model: $model"
print "  Context: ${DS4_CTX:-100000} tokens"
print "Press Ctrl-C to stop and unload the model."

exec ./ds4-server --metal \
  -m "$model" \
  --ctx "${DS4_CTX:-100000}" \
  --host "${DS4_HOST:-100.109.208.12}" \
  --port "${DS4_PORT:-8000}" \
  --kv-disk-dir "$cache_dir" \
  --kv-disk-space-mb "${DS4_KV_DISK_MB:-8192}"
