#!/bin/zsh

script_dir=${0:A:h}
cd "$script_dir" || exit 1

if pgrep -x ds4-server >/dev/null 2>&1; then
  print -u2 "A DwarfStar server is already running. Stop it before changing models."
  exit 1
fi

print "Choose the model that you will select in Pi GUI:"
print "  1) DeepSeek V4 Flash 0731 — 100K (recommended coding default)"
print "  2) DeepSeek V4 Vision Exp — 40K"
print "  3) DeepSeek V4.1 Flash + Vision — 16K, SSD streaming, experimental"
print "  4) GLM 5.3 Flash — 32K"
print "  5) GLM 5.3 Flash Vision — 32K"
print "  6) Qwen 3.8 Flash Next Q4 — 64K"
print "  7) Qwen 3.8 Flash Next Q4 Vision — 64K"
print "  8) Qwen 3.6 35B-A3B Q8 — 4K"
print "  9) Qwen 3.8 27B Q8 — 4K"
print " 10) Qwen 3.8 27B optimized Q4 — 4K"
read "choice?Model [1]: "
choice=${choice:-1}

typeset -a extra
use_disk_kv=1
case "$choice" in
  1) label="DeepSeek V4 Flash 0731"; ctx=100000
     model="$script_dir/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AOutQ4K-L27-42-chat-v2-imatrix-0731.gguf" ;;
  2) label="DeepSeek V4 Vision Exp"; ctx=40000
     model="$script_dir/gguf/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AOutQ4K-L27-42.gguf"
     extra=(--vision "$script_dir/gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf") ;;
  3) label="DeepSeek V4.1 Flash"; ctx=16384
     model="$script_dir/gguf/DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf"
     extra=(--engram "$script_dir/gguf/DeepSeek-V4.1-Flash-Engram.gguf"
            --vision "$script_dir/gguf/DeepSeek-V4.1-Flash-Vision-Encoder.gguf"
            --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 16GB) ;;
  4) label="GLM 5.3 Flash"; ctx=32768
     model="$script_dir/gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf"
     extra=(--mtp --mtp-exact-sampling) ;;
  5) label="GLM 5.3 Flash Vision"; ctx=32768
     model="$script_dir/gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf"
     extra=(--mtp --mtp-exact-sampling --vision "$script_dir/gguf/GLM-5.3-Flash-Vision-Encoder.gguf") ;;
  6) label="Qwen 3.8 Flash Next Q4"; ctx=65536
     model="$script_dir/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf"
     extra=(--ple "$script_dir/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" --mtp --mtp-exact-sampling) ;;
  7) label="Qwen 3.8 Flash Next Q4 Vision"; ctx=65536
     model="$script_dir/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf"
     extra=(--ple "$script_dir/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" --mtp --mtp-exact-sampling
            --vision "$script_dir/gguf/mmproj-Qwen3.8-Flash-Next-F16.gguf") ;;
  8) label="Qwen 3.6 35B-A3B Q8"; ctx=4096; use_disk_kv=0
     model="$script_dir/gguf/qwen-small-quality/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" ;;
  9) label="Qwen 3.8 27B Q8"; ctx=4096; use_disk_kv=0
     model="$script_dir/gguf/qwen-small-quality/Qwen3.8-27B-Q8_0.gguf"
     export DS4_QWEN_MTP_HEAD="$script_dir/gguf/qwen-small-quality/mtp-Qwen3.8-27B-Q4_64A.gguf" ;;
 10) label="Qwen 3.8 27B optimized Q4"; ctx=4096; use_disk_kv=0
     model="$script_dir/gguf/qwen-small-quality/Qwen3.8-27B-Q4_64A.gguf"
     export DS4_QWEN_MTP_HEAD="$script_dir/gguf/qwen-small-quality/mtp-Qwen3.8-27B-Q4_64A.gguf" ;;
  *) print -u2 "Invalid selection: $choice"; exit 1 ;;
esac

if [[ "$choice" != 9 && "$choice" != 10 ]]; then
  unset DS4_QWEN_MTP_HEAD
fi

cache_dir=${DS4_PI_KV_DIR:-"/Users/brianfarley/Library/Caches/ds4-pi-kv"}

if [[ ! -s "$model" ]]; then
  print -u2 "Missing DwarfStar coding model: $model"
  exit 1
fi

mkdir -p "$cache_dir" || exit 1

print "Starting $label for Pi GUI"
print "  API: http://${DS4_HOST:-100.109.208.12}:${DS4_PORT:-8000}/v1"
print "  Model: $model"
print "  Context: ${DS4_CTX:-$ctx} tokens"
print "Press Ctrl-C to stop and unload the model."

typeset -a args
args=(--metal -m "$model" --ctx "${DS4_CTX:-$ctx}"
      --host "${DS4_HOST:-100.109.208.12}" --port "${DS4_PORT:-8000}")
if [[ "$use_disk_kv" == 1 ]]; then
  args+=(--kv-disk-dir "$cache_dir" --kv-disk-space-mb "${DS4_KV_DISK_MB:-8192}")
fi
args+=("${extra[@]}")
exec ./ds4-server "${args[@]}"
