#!/bin/sh
# Run from the repository root. One model process at a time; no production edits.
# Usage: sh tests/run_qwen_q4k_wide_oracle.sh MODEL.gguf PLE.gguf [OUTPUT_DIR]
set -eu
model=${1:?model GGUF required}
export HOT_PLE=${2:?external PLE GGUF required}
result_dir=${3:-$(mktemp -d /tmp/qwen-q4k-wide-oracle.XXXXXX)}
mkdir -p "$result_dir"
make tests/test_qwen_mtp_port
export HOT_LOOKUP=1 HOT_REFERENCE_PLAIN=1 HOT_RELAX_ROUTE=1
export HOT_RELAX_LOGITS=1 HOT_REWIND=1 HOT_NO_WARM=1
export DS4_QWEN4_MOE_MM_NAX=1
copy_task=$(cat speed-bench/qwen_prompt_lookup_copy_chat.txt)
for context in 2048 8192; do
    HOT_CTX=$context HOT_PAD=1 HOT_TOKENS=256 HOT_TASK="$copy_task" \
        ./tests/test_qwen_mtp_port "$model" > "$result_dir/copy-$context.log" 2>&1
    HOT_CTX=$context HOT_PAD=1 HOT_TOKENS=128 \
        HOT_TASK='Write a Python LRU cache implementation with a doubly linked list, a dictionary, and unit tests covering updates, eviction, and capacity one.' \
        ./tests/test_qwen_mtp_port "$model" > "$result_dir/code-$context.log" 2>&1
done
printf 'Exact oracle logs: %s\n' "$result_dir"
