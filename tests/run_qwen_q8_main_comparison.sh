#!/bin/sh
# Matched production/candidate CLI comparison; one model process at a time.
set -eu
main=${1:?main checkout}
candidate=${2:?candidate checkout}
result=${3:?new output directory}
mkdir -p "$result"
model="$main/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf"
ple="$main/gguf/qwen38-q4k-tensor/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
unset DS4_QWEN_GROUP4 DS4_QWEN_MXFP4_PAIR DS4_QWEN_Q4K_PAIR DS4_QWEN_EXPERT_OVERLAP DS4_QWEN_GPU_BREAKDOWN DS4_QWEN_Q8_MULTI
for task in copy code story; do
    case "$task" in
        copy) prompt=$(cat "$candidate/speed-bench/qwen_prompt_lookup_copy_chat.txt"); tokens=512 ;;
        code) prompt='Write a Python LRU cache implementation with a doubly linked list, a dictionary, and unit tests covering updates, eviction, and capacity one.'; tokens=256 ;;
        story) prompt='Write an original story about a lighthouse keeper who receives a message from tomorrow. Use vivid dialogue, a surprising ending, and no introductory explanation.'; tokens=256 ;;
    esac
    run=0
    for arm in main candidate candidate main; do
        run=$((run+1))
        if [ "$arm" = main ]; then checkout=$main; else checkout=$candidate; fi
        (cd "$checkout" && DS4_QWEN4_MOE_MM_NAX=1 ./ds4 \
            --model "$model" --ple "$ple" --metal --ctx 4096 --temp 0 --nothink \
            --mtp-timing -n "$tokens" -p "$prompt") \
            > "$result/$task-$run-$arm.out" 2> "$result/$task-$run-$arm.log"
        printf '%s %s %s\n' "$task" "$run" "$arm"
    done
done
