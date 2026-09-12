#!/bin/sh
# Short SSD regression, not a storage throughput benchmark (OS cache is warm).
set -eu
main=${1:?main checkout}
port=${2:?port checkout}
result=${3:?new result directory}
mkdir -p "$result"
# The existing SSD backend supports IQ2_XXS gate/up, not the resident Q4_K quant.
model="$main/gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf"
ple="$main/gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
for arm in main port; do
    if [ "$arm" = main ]; then checkout=$main; else checkout=$port; fi
    (cd "$checkout" && ./ds4 --model "$model" --ple "$ple" --metal \
        --ssd-streaming --ssd-streaming-cache-experts 4GB --ctx 2048 \
        --temp 0 --nothink --mtp-timing -n 32 \
        -p 'Write a Python function to reverse a linked list. Return only code.') \
        > "$result/$arm.out" 2> "$result/$arm.log"
done
cmp "$result/main.out" "$result/port.out"
printf 'QWEN_Q8_SSD_OUTPUT_PARITY_PASS\n'
