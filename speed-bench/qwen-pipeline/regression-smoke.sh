#!/bin/bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 BASELINE_BUILD_DIR CANDIDATE_BUILD_DIR [MODEL_ROOT]" >&2
  exit 2
fi
baseline=$1
experiment=$2
root=${3:-/Users/brianfarley/Desktop/ds4}
result=/private/tmp/qwen-promotion-other-models
mkdir -p "$result"
for family in ds4 glm; do
  if [[ $family == ds4 ]]; then
    model="$root/ds4flash.gguf"
  else
    model="$root/gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf"
  fi
  (cd "$baseline" && ./ds4 -m "$model" --metal --ctx 2048 -n 32 --temp 0 --nothink -p 'Write a Python function to add two integers. Briefly explain it.' > "$result/$family-base.out" 2> "$result/$family-base.log")
  (cd "$experiment" && ./ds4 -m "$model" --metal --ctx 2048 -n 32 --temp 0 --nothink -p 'Write a Python function to add two integers. Briefly explain it.' > "$result/$family-candidate.out" 2> "$result/$family-candidate.log")
  cmp "$result/$family-base.out" "$result/$family-candidate.out"
  printf '%s output parity PASS\n' "$family"
  shasum -a 256 "$result/$family-base.out" "$result/$family-candidate.out"
done
