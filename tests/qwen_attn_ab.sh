#!/bin/sh
# Serial, same-input A/B. Large models must never overlap.
set -eu
candidate=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
main=$(dirname "$(git -C "$candidate" rev-parse --path-format=absolute --git-common-dir)")
baseline=${QWEN_AB_BASELINE_ROOT:-$main}
case "$(git -C "$baseline" rev-parse HEAD)" in
 13517e9*) ;;
 *) echo "Set QWEN_AB_BASELINE_ROOT to a built checkout of 13517e9; updated main is not the baseline." >&2; exit 2;;
esac
cd "$candidate"
for model in 35 27; do
 for arm in ${QWEN_AB_ORDER:-base candidate}; do
  binary="$baseline"
  if [ "$arm" = candidate ]; then binary="$candidate"; fi
  if [ "$model" = 35 ]; then
   HEALTH_MODEL_FILE="$main/gguf/qwen-small-quality/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" \
   HEALTH_LABEL="attention-$model-$arm${QWEN_AB_SUFFIX:-}" HEALTH_BINARY_ROOT="$binary" \
   HEALTH_CHAT=1 HEALTH_CONTEXTS="${QWEN_AB_CONTEXTS:-short,2048}" HEALTH_TASKS=code,story,json HEALTH_REPEAT="${QWEN_AB_REPEAT:-2}" \
   python3 tests/model_api_health_probe.py qwen27-q8 > "/tmp/qwen-attn-api-$model-$arm${QWEN_AB_SUFFIX:-}.log" 2>&1
  else
   HEALTH_LABEL="attention-$model-$arm${QWEN_AB_SUFFIX:-}" HEALTH_BINARY_ROOT="$binary" \
   HEALTH_CHAT=1 HEALTH_CONTEXTS="${QWEN_AB_CONTEXTS:-short,2048}" HEALTH_TASKS=code,story,json HEALTH_REPEAT="${QWEN_AB_REPEAT:-2}" \
   python3 tests/model_api_health_probe.py qwen27-q8 > "/tmp/qwen-attn-api-$model-$arm${QWEN_AB_SUFFIX:-}.log" 2>&1
  fi
 done
done
