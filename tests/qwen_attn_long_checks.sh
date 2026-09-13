#!/bin/sh
set -eu
candidate=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
main=$(dirname "$(git -C "$candidate" rev-parse --path-format=absolute --git-common-dir)")
baseline=${QWEN_AB_BASELINE_ROOT:-$main}
case "$(git -C "$baseline" rev-parse HEAD)" in
 13517e9*) ;;
 *) echo "Set QWEN_AB_BASELINE_ROOT to a built checkout of 13517e9; updated main is not the baseline." >&2; exit 2;;
esac
for model in 35 27; do
 weights="$main/gguf/qwen-small-quality/Qwen3.8-27B-Q8_0.gguf"
 if [ "$model" = 35 ]; then weights="$main/gguf/qwen-small-quality/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf"; fi
 cd "$baseline"
 PROBE_CONTEXT=3840 DS4_QWEN_NEXTN_DRAFT=0 /tmp/qwen-attn-baseline-probe "$weights" "/tmp/qwen-attn-$model-3840.ids" "/tmp/qwen-attn-$model-3840-base.logits" write > "/tmp/qwen-attn-$model-3840-base.log" 2>&1
 cd "$candidate"
 DS4_QWEN_NEXTN_DRAFT=0 tests/qwen_fixed_token_probe "$weights" "/tmp/qwen-attn-$model-3840.ids" "/tmp/qwen-attn-$model-3840-candidate.logits" read > "/tmp/qwen-attn-$model-3840-candidate.log" 2>&1
 python3 tests/compare_qwen_fixed_logits.py "/tmp/qwen-attn-$model-3840-base.logits" "/tmp/qwen-attn-$model-3840-candidate.logits"
 QWEN_TEST_CONTEXT=3840 DS4_QWEN_MTP_HEAD="$main/gguf/qwen-small-quality/mtp-Qwen3.8-27B-Q4_64A.gguf" \
 tests/test_qwen_small_runtime "$weights" > "/tmp/qwen-attn-$model-3840-runtime.log" 2>&1
done
