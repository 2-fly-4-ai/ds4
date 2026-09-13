#!/bin/sh
# Run serially after the API matrix. Never overlap large model processes.
set -eu
cd "$(dirname "$0")/.."
out=speed-bench/qwen35-port
models=/Users/brianfarley/Desktop/ds4/gguf/qwen-small-quality
./tests/test_qwen35_gdn_port > "$out/gdn-final.log" 2>&1
./tests/test_qwen35_moe_port > "$out/moe-final.log" 2>&1
./tests/test_qwen_small_runtime "$models/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" "$models/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" > "$out/reopen-35b.log" 2>&1
DS4_QWEN_MTP_HEAD="$models/mtp-Qwen3.8-27B-Q4_64A.gguf" ./tests/test_qwen_small_runtime "$models/Qwen3.8-27B-Q8_0.gguf" "$models/Qwen3.8-27B-Q4_64A.gguf" "$models/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" > "$out/reopen-27b.log" 2>&1
HEALTH_MODEL_FILE="$models/35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf" HEALTH_LABEL=35b-reference HEALTH_BINARY_ROOT=/Users/brianfarley/Desktop/ds4-qwen-small-reference HEALTH_MTP_OFF=1 HEALTH_CHAT=1 python3 tests/model_api_health_probe.py qwen27-q8 > "$out/reference-35b.log" 2>&1
for model in qwen-next-q4 ds4-0731 ds4-vision-exp; do
    HEALTH_LABEL=port-regression python3 tests/model_api_health_probe.py "$model" context-sweep > "$out/regression-$model.log" 2>&1
done
HEALTH_LABEL=port-regression python3 tests/model_api_health_probe.py glm53 glm-low-scratch > "$out/regression-glm53.log" 2>&1
# Existing short-prompt SSD path only. No encoder residency or large prefills.
HEALTH_LABEL=port-regression python3 tests/model_api_health_probe.py ds41-ssd-safe v41-16gb > "$out/regression-ds41.log" 2>&1
./ds4_test --server > "$out/server-tests-final.log" 2>&1
python3 tests/summarize_qwen_runtime.py > "$out/RESULTS.md"
