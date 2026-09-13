# Qwen API protocol repair — validation

Validation was performed in an isolated worktree before integration. No quant or kernel changes. Output caps are 256 tokens; nonempty streams alone do not certify complete answers.

| Run | Complete nonempty | Leaked ChatML boundaries | JSON checks | Edit checks |
|---|---:|---:|---:|---:|
| ds4-0731-regression | 5/5 | 0 | 0/1 | 1/1 |
| ds4-vision-exp-regression | 5/5 | 0 | 1/1 | 1/1 |
| ds41-ssd-safe-regression | 5/5 | 0 | 1/1 | 1/1 |
| glm53-regression | 5/5 | 0 | 1/1 | 1/1 |
| glm53-regression-guard-stop | 1/2 | 0 | 0/0 | 0/0 |
| qwen-next-q4-regression | 5/5 | 0 | 1/1 | 1/1 |
| qwen27-q4-64a-cache-repeat | 2/2 | 0 | 2/2 | 0/0 |
| qwen27-q4-64a-final-chat | 13/13 | 0 | 2/3 | 2/2 |
| qwen27-q4-64a-no-mtp | 5/5 | 0 | 1/1 | 1/1 |
| qwen27-q8-cache-repeat | 2/2 | 0 | 2/2 | 0/0 |
| qwen27-q8-final-chat | 13/13 | 0 | 3/3 | 2/2 |
| qwen27-q8-no-mtp | 5/5 | 0 | 1/1 | 1/1 |
| qwen27-q8-thinking | 1/1 | 0 | 0/0 | 0/0 |

## MTP-on/off exact text comparison

| Model | Identical short greedy cases |
|---|---:|
| qwen27-q8 | 5/5 |
| qwen27-q4-64a | 5/5 |

## Existing-model archived request replay

| Model | Identical text to main benchmark |
|---|---:|
| ds4-0731 | 5/5 |
| ds4-vision-exp | 5/5 |
| ds41-ssd-safe | 5/5 |
| glm53 | 5/5 |
| qwen-next-q4 | 5/5 |

## Short-request decode sanity check

Single-run medians across tasks, not a statistical speedup comparison. Model launch and prefill are excluded.

| Run | Median decode t/s |
|---|---:|
| ds4-0731-regression | 46.47 |
| ds4-vision-exp-regression | 46.76 |
| ds41-ssd-safe-regression | 12.15 |
| glm53-regression | 44.15 |
| qwen-next-q4-regression | 66.56 |
| qwen27-q4-64a-final-chat | 40.12 |
| qwen27-q4-64a-no-mtp | 25.80 |
| qwen27-q8-final-chat | 32.22 |
| qwen27-q8-no-mtp | 18.37 |
