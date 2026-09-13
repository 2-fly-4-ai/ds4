# Qwen 35B runtime port — prerequisite checkpoint

2026-09-14. Isolated from main 7f69f18 in port/qwen35-runtime-20260914.
**Not a finished model port, speedup claim, or merge candidate yet.**

## Verified in this checkpoint

- Added a separate 32-value-head/40-trunk-layer GDN Metal specialization.
  Existing 27B GDN source and entry points are unchanged.
- New wrapper takes caller-owned prefix snapshots rather than the 27B global
  snapshots, and rejects invalid layer/count/buffer/snapshot sizes before dispatch.
- 32 synthetic cases: 27B/35B, first/last layer, 1/2/8/17 rows, zero/nonzero
  initial recurrent state. Each runs batched and repeated single-token forms.
- All output/state comparisons pass the double-precision reference tolerance
  (5e-5 relative to tensor peak). Batched/scalar output, recurrent state and
  convolution history are byte-identical. Neighboring layer state is unchanged.
- For 35B, every 2/8-token prefix snapshot matches the corresponding scalar
  state exactly, including at layer 39. Invalid-input rejection checks pass.
- The initial synthetic run caught a K-offset typo in the new specialization;
  fixed before model inference. Final complete raw results: gdn-synthetic.log.
- `make -j4 tests/test_qwen35_gdn_port ds4 ds4-server ds4_test` passed.
  Existing unused-code warnings remain.
- `./ds4_test --server` passed (including its expected incomplete-tool-call diagnostic).
- Metadata/tokenizer-only `test_qwen_chat_tokens` passed for 27B Q8 and Q4_64A.
- Installed `35b-mtp/Qwen3.6-35B-A3B-Q8_0.gguf` passed `test_qwen35_contract`:
  metadata block_count=41, width=2048, experts=256, topk=8, 30 GDN trunk blocks,
  10 full-attention trunk blocks, native `blk.40.nextn.eh_proj.weight` present.
- Swap remained zero. No large CPU inference, model weight mutation, residency
  experiment, full-model generation, or performance benchmark was run.

## Still required — do not enable 35B by removing the loader refusal alone

1. Port validated 35B metadata/weight binding, routed and shared FFN scheduling.
   The reference implementation is at ../ds4-qwen-small-reference (945f28f).
   Its qwen_gpu_moe_ffn uses existing routed MoE kernels, selected-expert
   renormalized softmax routing, and a separately gated shared expert.
2. Make host pool allocation, attention dimensions, reset/rewind, and hidden
   capture shape-aware. Current target/MTP paths contain fixed 5120/64/48/24/4
   dimensions; 35B needs 2048/40/32/16/2. Current full-attention kernels already
   accept dimensions, but host wrappers hard-code the 27B values.
3. Keep trunk layer count separate from the bundled MTP block. Prove target
   logits and recurrent/KV state before enabling native MTP and auto routing.
4. Re-test the existing models with matched prompts and wall-time benchmarks;
   no inference-level regression/performance certification is claimed here.
5. Only then merge verified runtime changes. Main remains untouched at 7f69f18.
6. 27B prompt-cache reuse remains outstanding. Its global recurrent pool and
   session snapshot lifetime need an ownership-correct port, not cache reporting
   changes. JSON instruction-following misses are not yet established engine bugs.

The older broad runtime commit b2396f6 is a reference, not a cherry-pick:
it overlaps and would overwrite newer main optimizations if transplanted wholesale.
