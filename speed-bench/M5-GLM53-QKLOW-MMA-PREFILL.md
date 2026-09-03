# M5 GLM-5.3 QK-low MMA prefill campaign

Measured 2026-09-03 on a MacBook Pro M5 Max with 128 GiB RAM. macOS was in
Automatic power mode, and the user forced maximum fan speed during the final
balanced runs to reduce thermal-order noise. The work was developed on the
isolated `experiment/m5-attention-prefill` branch; no upstream branch was
merged.

## Retained implementation

GLM-5.3's active Q2/Q4K checkpoints keep `attn_k_b.weight` in Q8_0. The old
batch path assigned one workgroup to each `(head, token)` pair, so every prompt
token reread and dequantized the same K_b matrix. The retained Metal kernel
uses a 64-output by 32-token simdgroup-MMA tile. It stages both operands in
full F32, dequantizes each Q8_0 weight tile once, and reuses it across 32 prompt
rows.

The automatic route is deliberately narrow:

- Apple M5 with the Metal 4 tensor API enabled;
- GLM-5.3's 64-head, 512-wide latent, 256-wide QK-low shape;
- Q8_0 K_b rows of 272 bytes;
- batches of at least 32 tokens; and
- non-quality mode.

Every other shape and platform retains its previous kernel. Set
`DS4_METAL_DISABLE_M5_GLM53_QKLOW_MMA=1` for a same-binary rollback.

## Stage result

Layer-7 profiling at a 4096-token frontier measured the QK-low stage at
58.252 ms before and 6.070 ms after: 9.60x stage throughput, or 89.6% less
stage time. The attention-LORA work is now the largest attention-side cost.

## Balanced full-model results

The comparison harness alternated default and rollback in ABBA/BAAB order and
rejected any run whose full-vocabulary logits differed by one bit.

| Model/workload | Prefill interval | Rollback | Optimized | Gain |
| --- | ---: | ---: | ---: | ---: |
| SharedDownQ4K, final source | 4,096 | 555.81 t/s | 603.68 t/s | +8.61% |
| Standard Q2/Q4K, final source | 4,096 | 566.27 t/s | 619.89 t/s | +9.47% |
| SharedDownQ4K, code | 4,096 | 557.44 t/s | 604.90 t/s | +8.51% |
| SharedDownQ4K, prose | 4,096 | 489.23 t/s | 529.16 t/s | +8.16% |
| SharedDownQ4K, structured | 4,096 | 488.27 t/s | 531.33 t/s | +8.82% |
| SharedDownQ4K, 12K to 16K suffix | 4,096 | 409.68 t/s | 438.95 t/s | +7.15% |

An independent context sweep measured +8.02% at 512 tokens, +10.40% at 1024,
+8.69% at 2048, and +8.12% at 4096. A 33-token tail case gained 1.52%, which
supports the 32-token activation threshold without exposing the slower tiny
batches to the specialization.

All accepted matrix runs produced bit-identical full-vocabulary logits. The
campaign compared more than 18 million logits across short tails, irregular
batch lengths, 512-through-4096 context points, three prompt disciplines, both
installed GLM Q2/Q4K checkpoints, and a resumed 12K-to-16K suffix.

Reproduce the final 4K comparison with:

```sh
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  -m gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf \
  --prompt-file ds4.c --prefix-tokens 4096 --warmup-tokens 256 \
  --repeats 2 \
  --candidate-env DS4_METAL_DISABLE_M5_GLM53_QKLOW_MMA
```

In that command, `control` is the optimized default and `candidate` is the
rollback, so the harness prints a negative candidate delta.

## Rejected variants

Nothing below remains in the runtime:

- Half-precision MMA staging made the isolated QK-low stage 10.9x faster but
  changed every final logit in the 4K gate, with 1.288 MAE, 1.629 RMSE, an
  8.55 maximum error, and a changed top token.
- A 64-token MMA tile was -0.96% at 512, +2.12% at 1K, -0.91% at 2K, and flat
  at 4K. The stable 32-token tile won.
- A Q4-only QK-low kernel did not match the installed model's Q8_0 K_b tensor
  and its half-staging design was not numerically safe.
- Changing GLM's causal FlashAttention slices from 2K to 4K was bit-exact but
  0.52% slower. Changing them to 1K was bit-exact but 0.34% slower.
- Replacing causal FlashAttention with the indexed attention implementation
  changed all 154,880 final logits in the first comparison.
- A four-query/eight-SIMDgroup FlashAttention occupancy variant also changed
  all 154,880 final logits and was removed.

The DeepSeek attention investigation also retained no code. Four-SIMDgroup
attention was exact but 2.26% slower at 2K and 0.51% slower at 4K; a 128-key
tile failed exactness; skipping runs of empty block-map entries was exact but
flat at 4K and -0.06% at a 60K-to-64K suffix; and the reduced-memory
four-query/four-SIMDgroup shape was exact but 15.37% slower. Internal stage
profiling showed that copies and block-map preparation are small compared with
the attention math itself.

## Validation status

`make -j8` completed cleanly. The relevant Metal kernel,
Metal tensor-equivalence, short-prefill, long-context, snapshot, agent, and
server tests passed. The complete `make test` run ended with the same three
model-fixture golden failures already documented on the parent branch because
`ds4flash.gguf` points to the newer Vision-Exp late-Q4 checkpoint rather than
the repository's golden reference model. This GLM-only shape gate is not
entered by those DeepSeek fixture tests. The optional GLM continued-prefill
test also exposes an existing resumed-versus-cold divergence with this model;
it produces the identical token mismatch when the new kernel is forcibly
disabled, while the resumed 12K-to-16K default/rollback A/B remains bit-exact.

## Next attention project

The easy selector, tile-size, copy, and block-map ideas are now measured out.
A further material gain needs a new exact dataflow: most plausibly a GLM
shared-latent-KV kernel that reuses one K/V tile across both multiple query rows
and multiple heads, or a DeepSeek long-context attention kernel that reduces
the actual QK/softmax/V math rather than its setup. Both are larger kernel
projects and should keep the same rollback and full-logit acceptance gates.
