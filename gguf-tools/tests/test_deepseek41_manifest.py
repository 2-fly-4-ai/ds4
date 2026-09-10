#!/usr/bin/env python3

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek41_manifest import (
    BACKBONE_EXPERTS,
    BACKBONE_LAYERS,
    EXPECTED_TEXT_CONFIG,
    MTP_EXPERTS,
    MTP_LAYERS,
    VISION_LAYERS,
    tensor_role,
    validate_config,
    validate_index,
)


def complete_index():
    names = {
        "embed.weight", "norm.weight", "head.weight",
        "image_start", "image_end", "image_newline",
        "aligner.w1.weight", "aligner.w2.weight",
        "mtp.0.main_proj.weight", "mtp.0.main_norm.weight",
        "mtp.2.norm.weight", "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight", "mtp.2.confidence_head.proj.weight",
    }
    for layer in range(VISION_LAYERS):
        names.add(f"vision.blocks.{layer}.norm1.weight")
    for layer in range(BACKBONE_LAYERS):
        names.add(f"layers.{layer}.attn.wq_a.weight")
        for expert in range(BACKBONE_EXPERTS):
            for projection in ("w1", "w2", "w3"):
                names.add(f"layers.{layer}.ffn.experts.{expert}.{projection}.weight")
                names.add(f"layers.{layer}.ffn.experts.{expert}.{projection}.scale")
    for layer in (1, 14):
        names.add(f"layers.{layer}.engram.embed.weight")
    for layer in (2, 8, 14, 20):
        names.add(f"layers.{layer}.attn.compressor.wkv.weight")
    for layer in (2, 8, 14, 20, 24, 28, 32, 36):
        names.add(f"layers.{layer}.attn.indexer.wq_b.weight")
    for layer in range(MTP_LAYERS):
        names.add(f"mtp.{layer}.attn.wq_a.weight")
        for expert in range(MTP_EXPERTS):
            for projection in ("w1", "w2", "w3"):
                names.add(f"mtp.{layer}.ffn.experts.{expert}.{projection}.weight")
                names.add(f"mtp.{layer}.ffn.experts.{expert}.{projection}.scale")
    return {name: "model.safetensors" for name in names}


class DeepSeek41ManifestTests(unittest.TestCase):
    def test_release_config(self):
        config = {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": copy.deepcopy(EXPECTED_TEXT_CONFIG),
            "vision_config": {
                "model_type": "deepseek_v41_vision",
                "num_hidden_layers": 32,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "intermediate_size": 2816,
                "patch_size": 14,
                "downsample_ratio": 3,
                "max_image_tokens": 1024,
            },
        }
        validate_config(config)
        config["text_config"]["hidden_size"] = 4096
        with self.assertRaisesRegex(ValueError, "text_config.hidden_size"):
            validate_config(config)

    def test_complete_index_and_missing_expert(self):
        index = complete_index()
        validate_index(index)
        del index["layers.17.ffn.experts.93.w2.scale"]
        with self.assertRaisesRegex(ValueError, "layer 17 expert 93 tensors"):
            validate_index(index)

    def test_quantization_roles(self):
        self.assertEqual(tensor_role("layers.1.engram.embed.weight"), "engram_table")
        self.assertEqual(tensor_role("layers.4.ffn.experts.2.w1.weight"), "routed_expert")
        self.assertEqual(tensor_role("layers.8.attn.indexer.wq_b.weight"), "indexer")
        self.assertEqual(tensor_role("mtp.2.confidence_head.proj.weight"), "mtp")


if __name__ == "__main__":
    unittest.main()
