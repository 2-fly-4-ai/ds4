#!/usr/bin/env python3
"""Inspect and validate the official DeepSeek-V4.1-Flash checkpoint.

The V4.1 checkpoint is large enough that conversion failures must be caught from
its index and shard headers before any tensor payload is read.  This program is
also the source of truth for the tensor families used by the mixed-quant plan.
"""

import argparse
import json
import os
import re
import sys

from glm53_manifest import load_index, load_safetensors_header


BACKBONE_LAYERS = 40
BACKBONE_EXPERTS = 384
MTP_LAYERS = 3
MTP_EXPERTS = 128
VISION_LAYERS = 32
ENGRAM_LAYERS = {1, 14}
KV_SOURCE_LAYERS = {2, 8, 14, 20}
INDEX_SOURCE_LAYERS = {2, 8, 14, 20, 24, 28, 32, 36}

LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
VISION_RE = re.compile(r"^vision\.blocks\.(\d+)\.")


EXPECTED_TEXT_CONFIG = {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "rms_norm_eps": 1e-20,
    "max_position_embeddings": 1048576,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "sliding_window": 128,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
}


def fail(message):
    raise ValueError(message)


def expect_equal(actual, expected, label):
    if actual != expected:
        fail(f"{label}: got {actual!r}, expected {expected!r}")


def validate_config(config):
    expect_equal(config.get("model_type"), "deepseek_v41", "model_type")
    expect_equal(config.get("architectures"), ["DeepseekV41ForCausalLM"], "architectures")
    text = config.get("text_config")
    if not isinstance(text, dict):
        fail("text_config is missing")
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        expect_equal(text.get(key), expected, f"text_config.{key}")

    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        fail("vision_config is missing")
    expected_vision = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 32,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2816,
        "patch_size": 14,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
    }
    for key, expected in expected_vision.items():
        expect_equal(vision.get(key), expected, f"vision_config.{key}")


def _collect_experts(names, prefix_re):
    layers = set()
    ids = {}
    parts = {}
    for name in names:
        match = prefix_re.match(name)
        if not match:
            continue
        layer = int(match.group(1))
        layers.add(layer)
        expert = EXPERT_RE.match(match.group(2))
        if expert:
            expert_id = int(expert.group(1))
            ids.setdefault(layer, set()).add(expert_id)
            parts.setdefault((layer, expert_id), set()).add(
                (expert.group(2), expert.group(3))
            )
    return layers, ids, parts


def validate_index(weight_map):
    names = set(weight_map)
    allowed = (
        "vision.", "aligner.", "layers.", "mtp.", "embed.", "norm.",
        "head.",
    )
    scalar_top = {"image_start", "image_end", "image_newline"}
    unknown = sorted(
        name for name in names if name not in scalar_top and not name.startswith(allowed)
    )
    if unknown:
        fail(f"unknown top-level tensors: {unknown[:5]}")

    required = {
        "embed.weight",
        "norm.weight",
        "head.weight",
        "image_start",
        "image_end",
        "image_newline",
        "aligner.w1.weight",
        "aligner.w2.weight",
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight",
        "mtp.2.confidence_head.proj.weight",
    }
    missing = sorted(required - names)
    if missing:
        fail(f"missing required tensors: {missing}")

    vision_layers = {
        int(match.group(1)) for name in names if (match := VISION_RE.match(name))
    }
    expect_equal(vision_layers, set(range(VISION_LAYERS)), "vision layer set")

    layers, expert_ids, expert_parts = _collect_experts(names, LAYER_RE)
    expect_equal(layers, set(range(BACKBONE_LAYERS)), "backbone layer set")
    expected_parts = {
        (projection, field)
        for projection in ("w1", "w2", "w3")
        for field in ("weight", "scale")
    }
    for layer in range(BACKBONE_LAYERS):
        expect_equal(
            expert_ids.get(layer), set(range(BACKBONE_EXPERTS)),
            f"layer {layer} expert ids",
        )
        for expert in range(BACKBONE_EXPERTS):
            expect_equal(
                expert_parts.get((layer, expert)), expected_parts,
                f"layer {layer} expert {expert} tensors",
            )

    mtp_layers, mtp_expert_ids, mtp_expert_parts = _collect_experts(names, MTP_RE)
    expect_equal(mtp_layers, set(range(MTP_LAYERS)), "MTP layer set")
    for layer in range(MTP_LAYERS):
        expect_equal(
            mtp_expert_ids.get(layer), set(range(MTP_EXPERTS)),
            f"MTP layer {layer} expert ids",
        )
        for expert in range(MTP_EXPERTS):
            expect_equal(
                mtp_expert_parts.get((layer, expert)), expected_parts,
                f"MTP layer {layer} expert {expert} tensors",
            )

    engram_layers = {
        int(match.group(1))
        for name in names
        if (match := LAYER_RE.match(name)) and match.group(2).startswith("engram.")
    }
    expect_equal(engram_layers, ENGRAM_LAYERS, "Engram layer set")

    compressor_layers = {
        int(match.group(1))
        for name in names
        if (match := LAYER_RE.match(name)) and match.group(2).startswith("attn.compressor.")
    }
    expect_equal(compressor_layers, KV_SOURCE_LAYERS, "KV source layer set")

    indexer_layers = {
        int(match.group(1))
        for name in names
        if (match := LAYER_RE.match(name)) and match.group(2).startswith("attn.indexer.")
    }
    expect_equal(indexer_layers, INDEX_SOURCE_LAYERS, "index source layer set")


def tensor_role(name):
    if name.startswith("vision.") or name.startswith("aligner.") or name.startswith("image_"):
        return "vision"
    if ".engram.embed." in name:
        return "engram_table"
    if ".engram." in name:
        return "engram_projection"
    match = LAYER_RE.match(name)
    if match:
        tail = match.group(2)
        if EXPERT_RE.match(tail):
            return "routed_expert"
        if tail.startswith("ffn.shared_experts."):
            return "shared_expert"
        if tail.startswith("ffn.gate."):
            return "router"
        if tail.startswith("attn.indexer."):
            return "indexer"
        if tail.startswith("attn.compressor."):
            return "compressor"
        if tail.startswith("attn."):
            return "attention"
        if tail.startswith("hc_"):
            return "hyper_connection"
        return "backbone_other"
    if MTP_RE.match(name):
        return "mtp"
    if name == "embed.weight":
        return "embedding"
    if name == "head.weight":
        return "output"
    return "other"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", help="downloaded deepseek-ai/DeepSeek-V4.1-Flash directory")
    parser.add_argument(
        "--allow-missing-shards", action="store_true",
        help="validate the index and report headers for shards already downloaded",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="print one byte-total row per tensor role instead of every tensor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.hf_dir, "config.json"), "rb") as fp:
        validate_config(json.load(fp))
    document, weight_map = load_index(
        os.path.join(args.hf_dir, "model.safetensors.index.json")
    )
    validate_index(weight_map)

    shard_names = sorted(set(weight_map.values()))
    tensors = {}
    missing_shards = []
    for shard_name in shard_names:
        path = os.path.join(args.hf_dir, shard_name)
        if not os.path.isfile(path):
            missing_shards.append(shard_name)
            continue
        for name, info in load_safetensors_header(path).items():
            if weight_map.get(name) != shard_name:
                fail(f"{shard_name}: {name} is assigned to {weight_map.get(name)!r}")
            if name in tensors:
                fail(f"duplicate tensor in shard headers: {name}")
            tensors[name] = info

    if missing_shards and not args.allow_missing_shards:
        fail(f"missing {len(missing_shards)} shards; first is {missing_shards[0]}")

    expected_present = {
        name for name, shard in weight_map.items() if shard not in missing_shards
    }
    if set(tensors) != expected_present:
        missing = sorted(expected_present - set(tensors))
        extra = sorted(set(tensors) - expected_present)
        fail(f"header/index mismatch: missing={missing[:3]} extra={extra[:3]}")

    if args.summary:
        totals = {}
        counts = {}
        for name, info in tensors.items():
            role = tensor_role(name)
            totals[role] = totals.get(role, 0) + info["nbytes"]
            counts[role] = counts.get(role, 0) + 1
        print("role\ttensors\tbytes\tGiB")
        for role in sorted(totals):
            print(f"{role}\t{counts[role]}\t{totals[role]}\t{totals[role] / (1 << 30):.3f}")
    else:
        print("role\tname\tdtype\tshape\tshard\toffset\tnbytes")
        for name in sorted(weight_map):
            info = tensors.get(name)
            fields = (
                info["dtype"], "x".join(map(str, info["shape"])),
                str(info["offset"]), str(info["nbytes"]),
            ) if info else ("-", "-", "-", "-")
            print(
                f"{tensor_role(name)}\t{name}\t{fields[0]}\t{fields[1]}\t"
                f"{weight_map[name]}\t{fields[2]}\t{fields[3]}"
            )

    declared = document.get("metadata", {}).get("total_size")
    print(
        f"deepseek41-manifest: tensors={len(weight_map)} shards={len(shard_names)} "
        f"present_shards={len(shard_names) - len(missing_shards)} "
        f"present_bytes={sum(info['nbytes'] for info in tensors.values())} "
        f"declared_bytes={declared} missing_shards={len(missing_shards)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"deepseek41-manifest: error: {error}", file=sys.stderr)
        sys.exit(1)
