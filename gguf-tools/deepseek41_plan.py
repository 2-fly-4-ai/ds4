#!/usr/bin/env python3
"""Plan the split GGUF artifacts for DeepSeek V4.1 Flash.

This is intentionally header-only: it validates complete source coverage and
computes exact output payload sizes before the 510 GB checkpoint finishes
downloading.  The plan is also consumed by the resumable converter.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from collections import Counter

from deepseek41_manifest import (
    DEFAULT_REPO,
    DEFAULT_REVISION,
    EXPERT_RE,
    LAYER_RE,
    MTP_RE,
    load_checkpoint_headers,
    tensor_role,
    validate_config,
    validate_index,
)
from glm53_manifest import load_index


QTYPE_LAYOUT = {
    "F32": (1, 4),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "IQ2_XXS": (256, 66),
    "Q2_K": (256, 84),
    "MXFP4": (32, 17),
    "I8": (1, 1),
}


@dataclasses.dataclass(frozen=True)
class PlanEntry:
    artifact: str
    target: str
    sources: tuple[str, ...]
    shape: tuple[int, ...]
    qtype: str
    mode: str
    nbytes: int


def fail(message):
    raise ValueError(message)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def qtype_nbytes(qtype, shape):
    block, block_bytes = QTYPE_LAYOUT[qtype]
    elements = product(shape)
    if not shape or shape[0] % block or elements % block:
        fail(f"{qtype} is incompatible with shape {shape}")
    return elements // block * block_bytes


def dense_qtype(name, info):
    if name in ("embed.weight", "head.weight"):
        return "Q8_0", "quantize"
    if info["dtype"] == "F32":
        return "F32", "copy"
    if info["dtype"] == "BF16":
        # The shared DeepSeek ViT kernels consume RMS weights directly as
        # BF16. Main-model norms are deliberately expanded to F32, but doing
        # that to the vision sidecar changes both its ABI and arithmetic.
        if name.startswith("vision.") and name.endswith("norm.weight"):
            return "BF16", "copy"
        if name.endswith(("_norm.weight", ".norm.weight")) or \
                ".ffn.gate.weight" in name:
            return "F32", "bf16_to_f32"
        return "BF16", "copy"
    if info["dtype"] == "F8_E4M3":
        return "Q8_0", "fp8_to_q8"
    fail(f"unsupported dense dtype {info['dtype']} for {name}")


def expert_target(prefix, part):
    suffix = {"w1": "gate", "w2": "down", "w3": "up"}[part]
    return f"{prefix}.ffn_{suffix}_exps.weight"


def layer_target(prefix, tail):
    exact = {
        "attn.attn_sink": "attn_sinks.weight",
        "attn.kv_norm.weight": "attn_kv_a_norm.weight",
        "attn.q_norm.weight": "attn_q_a_norm.weight",
        "attn.wkv.weight": "attn_kv.weight",
        "attn.wo_a.weight": "attn_output_a.weight",
        "attn.wo_b.weight": "attn_output_b.weight",
        "attn.wq_a.weight": "attn_q_a.weight",
        "attn.wq_b.weight": "attn_q_b.weight",
        "attn.compressor.norm.weight": "attn_compressor_norm.weight",
        "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
        "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
        "attn.indexer.k_norm.weight": "indexer.k_norm.weight",
        "attn.indexer.weights_proj.weight": "indexer.proj.weight",
        "attn.indexer.wk.weight": "indexer.attn_k.weight",
        "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
        "attn_norm.weight": "attn_norm.weight",
        "ffn.gate.weight": "ffn_gate_inp.weight",
        "ffn.gate.bias": "exp_probs_b.bias",
        "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
        "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
        "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
        "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "hc_attn_base": "hc_attn_base.weight",
        "hc_attn_fn": "hc_attn_fn.weight",
        "hc_attn_scale": "hc_attn_scale.weight",
        "hc_ffn_base": "hc_ffn_base.weight",
        "hc_ffn_fn": "hc_ffn_fn.weight",
        "hc_ffn_scale": "hc_ffn_scale.weight",
        "engram.k_weight": "engram.k.weight",
        "engram.q_weight": "engram.q.weight",
        "engram.wkv.weight": "engram.wkv.weight",
        "main_norm.weight": "main_norm.weight",
        "main_proj.weight": "main_proj.weight",
        "norm.weight": "norm.weight",
        "markov_head.embed.weight": "markov_head.markov_w1.weight",
        "markov_head.head.weight": "markov_head.markov_w2.weight",
        "confidence_head.proj.weight": "confidence_head.proj.weight",
    }
    try:
        return prefix + "." + exact[tail]
    except KeyError:
        fail(f"no target tensor mapping for {prefix}: {tail}")


def regular_target(name):
    if name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "head.weight":
        return "output.weight"
    match = LAYER_RE.match(name)
    if match:
        return layer_target(f"blk.{int(match.group(1))}", match.group(2))
    match = MTP_RE.match(name)
    if match:
        return layer_target(f"mtp.{int(match.group(1))}", match.group(2))
    if name.startswith(("vision.", "aligner.")) or name.startswith("image_"):
        return name
    fail(f"no regular target mapping for {name}")


def artifact_for(name):
    role = tensor_role(name)
    if role == "vision":
        return "vision"
    if role == "engram_table":
        return "engram"
    if role == "mtp":
        return "dspark"
    return "main"


def build_expert_entry(tensors, artifact, layer_kind, layer, part, count, quant):
    source_prefix = f"{layer_kind}.{layer}.ffn.experts"
    sources = []
    first_shape = None
    for expert in range(count):
        weight = f"{source_prefix}.{expert}.{part}.weight"
        scale = f"{source_prefix}.{expert}.{part}.scale"
        weight_info = tensors[weight]
        scale_info = tensors[scale]
        if weight_info["dtype"] != "I8" or scale_info["dtype"] != "F8_E8M0":
            fail(f"{weight}: native expert is not packed FP4 + E8M0")
        out_rows, packed_columns = weight_info["shape"]
        logical_shape = (out_rows, packed_columns * 2)
        if scale_info["shape"] != [out_rows, packed_columns // 16]:
            fail(f"{scale}: scale shape does not cover 32-value blocks")
        if first_shape is None:
            first_shape = logical_shape
        elif logical_shape != first_shape:
            fail(f"{weight}: expert shape differs within stacked tensor")
        sources.extend((weight, scale))
    prefix = f"blk.{layer}" if layer_kind == "layers" else f"mtp.{layer}"
    shape = (first_shape[1], first_shape[0], count)
    if quant == "native":
        qtype, mode = "MXFP4", "repack_mxfp4"
    else:
        qtype = "Q2_K" if part == "w2" else "IQ2_XXS"
        mode = "fp4_to_q2"
    return PlanEntry(
        artifact,
        expert_target(prefix, part),
        tuple(sources),
        shape,
        qtype,
        mode,
        qtype_nbytes(qtype, shape),
    )


def build_plan(tensors, quant):
    plan = []
    consumed = set()
    for layer_kind, n_layers, n_experts, artifact in (
        ("layers", 40, 384, "main"),
        ("mtp", 3, 128, "dspark"),
    ):
        for layer in range(n_layers):
            for part in ("w1", "w2", "w3"):
                entry = build_expert_entry(
                    tensors, artifact, layer_kind, layer, part, n_experts, quant
                )
                plan.append(entry)
                consumed.update(entry.sources)

    for name in sorted(tensors):
        if name in consumed:
            continue
        artifact = artifact_for(name)
        info = tensors[name]
        if artifact == "engram" and ".engram.embed." in name:
            qtype, mode = "I8", "copy_fp8_sidecar"
        elif info["dtype"] == "F8_E8M0" and name.endswith(".scale"):
            weight = name[:-len("scale")] + "weight"
            if weight not in tensors:
                fail(f"orphan FP8 scale tensor: {name}")
            consumed.add(name)
            continue
        else:
            qtype, mode = dense_qtype(name, info)
        shape = tuple(reversed(info["shape"]))
        if artifact == "engram" and ".engram.embed." in name:
            layer = int(LAYER_RE.match(name).group(1))
            suffix = "weight.fp8" if name.endswith(".weight") else "scale.e8m0"
            target = f"engram.{layer}.{suffix}"
        else:
            target = regular_target(name)
        sources = (name,)
        if mode == "fp8_to_q8":
            scale = name[:-len("weight")] + "scale"
            if scale not in tensors:
                fail(f"missing E8M0 scale for {name}")
            sources = (name, scale)
            consumed.add(scale)
        plan.append(
            PlanEntry(
                artifact,
                target,
                sources,
                shape,
                qtype,
                mode,
                qtype_nbytes(qtype, shape),
            )
        )
        consumed.add(name)

    if consumed != set(tensors):
        missing = sorted(set(tensors) - consumed)
        extra = sorted(consumed - set(tensors))
        fail(f"source coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    targets = [(entry.artifact, entry.target) for entry in plan]
    if len(targets) != len(set(targets)):
        duplicates = [target for target, count in Counter(targets).items() if count > 1]
        fail(f"duplicate output tensors: {duplicates[:3]}")
    return sorted(plan, key=lambda entry: (entry.artifact, entry.target))


def print_summary(plan, quant, detailed):
    print(f"deepseek41-plan: quant={quant} tensors={len(plan)}")
    totals = Counter()
    type_totals = Counter()
    for entry in plan:
        totals[entry.artifact] += entry.nbytes
        type_totals[(entry.artifact, entry.qtype)] += entry.nbytes
        if detailed:
            print(
                f"{entry.artifact}\t{entry.qtype}\t{entry.nbytes}\t"
                f"{entry.mode}\t{entry.target}"
            )
    for artifact in ("main", "engram", "vision", "dspark"):
        print(
            f"artifact_bytes\t{artifact}\t{totals[artifact]}\t"
            f"{totals[artifact] / (1 << 30):.3f} GiB"
        )
        for (entry_artifact, qtype), nbytes in sorted(type_totals.items()):
            if entry_artifact == artifact:
                print(f"type_bytes\t{artifact}\t{qtype}\t{nbytes}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir")
    parser.add_argument("--quant", choices=("native", "q2"), default="native")
    parser.add_argument("--fetch-missing-headers", action="store_true")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--detailed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.hf_dir, "config.json"), "rb") as fp:
        validate_config(json.load(fp))
    _, weight_map = load_index(os.path.join(args.hf_dir, "model.safetensors.index.json"))
    validate_index(weight_map)
    tensors, missing = load_checkpoint_headers(
        args.hf_dir,
        weight_map,
        fetch_missing_headers=args.fetch_missing_headers,
        repo=args.repo,
        revision=args.revision,
    )
    if missing:
        fail(f"missing {len(missing)} shard headers; use --fetch-missing-headers")
    plan = build_plan(tensors, args.quant)
    print_summary(plan, args.quant, args.detailed)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"deepseek41-plan: error: {error}")
