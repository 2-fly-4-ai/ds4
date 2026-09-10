#!/usr/bin/env python3
"""Validate a converted DeepSeek V4.1 split GGUF against its source plan."""

from __future__ import annotations

import argparse
import json
import os
import struct

from deepseek41_convert import QTYPE_IDS, SourceDB, output_layout, prepare_plan
from deepseek41_manifest import DEFAULT_REVISION, validate_config
from deepseek41_plan import build_plan
from glm53_quantize import (
    GGUF_ALIGNMENT,
    GGUF_BOOL,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
    align,
    fail,
    read_exact,
    read_gguf_string,
    read_u32,
    read_u64,
    skip_gguf_value,
)


def read_metadata_value(fp, value_type):
    if value_type == GGUF_STRING:
        return read_gguf_string(fp, "GGUF metadata string")
    if value_type == GGUF_UINT32:
        return read_u32(fp, "GGUF metadata uint32")
    if value_type == GGUF_UINT64:
        return read_u64(fp, "GGUF metadata uint64")
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", read_exact(fp, 4, "GGUF metadata float32"))[0]
    if value_type == GGUF_BOOL:
        return bool(read_exact(fp, 1, "GGUF metadata bool")[0])
    skip_gguf_value(fp, value_type)
    return None


def validate(path, hf_dir, artifact, quant, revision, verify_copy_payloads):
    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    validate_config(config)
    db = SourceDB(hf_dir)
    try:
        raw_plan = [x for x in build_plan(db.tensors, quant)
                    if x.artifact == artifact]
        plan = prepare_plan(raw_plan)
        wanted = {
            "general.architecture",
            "general.alignment",
            "general.source.revision",
            "deepseek41.sidecar.kind",
        }
        if artifact == "vision":
            wanted.update({
                "deepseek41-vision.block_count",
                "deepseek41-vision.embedding_length",
                "deepseek41-vision.feed_forward_length",
                "deepseek41-vision.attention.head_count",
                "deepseek41-vision.projection_length",
                "deepseek41-vision.patch_size",
                "deepseek41-vision.downsample_ratio",
                "deepseek41-vision.image.max_tokens",
                "deepseek41-vision.image.min_pixels",
                "deepseek41-vision.image_token_id",
                "deepseek41-vision.attention.layer_norm_rms_epsilon",
            })

        with open(path, "rb") as fp:
            if read_exact(fp, 4, "GGUF magic") != b"GGUF":
                fail(f"{path}: not a GGUF file")
            version = read_u32(fp, "GGUF version")
            if version != 3:
                fail(f"expected GGUF v3, got v{version}")
            tensor_count = read_u64(fp, "GGUF tensor count")
            metadata_count = read_u64(fp, "GGUF metadata count")
            if tensor_count != len(plan):
                fail(f"tensor count {tensor_count} != expected {len(plan)}")

            selected = {}
            for _ in range(metadata_count):
                key = read_gguf_string(fp, "GGUF metadata key")
                value_type = read_u32(fp, "GGUF metadata type")
                if key in wanted:
                    selected[key] = read_metadata_value(fp, value_type)
                else:
                    skip_gguf_value(fp, value_type)

            expected_arch = "deepseek41" if artifact == "main" else f"deepseek41-{artifact}"
            if selected.get("general.architecture") != expected_arch:
                fail(f"unexpected architecture {selected.get('general.architecture')!r}")
            if selected.get("general.alignment") != GGUF_ALIGNMENT:
                fail(f"unexpected alignment {selected.get('general.alignment')!r}")
            if selected.get("general.source.revision") != revision:
                fail(f"unexpected source revision {selected.get('general.source.revision')!r}")
            if artifact != "main" and selected.get("deepseek41.sidecar.kind") != artifact:
                fail(f"unexpected sidecar kind {selected.get('deepseek41.sidecar.kind')!r}")
            if artifact == "vision":
                vision = config["vision_config"]
                exact = {
                    "deepseek41-vision.block_count": vision["num_hidden_layers"],
                    "deepseek41-vision.embedding_length": vision["hidden_size"],
                    "deepseek41-vision.feed_forward_length": vision["intermediate_size"],
                    "deepseek41-vision.attention.head_count": vision["num_attention_heads"],
                    "deepseek41-vision.projection_length": config["text_config"]["hidden_size"],
                    "deepseek41-vision.patch_size": vision["patch_size"],
                    "deepseek41-vision.downsample_ratio": vision["downsample_ratio"],
                    "deepseek41-vision.image.max_tokens": vision["max_image_tokens"],
                    "deepseek41-vision.image.min_pixels": vision["min_pixels"],
                    "deepseek41-vision.image_token_id": config["image_token_id"],
                }
                for key, value in exact.items():
                    if selected.get(key) != value:
                        fail(f"{key}={selected.get(key)!r}, expected {value!r}")
                epsilon = selected.get("deepseek41-vision.attention.layer_norm_rms_epsilon")
                if epsilon is None or abs(epsilon - 1.0e-6) > 1.0e-12:
                    fail(f"unexpected vision RMS epsilon {epsilon!r}")

            for index, entry in enumerate(plan):
                item = entry.plan
                name = read_gguf_string(fp, f"tensor {index} name")
                rank = read_u32(fp, f"tensor {index} rank")
                shape = tuple(read_u64(fp, f"tensor {index} dimension")
                              for _ in range(rank))
                qtype = read_u32(fp, f"tensor {index} type")
                offset = read_u64(fp, f"tensor {index} offset")
                if name != item.target:
                    fail(f"tensor {index} is {name!r}, expected {item.target!r}")
                if shape != item.shape:
                    fail(f"{name}: shape {shape} != {item.shape}")
                if qtype != QTYPE_IDS[item.qtype]:
                    fail(f"{name}: qtype {qtype} != {QTYPE_IDS[item.qtype]}")
                if offset != entry.offset:
                    fail(f"{name}: offset {offset} != {entry.offset}")

            data_offset = align(fp.tell())
            expected_data_offset, data_bytes = output_layout(plan, [])
            # output_layout with no metadata cannot reproduce the header start;
            # tensor offsets and exact final size still prove the payload layout.
            del expected_data_offset
            expected_size = data_offset + data_bytes
            actual_size = os.fstat(fp.fileno()).st_size
            if actual_size != expected_size:
                fail(f"file size {actual_size} != expected {expected_size}")

            verified = 0
            if verify_copy_payloads:
                for entry in plan:
                    item = entry.plan
                    if item.mode != "copy" or len(item.sources) != 1:
                        continue
                    fp.seek(data_offset + entry.offset)
                    output = read_exact(fp, item.nbytes, item.target)
                    source = db.read_range(item.sources[0], 0, item.nbytes)
                    if output != source:
                        fail(f"{item.target}: payload differs from source")
                    verified += item.nbytes
    finally:
        db.close()

    print(f"validated {path}: {len(plan)} tensors, {actual_size} bytes")
    if verify_copy_payloads:
        print(f"copy payloads matched source byte-for-byte: {verified} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--artifact", choices=("main", "engram", "vision", "dspark"), required=True)
    parser.add_argument("--quant", choices=("native", "q2"), default="q2")
    parser.add_argument("--source-revision", default=DEFAULT_REVISION)
    parser.add_argument("--verify-copy-payloads", action="store_true")
    args = parser.parse_args()
    validate(args.gguf, args.hf, args.artifact, args.quant,
             args.source_revision, args.verify_copy_payloads)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"deepseek41-validate: error: {error}")
