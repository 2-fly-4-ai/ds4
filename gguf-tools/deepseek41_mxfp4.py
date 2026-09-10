#!/usr/bin/env python3
"""Losslessly repack DeepSeek V4.1's native FP4 experts as GGUF MXFP4.

The checkpoint stores adjacent columns in each byte (low=2*j, high=2*j+1)
and keeps one E8M0 scale byte per 32 values.  GGUF MXFP4 stores columns
0..15 in the low nibbles and 16..31 in the high nibbles, preceded by the
same E8M0 byte.  No dequantization or requantization is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct

from glm53_manifest import load_index, load_safetensors_header


MXFP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def fail(message):
    raise ValueError(message)


def e8m0_value(code):
    bits = 0x00400000 if code == 0 else code << 23
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def adjacent_nibble(packed, column):
    byte = packed[column // 2]
    return (byte >> (4 * (column & 1))) & 0x0F


def repack_block(adjacent_packed, scale):
    if len(adjacent_packed) != 16:
        fail(f"an FP4 block must contain 16 packed bytes, got {len(adjacent_packed)}")
    out = bytearray((scale,))
    for column in range(16):
        lo = adjacent_nibble(adjacent_packed, column)
        hi = adjacent_nibble(adjacent_packed, column + 16)
        out.append(lo | (hi << 4))
    return bytes(out)


def repack_row(adjacent_packed, scales, columns):
    if columns <= 0 or columns % 32:
        fail(f"expert row width {columns} is not a positive multiple of 32")
    if len(adjacent_packed) != columns // 2:
        fail("packed FP4 row has the wrong byte count")
    if len(scales) != columns // 32:
        fail("E8M0 scale row has the wrong byte count")
    out = bytearray()
    for block, scale in enumerate(scales):
        start = block * 16
        out += repack_block(adjacent_packed[start:start + 16], scale)
    return bytes(out)


def decode_adjacent_block(packed, scale):
    factor = e8m0_value(scale)
    return [MXFP4_VALUES[adjacent_nibble(packed, i)] * factor for i in range(32)]


def decode_mxfp4_block(block):
    if len(block) != 17:
        fail("GGUF MXFP4 block must be 17 bytes")
    factor = e8m0_value(block[0])
    values = []
    for i in range(16):
        values.append(MXFP4_VALUES[block[1 + i] & 0x0F] * factor)
    for i in range(16):
        values.append(MXFP4_VALUES[block[1 + i] >> 4] * factor)
    return values


def validate_block(adjacent_packed, scale):
    converted = repack_block(adjacent_packed, scale)
    if decode_adjacent_block(adjacent_packed, scale) != decode_mxfp4_block(converted):
        fail("MXFP4 repack changed decoded values")


def load_tensor_bytes(hf_dir, tensor_name):
    _, weight_map = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
    try:
        shard = weight_map[tensor_name]
    except KeyError:
        fail(f"tensor is not in checkpoint index: {tensor_name}")
    path = os.path.join(hf_dir, shard)
    if not os.path.isfile(path):
        fail(f"source shard is not downloaded: {path}")
    header = load_safetensors_header(path)
    try:
        info = header[tensor_name]
    except KeyError:
        fail(f"tensor is not present in assigned shard: {tensor_name}")
    with open(path, "rb") as fp:
        fp.seek(info["offset"])
        data = fp.read(info["nbytes"])
    if len(data) != info["nbytes"]:
        fail(f"short read for {tensor_name}")
    return info, data


def validate_checkpoint_sample(hf_dir, tensor_name, rows, seed):
    weight_info, weights = load_tensor_bytes(hf_dir, tensor_name)
    scale_name = tensor_name[:-len("weight")] + "scale"
    scale_info, scales = load_tensor_bytes(hf_dir, scale_name)
    if weight_info["dtype"] != "I8" or scale_info["dtype"] != "F8_E8M0":
        fail(f"unexpected source dtypes: {weight_info['dtype']}, {scale_info['dtype']}")
    out_rows, packed_columns = weight_info["shape"]
    columns = packed_columns * 2
    if scale_info["shape"] != [out_rows, columns // 32]:
        fail(f"scale shape {scale_info['shape']} does not cover {weight_info['shape']}")
    rng = random.Random(seed)
    selected = sorted({0, out_rows - 1, *(rng.randrange(out_rows) for _ in range(rows))})
    packed_stride = packed_columns
    scale_stride = columns // 32
    for row in selected:
        packed = weights[row * packed_stride:(row + 1) * packed_stride]
        row_scales = scales[row * scale_stride:(row + 1) * scale_stride]
        converted = repack_row(packed, row_scales, columns)
        for block in range(columns // 32):
            source = packed[block * 16:(block + 1) * 16]
            target = converted[block * 17:(block + 1) * 17]
            if target[0] != row_scales[block]:
                fail(f"row {row} block {block}: E8M0 scale changed")
            if decode_adjacent_block(source, row_scales[block]) != decode_mxfp4_block(target):
                fail(f"row {row} block {block}: decoded FP4 values changed")
    return len(selected), out_rows, columns


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir")
    parser.add_argument("--tensor", default="layers.0.ffn.experts.0.w1.weight")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0xD541)
    return parser.parse_args()


def main():
    args = parse_args()
    checked, total, columns = validate_checkpoint_sample(
        args.hf_dir, args.tensor, args.rows, args.seed
    )
    print(
        f"deepseek41-mxfp4: {args.tensor}: {checked}/{total} rows sampled, "
        f"{columns} columns, bit-exact decoded values"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"deepseek41-mxfp4: error: {error}")
