#!/usr/bin/env python3
"""Repack Ivan Fioravanti's Qwen3.8 DS4 fast-pack for upstream-schema DS4.

The fast-pack deliberately uses a private four-artifact ABI and native HF
tensor ordering.  The optimized DS4 tree uses llama.cpp's ``qwen4exp`` GGUF
schema and a single mapped model.  This tool joins the base, PLE and optional
MTP artifacts while preserving quantized payloads whenever their layouts are
already compatible.  It also reverses the pack-only layout changes:

* unfold the hyper-connection /4 factor;
* reorder GDN value heads to llama.cpp's tiled broadcast order;
* convert raw A_log to -exp(A_log);
* remove Q4_0 expert-down padding (and requantize the MTP Q4_K down matrix);
* split the packed QSA q/k index projection; and
* join the two MTP input projections.

Use a known-good Qwen3.8 GGUF as ``--template``.  Its metadata and expected
trunk tensor-name set are used as a compatibility oracle.  Tensor payloads are
streamed to the output, so the conversion does not retain the model in RAM.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--ple", required=True, type=Path)
    p.add_argument("--mtp", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--llama-cpp", type=Path,
                   default=Path(os.environ.get("LLAMA_CPP", "~/.unsloth/llama.cpp")).expanduser())
    p.add_argument("--plan", action="store_true", help="validate and print the plan without writing")
    return p.parse_args()


@dataclass
class Entry:
    name: str
    tensor: object | None
    shape: tuple[int, ...]
    qtype: object
    operation: str = "copy"
    auxiliary: object | None = None


def bf16_f32(data: np.ndarray) -> np.ndarray:
    words = data.view(np.uint16).astype(np.uint32)
    return np.ascontiguousarray((words << 16).view(np.float32))


def value_head_permutation(head_dim: int) -> np.ndarray:
    # HF: [K0 v0..v2, K1 v0..v2, ...]; GGML repeat: [v0 K0..K15,
    # v1 K0..K15, v2 K0..K15].  Values in the result are source indexes.
    return np.arange(48 * head_dim).reshape(16, 3, head_dim).transpose(1, 0, 2).reshape(-1)


def custom_layer_name(name: str, layer: int) -> str | None:
    prefix = f"language_model.model.layers.{layer}."
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    if tail == "linear_attn.dt_bias":
        return f"blk.{layer}.ssm_dt.bias"
    routed = {
        "mlp.switch_mlp.gate_proj.weight": "ffn_gate_exps.weight",
        "mlp.switch_mlp.up_proj.weight": "ffn_up_exps.weight",
        "mlp.switch_mlp.down_proj.weight": "ffn_down_exps.weight",
    }
    if tail in routed:
        return f"blk.{layer}.{routed[tail]}"
    return None


def transformed_operation(raw: str, mapped: str, tensor: object, *, mtp: bool) -> tuple[str, object | None]:
    qtype = int(tensor.tensor_type)
    shape = tuple(int(x) for x in tensor.shape)

    if raw.endswith("linear_attn.in_proj_qkv.weight"):
        perm = np.concatenate((np.arange(4096), 4096 + value_head_permutation(128)))
        return "q8_rows", perm
    if raw.endswith("linear_attn.in_proj_z.weight"):
        return "q8_rows", value_head_permutation(128)
    if raw.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return "bf16_q8_rows", value_head_permutation(1)
    if raw.endswith("linear_attn.out_proj.weight"):
        perm = value_head_permutation(128).reshape(-1, 32)[:, 0] // 32
        return "q8_blocks", perm
    if raw.endswith("linear_attn.conv1d.weight"):
        perm = np.concatenate((np.arange(4096), 4096 + value_head_permutation(128)))
        return "conv_f32_rows", perm
    if raw.endswith("linear_attn.A_log"):
        return "a_log_f32", value_head_permutation(1)
    if raw.endswith("linear_attn.dt_bias"):
        return "bf16_f32_rows", value_head_permutation(1)

    if raw.endswith(".self_attn.indexer.index_qk_proj.weight"):
        raise AssertionError("the packed index projection must be split before operation selection")

    if raw.endswith(".ple.conv1d.weight"):
        return "conv_f32", None

    if qtype == 30 and raw.endswith((".mlp.gate.weight", ".mlp.shared_expert_gate.weight")):
        return "bf16_f32", None

    if qtype == 30 and "hyper_connection" in raw:
        if raw.endswith("hc_norm.weight"):
            return "bf16_f32", None
        scale = 4.0 if raw.endswith(("input_mix_weight_down.weight", "block_inject_weight.weight")) else 1.0
        return "bf16_f16", scale

    if qtype == 30 and raw.startswith("language_model.mtp.hyper_connection_mixer."):
        if raw.endswith("hc_norm.weight"):
            return "bf16_f32", None
        scale = 4.0 if raw.endswith("input_mix_weight_down.weight") else 1.0
        return "bf16_f16", scale

    if qtype == 30 and len(shape) == 1:
        return "bf16_f32", None

    if mapped.endswith("ffn_down_exps.weight") and shape[0] == 768:
        if qtype == 2:
            return "q4_0_strip_down", None
        if qtype == 12 and mtp:
            return "q4_k_down_to_q8", None
        raise ValueError(f"{raw}: unsupported padded expert-down qtype {qtype}")

    return "copy", None


def output_spec(entry: Entry, gguf: object) -> tuple[np.dtype, int, object | None]:
    op = entry.operation
    if op in {"bf16_f32", "bf16_f32_rows", "a_log_f32", "conv_f32", "conv_f32_rows"}:
        return np.dtype(np.float32), int(np.prod(entry.shape)) * 4, None
    if op == "bf16_f16":
        return np.dtype(np.float16), int(np.prod(entry.shape)) * 2, None
    if op == "bf16_q8_rows":
        n = int(np.prod(entry.shape))
        return np.dtype(np.uint8), n // 32 * 34, gguf.GGMLQuantizationType.Q8_0
    if op == "q4_0_strip_down":
        n = int(np.prod(entry.shape))
        return np.dtype(np.uint8), n // 32 * 18, gguf.GGMLQuantizationType.Q4_0
    if op == "q4_k_down_to_q8":
        n = int(np.prod(entry.shape))
        return np.dtype(np.uint8), n // 32 * 34, gguf.GGMLQuantizationType.Q8_0
    if op in {"q8_rows", "q8_blocks", "q8_concat"}:
        n = int(np.prod(entry.shape))
        return np.dtype(np.uint8), n // 32 * 34, gguf.GGMLQuantizationType.Q8_0
    assert entry.tensor is not None
    return entry.tensor.data.dtype, int(entry.tensor.n_bytes), entry.qtype


def materialize(entry: Entry, gguf: object) -> np.ndarray:
    op, tensor, aux = entry.operation, entry.tensor, entry.auxiliary
    if op == "q8_concat":
        a, b = aux
        return np.ascontiguousarray(np.concatenate((a.data, b.data), axis=-1))
    assert tensor is not None
    data = tensor.data
    if op == "copy":
        return data
    if op == "bf16_f32":
        return bf16_f32(data)
    if op == "bf16_f16":
        return np.ascontiguousarray((bf16_f32(data) * np.float32(aux)).astype(np.float16))
    if op == "bf16_f32_rows":
        return np.ascontiguousarray(bf16_f32(data).reshape(-1)[aux])
    if op == "a_log_f32":
        values = bf16_f32(data).reshape(-1)[aux]
        return np.ascontiguousarray(-np.exp(values, dtype=np.float32))
    if op == "bf16_q8_rows":
        values = bf16_f32(data).reshape(tuple(reversed(tuple(int(x) for x in tensor.shape))))
        values = np.ascontiguousarray(values[aux])
        return gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
    if op == "q8_rows":
        return np.ascontiguousarray(data[aux])
    if op == "q8_blocks":
        rows = data.reshape(data.shape[0], -1, 34)
        return np.ascontiguousarray(rows[:, aux, :].reshape(data.shape[0], -1))
    if op in {"conv_f32", "conv_f32_rows"}:
        values = bf16_f32(data).reshape(tuple(reversed(tuple(int(x) for x in tensor.shape)))).squeeze()
        if op == "conv_f32_rows":
            values = values[aux]
        return np.ascontiguousarray(values)
    if op == "q4_0_strip_down":
        rows = data.reshape(-1, 24, 18)
        return np.ascontiguousarray(rows[:, :20, :].reshape(*data.shape[:-1], 360))
    if op == "q4_k_down_to_q8":
        values = gguf.quants.dequantize(data, tensor.tensor_type)
        values = np.ascontiguousarray(values[..., :640])
        return gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
    raise AssertionError(f"unknown operation {op}")


def copy_template_metadata(writer: object, template: object, *, with_mtp: bool, gguf: object) -> None:
    skip = {
        "general.architecture", "general.name", "qwen4exp.block_count",
        "qwen4exp.nextn_predict_layers", "qwen4exp.attention.compress_ratios",
    }
    for key, field in template.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        subtype = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=subtype)
    writer.add_name("Qwen3.8 Flash Next DS4 fast-pack compatibility")
    writer.add_uint32("qwen4exp.block_count", 49 if with_mtp else 48)
    if with_mtp:
        writer.add_uint32("qwen4exp.nextn_predict_layers", 1)
    ratios = list(template.fields["qwen4exp.attention.compress_ratios"].contents())
    if with_mtp:
        ratios.append(4)
    writer.add_key_value("qwen4exp.attention.compress_ratios", ratios,
                         gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    sys.path.insert(0, str(args.llama_cpp))
    import gguf

    for path in (args.template, args.base, args.ple, args.mtp):
        if path is not None and not path.is_file():
            raise SystemExit(f"missing input: {path}")

    template = gguf.GGUFReader(args.template, "r")
    base = gguf.GGUFReader(args.base, "r")
    ple = gguf.GGUFReader(args.ple, "r")
    mtp_reader = gguf.GGUFReader(args.mtp, "r") if args.mtp else None
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, 49)

    entries: list[Entry] = []
    split_index = 0
    for tensor in base.tensors:
        raw = tensor.name
        mapped = None
        match = re.match(r"language_model\.model\.layers\.(\d+)\.", raw)
        if match:
            mapped = custom_layer_name(raw, int(match.group(1)))
        if mapped is None:
            stripped = raw.removeprefix("language_model.")
            mapped = name_map.get_name(stripped, try_suffixes=(".weight", ".bias"))
        if raw.endswith(".self_attn.indexer.index_qk_proj.weight"):
            layer = int(match.group(1)) if match else -1
            if layer < 0 or int(tensor.tensor_type) != 8 or tuple(map(int, tensor.shape)) != (2560, 640):
                raise ValueError(f"{raw}: invalid packed QSA index projection")
            for suffix, rows in (("q_proj", slice(0, 512)), ("k_proj", slice(512, 640))):
                entries.append(Entry(f"blk.{layer}.indexer.{suffix}.weight", tensor,
                                     (2560, 512 if suffix == "q_proj" else 128), tensor.tensor_type,
                                     "q8_rows", np.arange(640)[rows]))
            split_index += 1
            continue
        if mapped is None:
            raise ValueError(f"no upstream-schema tensor name for {raw}")
        shape = tuple(int(x) for x in tensor.shape)
        op, aux = transformed_operation(raw, mapped, tensor, mtp=False)
        if op == "q4_0_strip_down":
            shape = (640, shape[1], shape[2])
        elif op in {"conv_f32", "conv_f32_rows"}:
            shape = (4, shape[-1])
        entries.append(Entry(mapped, tensor, shape, tensor.tensor_type, op, aux))

    ple_weight = next((t for t in ple.tensors if t.name == "ple.weight"), None)
    if ple_weight is None or int(ple_weight.tensor_type) != 3 or tuple(map(int, ple_weight.shape)) != (160, 320001536):
        raise ValueError("PLE sidecar does not contain the expected Q4_1 ple.weight")
    entries.append(Entry("per_layer_token_embd.weight", ple_weight,
                         tuple(map(int, ple_weight.shape)), ple_weight.tensor_type))

    trunk_names = {entry.name for entry in entries}
    expected_names = {tensor.name for tensor in template.tensors}
    if len(entries) != len(trunk_names):
        raise ValueError("the converted trunk contains duplicate tensor names")
    if trunk_names != expected_names:
        missing = sorted(expected_names - trunk_names)
        extra = sorted(trunk_names - expected_names)
        raise ValueError(f"trunk schema mismatch; missing={missing[:12]}, extra={extra[:12]}")
    if split_index != 12:
        raise ValueError(f"expected 12 packed QSA index projections, found {split_index}")

    if mtp_reader is not None:
        tensors = {tensor.name: tensor for tensor in mtp_reader.tensors}
        fc_e = tensors.pop("language_model.mtp.fc_embedding.weight")
        fc_h = tensors.pop("language_model.mtp.fc_hidden.weight")
        if int(fc_e.tensor_type) != 8 or int(fc_h.tensor_type) != 8:
            raise ValueError("MTP input projections must be Q8_0")
        entries.append(Entry("blk.48.nextn.eh_proj.weight", None, (5120, 2560),
                             gguf.GGMLQuantizationType.Q8_0, "q8_concat", (fc_e, fc_h)))
        for raw, tensor in tensors.items():
            if raw == "language_model.mtp.pre_fc_norm_embedding.weight":
                mapped = "blk.48.nextn.enorm.weight"
            elif raw == "language_model.mtp.pre_fc_norm_hidden.weight":
                mapped = "blk.48.nextn.hnorm.weight"
            elif raw.startswith("language_model.mtp.hyper_connection_mixer."):
                tail = raw.rsplit(".", 2)[-2]
                part = {
                    "hc_norm": "hc_head_norm",
                    "input_mix_weight_down": "hc_head_down",
                    "input_mix_weight_up": "hc_head_up",
                }[tail]
                mapped = f"blk.48.nextn.{part}.weight"
            elif raw.startswith("language_model.mtp.layers.0."):
                normalized = "language_model.model.layers.48." + raw.split("language_model.mtp.layers.0.", 1)[1]
                mapped = custom_layer_name(normalized, 48)
                if mapped is None:
                    mapped = name_map.get_name(normalized.removeprefix("language_model."),
                                               try_suffixes=(".weight", ".bias"))
                raw = normalized
            else:
                raise ValueError(f"unknown MTP tensor {raw}")
            if raw.endswith(".self_attn.indexer.index_qk_proj.weight"):
                for suffix, rows in (("q_proj", slice(0, 512)), ("k_proj", slice(512, 640))):
                    entries.append(Entry(f"blk.48.indexer.{suffix}.weight", tensor,
                                         (2560, 512 if suffix == "q_proj" else 128), tensor.tensor_type,
                                         "q8_rows", np.arange(640)[rows]))
                continue
            if mapped is None:
                raise ValueError(f"no upstream-schema MTP name for {raw}")
            shape = tuple(int(x) for x in tensor.shape)
            op, aux = transformed_operation(raw, mapped, tensor, mtp=True)
            if op in {"conv_f32", "conv_f32_rows"}:
                shape = (4, shape[-1])
            elif op == "q4_k_down_to_q8":
                shape = (640, shape[1], shape[2])
            entries.append(Entry(mapped, tensor, shape, tensor.tensor_type, op, aux))

        mtp_names = [entry.name for entry in entries if entry.name.startswith("blk.48")]
        if len(mtp_names) != 32 or len(set(mtp_names)) != 32:
            raise ValueError(f"expected 32 unique MTP tensors, found {len(mtp_names)}/{len(set(mtp_names))}")

    counts: dict[str, int] = {}
    total = 0
    for entry in entries:
        _, nbytes, _ = output_spec(entry, gguf)
        counts[entry.operation] = counts.get(entry.operation, 0) + 1
        total += nbytes
    print(f"validated {len(entries)} tensors ({total / 1e9:.2f} GB payload)")
    for operation, count in sorted(counts.items()):
        print(f"  {operation:22s} {count:4d}")
    if args.plan:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    incomplete = args.out.with_name(args.out.name + ".incomplete")
    if incomplete.exists():
        raise SystemExit(f"refusing to overwrite incomplete output: {incomplete}")
    writer = gguf.GGUFWriter(incomplete, "qwen4exp", use_temp_file=False)
    copy_template_metadata(writer, template, with_mtp=mtp_reader is not None, gguf=gguf)
    for entry in entries:
        dtype, nbytes, raw_dtype = output_spec(entry, gguf)
        numpy_shape = tuple(reversed(entry.shape))
        if raw_dtype is not None:
            numpy_shape = gguf.quants.quant_shape_to_byte_shape(numpy_shape, raw_dtype)
        writer.add_tensor_info(entry.name, numpy_shape, dtype, nbytes, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for index, entry in enumerate(entries, 1):
        print(f"[{index:4d}/{len(entries)}] {entry.name} ({entry.operation})", flush=True)
        writer.write_tensor_data(materialize(entry, gguf))
    writer.close()
    os.replace(incomplete, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
