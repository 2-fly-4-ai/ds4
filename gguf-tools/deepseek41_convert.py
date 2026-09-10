#!/usr/bin/env python3
"""Convert official DeepSeek V4.1 Flash into resumable split GGUF artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import struct
import sys
import time

from deepseek41_manifest import DEFAULT_REVISION, load_checkpoint_headers, validate_config, validate_index
from deepseek41_mxfp4 import MXFP4_VALUES
from deepseek41_plan import PlanEntry, build_plan
from glm53_manifest import load_index
from glm53_quantize import (
    GGUF_ALIGNMENT, GGUF_ARRAY, GGUF_STRING, GGUF_VERSION,
    QTYPE_BF16, QTYPE_F32, QTYPE_I8, QTYPE_IQ2_XXS, QTYPE_Q2_K, QTYPE_Q8_0,
    Quantizer, align, kv_bool, kv_f32, kv_string, kv_u32, kv_u32_array, kv_u64,
    pack_string, read_exact, read_gguf_string, read_u32, read_u64, skip_gguf_value,
)

QTYPE_MXFP4 = 39
QTYPE_IDS = {"F32": QTYPE_F32, "BF16": QTYPE_BF16, "Q8_0": QTYPE_Q8_0,
             "IQ2_XXS": QTYPE_IQ2_XXS, "Q2_K": QTYPE_Q2_K,
             "I8": QTYPE_I8, "MXFP4": QTYPE_MXFP4}


def fail(message):
    raise ValueError(message)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


@dataclasses.dataclass(frozen=True)
class OutputEntry:
    plan: PlanEntry
    qtype: int
    offset: int


class SourceDB:
    """Read byte ranges from a fully downloaded and validated checkpoint."""

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        _, self.weight_map = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
        validate_index(self.weight_map)
        self.tensors, missing = load_checkpoint_headers(hf_dir, self.weight_map)
        if missing:
            fail(f"checkpoint download is incomplete: {len(missing)} shards missing; first is {missing[0]}")
        if set(self.tensors) != set(self.weight_map):
            absent = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"checkpoint headers are incomplete; first missing tensor is {absent[0]}")
        self._fds = {}

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            fail(f"source tensor not found: {name}")

    def _fd(self, shard):
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(os.path.join(self.hf_dir, shard), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def read_range(self, name, byte_start, byte_count):
        info = self.info(name)
        if byte_start < 0 or byte_count < 0 or byte_start + byte_count > info["nbytes"]:
            fail(f"invalid source range for {name}")
        data = os.pread(self._fd(info["shard"]), byte_count, info["offset"] + byte_start)
        if len(data) != byte_count:
            fail(f"short payload read for {name}: {len(data)} of {byte_count} bytes")
        return data

    def iter_read(self, name, chunk_size=16 << 20):
        info = self.info(name)
        for start in range(0, info["nbytes"], chunk_size):
            yield self.read_range(name, start, min(chunk_size, info["nbytes"] - start))

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def kv_string_array(key, values):
    return (pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(values))
            + b"".join(pack_string(value) for value in values))


def kv_u64_array(key, values):
    return (pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, 10, len(values))
            + struct.pack(f"<{len(values)}Q", *values))


def _is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def build_engram_layout(config):
    """Reproduce official engram.py's prime buckets and NumPy PCG64 multipliers."""
    import numpy as np

    text = config["text_config"]
    layers = text["engram_layer_ids"]
    max_ngram = text["engram_max_ngram_size"]
    n_heads = text["engram_n_heads"]
    seen, primes = set(), []
    for _ in layers:
        for _ in range(max_ngram - 1):
            current = text["engram_vocab_size"] - 1
            for _ in range(n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                primes.append(current)
    width = (max_ngram - 1) * n_heads
    offsets = []
    for layer in range(len(layers)):
        running = 0
        for prime in primes[layer * width:(layer + 1) * width]:
            offsets.append(running)
            running += prime
        if running != text["engram_num_embeddings"][layer]:
            fail(f"Engram bucket rows {running} do not match layer {layers[layer]} table")
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // text["engram_compressed_vocab_size"]) // 2)
    multipliers = []
    for layer in layers:
        generator = np.random.default_rng(10007 * layer)
        values = generator.integers(0, bound, size=max_ngram, dtype=np.int64)
        multipliers.extend((values * 2 + 1).tolist())
    return primes, offsets, multipliers


def build_compressed_token_map(hf_dir, expected_size):
    """Build the exact official NFKC/case/space-normalized tokenizer-id map."""
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as error:
        fail("DeepSeek V4.1 conversion requires the 'tokenizers' Python package")
    tokenizer = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    key_to_id, result = {}, []
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        result.append(key_to_id.setdefault(key, len(key_to_id)))
    if len(key_to_id) != expected_size:
        fail(f"compressed tokenizer vocabulary {len(key_to_id)} != {expected_size}")
    return result


def load_tokenizer_template_records(path):
    records, template_tokens = [], None
    with open(path, "rb") as fp:
        if read_exact(fp, 4, "GGUF magic") != b"GGUF":
            fail(f"{path}: not a GGUF file")
        version = read_u32(fp, "GGUF version")
        if version not in (2, 3):
            fail(f"{path}: unsupported GGUF version {version}")
        read_u64(fp, "GGUF tensor count")
        n_kv = read_u64(fp, "GGUF metadata count")
        for _ in range(n_kv):
            start = fp.tell()
            key = read_gguf_string(fp, "GGUF metadata key")
            value_type = read_u32(fp, "GGUF metadata type")
            if key == "tokenizer.ggml.tokens":
                if value_type != GGUF_ARRAY:
                    fail(f"{path}: tokenizer.ggml.tokens is not an array")
                element_type = read_u32(fp, "tokenizer token element type")
                count = read_u64(fp, "tokenizer token count")
                if element_type != GGUF_STRING or count > 1 << 20:
                    fail(f"{path}: invalid tokenizer token array")
                template_tokens = [read_gguf_string(fp, "tokenizer token") for _ in range(count)]
            else:
                skip_gguf_value(fp, value_type)
            end = fp.tell()
            if key.startswith("tokenizer.") and key != "tokenizer.chat_template":
                fp.seek(start)
                records.append((key, read_exact(fp, end - start, key)))
            fp.seek(end)
    if template_tokens is None:
        fail(f"{path}: tokenizer.ggml.tokens is missing")
    return records, template_tokens


def load_v41_tokens(hf_dir, vocab_size=129280):
    """Load official tokens, accepting identical duplicate added-token entries."""
    path = os.path.join(hf_dir, "tokenizer.json")
    with open(path, "rb") as fp:
        document = json.load(fp)
    vocab, added = document.get("model", {}).get("vocab"), document.get("added_tokens")
    if not isinstance(vocab, dict) or not isinstance(added, list):
        fail(f"{path}: unsupported tokenizer structure")
    tokens = [None] * vocab_size

    def assign(token_id, token, source):
        if not isinstance(token_id, int) or not isinstance(token, str):
            fail(f"{path}: invalid {source} token entry")
        if token_id < 0 or token_id >= vocab_size:
            fail(f"{path}: out-of-range token id {token_id}")
        if tokens[token_id] is not None and tokens[token_id] != token:
            fail(f"{path}: conflicting token id {token_id}: {tokens[token_id]!r} versus {token!r}")
        tokens[token_id] = token

    for token, token_id in vocab.items():
        assign(token_id, token, "base")
    for entry in added:
        if not isinstance(entry, dict):
            fail(f"{path}: invalid added token")
        assign(entry.get("id"), entry.get("content"), "added")
    missing = [i for i, token in enumerate(tokens) if token is None]
    if missing:
        fail(f"{path}: vocabulary has {len(missing)} holes; first is id {missing[0]}")
    return tokens


def tokenizer_records(hf_dir, template_path):
    records, old_tokens = load_tokenizer_template_records(template_path)
    tokens = load_v41_tokens(hf_dir)
    if len(old_tokens) != len(tokens):
        fail(f"tokenizer template has {len(old_tokens)} entries, expected {len(tokens)}")
    changed = [i for i, pair in enumerate(zip(old_tokens, tokens)) if pair[0] != pair[1]]
    if len(changed) != 9:
        fail(f"tokenizer template differs from V4.1 at {len(changed)} ids, expected 9")
    return [kv_string_array("tokenizer.ggml.tokens", tokens)
            if key == "tokenizer.ggml.tokens" else record for key, record in records]


def main_metadata(config, revision, hf_dir):
    text, rope = config["text_config"], config["text_config"]["rope_scaling"]
    token_map = build_compressed_token_map(
        hf_dir, text["engram_compressed_vocab_size"])
    primes, offsets, multipliers = build_engram_layout(config)
    return [
        kv_string("general.architecture", "deepseek41"),
        kv_string("general.name", "DeepSeek-V4.1-Flash"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.revision", revision),
        kv_u32("deepseek41.block_count", 43), kv_u32("deepseek41.trunk_block_count", 40),
        kv_u32("deepseek41.nextn_predict_layers", 3),
        kv_u64("deepseek41.context_length", text["max_position_embeddings"]),
        kv_u32("deepseek41.embedding_length", text["hidden_size"]),
        kv_u32("deepseek41.vocab_size", text["vocab_size"]),
        kv_u32("deepseek41.attention.head_count", text["num_attention_heads"]),
        kv_u32("deepseek41.attention.head_count_kv", text["num_key_value_heads"]),
        kv_u32("deepseek41.attention.key_length", text["head_dim"]),
        kv_u32("deepseek41.attention.value_length", text["head_dim"]),
        kv_u32("deepseek41.rope.dimension_count", text["qk_rope_head_dim"]),
        kv_u32("deepseek41.attention.q_lora_rank", text["q_lora_rank"]),
        kv_u32("deepseek41.attention.output_lora_rank", text["o_lora_rank"]),
        kv_u32("deepseek41.attention.output_group_count", text["o_groups"]),
        kv_u32("deepseek41.attention.sliding_window", text["sliding_window"]),
        kv_u32("deepseek41.attention.indexer.head_count", text["index_n_heads"]),
        kv_u32("deepseek41.attention.indexer.key_length", text["index_head_dim"]),
        kv_u32("deepseek41.attention.indexer.top_k", text["index_topk"]),
        kv_u32("deepseek41.attention.candidate_source_layer", text["candidate_source_layer_id"]),
        kv_u32("deepseek41.attention.candidate_top_k_blocks", text["candidate_topk_blocks"]),
        kv_u32("deepseek41.attention.candidate_block_size", text["candidate_block_size"]),
        kv_u32("deepseek41.expert_count", text["n_routed_experts"]),
        kv_u32("deepseek41.expert_used_count", text["num_experts_per_tok"]),
        kv_u32("deepseek41.expert_shared_count", text["n_shared_experts"]),
        kv_u32("deepseek41.expert_feed_forward_length", text["moe_intermediate_size"]),
        kv_f32("deepseek41.expert_weights_scale", text["routed_scaling_factor"]),
        kv_bool("deepseek41.expert_weights_norm", text["norm_topk_prob"]),
        kv_string("deepseek41.expert_scoring_func", text["scoring_func"]),
        kv_u32("deepseek41.hyper_connection.count", text["hc_mult"]),
        kv_u32("deepseek41.hyper_connection.sinkhorn_iterations", text["hc_sinkhorn_iters"]),
        kv_f32("deepseek41.attention.layer_norm_rms_epsilon", text["rms_norm_eps"]),
        kv_f32("deepseek41.hyper_connection.epsilon", 1.0e-6),
        kv_f32("deepseek41.swiglu_limit", text["swiglu_limit"]),
        kv_f32("deepseek41.rope.freq_base", text["rope_theta"]),
        kv_f32("deepseek41.rope.scaling.factor", rope["factor"]),
        kv_u64("deepseek41.rope.scaling.original_context_length", rope["original_max_position_embeddings"]),
        kv_f32("deepseek41.attention.compress_rope_freq_base", text["compress_rope_theta"]),
        kv_u32_array("deepseek41.attention.compress_ratios", text["compress_ratios"]),
        kv_u32_array("deepseek41.attention.kv_source_layers", text["kv_source_layer_ids"]),
        kv_u32_array("deepseek41.attention.index_source_layers", text["index_source_layer_ids"]),
        kv_u32("deepseek41.engram.max_ngram_size", text["engram_max_ngram_size"]),
        kv_u32("deepseek41.engram.head_count", text["engram_n_heads"]),
        kv_u32("deepseek41.engram.head_dim", text["engram_head_dim"]),
        kv_u32("deepseek41.engram.compressed_vocab_size", text["engram_compressed_vocab_size"]),
        kv_u32("deepseek41.engram.pad_token_id", 2),
        kv_u32_array("deepseek41.engram.layers", text["engram_layer_ids"]),
        kv_u32_array("deepseek41.engram.num_embeddings", text["engram_num_embeddings"]),
        kv_u32_array("deepseek41.engram.token_map", token_map),
        kv_u32_array("deepseek41.engram.hash_primes", primes),
        kv_u64_array("deepseek41.engram.hash_offsets", offsets),
        kv_u64_array("deepseek41.engram.hash_multipliers", multipliers),
        kv_string("deepseek41.engram.hash_contract_sha256",
                  "9a50b6f0ae6be53fa6aa7ea4c2c6c5e94b605b41241a00dbc2a1b5a4c2c81a18"),
        kv_u32("deepseek41.dspark.block_size", text["dspark_block_size"]),
        kv_u32("deepseek41.dspark.markov_rank", 256), kv_u32("deepseek41.dspark.noise_token_id", 128799),
        kv_u32("deepseek41.dspark.expert_count", text["dspark_n_routed_experts"]),
        kv_u32("deepseek41.dspark.expert_used_count", text["dspark_num_experts_per_tok"]),
        kv_u32_array("deepseek41.dspark.target_layers", text["dspark_target_layer_ids"]),
    ]


def sidecar_metadata(artifact, revision):
    return [kv_string("general.architecture", f"deepseek41-{artifact}"),
            kv_string("general.name", f"DeepSeek-V4.1-Flash-{artifact}"),
            kv_u32("general.alignment", GGUF_ALIGNMENT),
            kv_string("general.source.revision", revision),
            kv_string("deepseek41.sidecar.kind", artifact)]


def tensor_header(entry):
    item = entry.plan
    return (pack_string(item.target) + struct.pack("<I", len(item.shape))
            + struct.pack(f"<{len(item.shape)}Q", *item.shape)
            + struct.pack("<IQ", entry.qtype, entry.offset))


def prepare_plan(plan):
    result, offset = [], 0
    for item in plan:
        if item.qtype not in QTYPE_IDS:
            fail(f"unknown output type {item.qtype}")
        result.append(OutputEntry(item, QTYPE_IDS[item.qtype], offset))
        offset += align(item.nbytes)
    return result


def output_layout(plan, records):
    header = 4 + 4 + 8 + 8 + sum(map(len, records)) + sum(len(tensor_header(x)) for x in plan)
    return align(header), sum(align(x.plan.nbytes) for x in plan)


def e8m0_lut(np):
    bits = np.arange(256, dtype=np.uint32) << np.uint32(23)
    bits[0] = np.uint32(0x00400000)
    return bits.view(np.float32)


def decode_fp8_rows(db, weight_name, scale_name, row_start, row_count, quantizer):
    np, weight, scale = quantizer.np, db.info(weight_name), db.info(scale_name)
    if weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0":
        fail(f"{weight_name}: expected E4M3 weights and E8M0 scales")
    if len(weight["shape"]) != 2:
        fail(f"{weight_name}: FP8 source is not a matrix")
    rows, columns = weight["shape"]
    expected = [(rows + 31) // 32, (columns + 31) // 32]
    if scale["shape"] != expected:
        fail(f"{scale_name}: shape {scale['shape']} != {expected}")
    raw = db.read_range(weight_name, row_start * columns, row_count * columns)
    codes = np.frombuffer(raw, dtype=np.uint8).reshape(row_count, columns)
    if np.any((codes & 0x7F) == 0x7F):
        fail(f"{weight_name}: non-finite E4M3 code")
    scale_start, scale_end = row_start // 32, (row_start + row_count - 1) // 32 + 1
    raw_scales = db.read_range(scale_name, scale_start * expected[1],
                               (scale_end - scale_start) * expected[1])
    scales = np.frombuffer(raw_scales, dtype=np.uint8).reshape(-1, expected[1])
    selected = scales[np.arange(row_start, row_start + row_count) // 32 - scale_start]
    expanded = np.repeat(e8m0_lut(np)[selected], 32, axis=1)[:, :columns]
    return np.ascontiguousarray(quantizer.fp8_lut[codes] * expanded, dtype=np.float32)


def decode_fp4_rows(db, weight_name, scale_name, row_start, row_count, np):
    weight, scale = db.info(weight_name), db.info(scale_name)
    rows, packed_columns = weight["shape"]
    columns, blocks = packed_columns * 2, packed_columns // 16
    if weight["dtype"] != "I8" or scale["dtype"] != "F8_E8M0" or scale["shape"] != [rows, blocks]:
        fail(f"{weight_name}: invalid packed E2M1/E8M0 pair")
    packed = np.frombuffer(db.read_range(weight_name, row_start * packed_columns,
                                         row_count * packed_columns), dtype=np.uint8).reshape(row_count, blocks, 16)
    scales = np.frombuffer(db.read_range(scale_name, row_start * blocks,
                                         row_count * blocks), dtype=np.uint8).reshape(row_count, blocks)
    codes = np.empty((row_count, blocks, 32), dtype=np.uint8)
    codes[:, :, 0::2], codes[:, :, 1::2] = packed & 0x0F, packed >> 4
    values = np.asarray(MXFP4_VALUES, dtype=np.float32)[codes]
    values *= e8m0_lut(np)[scales, None]
    return np.ascontiguousarray(values.reshape(row_count, columns), dtype=np.float32)


def repack_mxfp4_rows(db, weight_name, scale_name, row_start, row_count, np):
    weight, scale = db.info(weight_name), db.info(scale_name)
    rows, packed_columns = weight["shape"]
    blocks = packed_columns // 16
    if weight["dtype"] != "I8" or scale["dtype"] != "F8_E8M0" or scale["shape"] != [rows, blocks]:
        fail(f"{weight_name}: invalid packed E2M1/E8M0 pair")
    packed = np.frombuffer(db.read_range(weight_name, row_start * packed_columns,
                                         row_count * packed_columns), dtype=np.uint8).reshape(row_count, blocks, 16)
    scales = np.frombuffer(db.read_range(scale_name, row_start * blocks,
                                         row_count * blocks), dtype=np.uint8).reshape(row_count, blocks)
    output = np.empty((row_count, blocks, 17), dtype=np.uint8)
    output[:, :, 0] = scales
    first, second = packed[:, :, :8], packed[:, :, 8:]
    output[:, :, 1::2] = (first & 0x0F) | ((second & 0x0F) << 4)
    output[:, :, 2::2] = (first >> 4) | (second & 0xF0)
    return output.tobytes()


def write_entry(fp, output, db, quantizer, chunk_rows):
    np, item, start = quantizer.np, output.plan, fp.tell()
    if item.mode in ("copy", "copy_fp8_sidecar"):
        for data in db.iter_read(item.sources[0]): fp.write(data)
    elif item.mode == "bf16_to_f32":
        info, chunk = db.info(item.sources[0]), 4 << 20
        if info["dtype"] != "BF16": fail(f"{item.sources[0]}: expected BF16")
        count = product(info["shape"])
        for pos in range(0, count, chunk):
            n = min(chunk, count - pos)
            bits = np.frombuffer(db.read_range(item.sources[0], pos * 2, n * 2), dtype="<u2").astype(np.uint32) << 16
            fp.write(bits.view(np.float32).astype("<f4", copy=False).tobytes())
    elif item.mode == "quantize":
        info = db.info(item.sources[0]); rows, columns = info["shape"]
        if info["dtype"] != "BF16": fail(f"{item.sources[0]}: expected BF16")
        for row in range(0, rows, chunk_rows):
            n = min(chunk_rows, rows - row)
            bits = np.frombuffer(db.read_range(item.sources[0], row * columns * 2, n * columns * 2), dtype="<u2").astype(np.uint32) << 16
            fp.write(quantizer.encode(bits.view(np.float32).reshape(n, columns), output.qtype))
    elif item.mode == "fp8_to_q8":
        rows = db.info(item.sources[0])["shape"][0]
        for row in range(0, rows, chunk_rows):
            n = min(chunk_rows, rows - row)
            fp.write(quantizer.encode(decode_fp8_rows(db, *item.sources, row, n, quantizer), output.qtype))
    elif item.mode in ("repack_mxfp4", "fp4_to_q2"):
        for source in range(0, len(item.sources), 2):
            weight, scale = item.sources[source:source + 2]
            rows, packed_columns = db.info(weight)["shape"]
            importance = None
            if item.mode == "fp4_to_q2" and output.qtype == QTYPE_IQ2_XXS:
                importance = np.zeros(packed_columns * 2, dtype=np.float32)
                for row in range(0, rows, chunk_rows):
                    n = min(chunk_rows, rows - row)
                    values = decode_fp4_rows(db, weight, scale, row, n, np)
                    importance += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float32)
            for row in range(0, rows, chunk_rows):
                n = min(chunk_rows, rows - row)
                if item.mode == "repack_mxfp4":
                    fp.write(repack_mxfp4_rows(db, weight, scale, row, n, np))
                else:
                    fp.write(quantizer.encode(decode_fp4_rows(db, weight, scale, row, n, np), output.qtype, importance))
    else:
        fail(f"{item.target}: unsupported mode {item.mode}")
    if fp.tell() - start != item.nbytes:
        fail(f"{item.target}: wrote {fp.tell() - start} bytes, expected {item.nbytes}")


def conversion_signature(plan, records):
    digest = hashlib.sha256()
    for record in records: digest.update(record)
    for entry in plan:
        digest.update(tensor_header(entry)); digest.update(entry.plan.mode.encode())
        for source in entry.plan.sources: digest.update(source.encode() + b"\0")
    return digest.hexdigest()


def save_resume(path, signature, completed):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fp:
        json.dump({"version": 1, "signature": signature, "completed": completed}, fp); fp.write("\n"); fp.flush(); os.fsync(fp.fileno())
    os.replace(temporary, path)


def load_resume(path, signature, count):
    with open(path, "r", encoding="utf-8") as fp: state = json.load(fp)
    completed = state.get("completed")
    if state.get("version") != 1 or state.get("signature") != signature:
        fail(f"resume journal does not match conversion: {path}")
    if not isinstance(completed, int) or not 0 <= completed <= count:
        fail(f"invalid completed count in {path}")
    return completed


def write_gguf(args, plan, records, db):
    prepared = prepare_plan(plan); data_offset, data_bytes = output_layout(prepared, records)
    if shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free < data_offset + data_bytes + (8 << 30):
        fail("insufficient free space for output plus 8 GiB reserve")
    library = args.quants_library or os.path.join(os.path.dirname(__file__), "libds4quants.dylib" if sys.platform == "darwin" else "libds4quants.so")
    if not os.path.isfile(library): fail(f"quantizer library not found: {library}; run make -C gguf-tools")
    quantizer = Quantizer(library)
    partial, journal = args.out + ".partial", args.out + ".partial.resume.json"
    signature = conversion_signature(prepared, records)
    if os.path.exists(args.out) and not args.overwrite: fail(f"output exists: {args.out}; use --overwrite")
    if args.overwrite:
        for path in (partial, journal):
            if os.path.exists(path): os.unlink(path)
    exists = os.path.exists(partial) or os.path.exists(journal)
    if exists and not args.resume: fail(f"partial output exists; use --resume: {partial}")
    if exists and not (os.path.isfile(partial) and os.path.isfile(journal)): fail("partial GGUF and journal must both exist")
    completed = load_resume(journal, signature, len(prepared)) if exists else 0
    if exists:
        expected = data_offset + (prepared[completed - 1].offset + align(prepared[completed - 1].plan.nbytes) if completed else 0)
        with open(partial, "r+b") as fp: fp.truncate(expected)
    else:
        with open(partial, "wb") as fp:
            fp.write(b"GGUF" + struct.pack("<IQQ", GGUF_VERSION, len(prepared), len(records)))
            for record in records: fp.write(record)
            for entry in prepared: fp.write(tensor_header(entry))
            if fp.tell() > data_offset: fail("GGUF header exceeds planned offset")
            fp.write(bytes(data_offset - fp.tell())); fp.flush(); os.fsync(fp.fileno())
        save_resume(journal, signature, 0)
    with open(partial, "r+b") as fp:
        prior = prepared[completed - 1] if completed else None
        fp.seek(data_offset + (prior.offset + align(prior.plan.nbytes) if prior else 0))
        started = time.monotonic()
        for number, entry in enumerate(prepared[completed:], completed + 1):
            if fp.tell() != data_offset + entry.offset: fail("output offset mismatch")
            tensor_started = time.monotonic(); write_entry(fp, entry, db, quantizer, args.rows_per_chunk)
            fp.write(bytes(align(entry.plan.nbytes) - entry.plan.nbytes)); fp.flush(); os.fsync(fp.fileno()); save_resume(journal, signature, number)
            print(f"[{number:4d}/{len(prepared):4d}] {entry.plan.target} {entry.plan.qtype} {entry.plan.nbytes/(1<<30):.3f} GiB {time.monotonic()-tensor_started:.1f}s total={(time.monotonic()-started)/60:.1f}m", file=sys.stderr, flush=True)
    os.replace(partial, args.out); os.unlink(journal)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True); parser.add_argument("--artifact", choices=("main", "engram", "vision", "dspark"), default="main")
    parser.add_argument("--quant", choices=("native", "q2"), default="native"); parser.add_argument("--tokenizer-template")
    parser.add_argument("--out"); parser.add_argument("--source-revision", default=DEFAULT_REVISION); parser.add_argument("--quants-library")
    parser.add_argument("--rows-per-chunk", type=int, default=128); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.rows_per_chunk <= 8192: parser.error("--rows-per-chunk must be 1..8192")
    if args.artifact == "main" and not args.tokenizer_template: parser.error("--tokenizer-template is required for main")
    if not args.dry_run and not args.out: parser.error("--out is required unless --dry-run")
    return args


def main():
    args = parse_args()
    with open(os.path.join(args.hf, "config.json"), "rb") as fp: config = json.load(fp)
    validate_config(config); db = SourceDB(args.hf)
    try:
        plan = [x for x in build_plan(db.tensors, args.quant) if x.artifact == args.artifact]
        records = (main_metadata(config, args.source_revision, args.hf) + tokenizer_records(args.hf, args.tokenizer_template)
                   if args.artifact == "main" else sidecar_metadata(args.artifact, args.source_revision))
        prepared = prepare_plan(plan); data_offset, data_bytes = output_layout(prepared, records)
        print(f"deepseek41-convert: artifact={args.artifact} quant={args.quant} tensors={len(plan)} bytes={data_offset+data_bytes} ({(data_offset+data_bytes)/(1<<30):.3f} GiB)")
        if not args.dry_run: write_gguf(args, plan, records, db); print(f"deepseek41-convert: wrote {args.out}", file=sys.stderr)
    finally: db.close()


if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"deepseek41-convert: error: {error}", file=sys.stderr); sys.exit(1)
