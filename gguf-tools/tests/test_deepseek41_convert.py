#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from deepseek41_convert import (build_engram_layout, decode_fp4_rows,
                                decode_fp8_rows, load_v41_tokens,
                                repack_mxfp4_rows)
from deepseek41_mxfp4 import decode_adjacent_block, decode_mxfp4_block
from glm53_quantize import Quantizer


class MemoryDB:
    def __init__(self, tensors): self.tensors = tensors
    def info(self, name): return self.tensors[name][0]
    def read_range(self, name, start, count): return self.tensors[name][1][start:start + count]


def fake_quantizer():
    result = Quantizer.__new__(Quantizer)
    result.np = np
    result.fp8_lut = result._build_fp8_lut()
    return result


class DeepSeek41ConverterTests(unittest.TestCase):
    def test_engram_layout_matches_official_bucket_totals_and_rng(self):
        config = {"text_config": {
            "engram_layer_ids": [1, 14], "engram_max_ngram_size": 4,
            "engram_n_heads": 8, "engram_vocab_size": 16000000,
            "engram_num_embeddings": [384006168, 384016682],
            "engram_compressed_vocab_size": 99092,
        }}
        primes, offsets, multipliers = build_engram_layout(config)
        self.assertEqual(len(primes), 48)
        self.assertEqual(len(offsets), 48)
        self.assertEqual(primes[:3], [16000057, 16000079, 16000081])
        self.assertEqual(primes[-3:], [16000877, 16000879, 16000889])
        self.assertEqual(offsets[24], 0)
        self.assertEqual(offsets[-1] + primes[-1], 384016682)
        self.assertEqual(multipliers,
                         [76632096046245, 4839876093313,
                          35959672319349, 73987337458391,
                          67716810739261, 51510806800915,
                          30921347202721, 82619226485591])

    def test_native_fp4_repack_preserves_every_value(self):
        rng = np.random.default_rng(41)
        packed = rng.integers(0, 256, size=(3, 32), dtype=np.uint8)
        scales = rng.integers(110, 130, size=(3, 2), dtype=np.uint8)
        db = MemoryDB({"w": ({"dtype": "I8", "shape": [3, 32]}, packed.tobytes()),
                       "s": ({"dtype": "F8_E8M0", "shape": [3, 2]}, scales.tobytes())})
        converted = np.frombuffer(repack_mxfp4_rows(db, "w", "s", 0, 3, np), dtype=np.uint8).reshape(3, 2, 17)
        source = packed.reshape(3, 2, 16)
        for row in range(3):
            for block in range(2):
                self.assertEqual(converted[row, block, 0], scales[row, block])
                self.assertEqual(decode_adjacent_block(source[row, block].tobytes(), int(scales[row, block])),
                                 decode_mxfp4_block(converted[row, block].tobytes()))

    def test_fp4_decoder_uses_adjacent_nibble_order(self):
        packed = bytes(range(16))
        db = MemoryDB({"w": ({"dtype": "I8", "shape": [1, 16]}, packed),
                       "s": ({"dtype": "F8_E8M0", "shape": [1, 1]}, bytes((127,)))})
        self.assertEqual(decode_fp4_rows(db, "w", "s", 0, 1, np)[0].tolist(),
                         decode_adjacent_block(packed, 127))

    def test_fp8_decoder_expands_scales_across_row_boundary(self):
        codes = np.full((33, 64), 0x38, dtype=np.uint8)
        scales = np.array([[127, 128], [129, 130]], dtype=np.uint8)
        db = MemoryDB({"w": ({"dtype": "F8_E4M3", "shape": [33, 64]}, codes.tobytes()),
                       "s": ({"dtype": "F8_E8M0", "shape": [2, 2]}, scales.tobytes())})
        decoded = decode_fp8_rows(db, "w", "s", 31, 2, fake_quantizer())
        np.testing.assert_array_equal(decoded[0, :32], np.ones(32, dtype=np.float32))
        np.testing.assert_array_equal(decoded[0, 32:], np.full(32, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(decoded[1, :32], np.full(32, 4.0, dtype=np.float32))
        np.testing.assert_array_equal(decoded[1, 32:], np.full(32, 8.0, dtype=np.float32))

    def test_token_loader_allows_identical_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "tokenizer.json"), "w") as fp:
                json.dump({"model": {"vocab": {"a": 0, "b": 1, "c": 2}},
                           "added_tokens": [{"id": 1, "content": "b"}]}, fp)
            self.assertEqual(load_v41_tokens(directory, 3), ["a", "b", "c"])

    def test_token_loader_rejects_conflicting_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "tokenizer.json"), "w") as fp:
                json.dump({"model": {"vocab": {"a": 0}},
                           "added_tokens": [{"id": 0, "content": "different"}]}, fp)
            with self.assertRaisesRegex(ValueError, "conflicting token id"):
                load_v41_tokens(directory, 1)


if __name__ == "__main__": unittest.main()
