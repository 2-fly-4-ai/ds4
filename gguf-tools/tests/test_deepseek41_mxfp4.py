#!/usr/bin/env python3

import random
import unittest

from deepseek41_mxfp4 import (
    decode_adjacent_block,
    decode_mxfp4_block,
    repack_block,
    repack_row,
)


class DeepSeek41MXFP4Tests(unittest.TestCase):
    def test_all_nibbles_and_scales_are_lossless(self):
        for scale in (0, 1, 100, 127, 128, 254):
            for offset in range(16):
                codes = [(i + offset) & 15 for i in range(32)]
                adjacent = bytes(
                    codes[i] | (codes[i + 1] << 4)
                    for i in range(0, 32, 2)
                )
                converted = repack_block(adjacent, scale)
                self.assertEqual(converted[0], scale)
                self.assertEqual(
                    decode_adjacent_block(adjacent, scale),
                    decode_mxfp4_block(converted),
                )

    def test_random_rows_are_lossless(self):
        rng = random.Random(0xD541)
        for columns in (32, 5120):
            adjacent = bytes(rng.randrange(256) for _ in range(columns // 2))
            scales = bytes(rng.randrange(255) for _ in range(columns // 32))
            converted = repack_row(adjacent, scales, columns)
            self.assertEqual(len(converted), columns // 32 * 17)
            for block in range(columns // 32):
                self.assertEqual(
                    decode_adjacent_block(
                        adjacent[block * 16:(block + 1) * 16], scales[block]
                    ),
                    decode_mxfp4_block(converted[block * 17:(block + 1) * 17]),
                )

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            repack_block(bytes(15), 127)
        with self.assertRaises(ValueError):
            repack_row(bytes(16), bytes(1), 31)
        with self.assertRaises(ValueError):
            repack_row(bytes(15), bytes(1), 32)


if __name__ == "__main__":
    unittest.main()
