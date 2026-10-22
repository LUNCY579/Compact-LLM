"""Independent binary16 numeric-reference checks; no model or asset inputs."""
from __future__ import annotations

import hashlib
import math
import random
import struct
import unittest

from experiments.rineya_quality_queue_v1.f16_inspection import inspect_finite_argmax_row


def encode(*words):
    return b"".join(struct.pack("<H", word) for word in words)


def numeric_first_max(payload):
    values = struct.unpack("<" + "e" * (len(payload) // 2), payload)
    return max(range(len(values)), key=values.__getitem__)


class F16InspectionTests(unittest.TestCase):
    def inspect(self, payload, **kwargs):
        return inspect_finite_argmax_row(payload, vocabulary_size=len(payload) // 2, **kwargs)

    def test_every_binary16_pattern_against_independent_numeric_reference(self):
        for word in range(65536):
            one = encode(word)
            value = struct.unpack("<e", one)[0]
            if not math.isfinite(value):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.inspect(one)
                continue
            for pair in (encode(word, 0), encode(0, word)):
                self.assertEqual(self.inspect(pair)["argmax_token_id"], numeric_first_max(pair), hex(word))

    def test_signed_zero_ties_and_equal_negative_or_positive_maxima(self):
        for words in ((0x8000, 0), (0, 0x8000), (0xBC00, 0xBC00, 0xC000),
                      (0x3C00, 0x3C00, 0x0001), (0xFBFF, 0x8001, 0x8001)):
            payload = encode(*words)
            self.assertEqual(self.inspect(payload)["argmax_token_id"], numeric_first_max(payload))

    def test_whole_row_rejects_nonfinite_even_after_a_known_high_maximum(self):
        for nonfinite in (0x7C00, 0xFC00, 0x7C01, 0x7FFF, 0xFE00):
            with self.assertRaisesRegex(ValueError, "token ID 2"):
                self.inspect(encode(0x7BFF, 0xBC00, nonfinite), expected_argmax_token_id=0)

    def test_unordered_finite_rows_and_a_full_sized_row_match_numeric_reference(self):
        rng = random.Random(101)
        for size in (3, 31, 1024, 151936):
            words = []
            while len(words) < size:
                word = rng.randrange(65536)
                if math.isfinite(struct.unpack("<e", encode(word))[0]):
                    words.append(word)
            payload = encode(*words)
            self.assertEqual(self.inspect(payload)["argmax_token_id"], numeric_first_max(payload))

    def test_strict_extent_types_expected_id_and_no_mutation(self):
        payload = encode(0x8000, 0, 0x3C00)
        before = bytes(payload)
        result = self.inspect(payload, expected_argmax_token_id=2)
        self.assertEqual(payload, before)
        self.assertEqual(result["raw_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["elements"], 3)
        self.assertEqual(result["scope"], "SUPPLIED_BYTES_ONLY")
        self.assertFalse(result["quality_claim"])
        self.assertFalse(result["native_execution_authorized"])
        for vocabulary in (True, 0, -1, 1.5, 1_000_001):
            with self.assertRaises(ValueError):
                inspect_finite_argmax_row(payload, vocabulary_size=vocabulary)
        for bad in (payload[:-1], payload + b"\0\0", bytearray(payload), memoryview(payload), None):
            with self.assertRaises(ValueError):
                inspect_finite_argmax_row(bad, vocabulary_size=3)
        for expected in (True, -1, 3, 2.0, "2", 1):
            with self.assertRaises(ValueError):
                self.inspect(payload, expected_argmax_token_id=expected)


if __name__ == "__main__":
    unittest.main()
