"""Metadata-only regression fragments; no semantic worlds or admission objects."""
import unittest
import strict_values as boundary


class StrictValuesTests(unittest.TestCase):
    def test_nested_boolean_integer_identity_differs(self):
        self.assertFalse(boundary.exact_identity({"ancestry": [{"ordinal": 1}]}, {"ancestry": [{"ordinal": True}]}))

    def test_nested_float_integer_identity_differs(self):
        self.assertFalse(boundary.exact_identity({"views": [0, 1]}, {"views": [0.0, 1.0]}))

    def test_dictionary_order_is_canonical(self):
        self.assertTrue(boundary.exact_identity({"a": 1, "b": False}, {"b": False, "a": 1}))

    def test_array_order_stays_semantic(self):
        self.assertFalse(boundary.exact_identity([0, 1], [1, 0]))

    def test_unicode_not_normalized(self):
        self.assertFalse(boundary.exact_identity("é", "e\u0301"))

    def test_duplicate_nested_key_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"a":{"x":0,"x":1}}')

    def test_nonstandard_nan_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"a":NaN}')

    def test_positive_overflow_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"a":1e999}')

    def test_negative_overflow_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"a":-1e999}')

    def test_escaped_surrogate_value_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"a":"\\ud800"}')

    def test_escaped_surrogate_key_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('{"\\udfff":0}')

    def test_literal_surrogate_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode('"\ud800"')

    def test_invalid_utf8_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode(b'"\xff"')

    def test_bom_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.strict_decode(b'\xef\xbb\xbf{}')

    def test_true_is_not_integer(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.exact_integer(True)

    def test_float_is_not_integer(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.exact_integer(1.0)

    def test_valid_integer_bounds(self):
        self.assertEqual(boundary.exact_integer(0, minimum=0, maximum=1), 0)

    def test_hash_suffix_rejected(self):
        for suffix in ("\n", "\r", " ", "\x00"):
            with self.assertRaises(boundary.BoundaryError):
                boundary.exact_sha256("a" * 64 + suffix)

    def test_unsupported_python_container_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.canonical_bytes((0, 1))

    def test_nonstring_key_rejected(self):
        with self.assertRaises(boundary.BoundaryError):
            boundary.canonical_bytes({1: "x"})

    def test_valid_json_round_trip(self):
        value = {"a": [None, True, 1, 1.0, "🟦"], "b": {"x": -100}}
        self.assertTrue(boundary.exact_identity(value, boundary.strict_decode(boundary.canonical_bytes(value))))


if __name__ == "__main__":
    unittest.main()
