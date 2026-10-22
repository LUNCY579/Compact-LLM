"""Byte preservation and rejection fixtures; no model or numeric F16 conversion."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from unittest.mock import patch
import zlib

from experiments.rineya_quality_queue_v1 import vector_archive as archive


def words(*values):
    return b"".join(value.to_bytes(2, "little") for value in values)


def row(payload, slot="capture-0", domain="fixture-logits", source="a" * 64):
    return {"payload": payload, "slot": slot, "domain": domain, "source_identity": source}


def seal(value):
    manifest = value["manifest"]
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return manifest["manifest_sha256"]


class VectorArchiveTests(unittest.TestCase):
    def setUp(self):
        self.payload = words(0x8000, 0x7C01, 0xFE01, 0x0001, 0x7BFF, 0xFFFF)
        self.before_runtime = set(sys.modules) & {"torch", "numpy", "transformers", "safetensors"}
        self.addCleanup(lambda: self.assertEqual(
            self.before_runtime, set(sys.modules) & {"torch", "numpy", "transformers", "safetensors"}))

    def pack(self, rows=None, **options):
        return archive.pack_rows(rows or [row(self.payload)], vocabulary_size=6, unknown_index=2, **options)

    def restore(self, value, expected=None, **options):
        return archive.restore_rows(value, expected_manifest_sha256=expected or value["manifest"]["manifest_sha256"], **options)

    def test_opaque_edge_bits_and_boundary_slots_round_trip_without_mutation(self):
        payload = words(0x0000, 0x8000, 0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFE01, 0xFFFF, 0x0001, 0x03FF, 0x7BFF)
        for index in (0, 5, 10):
            for codec in ("none", "zlib"):
                with self.subTest(index=index, codec=codec):
                    original = [row(payload)]
                    before = copy.deepcopy(original)
                    packed = archive.pack_rows(original, vocabulary_size=11, unknown_index=index, codec=codec)
                    saved = copy.deepcopy(packed)
                    self.assertEqual(self.restore(packed), original)
                    self.assertEqual(original, before)
                    self.assertEqual(packed, saved)
                    self.assertFalse(packed["manifest"]["model_quality"])
                    self.assertTrue(all(flag is False for flag in packed["manifest"]["permissions"].values()))

    def test_all_65536_unknown_bit_patterns_restore_exactly(self):
        for start in range(0, 65536, archive.MAX_ROWS):
            rows = [row(words(0x8000, value, 0x7C01), slot="bits-" + str(value))
                    for value in range(start, start + archive.MAX_ROWS)]
            packed = archive.pack_rows(rows, vocabulary_size=3, unknown_index=1)
            self.assertEqual(len(packed["blobs"]), 1)
            self.assertEqual(self.restore(packed), rows)

    def test_only_unknown_patch_changes_coalesce_and_preserve_order_provenance(self):
        changed = self.payload[:4] + b"\x00\x80" + self.payload[6:]
        rows = [row(self.payload, "first"), row(changed, "second", "another-domain", "b" * 64), row(self.payload, "third")]
        packed = self.pack(rows)
        self.assertEqual(len(packed["blobs"]), 1)
        self.assertEqual(len(packed["manifest"]["rows"]), 3)
        self.assertEqual(self.restore(packed), rows)
        self.assertNotEqual(packed["manifest"]["rows"][0]["raw_sha256"], packed["manifest"]["rows"][1]["raw_sha256"])

    def test_single_off_channel_byte_change_is_a_different_blob(self):
        changed = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        rows = [row(self.payload, "first"), row(changed, "second")]
        packed = self.pack(rows)
        self.assertEqual(len(packed["blobs"]), 2)
        self.assertEqual(self.restore(packed), rows)
        # A digest collision must not turn actual unequal bytes into dedup.
        with patch.object(archive, "_sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "collision"):
                self.pack(rows)

    def test_both_lossless_codecs_have_same_raw_rows_and_canonical_addresses(self):
        rows = [row(words(*([0] * 100)), "zeros"), row(words(*range(100)), "varied")]
        packed = [archive.pack_rows(rows, vocabulary_size=100, unknown_index=19, codec=codec) for codec in ("none", "zlib")]
        self.assertEqual(set(packed[0]["blobs"]), set(packed[1]["blobs"]))
        for value in packed:
            self.assertEqual(self.restore(value), rows)

    def test_metadata_reseal_cannot_replace_the_trusted_root(self):
        base = self.pack([row(self.payload, "first"), row(self.payload, "second")])
        trusted = base["manifest"]["manifest_sha256"]
        mutations = [
            lambda m: m["rows"].reverse(),
            lambda m: m["rows"][0].update(slot="different"),
            lambda m: m["rows"][0].update(domain="different"),
            lambda m: m["rows"][0].update(source_identity="b" * 64),
            lambda m: m.update(unknown_index=3),
        ]
        for mutate in mutations:
            value = copy.deepcopy(base)
            mutate(value["manifest"])
            seal(value)
            with self.subTest(manifest=value["manifest"]["manifest_sha256"]):
                with self.assertRaisesRegex(ValueError, "trusted manifest root"):
                    self.restore(value, trusted)

    def test_incorrect_patch_raw_hash_and_corrupt_blob_are_rejected(self):
        for field, replacement in (("unknown_patch_hex", "0000"), ("raw_sha256", "0" * 64)):
            value = self.pack()
            value["manifest"]["rows"][0][field] = replacement
            seal(value)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "raw row hash/patch"):
                    self.restore(value)
        value = self.pack()
        key = next(iter(value["blobs"]))
        value["blobs"][key] = bytes([value["blobs"][key][0] ^ 1]) + value["blobs"][key][1:]
        with self.assertRaisesRegex(ValueError, "stored blob hash"):
            self.restore(value)

    def test_self_consistent_nonzero_canonical_slot_and_bad_canonical_hash_reject(self):
        value = self.pack()
        old_key = next(iter(value["blobs"]))
        blob = value["blobs"].pop(old_key)
        bad = blob[:4] + b"\x01\x00" + blob[6:]
        key = hashlib.sha256(bad).hexdigest()
        value["blobs"][key] = bad
        value["manifest"]["rows"][0]["canonical_sha256"] = key
        value["manifest"]["blobs"][0].update(canonical_sha256=key, stored_sha256=key)
        seal(value)
        with self.assertRaisesRegex(ValueError, "slot is not zero"):
            self.restore(value)
        value = self.pack()
        key = next(iter(value["blobs"]))
        bad = b"\x01" + value["blobs"][key][1:]
        value["blobs"][key] = bad
        value["manifest"]["blobs"][0]["stored_sha256"] = hashlib.sha256(bad).hexdigest()
        seal(value)
        with self.assertRaisesRegex(ValueError, "canonical blob hash"):
            self.restore(value)

    def test_zlib_truncation_trailing_concatenation_and_overlong_output_reject(self):
        base = self.pack(codec="zlib")
        key = next(iter(base["blobs"]))
        stream = base["blobs"][key]
        bad_streams = [stream[:-1], stream + b"trailing", stream + stream,
                       zlib.compress(b"\0" * 100000), b"not zlib"]
        for bad in bad_streams:
            value = copy.deepcopy(base)
            value["blobs"][key] = bad
            descriptor = value["manifest"]["blobs"][0]
            descriptor.update(stored_bytes=len(bad), stored_sha256=hashlib.sha256(bad).hexdigest())
            value["manifest"]["stored_blob_bytes"] = len(bad)
            seal(value)
            with self.subTest(length=len(bad)):
                with self.assertRaisesRegex(ValueError, "zlib"):
                    self.restore(value)

    def test_strict_shapes_payload_types_logical_bounds_and_duplicate_slots(self):
        for vocabulary, index in ((True, 0), (0, 0), (6.0, 2), (6, True), (6, -1), (6, 6)):
            with self.subTest(vocabulary=vocabulary, index=index):
                with self.assertRaises(ValueError):
                    archive.pack_rows([row(self.payload)], vocabulary_size=vocabulary, unknown_index=index)
        for payload in (self.payload[:-1], self.payload + b"\0\0", bytearray(self.payload), memoryview(self.payload), None):
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(ValueError):
                    self.pack([row(payload)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.pack([row(self.payload), row(self.payload)])
        packed = self.pack([row(self.payload, "first"), row(self.payload, "second")])
        with self.assertRaisesRegex(ValueError, "logical output"):
            self.restore(packed, max_logical_bytes=12)
        for allowance in (True, 0, -1, 1.5):
            with self.subTest(allowance=allowance):
                with self.assertRaises(ValueError):
                    self.pack(max_logical_bytes=allowance)

    def test_path_labels_noncanonical_hashes_inventory_and_unknown_keys_reject(self):
        for label in ("../x", "absolute/path", r"C:\x", "C:drive", "..", "", "a..b"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.pack([row(self.payload, label)])
        for mutation in (
            lambda m: m["rows"][0].update(canonical_sha256="A" * 64),
            lambda m: m["rows"][0].update(unknown_patch_hex="ABCD"),
            lambda m: m["rows"][0].update(source_identity="../blob"),
            lambda m: m.update(shape=[True]),
            lambda m: m.update(dtype="F32"),
            lambda m: m.update(logical_bytes=True),
            lambda m: m.update(path="outside-file"),
        ):
            value = self.pack()
            mutation(value["manifest"])
            seal(value)
            with self.assertRaises(ValueError):
                self.restore(value)
        value = self.pack()
        del value["blobs"][next(iter(value["blobs"]))]
        with self.assertRaises(ValueError):
            self.restore(value)
        value = self.pack()
        value["blobs"]["../blob"] = b"unsafe"
        with self.assertRaises(ValueError):
            self.restore(value)
        value = self.pack()
        value["manifest"]["blobs"].append(copy.deepcopy(value["manifest"]["blobs"][0]))
        seal(value)
        with self.assertRaises(ValueError):
            self.restore(value)

    def test_resealed_open_permissions_quality_and_missing_trusted_root_reject(self):
        for field in archive._PERMISSIONS:
            value = self.pack()
            value["manifest"]["permissions"][field] = True
            seal(value)
            with self.assertRaisesRegex(ValueError, "stay closed"):
                self.restore(value)
        value = self.pack()
        value["manifest"]["model_quality"] = True
        seal(value)
        with self.assertRaisesRegex(ValueError, "stay closed"):
            self.restore(value)
        for expected in (None, True, "", "A" * 64, "f" * 64):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    archive.restore_rows(self.pack(), expected_manifest_sha256=expected)


if __name__ == "__main__":
    unittest.main()
