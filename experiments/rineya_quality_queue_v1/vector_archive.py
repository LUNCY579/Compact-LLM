"""Bounded, in-memory lossless packing of opaque little-endian F16 rows.

Only the two bytes at UNKNOWN are replaced by zero in a canonical blob. Every
logical row retains its original patch, raw hash, order and supplied provenance.
No floating-point conversion, file input, dispatcher or model API exists here.
The caller must retain a trusted manifest digest separately for restoration;
an embedded digest alone authenticates neither metadata nor source identity.
This fixture prototype grants no quality/native permission or storage-fit claim.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from typing import Any

SCHEMA = "rineya.lossless_f16_rows/1"
MAX_ROWS = 4096
MAX_VOCABULARY = 1_000_000
DEFAULT_MAX_LOGICAL_BYTES = 64 * 2**20
MAX_LOGICAL_BYTES_BOUND = 2**40
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PATCH = re.compile(r"[0-9a-f]{4}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PERMISSIONS = {
    "native_execution": False, "training": False, "protected_data_read": False,
    "quality_claim": False, "release": False, "deployment": False,
}
_ROW_KEYS = {"slot", "domain", "source_identity", "raw_sha256", "canonical_sha256", "unknown_patch_hex"}
_BLOB_KEYS = {"canonical_sha256", "codec", "raw_bytes", "stored_bytes", "stored_sha256"}
_MANIFEST_KEYS = {
    "schema", "dtype", "byte_order", "shape", "unknown_index", "rows", "blobs",
    "logical_bytes", "unique_canonical_bytes", "stored_blob_bytes", "permissions",
    "model_quality", "manifest_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def _hash(value: Any) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _label(value: Any) -> bool:
    # Labels are opaque logical identities, never paths or filenames. In
    # particular separators, drive colons and dot traversal are not admitted.
    return type(value) is str and _LABEL.fullmatch(value) is not None and ".." not in value


def _dimensions(vocabulary_size: Any, unknown_index: Any, count: int, max_logical_bytes: Any) -> int:
    _require(type(vocabulary_size) is int and 1 <= vocabulary_size <= MAX_VOCABULARY,
             "bounded positive integer vocabulary_size required")
    _require(type(unknown_index) is int and 0 <= unknown_index < vocabulary_size,
             "UNKNOWN index outside row")
    _require(type(count) is int and 1 <= count <= MAX_ROWS, "bounded nonempty row list required")
    _require(type(max_logical_bytes) is int and 1 <= max_logical_bytes <= MAX_LOGICAL_BYTES_BOUND,
             "bounded positive integer logical-byte allowance required")
    row_bytes = 2 * vocabulary_size
    _require(count * row_bytes <= max_logical_bytes, "logical output byte allowance exceeded")
    return row_bytes


def _metadata(row: dict[str, Any]) -> None:
    _require(_label(row.get("slot")) and _label(row.get("domain")), "safe nonempty slot/domain labels required")
    _require(_hash(row.get("source_identity")), "canonical source_identity SHA-256 required")


def _root(manifest: dict[str, Any]) -> str:
    return _digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def pack_rows(rows: list[dict[str, Any]], *, vocabulary_size: int, unknown_index: int,
              codec: str = "none", max_logical_bytes: int = DEFAULT_MAX_LOGICAL_BYTES) -> dict[str, Any]:
    """Pack explicit byte rows; source_identity is supplied metadata, not attestation.

    Each input has exactly payload(bytes), slot, domain, source_identity. Slots
    must be globally unique even when row bytes are identical. Return contains
    a JSON-compatible manifest plus a SHA-keyed bytes mapping; original inputs
    are untouched. Compression may expand data; dedup is observed byte equality.
    """
    _require(type(rows) is list, "explicit row list required")
    row_bytes = _dimensions(vocabulary_size, unknown_index, len(rows), max_logical_bytes)
    _require(type(codec) is str and codec in {"none", "zlib"}, "unknown lossless codec")
    offset = 2 * unknown_index
    slots: set[str] = set()
    canonical_values: dict[str, bytes] = {}
    stored: dict[str, bytes] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    records = []
    for row in rows:
        _require(type(row) is dict and set(row) == {"payload", "slot", "domain", "source_identity"},
                 "exact input row keys required")
        _metadata(row)
        _require(row["slot"] not in slots, "duplicate logical slot")
        slots.add(row["slot"])
        payload = row["payload"]
        _require(type(payload) is bytes and len(payload) == row_bytes, "exact F16 row byte length required")
        canonical = payload[:offset] + b"\x00\x00" + payload[offset + 2:]
        canonical_sha = _sha(canonical)
        if canonical_sha in canonical_values:
            _require(canonical_values[canonical_sha] == canonical, "canonical digest collision; unequal rows cannot deduplicate")
        else:
            canonical_values[canonical_sha] = canonical
            encoded = canonical if codec == "none" else zlib.compress(canonical, level=9)
            stored[canonical_sha] = encoded
            descriptors[canonical_sha] = {
                "canonical_sha256": canonical_sha, "codec": codec, "raw_bytes": row_bytes,
                "stored_bytes": len(encoded), "stored_sha256": _sha(encoded),
            }
        records.append({
            "slot": row["slot"], "domain": row["domain"], "source_identity": row["source_identity"],
            "raw_sha256": _sha(payload), "canonical_sha256": canonical_sha,
            "unknown_patch_hex": payload[offset:offset + 2].hex(),
        })
    manifest = {
        "schema": SCHEMA, "dtype": "F16", "byte_order": "little", "shape": [vocabulary_size],
        "unknown_index": unknown_index, "rows": records,
        "blobs": [descriptors[key] for key in sorted(descriptors)],
        "logical_bytes": len(rows) * row_bytes,
        "unique_canonical_bytes": len(stored) * row_bytes,
        "stored_blob_bytes": sum(map(len, stored.values())),
        "permissions": dict(_PERMISSIONS), "model_quality": False,
    }
    manifest["manifest_sha256"] = _root(manifest)
    return {"manifest": manifest, "blobs": stored}


def _decode(payload: bytes, codec: str, row_bytes: int) -> bytes:
    if codec == "none":
        _require(len(payload) == row_bytes, "uncompressed blob length mismatch")
        return payload
    _require(codec == "zlib", "unknown lossless codec")
    decoder = zlib.decompressobj()
    try:
        value = decoder.decompress(payload, row_bytes + 1)
    except zlib.error as error:
        raise ValueError("invalid zlib blob") from error
    # Do not call an unbounded flush. Reject bombs, truncation, concatenation and
    # trailing bytes, including a second valid member after the first one.
    _require(len(value) == row_bytes and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
             "zlib blob must be one complete exact-length stream")
    return value


def restore_rows(archive: dict[str, Any], *, expected_manifest_sha256: str,
                 max_logical_bytes: int = DEFAULT_MAX_LOGICAL_BYTES) -> list[dict[str, Any]]:
    """Verify a trusted manifest root and reconstruct every original row exactly.

    expected_manifest_sha256 must come from a previously trusted binding, not
    be copied from an untrusted incoming manifest. No scientific authenticity
    follows from a source_identity string or a successful byte round trip.
    """
    _require(_hash(expected_manifest_sha256), "trusted canonical manifest SHA-256 required")
    _require(type(archive) is dict and set(archive) == {"manifest", "blobs"}, "exact archive keys required")
    manifest, blobs = archive["manifest"], archive["blobs"]
    _require(type(manifest) is dict and set(manifest) == _MANIFEST_KEYS, "exact manifest keys required")
    _require(manifest["schema"] == SCHEMA and manifest["dtype"] == "F16" and manifest["byte_order"] == "little",
             "unsupported archive schema/dtype/byte order")
    _require(type(manifest["permissions"]) is dict and manifest["permissions"] == _PERMISSIONS
             and all(value is False for value in manifest["permissions"].values()) and manifest["model_quality"] is False,
             "archive permissions and model quality must stay closed")
    _require(type(manifest["shape"]) is list and len(manifest["shape"]) == 1, "one vocabulary-row shape required")
    rows, descriptors = manifest["rows"], manifest["blobs"]
    _require(type(rows) is list, "explicit row manifest list required")
    row_bytes = _dimensions(manifest["shape"][0], manifest["unknown_index"], len(rows), max_logical_bytes)
    _require(type(descriptors) is list and 1 <= len(descriptors) <= len(rows), "bounded nonempty blob descriptor list required")
    _require(type(blobs) is dict and len(blobs) == len(descriptors), "blob inventory cardinality mismatch")
    slots, references = set(), set()
    for row in rows:
        _require(type(row) is dict and set(row) == _ROW_KEYS, "exact logical row metadata required")
        _metadata(row)
        _require(row["slot"] not in slots, "duplicate logical slot")
        slots.add(row["slot"])
        _require(_hash(row["raw_sha256"]) and _hash(row["canonical_sha256"]), "noncanonical row SHA-256")
        _require(type(row["unknown_patch_hex"]) is str and _PATCH.fullmatch(row["unknown_patch_hex"]) is not None,
                 "exact canonical two-byte UNKNOWN patch required")
        references.add(row["canonical_sha256"])
    descriptor_keys, stored_total = [], 0
    for descriptor in descriptors:
        _require(type(descriptor) is dict and set(descriptor) == _BLOB_KEYS, "exact blob metadata required")
        key = descriptor["canonical_sha256"]
        _require(_hash(key) and _hash(descriptor["stored_sha256"]), "noncanonical blob SHA-256/path key")
        _require(type(descriptor["codec"]) is str and descriptor["codec"] in {"none", "zlib"}, "unknown lossless codec")
        _require(type(descriptor["raw_bytes"]) is int and descriptor["raw_bytes"] == row_bytes, "canonical blob row length mismatch")
        # zlib can have a small expansion; this conservative finite ceiling is
        # deliberately separate from an assertion of compression or storage fit.
        _require(type(descriptor["stored_bytes"]) is int and 1 <= descriptor["stored_bytes"] <= row_bytes * 2 + 1024,
                 "stored blob byte bound exceeded")
        _require(key not in descriptor_keys, "duplicate canonical blob descriptor")
        descriptor_keys.append(key)
        stored_total += descriptor["stored_bytes"]
    _require(descriptor_keys == sorted(descriptor_keys), "canonical blob descriptor ordering required")
    _require(set(blobs) == set(descriptor_keys) == references, "missing, orphan or malformed blob key")
    for field, expected in (("logical_bytes", len(rows) * row_bytes),
                            ("unique_canonical_bytes", len(descriptors) * row_bytes),
                            ("stored_blob_bytes", stored_total)):
        _require(type(manifest[field]) is int and manifest[field] == expected, "archive byte counter mismatch: " + field)
    _require(_hash(manifest["manifest_sha256"]), "noncanonical embedded manifest SHA-256")
    try:
        root = _root(manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest is not canonical JSON metadata") from error
    _require(root == manifest["manifest_sha256"] == expected_manifest_sha256, "trusted manifest root mismatch")
    offset = 2 * manifest["unknown_index"]
    decoded = {}
    for descriptor in descriptors:
        key = descriptor["canonical_sha256"]
        payload = blobs[key]
        _require(type(payload) is bytes and len(payload) == descriptor["stored_bytes"], "stored blob length/type mismatch")
        _require(_sha(payload) == descriptor["stored_sha256"], "stored blob hash mismatch")
        canonical = _decode(payload, descriptor["codec"], row_bytes)
        _require(_sha(canonical) == key, "canonical blob hash mismatch")
        _require(canonical[offset:offset + 2] == b"\x00\x00", "canonical UNKNOWN slot is not zero")
        decoded[key] = canonical
    restored = []
    for row in rows:
        canonical = decoded[row["canonical_sha256"]]
        patch = bytes.fromhex(row["unknown_patch_hex"])
        payload = canonical[:offset] + patch + canonical[offset + 2:]
        _require(_sha(payload) == row["raw_sha256"], "restored raw row hash/patch mismatch")
        restored.append({"slot": row["slot"], "domain": row["domain"], "source_identity": row["source_identity"], "payload": payload})
    return restored
