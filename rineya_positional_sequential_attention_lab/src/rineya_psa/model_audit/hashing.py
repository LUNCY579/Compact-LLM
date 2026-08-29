"""Content-addressed identity for checkpoint files, tensors and tokenizers.

Torch-free on purpose. Phase 1 runs while another process owns the GPU, so this
module reads safetensors payloads through ``numpy.memmap`` and decodes bfloat16 by
hand (``uint16 << 16`` reinterpreted as float32, which is lossless) instead of
paying for a torch import and a CUDA context. SPEC.md §2 also forbids importing
torch at module scope outside a short whitelist that does not include
``model_audit``.

The safetensors header is parsed directly rather than through the ``safetensors``
package so that whole-file and per-tensor digests can be produced in a *single*
sequential pass: the panel is ~6.5 GB and reading it twice is pure waste.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 4 MiB: big enough to amortise syscalls, small enough that the resident set stays
# flat while streaming a 2.4 GB float32 checkpoint.
CHUNK_BYTES: int = 1 << 22

# A safetensors header is JSON; anything above this is a corrupt or hostile file.
MAX_HEADER_BYTES: int = 1 << 28

TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)

# transformers >= 5 stores the chat template beside the tokenizer config instead of
# inside it. Both spellings must hash the same or Halo would look like a different
# tokenizer than Qwen3 for a purely cosmetic reason.
CHAT_TEMPLATE_FILE: str = "chat_template.jinja"

_ST_NUMPY: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}

# BF16 has no numpy dtype; it is handled by widening. Keep its itemsize alongside
# the natively supported ones so offset arithmetic stays uniform.
_ST_ITEMSIZE: dict[str, int] = {name: dt.itemsize for name, dt in _ST_NUMPY.items()}
_ST_ITEMSIZE["BF16"] = 2

FLOAT_DTYPES: frozenset[str] = frozenset({"F64", "F32", "F16", "BF16"})


# --------------------------------------------------------------------------- #
# generic digests
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = CHUNK_BYTES) -> str:
    """SHA-256 of a whole file, read in chunks so a 2.4 GB core never lands in RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON text: sorted keys, no incidental whitespace, unicode preserved.

    Two configs that differ only in key order or indentation must hash the same,
    otherwise every re-serialisation by a newer ``transformers`` would read as a
    checkpoint change.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# safetensors layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorLocation:
    """Where one tensor's raw bytes live, and how to interpret them."""

    name: str
    file: Path
    st_dtype: str
    shape: tuple[int, ...]
    begin: int  # absolute byte offset into ``file``
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def n_params(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def itemsize(self) -> int:
        return _ST_ITEMSIZE[self.st_dtype]

    @property
    def is_float(self) -> bool:
        return self.st_dtype in FLOAT_DTYPES


def read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Return ``(tensor_entries, metadata, data_start)`` for a safetensors file.

    ``data_start`` is the absolute offset the per-tensor ``data_offsets`` are
    relative to (8 header-length bytes plus the JSON header).
    """
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        if not 0 < header_len <= MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible safetensors header length {header_len}")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not a JSON object")
    metadata = header.pop("__metadata__", {}) or {}
    return header, dict(metadata), 8 + header_len


def tensor_locations(path: str | Path) -> dict[str, TensorLocation]:
    """Map tensor name -> location for one safetensors shard."""
    path = Path(path)
    entries, _meta, data_start = read_safetensors_header(path)
    out: dict[str, TensorLocation] = {}
    for name, entry in entries.items():
        st_dtype = str(entry["dtype"])
        if st_dtype not in _ST_ITEMSIZE:
            raise ValueError(f"{path}:{name}: unsupported safetensors dtype {st_dtype!r}")
        begin, end = (int(x) for x in entry["data_offsets"])
        shape = tuple(int(d) for d in entry["shape"])
        loc = TensorLocation(
            name=name,
            file=path,
            st_dtype=st_dtype,
            shape=shape,
            begin=data_start + begin,
            end=data_start + end,
        )
        expected = loc.n_params * loc.itemsize
        if loc.nbytes != expected:
            raise ValueError(
                f"{path}:{name}: header claims {loc.nbytes} bytes but "
                f"{shape} x {st_dtype} needs {expected}"
            )
        out[name] = loc
    return out


def discover_model_files(model_dir: str | Path) -> list[Path]:
    """Weight shards in load order. Single-file and sharded layouts both work."""
    model_dir = Path(model_dir)
    shards = sorted(p for p in model_dir.glob("*.safetensors") if p.is_file())
    if shards:
        return shards
    return sorted(p for p in model_dir.glob("*.bin") if p.is_file())


def collect_tensor_locations(files: Iterable[str | Path]) -> dict[str, TensorLocation]:
    """Merge shards into one name -> location map, rejecting duplicate names."""
    merged: dict[str, TensorLocation] = {}
    for f in files:
        for name, loc in tensor_locations(f).items():
            if name in merged:
                raise ValueError(
                    f"tensor {name!r} appears in both {merged[name].file.name} and "
                    f"{Path(f).name}; the shard index is inconsistent"
                )
            merged[name] = loc
    return merged


def open_memmap(path: str | Path) -> np.memmap:
    """Read-only byte view of a whole file. One per file, sliced per tensor."""
    return np.memmap(path, dtype=np.uint8, mode="r")


def decode_to_float64(raw: np.ndarray, st_dtype: str) -> np.ndarray:
    """Reinterpret a 1-D uint8 view as ``st_dtype`` and widen to float64.

    bfloat16 is widened by left-shifting into the float32 exponent/mantissa layout,
    which is exact: bf16 *is* the top 16 bits of a float32.
    """
    if raw.dtype != np.uint8:
        raise TypeError("decode_to_float64 expects a uint8 view")
    if st_dtype == "BF16":
        wide = raw.view("<u2").astype("<u4")
        np.left_shift(wide, 16, out=wide)
        return wide.view("<f4").astype(np.float64)
    try:
        nd = _ST_NUMPY[st_dtype]
    except KeyError:
        raise ValueError(f"cannot decode safetensors dtype {st_dtype!r} to float64") from None
    return raw.view(nd).astype(np.float64)


# --------------------------------------------------------------------------- #
# one-pass file + tensor hashing
# --------------------------------------------------------------------------- #


def hash_safetensors_file(
    path: str | Path, *, chunk_size: int = CHUNK_BYTES
) -> tuple[str, dict[str, str]]:
    """SHA-256 of the whole file *and* of every tensor payload, in one read.

    Tensors are laid out contiguously and in offset order, so a single sequential
    walk can feed the file digest continuously while forking each tensor's byte
    range into its own digest. Header bytes and any inter-tensor padding count
    towards the file digest only.
    """
    path = Path(path)
    locs = tensor_locations(path)
    ordered = sorted(locs.values(), key=lambda loc: loc.begin)

    file_hash = hashlib.sha256()
    per_tensor: dict[str, str] = {}
    pos = 0
    with open(path, "rb") as fh:
        for loc in ordered:
            if loc.begin < pos:
                raise ValueError(f"{path}:{loc.name}: overlapping tensor byte ranges")
            _consume(fh, file_hash, None, loc.begin - pos, chunk_size)
            th = hashlib.sha256()
            _consume(fh, file_hash, th, loc.nbytes, chunk_size)
            per_tensor[loc.name] = th.hexdigest()
            pos = loc.end
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            file_hash.update(block)
    return file_hash.hexdigest(), per_tensor


def _consume(fh: Any, primary: Any, secondary: Any, count: int, chunk_size: int) -> None:
    """Read exactly ``count`` bytes, feeding one or two digests."""
    remaining = count
    while remaining > 0:
        block = fh.read(min(chunk_size, remaining))
        if not block:
            raise ValueError("unexpected end of file while hashing")
        primary.update(block)
        if secondary is not None:
            secondary.update(block)
        remaining -= len(block)


def sha256_tensor(loc: TensorLocation, *, chunk_size: int = CHUNK_BYTES) -> str:
    """SHA-256 of one tensor's raw bytes. Used for spot checks and tied-head tests."""
    h = hashlib.sha256()
    with open(loc.file, "rb") as fh:
        fh.seek(loc.begin)
        _consume(fh, h, None, loc.nbytes, chunk_size)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# directory inventory
# --------------------------------------------------------------------------- #

# Hugging Face's local download bookkeeping. It records fetch times, not model
# content, so including it would make the manifest machine-specific.
IGNORED_DIRS: frozenset[str] = frozenset({".cache", ".git", "__pycache__"})


def hash_directory(
    model_dir: str | Path,
    *,
    chunk_size: int = CHUNK_BYTES,
    precomputed: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Inventory every content file under ``model_dir`` (recursive, sorted).

    ``precomputed`` lets the caller pass digests already produced by
    :func:`hash_safetensors_file` so the big shards are not read a second time.
    """
    model_dir = Path(model_dir)
    precomputed = dict(precomputed or {})
    rows: list[dict[str, Any]] = []
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(model_dir).parts
        if any(part in IGNORED_DIRS for part in rel_parts[:-1]):
            continue
        rel = "/".join(rel_parts)
        digest = precomputed.get(rel) or sha256_file(p, chunk_size=chunk_size)
        rows.append({"name": rel, "size_bytes": p.stat().st_size, "sha256": digest})
    return rows


def directory_fingerprint(model_dir: str | Path) -> str:
    """Cheap change detector: names, sizes and mtimes, no payload reads.

    Used only to decide whether a cached audit is still valid. The audit itself
    always carries real SHA-256s.
    """
    model_dir = Path(model_dir)
    items: list[tuple[str, int, int]] = []
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(model_dir).parts
        if any(part in IGNORED_DIRS for part in rel_parts[:-1]):
            continue
        st = p.stat()
        items.append(("/".join(rel_parts), st.st_size, st.st_mtime_ns))
    return sha256_json(items)


# --------------------------------------------------------------------------- #
# tokenizer identity
# --------------------------------------------------------------------------- #

# Keys that describe *how this copy was loaded*, not what the tokenizer does.
VOLATILE_TOKENIZER_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "is_local",
        "local_files_only",
        "use_fast",
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "name_or_path",
        "_name_or_path",
        "special_tokens_map_file",
        "processor_class",
        "transformers_version",
    }
)

_TOKEN_FIELDS: tuple[str, ...] = (
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "sep_token",
    "cls_token",
    "mask_token",
)


def _token_content(value: Any) -> str | None:
    """Reduce a special-token spec to its surface string.

    A token may be stored as a plain string or as a serialised ``AddedToken``
    dict; the two spellings mean the same thing.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        content = value.get("content")
        return None if content is None else str(content)
    return str(value)


def _normalize_chat_template(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        # transformers allows a list of {name, template} entries.
        return {
            str(entry.get("name")): str(entry.get("template", "")).strip()
            for entry in value
            if isinstance(entry, Mapping)
        }
    return value


def normalize_tokenizer_config(
    cfg: Mapping[str, Any],
    *,
    chat_template: str | None = None,
    added_tokens: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Canonical view of ``tokenizer_config.json``, comparable across transformers versions.

    Normalises four real spelling differences seen in this panel: the
    ``extra_special_tokens`` rename, ``AddedToken`` dicts vs bare strings, the
    chat template moving out to ``chat_template.jinja``, and
    ``added_tokens_decoder`` being omitted in favour of ``tokenizer.json``'s
    ``added_tokens``.
    """
    out: dict[str, Any] = {
        k: v for k, v in dict(cfg).items() if k not in VOLATILE_TOKENIZER_CONFIG_KEYS
    }

    extra = out.pop("extra_special_tokens", None)
    if extra is not None and "additional_special_tokens" not in out:
        out["additional_special_tokens"] = extra
    special = out.get("additional_special_tokens")
    if isinstance(special, Sequence) and not isinstance(special, str):
        contents = [_token_content(x) for x in special]
        out["additional_special_tokens"] = sorted(c for c in contents if c is not None)

    for field in _TOKEN_FIELDS:
        if field in out:
            out[field] = _token_content(out[field])

    decoder = out.pop("added_tokens_decoder", None)
    resolved: dict[int, str] = {}
    if isinstance(decoder, Mapping) and decoder:
        for tid, spec in decoder.items():
            content = _token_content(spec)
            if content is not None:
                resolved[int(tid)] = content
    elif added_tokens:
        resolved = {int(k): str(v) for k, v in added_tokens.items()}
    out["added_tokens"] = [[tid, resolved[tid]] for tid in sorted(resolved)]

    template = out.get("chat_template")
    if template is None:
        template = chat_template
    out["chat_template"] = _normalize_chat_template(template)
    return out


def tokenizer_json_structure(tokenizer_json: Mapping[str, Any]) -> dict[str, Any]:
    """Everything in ``tokenizer.json`` except the vocabulary itself.

    The vocab and merge table are ~11 MB and are already covered by their own
    digests. What is left -- normalizer, pre-tokenizer, post-processor, decoder,
    padding, truncation -- is small, is what actually differs between
    re-serialisations, and is worth carrying in the manifest so a tokenizer
    mismatch can be reported as a field name rather than as "the hashes differ".
    """
    if not tokenizer_json:
        return {}
    out = {k: v for k, v in tokenizer_json.items() if k != "model"}
    model = tokenizer_json.get("model") or {}
    if isinstance(model, Mapping):
        trimmed = {k: v for k, v in model.items() if k not in {"vocab", "merges"}}
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        trimmed["vocab_size"] = len(vocab)
        trimmed["n_merges"] = len(merges)
        out["model"] = trimmed
    return out


def added_tokens_from_tokenizer_json(tokenizer_json: Mapping[str, Any]) -> dict[int, str]:
    entries = tokenizer_json.get("added_tokens") or []
    out: dict[int, str] = {}
    for entry in entries:
        if isinstance(entry, Mapping) and "id" in entry and "content" in entry:
            out[int(entry["id"])] = str(entry["content"])
    return out


def tokenizer_identity(
    model_dir: str | Path, *, file_sha256: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Hashes that pin a tokenizer's identity, at three strictnesses.

    * ``identity_sha256`` — the brief's definition: raw ``tokenizer.json`` +
      ``vocab.json`` + ``merges.txt`` bytes plus the normalised config.
    * ``semantic_sha256`` — same, but ``tokenizer.json`` is canonicalised as JSON,
      so a pure re-serialisation does not read as a different tokenizer.
    * ``vocab_merges_sha256`` — vocabulary only; equal across the whole panel, and
      the thing that actually has to match for token ids to be comparable.
    """
    model_dir = Path(model_dir)
    known = dict(file_sha256 or {})

    present: dict[str, bool] = {}
    digests: dict[str, str | None] = {}
    for fname in (*TOKENIZER_FILES, CHAT_TEMPLATE_FILE):
        p = model_dir / fname
        present[fname] = p.is_file()
        digests[fname] = (known.get(fname) or sha256_file(p)) if p.is_file() else None

    tok_json: dict[str, Any] = {}
    if present["tokenizer.json"]:
        tok_json = read_json(model_dir / "tokenizer.json")
    raw_cfg: dict[str, Any] = {}
    if present["tokenizer_config.json"]:
        raw_cfg = read_json(model_dir / "tokenizer_config.json")
    template_text: str | None = None
    if present[CHAT_TEMPLATE_FILE]:
        template_text = (model_dir / CHAT_TEMPLATE_FILE).read_text(encoding="utf-8")

    normalized = normalize_tokenizer_config(
        raw_cfg,
        chat_template=template_text,
        added_tokens=added_tokens_from_tokenizer_json(tok_json),
    )
    normalized_sha = sha256_json(normalized)
    tok_json_canonical_sha = sha256_json(tok_json) if tok_json else None

    identity_sha = sha256_json(
        {
            "tokenizer.json": digests["tokenizer.json"],
            "vocab.json": digests["vocab.json"],
            "merges.txt": digests["merges.txt"],
            "tokenizer_config.normalized": normalized_sha,
        }
    )
    semantic_sha = sha256_json(
        {
            "tokenizer.json.canonical": tok_json_canonical_sha,
            "vocab.json": digests["vocab.json"],
            "merges.txt": digests["merges.txt"],
            "tokenizer_config.normalized": normalized_sha,
        }
    )
    vocab_merges_sha = sha256_json(
        {"vocab.json": digests["vocab.json"], "merges.txt": digests["merges.txt"]}
    )

    structure = tokenizer_json_structure(tok_json)
    return {
        "files_present": present,
        "file_sha256": digests,
        "tokenizer_json_structure": structure,
        "tokenizer_json_structure_sha256": sha256_json(structure),
        "tokenizer_class": raw_cfg.get("tokenizer_class"),
        "model_max_length": raw_cfg.get("model_max_length"),
        "chat_template_location": (
            "chat_template.jinja"
            if (raw_cfg.get("chat_template") is None and template_text is not None)
            else ("tokenizer_config.json" if raw_cfg.get("chat_template") is not None else None)
        ),
        "config_normalized": normalized,
        "config_normalized_sha256": normalized_sha,
        "tokenizer_json_canonical_sha256": tok_json_canonical_sha,
        "vocab_merges_sha256": vocab_merges_sha,
        "identity_sha256": identity_sha,
        "semantic_sha256": semantic_sha,
    }


def json_diff_paths(a: Any, b: Any, *, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON-ish trees disagree. Used for config diffs."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(child)
            else:
                out.extend(json_diff_paths(a[key], b[key], prefix=child))
        return out
    if (
        isinstance(a, Sequence)
        and isinstance(b, Sequence)
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):
            return [prefix or "<root>"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(json_diff_paths(x, y, prefix=f"{prefix}[{i}]"))
        return out
    return [] if a == b else [prefix or "<root>"]
