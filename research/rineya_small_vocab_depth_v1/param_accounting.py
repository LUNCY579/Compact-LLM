"""SMALLVD-001 Phase 1 -- exact parameter accounting for the teacher and every candidate student.

THREE INDEPENDENT COUNTS that must agree, so no single one is trusted:
  1. HEADER   -- unique physical tensors in a safetensors header. The format cannot alias, so a
                 tied head shows up as an ABSENT lm_head.weight. It is counted once.
  2. ANALYTIC -- closed form from config.json dimensions. Used for candidates that do not exist
                 on disk yet.
  3. TORCH    -- `unique_params_torch` deduplicates a live module by storage pointer. A plain
                 state_dict sum counts the tied embedding twice, which is the "blind alias sum"
                 the brief warns about. It lives in `surgery.py`'s environment; this file does not
                 import torch.

SERIALIZED SIZE IS A MEASUREMENT, NOT AN ESTIMATE. `safetensors_layout` rebuilds the exact header
bytes that safetensors 0.7.0 writes: lexicographic key order for one dtype, compact JSON,
`__metadata__` first, space padding to 8 bytes. It is asserted BYTE-FOR-BYTE against the teacher's
real header before any predicted size is printed. Built students are measured on disk as well
(`surgery.py`), and those two numbers must match exactly.

Pure python on purpose: it reads 35 KB of header and nothing else.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2,
               "I8": 1, "U8": 1, "BOOL": 1}
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Candidate grid from the brief. Vocabulary sizes are TOTAL EMBEDDING ROWS (regular + added
# tokens + any padding). 98304/65536/49152/32768 are multiples of 128, so no padding row is
# needed, and 65536 lets token ids fit in uint16.
VOCAB_GRID = {"V0": 151936, "V1": 98304, "V2": 65536, "V3": 49152, "V4": 32768}
DEPTH_GRID = (28, 24, 20, 18, 16, 12)


def read_header(path: str | Path) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), raw


def classify(name: str) -> tuple[str, int | None]:
    """Component family and block index. Every tensor must land in exactly one family."""
    m = LAYER_RE.match(name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if rest in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            return "qk_norm", layer
        if rest.startswith("self_attn.") and rest.split(".")[1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return "attention", layer
        if rest.startswith("mlp.") and rest.split(".")[1] in ("gate_proj", "up_proj", "down_proj"):
            return "ffn", layer
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return "block_norm", layer
        return "other_block", layer
    if name == "model.embed_tokens.weight":
        return "embedding", None
    if name == "lm_head.weight":
        return "lm_head", None
    if name == "model.norm.weight":
        return "final_norm", None
    return "other", None


def numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def account_header(header: dict, config: dict) -> dict:
    """Unique physical parameters from a safetensors header plus the config's tying flag."""
    fam, per_block, dtypes = {}, {}, set()
    for name, info in header.items():
        if name == "__metadata__":
            continue
        f, layer = classify(name)
        n = numel(info["shape"])
        fam[f] = fam.get(f, 0) + n
        dtypes.add(info["dtype"])
        if layer is not None:
            per_block[layer] = per_block.get(layer, 0) + n
    has_head = "lm_head.weight" in header
    tied = bool(config.get("tie_word_embeddings"))
    if tied and has_head:
        # Both present AND tied would mean the file stores an alias twice. safetensors refuses
        # to write shared storage, so this is a malformed checkpoint, not a counting choice.
        raise ValueError("tie_word_embeddings=true but lm_head.weight is stored separately")
    if not tied and not has_head:
        raise ValueError("untied config but no lm_head.weight in the checkpoint")
    total = sum(fam.values())
    return {
        "unique_physical_parameters": total,
        "by_family": dict(sorted(fam.items())),
        "embedding_and_head": fam.get("embedding", 0) + fam.get("lm_head", 0),
        "head_tied_to_embedding": tied,
        "lm_head_physically_stored": has_head,
        "naive_alias_sum_if_tied_head_materialised": total + (fam.get("embedding", 0) if tied else 0),
        "per_block": {str(k): per_block[k] for k in sorted(per_block)},
        "n_blocks": len(per_block),
        "dtypes": sorted(dtypes),
        "data_bytes": sum(numel(i["shape"]) * DTYPE_BYTES[i["dtype"]]
                          for k, i in header.items() if k != "__metadata__"),
    }


def block_params(config: dict) -> dict:
    d, f = config["hidden_size"], config["intermediate_size"]
    nh, nkv = config["num_attention_heads"], config["num_key_value_heads"]
    hd = config.get("head_dim") or d // nh
    bias = bool(config.get("attention_bias"))
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    if bias:
        attn += nh * hd + 2 * nkv * hd
    return {"attention": attn, "qk_norm": 2 * hd, "ffn": 3 * d * f, "block_norm": 2 * d}


def analytic(config: dict, vocab_rows: int, n_layers: int) -> dict:
    d = config["hidden_size"]
    blk = block_params(config)
    per_block = sum(blk.values())
    tied = bool(config.get("tie_word_embeddings"))
    fam = {k: v * n_layers for k, v in blk.items()}
    fam["embedding"] = vocab_rows * d
    if not tied:
        fam["lm_head"] = vocab_rows * d
    fam["final_norm"] = d
    total = sum(fam.values())
    return {"vocab_rows": vocab_rows, "n_layers": n_layers, "per_block": per_block,
            "by_family": dict(sorted(fam.items())), "unique_physical_parameters": total,
            "embedding_and_head": fam["embedding"] + fam.get("lm_head", 0)}


def student_shapes(teacher_header: dict, vocab_rows: int, kept_layers: list[int]) -> dict:
    """Name -> (dtype, shape) of a student: vocabulary rows replaced, kept blocks RENUMBERED
    0..L-1 in their original order. Mirrors exactly what `surgery.py` writes."""
    out = {}
    pos = {old: new for new, old in enumerate(kept_layers)}
    for name, info in teacher_header.items():
        if name == "__metadata__":
            continue
        m = LAYER_RE.match(name)
        shape = list(info["shape"])
        if m:
            old = int(m.group(1))
            if old not in pos:
                continue
            name = f"model.layers.{pos[old]}.{m.group(2)}"
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            shape = [vocab_rows] + shape[1:]
        out[name] = (info["dtype"], shape)
    return out


def safetensors_layout(shapes: dict, metadata: dict | None = None) -> tuple[bytes, int]:
    """Exact header bytes and total file size that safetensors 0.7.0 `save_file` produces.

    Order is dtype-alignment DESCENDING then name -- identical to lexicographic when every tensor
    shares one dtype, which is asserted rather than assumed."""
    if len({dt for dt, _ in shapes.values()}) != 1:
        raise NotImplementedError("mixed-dtype layout ordering is not reproduced here")
    parts, off = [], 0
    if metadata is not None:
        parts.append('"__metadata__":' + json.dumps(metadata, separators=(",", ":")))
    for name in sorted(shapes):
        dt, shape = shapes[name]
        n = numel(shape) * DTYPE_BYTES[dt]
        parts.append(json.dumps(name) + ":" + json.dumps(
            {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}, separators=(",", ":")))
        off += n
    raw = ("{" + ",".join(parts) + "}").encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return raw, 8 + len(raw) + off


def assert_layout_reproduces(path: str | Path) -> dict:
    """The predictor is only trusted after it reproduces a REAL header byte-for-byte."""
    header, raw = read_header(path)
    shapes = {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
    pred_raw, pred_size = safetensors_layout(shapes, header.get("__metadata__"))
    real_size = Path(path).stat().st_size
    ok = pred_raw == raw and pred_size == real_size
    if not ok:
        raise AssertionError(f"layout predictor does NOT reproduce {path}: "
                             f"header {len(pred_raw)} vs {len(raw)}, size {pred_size} vs {real_size}")
    return {"file": str(path), "header_bytes": len(raw), "file_bytes": real_size,
            "header_byte_identical": True}


def evenly_kept(n_teacher: int, n_keep: int) -> list[int]:
    """Placeholder layer set for SIZE accounting only -- sizes do not depend on WHICH layers are
    kept. The real selection is evidence-based (layer_select.py) and never uses this."""
    if n_keep == n_teacher:
        return list(range(n_teacher))
    step = (n_teacher - 1) / (n_keep - 1)
    kept = sorted({round(i * step) for i in range(n_keep)})
    assert len(kept) == n_keep
    return kept


def grid(teacher_dir: Path) -> dict:
    config = json.loads((teacher_dir / "config.json").read_text(encoding="utf-8"))
    header, _ = read_header(teacher_dir / "model.safetensors")
    measured = account_header(header, config)
    t = analytic(config, config["vocab_size"], config["num_hidden_layers"])
    for key in ("unique_physical_parameters", "by_family"):
        if measured[key] != t[key]:
            raise AssertionError(f"HEADER and ANALYTIC disagree on {key}: {measured[key]} vs {t[key]}")
    reproduce = assert_layout_reproduces(teacher_dir / "model.safetensors")
    rows = []
    for vname, v in VOCAB_GRID.items():
        for L in DEPTH_GRID:
            a = analytic(config, v, L)
            shapes = student_shapes(header, v, evenly_kept(config["num_hidden_layers"], L))
            _, size = safetensors_layout(shapes, header.get("__metadata__"))
            total = a["unique_physical_parameters"]
            rows.append({
                "vocab": vname, "vocab_rows": v, "n_layers": L,
                "unique_physical_parameters": total,
                "embedding_parameters": a["embedding_and_head"],
                "block_parameters": a["per_block"] * L,
                "embedding_share": round(a["embedding_and_head"] / total, 4),
                "fraction_of_teacher": round(total / t["unique_physical_parameters"], 4),
                "reduction_from_vocab": t["embedding_and_head"] - a["embedding_and_head"],
                "reduction_from_depth": t["per_block"] * (config["num_hidden_layers"] - L),
                "bf16_safetensors_bytes_exact": size,
                "fp32_weight_bytes": total * 4,
            })
    return {"teacher_dir": str(teacher_dir), "teacher": measured, "teacher_analytic": t,
            "block_breakdown": block_params(config), "layout_check": reproduce, "grid": rows}


def render_md(res: dict) -> str:
    t = res["teacher"]
    lines = [
        "| vocab | rows | layers | unique params | embedding | blocks | emb share | vs teacher | BF16 file bytes (exact) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res["grid"]:
        lines.append(f"| {r['vocab']} | {r['vocab_rows']:,} | {r['n_layers']} | "
                     f"{r['unique_physical_parameters']:,} | {r['embedding_parameters']:,} | "
                     f"{r['block_parameters']:,} | {r['embedding_share']:.3f} | "
                     f"{r['fraction_of_teacher']:.3f} | {r['bf16_safetensors_bytes_exact']:,} |")
    head = (f"Teacher unique physical parameters **{t['unique_physical_parameters']:,}** "
            f"(tied head counted once; a naive alias sum would say "
            f"{t['naive_alias_sum_if_tied_head_materialised']:,}).\n\n")
    return head + "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    a = ap.parse_args(argv)
    res = grid(a.teacher)
    if a.out_json:
        with open(a.out_json, "x", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, allow_nan=False)
    if a.out_md:
        with open(a.out_md, "x", encoding="utf-8", newline="\n") as f:
            f.write(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
