"""SMALLVD-001 Phase 0 -- pinned teacher lineage (gate G0).

Records, for the teacher directory given on the command line:
  * sha256 of EVERY file in the directory (weights, tokenizer, configs, provenance JSON)
  * the weights' DATA-SECTION sha256 (identity of record; whole-file sha can vary with
    `__metadata__` ordering) next to the whole-file sha, and any recorded hashes to compare against
  * config.json, generation_config.json, the chat template (text + sha256), tokenizer structure
    (model type, regular vocab, merges, every added token with id and special flag, normalizer,
    pre-tokenizer regex, decoder), embedding rows vs tokenizer ids (padding rows)
  * the safetensors tensor inventory and the exact parameter accounting (param_accounting)
  * git provenance of the directory inside its repository (tracked? last commit touching it?)

Pure python, no torch: G0 is about identity, and loading the model is G0b (baseline
reproduction), which is a separate, lease-holding step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import param_accounting as pa


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def data_section_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n)
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # provenance is best-effort metadata, never a gate by itself
        return f"ERROR: {e}"


def tokenizer_summary(d: Path) -> dict:
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    m = tj["model"]
    return {
        "model_type": m["type"], "regular_vocab": len(m["vocab"]), "merges": len(m["merges"]),
        "byte_fallback": m.get("byte_fallback"), "ignore_merges": m.get("ignore_merges"),
        "normalizer": tj.get("normalizer"), "pre_tokenizer": tj.get("pre_tokenizer"),
        "decoder": tj.get("decoder"), "post_processor": tj.get("post_processor"),
        "added_tokens": [{"id": a["id"], "content": a["content"], "special": a["special"]}
                         for a in tj["added_tokens"]],
        "tokenizer_class": tc.get("tokenizer_class"), "eos_token": tc.get("eos_token"),
        "pad_token": tc.get("pad_token"), "bos_token": tc.get("bos_token"),
        "split_special_tokens": tc.get("split_special_tokens"),
        "chat_template_sha256": hashlib.sha256(tc["chat_template"].encode("utf-8")).hexdigest(),
        "chat_template_chars": len(tc["chat_template"]),
        "max_tokenizer_id": max(max(m["vocab"].values()), max(a["id"] for a in tj["added_tokens"])),
    }


def lineage(teacher: Path, recorded: dict | None = None) -> dict:
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(teacher.iterdir()) if p.is_file()}
    cfg = json.loads((teacher / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((teacher / "generation_config.json").read_text(encoding="utf-8"))
    header, _ = pa.read_header(teacher / "model.safetensors")
    acc = pa.account_header(header, cfg)
    tok = tokenizer_summary(teacher)
    repo = teacher
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    rel = teacher.relative_to(repo).as_posix() if (repo / ".git").exists() else None
    out = {
        "schema": 1, "record": "SMALLVD_TEACHER_LINEAGE", "experiment": "SMALLVD-001",
        "teacher_dir": str(teacher), "files": files,
        "weights_data_section_sha256": data_section_sha256(teacher / "model.safetensors"),
        "config": cfg, "generation_config": gen, "tokenizer": tok,
        "embedding_rows": cfg["vocab_size"],
        "padding_rows_never_emitted": cfg["vocab_size"] - (tok["max_tokenizer_id"] + 1),
        "architecture": {k: cfg[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                             "num_attention_heads", "num_key_value_heads", "head_dim",
                                             "rope_theta", "rope_scaling", "rms_norm_eps", "hidden_act",
                                             "tie_word_embeddings", "max_position_embeddings")},
        "parameter_accounting": acc,
        "tensor_inventory": {"n_tensors": sum(1 for k in header if k != "__metadata__"),
                             "metadata": header.get("__metadata__"),
                             "lm_head_stored": "lm_head.weight" in header},
        "git": {"repo": str(repo) if rel else None, "path_in_repo": rel,
                "head": git(repo, "rev-parse", "HEAD") if rel else None,
                "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD") if rel else None,
                "tracked_files": git(repo, "ls-files", "--", rel).splitlines() if rel else None,
                "last_commit_touching": git(repo, "log", "-1", "--format=%H %cI %s", "--", rel) if rel else None},
        "quality_claim": False, "release_permitted": False,
    }
    if recorded:
        out["recorded_hash_comparison"] = {
            k: {"recorded": v, "measured": files.get(k, {}).get("sha256"),
                "match": files.get(k, {}).get("sha256") == v} for k, v in recorded.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--recorded", type=Path, help="JSON {filename: recorded sha256}")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rec = json.loads(a.recorded.read_text(encoding="utf-8")) if a.recorded else None
    res = lineage(a.teacher, rec)
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, allow_nan=False)
    bad = [k for k, v in (res.get("recorded_hash_comparison") or {}).items() if not v["match"]]
    print(json.dumps({"out": str(a.out), "weights_sha256": res["files"]["model.safetensors"]["sha256"],
                      "unique_params": res["parameter_accounting"]["unique_physical_parameters"],
                      "recorded_mismatches": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
