"""Prospective exclusive OPEN corpus export and tokenizer binding; no weights."""
import argparse
import hashlib
import json
import os
import platform
import sys
from importlib import metadata, util
from pathlib import Path
from datetime import datetime, timezone
from .fresh_scope import V3, SOURCE, INTERPRETER
from .short_response_panel import build_panel, solve_prompt

OUT = V3 / "data/fresh_open_replication_001"
RAW_KEYS = {"case_id", "raw_prompt", "raw_prompt_sha256"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def tokenizer_identity_before_import():
    proof_path = V3 / "probe_admission_v3.json"
    proof_sha = "b0b08bb354ce7bc48c760f59a87a7e7d8b8ba432aedddee3e05b55a0f0221131"
    if sha(proof_path) != proof_sha or sys.executable != INTERPRETER:
        raise PermissionError("qualified asset manifest/canonical interpreter changed")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof["source"] != str(SOURCE) or proof["interpreter"] != INTERPRETER or len(proof["checkpoint_files"]) != 10:
        raise PermissionError("qualified canonical ten-asset binding differs")
    measured_assets = {name: sha(SOURCE / name) for name in proof["checkpoint_files"]}
    if measured_assets != proof["checkpoint_files"]:
        raise PermissionError("canonical assets changed before tokenizer import")
    runtime = {"python": platform.python_version(), "torch": metadata.version("torch"),
               "transformers": metadata.version("transformers"), "tokenizers": metadata.version("tokenizers")}
    if any(runtime[k] != v for k, v in {"python": "3.11.15", "torch": "2.12.1+cu126", "transformers": "5.14.1"}.items()):
        raise PermissionError("qualified borrowed tokenizer runtime changed")
    package = Path(INTERPRETER).parents[1] / "Lib/site-packages/transformers"
    spec = util.find_spec("transformers")
    if spec is None or Path(spec.origin).resolve() != (package / "__init__.py").resolve():
        raise PermissionError("borrowed package import path differs")
    qwen_sha = sha(package / "models/qwen3/modeling_qwen3.py")
    if qwen_sha != "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab":
        raise PermissionError("qualified Qwen implementation differs")
    return {"qualified_manifest_sha256": proof_sha, "checkpoint_files": measured_assets,
            "runtime": runtime, "qwen_implementation_sha256": qwen_sha,
            "interpreter": sys.executable, "canonical_source": str(SOURCE)}


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def prompt_projection(rows):
    result = []
    for row in rows:
        prompt = row["prompt"]
        if type(prompt) is not str or not prompt:
            raise ValueError("nonempty prompt string required")
        result.append({"case_id": row["case_id"], "raw_prompt": prompt,
                       "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    if len({r["case_id"] for r in result}) != len(result):
        raise ValueError("duplicate prompt IDs")
    return result


def token_projection(raw_rows, tokenizer):
    result = []
    for row in raw_rows:
        if set(row) != RAW_KEYS:
            raise PermissionError("prompt-only input schema required")
        if (type(row["case_id"]) is not str or not row["case_id"] or type(row["raw_prompt"]) is not str
                or hashlib.sha256(row["raw_prompt"].encode("utf-8")).hexdigest() != row["raw_prompt_sha256"]):
            raise PermissionError("actual original prompt hash and opaque string ID required")
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["raw_prompt"]}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered)["input_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 350 or any(type(i) is not int or not 0 <= i < 151936 for i in ids):
            raise ValueError("complete rendered input exceeds fixed350/357 envelope; no truncation")
        result.append({**row, "rendered_prompt": rendered,
                       "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                       "input_token_ids": ids, "input_tokens": len(ids)})
    return result


def build(out, source_review, primary_freeze):
    from .fresh_open_bank import generate_registered_bank, audit_rows
    out = out.resolve()
    if out != OUT.resolve() or out.exists():
        raise PermissionError("exclusive exact fresh corpus directory required")
    # No scientific nonce generation occurs until both exact producer gates pass.
    rows = generate_registered_bank(source_review, primary_freeze)
    out.mkdir(parents=True, exist_ok=False)
    panel = build_panel()
    audit = audit_rows(rows)
    if len(panel) != 24 or any(solve_prompt(r["prompt"]) != r["gold"] for r in panel):
        raise ValueError("complete independent panel copy control differs")
    write(out / "SCIENTIFIC_LABELS.json", rows)
    write(out / "PRESERVATION_LABELS.json", panel)
    write(out / "CPU_AUDIT.json", audit)
    manifest = {"status": "CPU_FRESH_CORPUS_PREPARATION_INVALID", "native_model_loads": 0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_review_sha256": sha(source_review), "selected_freeze_sha256": sha(primary_freeze),
                "scientific_cases": len(rows), "preservation_cases": len(panel),
                "scorer_source_sha256": sha(V3 / "fresh_open_bank.py"),
                "panel_source_sha256": sha(V3 / "short_response_panel.py"),
                "provenance_limit": "New parents/generator code; shared primitives and same LLM author/verifier disclosed. Not independent human/source confirmation.",
                "copy_panel_limit": "Visible copying targets by design, perfect copy floor; no reasoning/general-language claim."}
    if audit["errors"]:
        write(out / "CORPUS_MANIFEST.json", manifest)
        raise ValueError("fixed-nonce CPU audit failed; preserve full corpus, no reseed")
    manifest["status"] = "CPU_FRESH_CORPUS_VERIFIED_TOKEN_BINDING_PENDING"
    manifest["files"] = {name: sha(out / name) for name in ("SCIENTIFIC_LABELS.json", "PRESERVATION_LABELS.json", "CPU_AUDIT.json")}
    write(out / "CORPUS_MANIFEST.json", manifest)
    return manifest


def bind(out):
    out = out.resolve()
    if out != OUT.resolve() or not out.is_dir() or (out / "model_inputs").exists():
        raise PermissionError("exact complete corpus/new token directory required")
    corpus = json.loads((out / "CORPUS_MANIFEST.json").read_text(encoding="utf-8"))
    if corpus["status"] != "CPU_FRESH_CORPUS_VERIFIED_TOKEN_BINDING_PENDING" or any(sha(out / name) != h for name, h in corpus["files"].items()):
        raise PermissionError("complete fixed corpus integrity required")
    dest = out / "model_inputs"
    identity = tokenizer_identity_before_import()
    dest.mkdir(exist_ok=False)
    cache = out / "tokenizer_cache"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(cache),
                      HF_HUB_CACHE=str(cache / "hub"), HF_MODULES_CACHE=str(cache / "modules"),
                      TOKENIZERS_PARALLELISM="false")
    # Tokenizer only; neither AutoModel nor canonical custom model is imported.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
    if len(tokenizer) != 151669:
        raise PermissionError("qualified tokenizer vocabulary differs")
    shape_audit = []
    for tag, expected_count in (("SCIENTIFIC", 432), ("PRESERVATION", 24)):
        labels = json.loads((out / (tag + "_LABELS.json")).read_text(encoding="utf-8"))
        if len(labels) != expected_count:
            raise ValueError("complete case count required")
        raw = prompt_projection(labels)
        tokens = token_projection(raw, tokenizer)
        for label, row in zip(labels, tokens, strict=True):
            target_ids = tokenizer.encode(label["gold"], add_special_tokens=False)
            if not 1 <= len(target_ids) + 1 <= 8:
                raise ValueError("full gold plus EOS exceeds frozen8 output cap")
            shape_audit.append({"case_id": row["case_id"], "input_tokens": row["input_tokens"],
                                "gold_plus_EOS_tokens": len(target_ids) + 1, "model_loads": 0})
        write(dest / (tag + "_PROMPTS.json"), raw)
        write(dest / (tag + "_TOKEN_BINDING.json"), {"status": "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING",
              "canonical_source": str(SOURCE), "model_loads": 0, "rows": tokens})
    files = {name: sha(dest / name) for name in ("SCIENTIFIC_PROMPTS.json", "SCIENTIFIC_TOKEN_BINDING.json",
              "PRESERVATION_PROMPTS.json", "PRESERVATION_TOKEN_BINDING.json")}
    identity_after = tokenizer_identity_before_import()
    if identity_after != identity:
        raise PermissionError("canonical asset/borrowed runtime/source identity changed during tokenization")
    manifest = {"status": "FROZEN_FRESH_OPEN_MODEL_EXPORT", "scientific_cases": 432, "preservation_cases": 24,
                "maximum_input_tokens": 350, "maximum_forward_prefix_tokens": 357, "labels_excluded": True,
                "tokenizer_source": str(SOURCE), "model_loads": 0, "files": files,
                "tokenizer_identity": {**identity, "class": type(tokenizer).__name__, "vocabulary": len(tokenizer)},
                "tokenizer_identity_after": identity_after,
                "checkpoint_files_after": identity_after["checkpoint_files"]}
    if manifest["checkpoint_files_after"] != identity["checkpoint_files"]:
        raise PermissionError("canonical assets changed during tokenizer binding")
    write(dest / "MODEL_EXPORT_MANIFEST.json", manifest)
    write(out / "TOKEN_SHAPE_AUDIT.json", shape_audit)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "bind"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--primary-freeze", type=Path)
    args = parser.parse_args()
    if args.stage == "build" and (args.source_review is None or args.primary_freeze is None):
        parser.error("build requires exact source review and primary freeze")
    result = build(args.out, args.source_review, args.primary_freeze) if args.stage == "build" else bind(args.out)
    print(json.dumps({"status": result["status"], "model_loads": 0}))
