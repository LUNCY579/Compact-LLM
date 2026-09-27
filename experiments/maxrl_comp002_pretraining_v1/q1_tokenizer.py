"""Audit actual local Qwen3 rendering and prompt lengths without a model load.

Only policy messages are fed to the tokenizer. TRAIN is restricted to the two
atomic donor manifests. CALIBRATION, DEV, guard_dev and replay_train are exposed
planning banks; TEST/CHALLENGE and guard_test/challenge stay unopened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics

from transformers import AutoTokenizer

EXPECTED = {"train_atomic_R_S": 6000, "calibration": 1250,
            "dev": 4150, "guard_dev": 1200, "replay_train": 1600}
ROLE_FILE = {"train_atomic_R_S": "policy/train.jsonl", "calibration": "policy/calibration.jsonl",
             "dev": "policy/dev.jsonl", "guard_dev": "policy/guard_dev.jsonl",
             "replay_train": "policy/replay_train.jsonl"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    at = fraction * (len(ordered) - 1)
    lo = int(at)
    return ordered[lo] + (at - lo) * (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--asset-receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--response-budget", type=int, default=128)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("receipt exists (create-only)")
    data = args.data.resolve(strict=True)
    model = args.model.resolve(strict=True)
    asset = json.loads(args.asset_receipt.read_text(encoding="utf-8"))
    if asset["model_path"] != str(model) or asset["stage"] != "Q1_ASSET_CANDIDATE":
        raise SystemExit("asset candidate receipt mismatch")
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt"):
        if sha256(model / name) != asset["assets"][name]["sha256"]:
            raise SystemExit(f"asset drift: {name}")
    if not (0 < args.response_budget < 40960):
        raise SystemExit("invalid response budget")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    atom_ids = set()
    for skill in ("R", "S"):
        atom_ids.update(json.loads((data / "manifests" / f"atomic_{skill}.json").read_text(encoding="utf-8"))["ids"])
    if len(atom_ids) != 6000:
        raise SystemExit("atomic donor ID scope mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != asset["tokenizer_template_sha256"]:
        raise SystemExit("loaded template differs from asset binding")
    tok_ids = {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
               "pad": tokenizer.pad_token_id, "im_start": tokenizer.convert_tokens_to_ids("<|im_start|>"),
               "im_end": tokenizer.convert_tokens_to_ids("<|im_end|>"),
               "think": tokenizer.convert_tokens_to_ids("<think>"),
               "end_think": tokenizer.convert_tokens_to_ids("</think>")}
    if tok_ids["eos"] != 151645 or tok_ids["pad"] != 151643 or tok_ids["im_start"] != 151644:
        raise SystemExit(f"unexpected tokenizer IDs: {tok_ids}")
    if tok_ids["think"] == tokenizer.unk_token_id or tok_ids["end_think"] == tokenizer.unk_token_id:
        raise SystemExit("thinking marker token missing")
    out_roles = {}
    for role, relative in ROLE_FILE.items():
        path = data / relative
        expected = manifest["files"][relative]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"policy bank drift: {relative}")
        lengths, rendered_hashes, seen = [], [], set()
        longest = None
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if role == "train_atomic_R_S" and row["id"] not in atom_ids:
                    continue
                if row["id"] in seen or len(row["messages"]) != 2 or [m["role"] for m in row["messages"]] != ["system", "user"]:
                    raise SystemExit(f"bad policy row {row['id']}")
                seen.add(row["id"])
                rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False)
                if ids != tokenizer(rendered, add_special_tokens=False)["input_ids"]:
                    raise SystemExit(f"render/encode mismatch: {row['id']}")
                counts = {key: ids.count(tok_ids[key]) for key in ("im_start", "im_end", "think", "end_think")}
                if counts != {"im_start": 3, "im_end": 2, "think": 1, "end_think": 1}:
                    raise SystemExit(f"unexpected chat delimiters: {row['id']} {counts}")
                if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
                    raise SystemExit(f"unexpected disabled-thinking prefix: {row['id']}")
                if len(ids) + args.response_budget > asset["config_max_position_embeddings"]:
                    raise SystemExit(f"context overflow: {row['id']} length={len(ids)}")
                lengths.append(len(ids))
                rendered_hashes.append((row["id"], hashlib.sha256(rendered.encode()).hexdigest()))
                if longest is None or len(ids) > longest["tokens"]:
                    longest = {"id": row["id"], "tokens": len(ids),
                               "render_sha256": rendered_hashes[-1][1]}
        if len(lengths) != EXPECTED[role]:
            raise SystemExit(f"row count mismatch: {role} {len(lengths)}")
        digest = hashlib.sha256(json.dumps(sorted(rendered_hashes), separators=(",", ":")).encode()).hexdigest()
        out_roles[role] = {"rows": len(lengths), "policy_sha256": expected["sha256"],
                           "prompt_render_digest": digest, "tokens_min": min(lengths),
                           "tokens_median": statistics.median(lengths), "tokens_p95": percentile(lengths, 0.95),
                           "tokens_max": max(lengths), "longest": longest,
                           "context_headroom_at_max": asset["config_max_position_embeddings"] - max(lengths) - args.response_budget}
        print(f"{role}: n={len(lengths)} p95={out_roles[role]['tokens_p95']:.1f} max={max(lengths)}", flush=True)
    receipt = {"experiment": "COMP-RS-001", "stage": "Q1_TOKENIZER_PROMPT_LENGTHS",
               "asset_receipt_sha256": sha256(args.asset_receipt),
               "tokenizer_class": type(tokenizer).__name__, "tokenizer_ids": tok_ids,
               "tokenizer_model_max_length": tokenizer.model_max_length,
               "generation_eos_ids": asset["generation_token_ids"]["eos_token_id"],
               "chat_template_sha256": asset["tokenizer_template_sha256"],
               "response_budget_candidate": args.response_budget,
               "roles": out_roles, "runtime_python": platform.python_version(),
               "native_model_loads": 0, "native_qualified": False, "training_authorized": False,
               "note": "Tokenizer-only development scope. No model generation, gradient, optimizer, test or challenge read."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"stage": receipt["stage"], "max_prompt": max(v["tokens_max"] for v in out_roles.values()),
                      "native_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
