#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_.$%+\-]+", text.lower()))


def latest_user(row: dict[str, Any]) -> str:
    users = [m["content"] for m in row.get("messages", []) if m.get("role") == "user"]
    return norm(users[-1]) if users else ""


def prompt_norm(row: dict[str, Any]) -> str:
    return norm("\n".join(f"{m['role']}:{m['content']}" for m in row["messages"][:-1]))


def qwen_prompt(row: dict[str, Any]) -> str:
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        for m in row["messages"][:-1]
    ) + "<|im_start|>assistant\n"


def proxy_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\w\s]", text))


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural and linkage audit for Rineya v2.0.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--eval-root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()

    files = {
        "reasoning": root / "data/sft/rineya_v2_reasoning_train.jsonl",
        "communication": root / "data/sft/rineya_v2_natural_communication_train.jsonl",
        "integrated": root / "data/sft/rineya_v2_integrated_reasoning_dialogue_train.jsonl",
        "new": root / "data/sft/rineya_v2_new_all_train.jsonl",
        "preservation": root / "data/sft/rineya_v1_preservation_slice_8000.jsonl",
        "combined": root / "data/sft/rineya_v2_recommended_combined_40000.jsonl",
        "dpo": root / "data/dpo/rineya_v2_preferences_train_12000.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in files.items()}
    expected = {"reasoning": 16000, "communication": 9600, "integrated": 6400,
                "new": 32000, "preservation": 8000, "combined": 40000, "dpo": 12000}
    errors: list[str] = []
    for name, count in expected.items():
        if len(rows[name]) != count:
            errors.append(f"{name}: expected {count}, got {len(rows[name])}")

    control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    bad_singular = re.compile(r"\b1\s+(hours|minutes|seconds|meters|kilometers|kilograms|liters|cups|cells|stops)\b", re.I)
    bad_plural = re.compile(r"\b(?:0|[2-9]|[1-9]\d+)\s+(hour|minute|second|meter|kilometer|kilogram|liter|cup|cell|stop)\b", re.I)
    sft_report: dict[str, Any] = {}
    for name in ("reasoning", "communication", "integrated", "new", "preservation", "combined"):
        ids: set[str] = set()
        prompts: set[str] = set()
        latest: set[str] = set()
        counters = Counter()
        for line_no, row in enumerate(rows[name], 1):
            rid = row.get("id")
            if not isinstance(rid, str) or not rid:
                errors.append(f"{name}:{line_no}: missing id")
            elif rid in ids:
                errors.append(f"{name}:{line_no}: duplicate id {rid}")
            ids.add(rid)
            messages = row.get("messages", [])
            roles = [m.get("role") for m in messages]
            expected_roles = ["system"] + [v for _ in range((len(roles) - 1) // 2) for v in ("user", "assistant")]
            if roles != expected_roles:
                counters["role_order_errors"] += 1
            if not roles or roles[-1] != "assistant":
                counters["final_role_errors"] += 1
            if row.get("metadata", {}).get("split") != "train":
                counters["split_errors"] += 1
            if "question" in row:
                counters["top_level_question_fields"] += 1
            target = messages[-1].get("content", "") if messages else ""
            if "<think>" in target.lower() or "</think>" in target.lower():
                counters["hidden_reasoning_targets"] += 1
            text = "\n".join(str(m.get("content", "")) for m in messages)
            if control_re.search(text) or "�" in text:
                counters["control_or_replacement_character_issues"] += 1
            if bad_singular.search(text) or bad_plural.search(text):
                counters["numeric_singular_plural_issues"] += 1
            p = prompt_norm(row)
            if p in prompts:
                counters["duplicate_rendered_prompts"] += 1
            prompts.add(p)
            u = latest_user(row)
            if u in latest:
                counters["duplicate_latest_user_rows"] += 1
                if name in {"reasoning", "communication", "integrated", "new"}:
                    counters["unexpected_duplicate_latest_user_rows"] += 1
            latest.add(u)
        hard_fields = [k for k in counters if k != "duplicate_latest_user_rows"]
        if any(counters[k] for k in hard_fields):
            errors.append(f"{name}: contract counters {dict(counters)}")
        sft_report[name] = {
            "rows": len(rows[name]), "unique_ids": len(ids),
            "unique_rendered_prompts": len(prompts), "unique_latest_user_prompts": len(latest),
            **dict(counters),
        }

    component = {row["id"]: row for key in ("reasoning", "communication", "integrated") for row in rows[key]}
    new = {row["id"]: row for row in rows["new"]}
    preservation = {row["id"]: row for row in rows["preservation"]}
    combined = {row["id"]: row for row in rows["combined"]}
    if len(component) != 32000 or set(component) != set(new):
        errors.append("component/new union mismatch")
    elif any(component[rid] != new[rid] for rid in component):
        errors.append("component/new row-content mismatch")
    if set(new) & set(preservation):
        errors.append("new and preservation IDs overlap")
    if set(combined) != set(new) | set(preservation):
        errors.append("combined does not equal new plus preservation")

    class_counts = Counter(row["metadata"]["reply_class"] for row in rows["new"])
    group_counts = Counter(row["metadata"]["capability_group"] for row in rows["new"])
    if len(class_counts) != 40 or set(class_counts.values()) != {800}:
        errors.append("new SFT is not 40 classes x 800")
    if group_counts != Counter({"reasoning": 16000, "communication": 9600, "integrated": 6400}):
        errors.append(f"unexpected new group counts: {dict(group_counts)}")

    dpo_ids: set[str] = set()
    dpo_prompts: set[str] = set()
    dpo_classes = Counter()
    dpo_counters = Counter()
    for line_no, row in enumerate(rows["dpo"], 1):
        rid = row.get("id")
        if rid in dpo_ids:
            dpo_counters["duplicate_ids"] += 1
        dpo_ids.add(rid)
        prompt = row.get("prompt", "")
        if prompt in dpo_prompts:
            dpo_counters["duplicate_prompts"] += 1
        dpo_prompts.add(prompt)
        if not prompt.endswith("<|im_start|>assistant\n"):
            dpo_counters["bad_assistant_prefix"] += 1
        if row.get("chosen", "").strip() == row.get("rejected", "").strip():
            dpo_counters["chosen_equals_rejected"] += 1
        source_id = row.get("metadata", {}).get("source_record_id")
        source = new.get(source_id)
        if source is None or row.get("chosen") != source["messages"][-1]["content"]:
            dpo_counters["chosen_source_mismatches"] += 1
        if source is None or prompt != qwen_prompt(source):
            dpo_counters["prompt_source_mismatches"] += 1
        text = prompt + row.get("chosen", "") + row.get("rejected", "")
        if control_re.search(text) or "�" in text:
            dpo_counters["control_or_replacement_character_issues"] += 1
        if bad_singular.search(text) or bad_plural.search(text):
            dpo_counters["numeric_singular_plural_issues"] += 1
        dpo_classes[row.get("metadata", {}).get("preference_dimension", "missing")] += 1
    if any(dpo_counters.values()):
        errors.append(f"DPO contract counters {dict(dpo_counters)}")
    if len(dpo_classes) != 40 or set(dpo_classes.values()) != {300}:
        errors.append("DPO is not 40 classes x 300")

    eval_files: list[str] = []
    eval_texts: set[str] = set()
    if args.eval_root:
        for path in sorted(args.eval_root.rglob("*.jsonl")):
            eval_files.append(str(path.resolve()))
            for row in read_jsonl(path):
                if isinstance(row.get("messages"), list):
                    users = [m.get("content", "") for m in row["messages"] if m.get("role") == "user"]
                    if users:
                        eval_texts.add(norm(users[-1]))
                for key in ("prompt", "question", "input", "user", "query", "instruction"):
                    value = row.get(key)
                    if isinstance(value, str) and norm(value):
                        eval_texts.add(norm(value))
    eval_overlap = sum(latest_user(row) in eval_texts for row in rows["new"]) if eval_texts else None
    if eval_overlap:
        errors.append(f"exact latest-user eval overlap: {eval_overlap}")

    lengths = sorted(proxy_tokens(row["messages"][-1]["content"]) for row in rows["new"])
    multi_turn = sum(sum(m["role"] == "user" for m in row["messages"]) > 1 for row in rows["new"])
    report = {
        "status": "PASS" if not errors else "FAIL",
        "row_counts": {name: len(value) for name, value in rows.items()},
        "sft_contract": sft_report,
        "union_integrity": {
            "component_ids": len(component), "new_ids": len(new),
            "preservation_ids": len(preservation), "combined_ids": len(combined),
            "new_preservation_id_overlap": len(set(new) & set(preservation)),
        },
        "new_class_distribution": dict(sorted(class_counts.items())),
        "new_group_distribution": dict(sorted(group_counts.items())),
        "new_multi_turn_rows": multi_turn,
        "new_target_proxy_tokens": {
            "min": lengths[0], "median": lengths[len(lengths)//2],
            "p90": lengths[(9*len(lengths)+9)//10-1], "max": lengths[-1],
        },
        "dpo": {
            "rows": len(rows["dpo"]), "unique_ids": len(dpo_ids), "unique_prompts": len(dpo_prompts),
            "class_distribution": dict(sorted(dpo_classes.items())), **dict(dpo_counters),
        },
        "eval_snapshot": {
            "root": str(args.eval_root.resolve()) if args.eval_root else None,
            "files": eval_files, "normalized_texts": len(eval_texts),
            "exact_latest_user_overlap": eval_overlap,
        },
        "errors": errors,
        "limitations": [
            "Automated checks do not establish that every open-ended social target is uniquely optimal.",
            "Human review remains required for safety, empathy, de-escalation, roleplay, informal language, and DPO preference quality.",
            "Re-run exact overlap against the repository revision actually used for training.",
        ],
    }
    output = args.output or root / "reports/full_quality_audit_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
