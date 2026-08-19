#!/usr/bin/env python3
"""Evaluate model predictions against canonical and accepted short answers."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import generate_dataset as g


def extract_final(text: str) -> str:
    text = text.strip()
    m = re.search(r"<final>(.*?)</final>", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if re.search(r"\bFinal\s*:", text, flags=re.I):
        return re.split(r"\bFinal\s*:\s*", text, flags=re.I)[-1].strip()
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True, help='JSONL rows with fields "id" and "prediction"')
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    gold = {}
    with args.gold.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line); gold[row["id"]] = row
    preds = {}
    duplicate_prediction_ids = []
    with args.predictions.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line); rid = row["id"]
            if rid in preds: duplicate_prediction_ids.append(rid)
            preds[rid] = str(row.get("prediction", row.get("output", "")))

    buckets = collections.defaultdict(lambda: [0, 0])
    correct = 0; total = len(gold); missing = []; extra = sorted(set(preds) - set(gold)); output_words = []
    errors = []
    for rid, row in gold.items():
        if rid not in preds:
            missing.append(rid); pred = ""
        else:
            pred = extract_final(preds[rid])
        output_words.append(len(pred.split()))
        norm = g.normalize_answer(pred)
        ok = any(norm == g.normalize_answer(a) for a in row["accepted_answers"])
        correct += int(ok)
        for key in [("family", row["family"]), ("difficulty", row["difficulty"]), ("answer_type", row["answer_type"])]:
            buckets[key][0] += int(ok); buckets[key][1] += 1
        if not ok and len(errors) < 100:
            errors.append({"id": rid, "family": row["family"], "gold": row["final_answer"], "prediction": pred})

    def bucket_report(kind):
        return {name: {"correct": c, "total": n, "accuracy": round(c/n, 6) if n else 0.0}
                for (k, name), (c, n) in sorted(buckets.items()) if k == kind}
    report = {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct, "total": total,
        "coverage": round((total-len(missing))/total, 6) if total else 0.0,
        "missing_prediction_count": len(missing), "extra_prediction_count": len(extra),
        "duplicate_prediction_id_count": len(duplicate_prediction_ids),
        "mean_extracted_output_words": round(sum(output_words)/len(output_words), 4) if output_words else 0.0,
        "by_family": bucket_report("family"),
        "by_difficulty": bucket_report("difficulty"),
        "by_answer_type": bucket_report("answer_type"),
        "sample_errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
