"""Evaluate explicit local transition-validity predictions (not free-form CoT)."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import read_jsonl, loads, write_json, dumps


def evaluate_process(references: Path, predictions: Path) -> dict:
    gold = {}
    for row in read_jsonl(references):
        if row["id"] in gold or type(row["valid"]) is not bool:
            raise ValueError("Invalid process references")
        gold[row["id"]] = row["valid"]
    seen = {}; invalid = 0
    for row in read_jsonl(predictions):
        if set(row) != {"id", "prediction"} or not isinstance(row["id"], str) or row["id"] not in gold or row["id"] in seen:
            raise ValueError("Unknown or duplicate process prediction")
        try:
            obj = loads(row["prediction"])
            if not isinstance(obj, dict) or set(obj) != {"valid"} or type(obj["valid"]) is not bool:
                raise ValueError("Invalid classification")
            seen[row["id"]] = obj["valid"]
        except (TypeError, ValueError, RecursionError):
            seen[row["id"]] = None; invalid += 1
    tp = sum(value is True and seen.get(key) is True for key, value in gold.items())
    tn = sum(value is False and seen.get(key) is False for key, value in gold.items())
    positives = sum(gold.values()); negatives = len(gold)-positives
    return {"rows": len(gold), "received": len(seen), "invalid_format": invalid,
            "accuracy_missing_as_wrong": (tp+tn)/len(gold) if gold else None,
            "valid_recall": tp/positives if positives else None, "invalid_recall": tn/negatives if negatives else None,
            "scope": "synthetic local transition classification; not general process/trace faithfulness"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True); p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Output exists")
    result = evaluate_process(a.references, a.predictions)
    write_json(a.output, result); print(dumps(result))


if __name__ == "__main__":
    main()
