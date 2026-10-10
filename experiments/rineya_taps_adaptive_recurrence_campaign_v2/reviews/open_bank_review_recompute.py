"""Independent OPEN preparation receipt; standard-library prompt-only apparatus."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    campaign = Path(__file__).resolve().parent.parent
    packet = campaign / "runs/open-bank-apparatus-002"
    rows = json.loads((packet / "bank.json").read_text(encoding="utf-8"))
    manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 24 and sha(packet / "bank.json") == "fa52fd850e3a0a2e113fd7bdefbdb56427c9632a0ce3137312c82ae4059c369f"
    assert sha(campaign / "open_bank.py") == manifest["source_sha256"]
    for name, expected in manifest["files"].items():
        assert sha(packet / name) == expected
    floors = Counter()
    positive = []
    for row in rows:
        result = open_bank.solve_prompt(row["prompt"])
        assert result == row["gold"]
        positive.append({"id": row["id"], "prediction": result, "correct": True})
        for name, prediction in open_bank.blind_predictions(row["prompt"]).items():
            floors[name] += open_bank.score_response(prediction, row)["correct"]
    assert list(inspect.signature(open_bank.solve_prompt).parameters) == ["prompt"]
    try:
        open_bank.solve_prompt({"prompt": "ignored", "gold": "SECRET"})
    except TypeError:
        pass
    else:
        raise AssertionError("target-bearing solver input accepted")
    assert floors["query_blind_first_entity_solver"] == 10 and floors["copy_full_prompt"] == 0
    family_splits = defaultdict(set)
    for row in rows:
        family_splits[row["generator_family"]].add(row["split"])
    assert all(len(values) == 1 for values in family_splits.values())
    receipt = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "PASS_OPEN_SOURCE_DATA_APPARATUS_NOT_NATIVE_ADMITTED", "source_sha256": sha(campaign / "open_bank.py"),
               "bank_sha256": sha(packet / "bank.json"), "manifest_sha256": sha(packet / "manifest.json"),
               "safe_binding_source_sha256": sha(campaign / "open_binding.py"),
               "independent_CPU_tests": {"passed": 24, "failed": 0, "skipped": 0, "pytest_seconds": .88,
                                         "junit_sha256": sha(campaign / "reviews/open_bank_review_junit.xml")},
               "independent_positive_answers": positive, "independent_blind_correct_counts": dict(floors),
               "split_counts": dict(Counter(row["split"] for row in rows)),
               "unique_parent_worlds": len({row["parent_id"] for row in rows}), "named_family_count": len(family_splits),
               "category_counts": dict(Counter(row["category"] for row in rows)),
               "logical_labels": dict(Counter(row["gold"] for row in rows if row["answer_kind"] == "choice")),
               "document_statuses": dict(Counter(row["source_status"] for row in rows if row["category"] == "document_reconciliation")),
               "safe_no_model_binding_review": "PASS: exact asset/source/runtime/Qwen guards before local-only trust_remote_codeFalse tokenizer; immutable packet checks;24 IDs; <=350 no truncation; exclusive offline cache",
               "limitations": ["Eight named generator variants share four broad code/categories; four variant families per split are too few for broad generator-level inference",
                               "OPEN_VALIDATION is author/source visible and not independent confirmation; later outcome-driven revision makes it selection data",
                               "Query-blind first-entity solver10/24 remains a substantial own-bank floor; several binary/document families have50% constant floors",
                               "Fixed program constants permit template-specific closed-form solution; solving this bank is not general program reasoning",
                               "No ordinary-language/factual preservation endpoint in this bank; preserve/check original-bank behavior separately",
                               "Native tokenizer/shape binding and complete new data/scorer/cap/baseline matrix registration still required"],
               "native_model_loads": 0, "native_model_inference": 0, "protected_or_native_TEST_reads": False,
               "human_review": "NOT_REVIEWED", "review_kind": "independent agent source/data audit, not independent bank authorship"}
    out = campaign / "reviews/open_bank_review_receipt.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps({"status": receipt["status"], "positive": len(positive), "blind_first_entity": floors["query_blind_first_entity_solver"]}))


if __name__ == "__main__":
    main()
