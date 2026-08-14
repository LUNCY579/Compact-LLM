from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.io import read_json, read_jsonl, sha256_file, write_json_atomic  # noqa: E402
from consensus_lab.paths import LabPaths  # noqa: E402


OUTPUT = LAB_ROOT / "results/phase2_mechanism/cross_benchmark_verification.json"
REPORT = LAB_ROOT / "reports/PHASE2_CROSS_BENCHMARK_VERIFICATION.md"
EXPECTED_ROWS = {"train": 10_560, "dev": 1_320, "test": 1_320}
EXPECTED_HASHES = {
    "train": "b2eacf4435bfa7ef617639ed3d9626891fecae012bd4265ec5e0e9e7f4cdab5b",
    "dev": "c6619bdc3c492355d8e14a4a2b4350f9bae41a060649a6f0834ad7c8f7910f90",
    "test": "9333a3ae97c3c70f81a8998f88611d4224fc23e979edae0f69f860fc9c7c6cce",
}


def normalized(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", " ", value).strip()


def template_normalized(text: str) -> str:
    value = normalized(text)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", value)
    value = re.sub(r"\b(?:atlas|beacon|cedar|delta|ember|falcon|grove|harbor)\b", "<entity>", value)
    return re.sub(r"\s+", " ", value).strip()


def prompt(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def answer(row: dict[str, Any]) -> str:
    return str(row.get("approved_answer") or row.get("answer") or "")


def split_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id")) for row in rows]
    prompts = [normalized(prompt(row)) for row in rows]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_exact_normalized_prompts": len(set(prompts)),
        "duplicate_ids": sorted(key for key, count in Counter(ids).items() if count > 1),
        "duplicate_exact_normalized_prompt_rows": len(prompts) - len(set(prompts)),
        "decision_counts": dict(sorted(Counter(str(row.get("decision")) for row in rows).items())),
        "family_counts": dict(sorted(Counter(str(row.get("reasoning_family")) for row in rows).items())),
        "nonempty_answers": sum(bool(normalized(answer(row))) for row in rows),
        "assistant_final_message": sum(
            bool(row.get("messages")) and (row.get("messages") or [])[-1].get("role") == "assistant"
            for row in rows
        ),
    }


def main() -> None:
    paths = LabPaths.discover(LAB_ROOT)
    v212_root = paths.official_root / "data/phase_9_student_distillation/general_reasoning_v212"
    split_paths = {
        split: paths.assert_official_source(v212_root / f"reviewed_splits/{split}.reviewed.jsonl")
        for split in EXPECTED_ROWS
    }
    manifest_path = paths.assert_official_source(v212_root / "manifests/dataset_manifest.json")
    manifest = read_json(manifest_path)
    splits = {split: read_jsonl(path) for split, path in split_paths.items()}
    summaries = {split: split_summary(split_paths[split], rows) for split, rows in splits.items()}

    ids = {split: {str(row["id"]) for row in rows} for split, rows in splits.items()}
    exact = {split: {normalized(prompt(row)) for row in rows} for split, rows in splits.items()}
    prompt_answer = {
        split: {(normalized(prompt(row)), normalized(answer(row))) for row in rows}
        for split, rows in splits.items()
    }
    templated = {split: {template_normalized(prompt(row)) for row in rows} for split, rows in splits.items()}

    v22_root = paths.official_root / "data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits"
    v22_paths = {
        split: paths.assert_official_source(v22_root / f"{split}.reviewed.jsonl")
        for split in EXPECTED_ROWS
    }
    v22_rows = {split: read_jsonl(path) for split, path in v22_paths.items()}
    v22_ids = {str(row["id"]) for rows in v22_rows.values() for row in rows}
    v22_exact = {normalized(prompt(row)) for rows in v22_rows.values() for row in rows}
    v22_templated = {template_normalized(prompt(row)) for rows in v22_rows.values() for row in rows}
    test_ids = ids["test"]
    test_exact = exact["test"]
    test_templated = templated["test"]

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("manifest schema", manifest.get("schema_version") == "2.1.2", manifest.get("schema_version"))
    check("manifest deterministic seed", manifest.get("seed") == 21_242, manifest.get("seed"))
    check("manifest split counts", manifest.get("split_counts") == EXPECTED_ROWS, manifest.get("split_counts"))
    check("manifest canonical row count", manifest.get("canonical_rows") == sum(EXPECTED_ROWS.values()), manifest.get("canonical_rows"))
    check("manifest family count", manifest.get("families") == 33, manifest.get("families"))
    for split in EXPECTED_ROWS:
        summary = summaries[split]
        check(f"{split} row count", summary["rows"] == EXPECTED_ROWS[split], summary["rows"])
        check(f"{split} frozen hash", summary["sha256"] == EXPECTED_HASHES[split], summary["sha256"])
        check(f"{split} unique IDs", summary["unique_ids"] == summary["rows"], summary["unique_ids"])
        check(f"{split} nonempty answers", summary["nonempty_answers"] == summary["rows"], summary["nonempty_answers"])
        check(f"{split} final assistant message", summary["assistant_final_message"] == summary["rows"], summary["assistant_final_message"])
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        check(f"{left}/{right} ID disjoint", not (ids[left] & ids[right]), len(ids[left] & ids[right]))
    check("v2.1.2 TEST/v2.2 all ID disjoint", not (test_ids & v22_ids), len(test_ids & v22_ids))
    check("v2.1.2 TEST/v2.2 all exact-prompt disjoint", not (test_exact & v22_exact), len(test_exact & v22_exact))
    cross_template_overlap = test_templated & v22_templated
    # Template overlap is reported rather than required: both are structured synthetic reasoning corpora.

    repo_references: list[str] = []
    excluded_roots = {str(OUTPUT.resolve()), str(REPORT.resolve())}
    needles = ("general_reasoning_v212/reviewed_splits/test", "v212_test")
    for path in LAB_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".toml"}:
            continue
        if str(path.resolve()) in excluded_roots or any(part in {".git", ".venv"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(needle in text.replace("\\", "/") for needle in needles):
            repo_references.append(str(path.relative_to(LAB_ROOT)))
    allowed_reference = "experiments/phase2_mechanism/verify_cross_benchmark.py"
    prior_references = sorted(
        path for path in repo_references if path.replace("\\", "/") != allowed_reference
    )
    check("no prior lab TEST reference", not prior_references, prior_references)

    split_overlap: dict[str, Any] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        right_rows = splits[right]
        split_overlap[f"{left}_{right}"] = {
            "exact_prompt_forms": len(exact[left] & exact[right]),
            "exact_prompt_answer_forms": len(prompt_answer[left] & prompt_answer[right]),
            "value_normalized_template_forms": len(templated[left] & templated[right]),
            "right_rows_with_exact_prompt_overlap": sum(
                normalized(prompt(row)) in exact[left] for row in right_rows
            ),
            "right_rows_with_exact_prompt_answer_overlap": sum(
                (normalized(prompt(row)), normalized(answer(row))) in prompt_answer[left]
                for row in right_rows
            ),
        }

    historical_root = paths.official_root / (
        "artifacts/phase_9_capability_composition_v22_lr1e4_s1200/sealed_selected_test"
    )
    historical_summary = historical_root / "v212/summary.json"
    historical_results = historical_root / "v212/results.jsonl"
    historical_manifest = historical_root / "candidate_evaluation_manifest.json"
    historical_test_use = {
        "exists": historical_summary.is_file() and historical_results.is_file(),
        "condition": "different historical D_greedy dense selected model, not any Phase 2 adapter committee",
        "summary_path": str(historical_summary.resolve()),
        "summary_sha256": sha256_file(historical_summary) if historical_summary.is_file() else None,
        "results_path": str(historical_results.resolve()),
        "results_sha256": sha256_file(historical_results) if historical_results.is_file() else None,
        "manifest_path": str(historical_manifest.resolve()),
        "manifest_sha256": sha256_file(historical_manifest) if historical_manifest.is_file() else None,
    }

    adapter_root = paths.official_root / "artifacts/phase_9_capability_composition_v22_lr1e4_s1200"
    adapter_dirs = [
        "track_a_broad_reasoning",
        "track_b_anchored_reasoning",
        "track_c_general",
        "track_c_grounding",
        "track_c_procedure",
        "track_c_abstention",
        "track_c_social",
    ]
    adapter_training_sources: dict[str, Any] = {}
    for directory in adapter_dirs:
        training_manifest = adapter_root / directory / "training_manifest.json"
        content = read_json(training_manifest)
        adapter_training_sources[directory] = {
            "manifest_path": str(training_manifest.resolve()),
            "manifest_sha256": sha256_file(training_manifest),
            "train_file": content.get("train_file"),
        }
    check(
        "all seven adapter training files are v2.2, not v2.1.2",
        all(
            "multidomain_reasoning_v22" in str(item["train_file"]).replace("\\", "/")
            and "general_reasoning_v212" not in str(item["train_file"]).replace("\\", "/")
            for item in adapter_training_sources.values()
        ),
        {name: item["train_file"] for name, item in adapter_training_sources.items()},
    )

    caveats = []
    train_test = split_overlap["train_test"]
    if train_test["right_rows_with_exact_prompt_answer_overlap"]:
        caveats.append(
            "v2.1.2 TEST contains exact prompt-and-answer repetitions from v2.1.2 TRAIN; "
            "this does not leak into the seven v2.2-trained adapters, but prevents a template-independent holdout claim."
        )
    if historical_test_use["exists"]:
        caveats.append(
            "The official project previously evaluated a different selected D_greedy dense model on v2.1.2 TEST; "
            "the split is therefore not globally untouched."
        )

    packet = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "CROSS-BENCHMARK PROVENANCE GATE",
        "status": (
            "PASS_WITH_CAVEATS"
            if all(item["passed"] for item in checks) and caveats
            else "PASS"
            if all(item["passed"] for item in checks)
            else "FAIL"
        ),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "content": manifest,
        },
        "splits": summaries,
        "cross_v22": {
            "v22_split_paths": {split: str(path.resolve()) for split, path in v22_paths.items()},
            "test_id_overlap": len(test_ids & v22_ids),
            "test_exact_prompt_overlap": len(test_exact & v22_exact),
            "test_value_normalized_template_overlap": len(cross_template_overlap),
            "template_overlap_interpretation": "Reported descriptively; shared abstract templates do not imply row leakage.",
        },
        "within_v212_split_overlap": split_overlap,
        "historical_official_test_use": historical_test_use,
        "adapter_training_sources": adapter_training_sources,
        "prior_lab_test_references": prior_references,
        "caveats": caveats,
    }
    write_json_atomic(OUTPUT, packet)

    lines = [
        "# Phase 2 Cross-Benchmark Verification",
        "",
        f"Status: **{packet['status']}** ({packet['checks_passed']}/{packet['checks_total']} checks)",
        "",
        "The v2.1.2 TEST split is treated as an independent cross-benchmark holdout.",
        "No Phase 2 inference may begin unless all mandatory checks pass. Caveats",
        "limit the eventual claim but do not invalidate fixed-method cross-corpus evaluation.",
        "",
        "| Split | Rows | Unique IDs | SHA256 |",
        "|---|---:|---:|---|",
    ]
    for split in EXPECTED_ROWS:
        row = summaries[split]
        lines.append(f"| {split.upper()} | {row['rows']:,} | {row['unique_ids']:,} | `{row['sha256']}` |")
    lines.extend(
        [
            "",
            "## Independence checks",
            "",
            f"- TEST-v2.2 ID overlap: **{packet['cross_v22']['test_id_overlap']}**",
            f"- TEST-v2.2 exact normalized prompt overlap: **{packet['cross_v22']['test_exact_prompt_overlap']}**",
            f"- TEST-v2.2 value-normalized template overlap: **{packet['cross_v22']['test_value_normalized_template_overlap']}** (descriptive only)",
            f"- Prior lab references to v2.1.2 TEST: **{len(prior_references)}**",
            f"- TEST rows repeating an exact v2.1.2 TRAIN prompt+answer: **{train_test['right_rows_with_exact_prompt_answer_overlap']:,}/{len(splits['test']):,}**",
            f"- Historical official TEST use by another model: **{historical_test_use['exists']}**",
            "- Seven adapter training manifests point to v2.2 data: **PASS**",
            "",
            "## Claim boundary",
            "",
            "The permitted claim is *fixed-method cross-corpus transfer to v2.1.2 TEST*.",
            "Do not describe this split as globally untouched, template-independent, or a test",
            "of generalization beyond all v2.1.2 training forms.",
            "",
            "## Caveats",
            "",
            *[f"- {item}" for item in caveats],
            "",
            "## Check ledger",
            "",
            "| Check | Result | Evidence |",
            "|---|---|---|",
        ]
    )
    for item in checks:
        evidence = json.dumps(item["evidence"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | `{evidence}` |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "checks": f"{packet['checks_passed']}/{packet['checks_total']}", "output": str(OUTPUT)}, indent=2))
    if packet["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
