from __future__ import annotations

"""Build category-neutral Evaluator-v4.2 benchmark contracts.

The ``category`` field is preserved only as reporting metadata.  This migration
never derives a requirement, scoring weight, question rule, or conversational
mode from category.  Existing v4.1 requirements are retained only when they
encode an objectively checkable fact, safety boundary, context fact, explicit
system constraint, or scope statement visible in the individual case.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# These v4.1 items were generated from broad category expectations rather than
# an objective fact or explicit per-case boundary.  v4.2 removes them.
REMOVE_LABELS = {
    "clarification_request_missing",
    "required_clarification_question_missing",
    "required_follow_up_question_missing",
    "required_question_missing",
    "forbidden_question_present",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def migrate_contract(record: dict[str, Any]) -> dict[str, Any]:
    old = dict(record.get("response_contract") or {})
    requirements = [
        dict(item) for item in old.get("requirements") or []
        if str(item.get("label")) not in REMOVE_LABELS
    ]
    prohibitions = [
        dict(item) for item in old.get("prohibitions") or []
        if str(item.get("label")) not in REMOVE_LABELS
    ]
    retained_ids = {str(item.get("id")) for item in requirements + prohibitions}
    domains = {
        str(domain): [str(identifier) for identifier in identifiers or [] if str(identifier) in retained_ids]
        for domain, identifiers in (old.get("domains") or {}).items()
    }
    domains = {domain: identifiers for domain, identifiers in domains.items() if identifiers}
    strict = bool(requirements or prohibitions or old.get("user_echo_forbidden"))
    contract: dict[str, Any] = {
        "schema_version": "4.2",
        "open_ended": not strict,
        "min_tokens": int(old.get("min_tokens", 1)),
        "max_tokens": int(old.get("max_tokens", 32)),
        "expected_language": old.get("expected_language", "en"),
        "requirements": requirements,
        "prohibitions": prohibitions,
        "recommended": [],
        "domains": domains,
        "category_is_reporting_only": True,
        "contract_authoring_basis": (
            "Human-reviewed case-level prompt, conversation context, or system constraint. "
            "No requirement or scoring rule is inferred from category or reference wording."
        ),
    }
    if old.get("user_echo_forbidden"):
        contract["user_echo_forbidden"] = True
    return contract


def migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    output["response_contract"] = migrate_contract(record)
    metadata = dict(output.get("metadata") or {})
    metadata["category_usage"] = "reporting_only"
    output["metadata"] = metadata
    return output


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compatibility name used by probe builders; never infers from category."""

    return migrate_record(record)


def manifest(source: Path, destination: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({str(row.get("category", "uncategorized")) for row in rows})
    strict = sum(1 for row in rows if not bool((row.get("response_contract") or {}).get("open_ended", True)))
    labels = sorted({
        str(item.get("label"))
        for row in rows
        for key in ("requirements", "prohibitions")
        for item in (row.get("response_contract") or {}).get(key) or []
    })
    return {
        "schema_version": "4.2",
        "source": str(source),
        "destination": str(destination),
        "case_count": len(rows),
        "category_count": len(categories),
        "categories_are_reporting_only": True,
        "strict_case_count": strict,
        "open_case_count": len(rows) - strict,
        "retained_objective_contract_labels": labels,
        "removed_category_derived_labels": sorted(REMOVE_LABELS),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build category-neutral v4.2 benchmark contracts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    rows = [migrate_record(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    out_manifest = args.manifest or args.output.with_name(args.output.stem + "_manifest.json")
    out_manifest.write_text(json.dumps(manifest(args.input, args.output, rows), indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} category-neutral cases to {args.output}")
    print(out_manifest)


if __name__ == "__main__":
    main()
