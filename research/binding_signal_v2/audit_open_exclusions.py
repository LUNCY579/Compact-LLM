"""Read-only candidate contact audit against exactly three explicitly OPEN banks.

Reuses the pinned pilot's text_contact.py without changing it. The filtering
helper's clean-row return is deliberately discarded: collisions block the audit,
and no candidate row is rewritten, removed, or promoted to training permission.

CLI: python audit_open_exclusions.py [--candidate DIR] [--out NEW_JSON_PATH]
Omit --out to print the report without creating a file. The intended saved path
is candidate-v2/OPEN_EXCLUSION_AUDIT.json; outputs are always create-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import unicodedata


BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
PILOT = BASE / "ChatBubble-retention-pilot-20260917"
DEFAULT_CANDIDATE = RECORDS / "binding-signal-prep-20260920/candidate-v2"
TEXT_CONTACT_PATH = PILOT / "experiments/reasoning_posttraining_v1/text_contact.py"
OPEN_BANK_PATHS = {
    "process-transition-probe-001": RECORDS / "process-transition-probe-001/CANDIDATE_VIEWS.jsonl",
    "select-views-001": RECORDS / "select-views-001/SELECT_VIEWS.jsonl",
    "arithmetic-monitor-open": PILOT / "experiments/architecture_benchmark_001/data/monitor_probe.jsonl",
}
EXPECTED_BANK_ROWS = {"process-transition-probe-001": 384, "select-views-001": 384, "arithmetic-monitor-open": 150}
EXPECTED_BANK_SHA256 = {
    "process-transition-probe-001": "45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e",
    "select-views-001": "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3",
    "arithmetic-monitor-open": "5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266",
}
SPLIT_FILES = {"train": "TRAIN.jsonl", "dev": "DEV.jsonl"}
ID_FIELDS = {"id", "row_id", "parent_id", "group_id", "source_id", "base_world_id"}
ID_LIST_FIELDS = {"row_ids", "parent_ids", "group_ids", "source_ids", "base_world_ids"}
JSON_LIMIT = 8 * 1024 * 1024
ROW_LIMIT = 1024 * 1024
FILE_LIMIT = 64 * 1024 * 1024
ROW_COUNT_LIMIT = 20000


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def decode(value: str):
    def invalid(token):
        raise AuditError(f"nonfinite JSON constant: {token}")
    return json.loads(value, object_pairs_hook=_object_pairs, parse_constant=invalid)


def regular(path: Path) -> Path:
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "symlink/traversal input paths are not accepted")
    require(path.is_file(), f"missing regular input file: {path}")
    return path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"invalid/missing SHA256: {label}")
    return value


def read_json(path: Path) -> dict:
    path = regular(path)
    require(path.stat().st_size <= JSON_LIMIT, "JSON object exceeds audit reader bound")
    value = decode(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def user_turns(row: dict) -> list[str]:
    messages = row.get("messages", row.get("prompt"))
    require(isinstance(messages, list) and all(isinstance(m, dict) for m in messages),
            "audit requires explicit chat messages")
    turns = [m.get("content") for m in messages if m.get("role") == "user"]
    require(bool(turns) and all(isinstance(t, str) and t.strip() for t in turns),
            "audit requires nonempty user text")
    return turns


def normalized_text(text: str) -> str:
    """NFKC, Unicode casefold, whitespace collapse; keep punctuation and digits."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_sha256(row: dict) -> str:
    # Preserve user-turn boundaries; shared system/assistant text never contributes.
    return hashlib.sha256(canonical([normalized_text(t) for t in user_turns(row)])).hexdigest()


def identifiers(row: dict) -> dict[str, str]:
    """Collect explicitly named identity fields, including nested provenance maps.

    Match values across field types: moving an evaluation ID into source_id or
    parent_id cannot evade exclusion. Names, roles and arbitrary prose are not IDs.
    """
    result = {}

    def walk(value, prefix="", depth=0):
        require(depth <= 32, "candidate identity nesting exceeds audit bound")
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key in ID_FIELDS:
                    require(isinstance(child, str) and bool(child.strip()), f"invalid identity field: {path}")
                    result[path] = child
                elif key in ID_LIST_FIELDS:
                    require(isinstance(child, list) and all(isinstance(v, str) and v.strip() for v in child),
                            f"invalid identity list: {path}")
                    for index, item in enumerate(child):
                        result[f"{path}[{index}]"] = item
                elif isinstance(child, (dict, list)):
                    walk(child, path, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    walk(child, f"{prefix}[{index}]", depth + 1)
    walk(row)
    return result


def read_rows(path: Path) -> list[dict]:
    path = regular(path)
    require(path.stat().st_size <= FILE_LIMIT, "JSONL exceeds bounded open-audit file size")
    rows, ids = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            require(len(line.encode("utf-8")) <= ROW_LIMIT, "JSONL row exceeds audit bound")
            require(bool(line.strip()), "blank JSONL rows cannot establish exact row inventory")
            value = decode(line)
            require(isinstance(value, dict), "JSONL row must be an object")
            row_id = value.get("id")
            require(isinstance(row_id, str) and bool(row_id.strip()) and row_id not in ids,
                    "missing or duplicate row ID")
            user_turns(value); identifiers(value)
            ids.add(row_id); rows.append(value)
            require(len(rows) <= ROW_COUNT_LIMIT, "JSONL row count exceeds audit bound")
    require(bool(rows), "empty candidate/bank cannot establish exclusion")
    return rows


def _load_guard():
    path = regular(TEXT_CONTACT_PATH)
    spec = importlib.util.spec_from_file_location("binding_signal_open_text_contact", path)
    require(spec is not None and spec.loader is not None, "pilot text-contact module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.POLICY == "user_text_contact_v1_id_8gram_digit_blinded_stem",
            "unexpected pilot text-contact policy")
    return module


def compare(rows: list[dict], panel: list[dict], guard) -> dict:
    id_index, prompt_index, turn_index = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in panel:
        for field, value in identifiers(row).items():
            id_index[value].append({"bank_row_id": row["id"], "bank_field": field})
        prompt_index[prompt_sha256(row)].append(row["id"])
        for index, text in enumerate(user_turns(row)):
            text_sha = hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()
            turn_index[text_sha].append({"bank_row_id": row["id"], "bank_user_turn": index})
    before = hashlib.sha256(canonical(rows)).hexdigest()
    ignored_clean_rows, guard_report = guard.audit_and_filter(rows, panel)
    del ignored_clean_rows
    require(hashlib.sha256(canonical(rows)).hexdigest() == before, "text-contact helper mutated candidate rows")
    require(guard_report["input_rows"] == len(rows) and guard_report["panel_rows"] == len(panel),
            "text-contact helper changed input inventory")
    contacts = {r["id"]: r for r in guard_report["dropped"]}
    require(len(contacts) == guard_report["dropped_rows"], "text-contact report repeats candidate IDs")
    collisions, counts = [], Counter()
    for row in rows:
        reasons, id_matches, turn_matches = set(), [], []
        for field, value in identifiers(row).items():
            for match in id_index.get(value, []):
                id_matches.append({"candidate_field": field, "value": value, **match})
        if id_matches:
            reasons.add("identifier_contact")
        full_sha = prompt_sha256(row)
        prompt_matches = sorted(prompt_index.get(full_sha, []))
        if prompt_matches:
            reasons.add("normalized_exact_user_prompt")
        for index, text in enumerate(user_turns(row)):
            text_sha = hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()
            for match in turn_index.get(text_sha, []):
                turn_matches.append({"candidate_user_turn": index, "normalized_text_sha256": text_sha, **match})
        if turn_matches:
            reasons.add("normalized_exact_user_turn")
        contact = contacts.get(row["id"], {})
        reasons.update(contact.get("reasons", []))
        if reasons:
            counts.update(reasons)
            collisions.append({"candidate_id": row["id"], "reasons": sorted(reasons),
                               "identifier_matches": id_matches, "normalized_user_prompt_sha256": full_sha,
                               "normalized_prompt_bank_ids": prompt_matches,
                               "normalized_turn_matches": turn_matches,
                               "digit_blinded_stem_bank_ids": contact.get("matched_panel_stem_ids", [])})
    return {"candidate_rows": len(rows), "open_bank_rows": len(panel),
            "collision_rows": len(collisions), "reason_counts": dict(sorted(counts.items())),
            "collisions": collisions}


def audit_candidate(candidate: Path) -> dict:
    """Return a report, never a filtered dataset or a training authorization."""
    candidate = Path(candidate).absolute()
    build_path = candidate / "BUILD.json"
    paths = {"candidate_BUILD": build_path, "text_contact_code": TEXT_CONTACT_PATH,
             **{f"candidate_{split}": candidate / name for split, name in SPLIT_FILES.items()},
             **{f"open_bank_{label}": path for label, path in OPEN_BANK_PATHS.items()}}
    input_hashes = {name: {"path": str(regular(path)), "sha256": sha(path), "bytes": regular(path).stat().st_size}
                    for name, path in paths.items()}
    require(set(OPEN_BANK_PATHS) == set(EXPECTED_BANK_SHA256) and
            all(input_hashes[f"open_bank_{name}"]["sha256"] == expected for name, expected in EXPECTED_BANK_SHA256.items()),
            "an explicitly open bank differs from its frozen SHA256")
    build = read_json(build_path)
    require(type(build.get("schema")) is int and build["schema"] == 1 and
            build.get("record") == "BINDING_SIGNAL_V2_DATA_READINESS", "unexpected candidate BUILD schema")
    files = build.get("files_sha256")
    require(isinstance(files, dict) and set(files) == set(SPLIT_FILES.values()),
            "candidate BUILD must bind exactly TRAIN.jsonl and DEV.jsonl")
    data, all_ids = {}, set()
    for split, name in SPLIT_FILES.items():
        require(input_hashes[f"candidate_{split}"]["sha256"] == _sha(files[name], name),
                f"candidate BUILD hash mismatch: {name}")
        rows = read_rows(candidate / name)
        require(isinstance(build.get("rows"), dict) and type(build["rows"].get(split)) is int and
                build["rows"][split] == len(rows), "candidate BUILD row counts disagree")
        require(all(r.get("split") == r.get("source_split") == split for r in rows), "candidate split identity differs")
        ids = {r["id"] for r in rows}
        require(not ids & all_ids, "candidate TRAIN/DEV repeat a row ID")
        all_ids.update(ids); data[split] = rows
    panels = {name: read_rows(path) for name, path in OPEN_BANK_PATHS.items()}
    require(set(panels) == set(EXPECTED_BANK_ROWS) and
            all(len(rows) == EXPECTED_BANK_ROWS[name] for name, rows in panels.items()),
            "the three complete open-bank row inventories are required")
    guard = _load_guard()
    comparisons, contacted = {}, {}
    for split, rows in data.items():
        comparisons[split] = {name: compare(rows, panel, guard) for name, panel in panels.items()}
        contacted[split] = sorted({r["candidate_id"] for audit in comparisons[split].values() for r in audit["collisions"]})
    for name, path in paths.items():
        require(sha(path) == input_hashes[name]["sha256"], f"audit input changed while being read: {name}")
    blocked = any(contacted.values())
    return {"schema": 1, "record": "BINDING_SIGNAL_OPEN_EXCLUSION_AUDIT",
            "status": "BLOCKED_ON_OPEN_EXCLUSION_CONTACT" if blocked else "PASS_OPEN_EXCLUSIONS_BUILD_ONLY",
            "created_utc": datetime.now(timezone.utc).isoformat(), "candidate": str(candidate),
            "input_files": input_hashes, "auditor_sha256": sha(Path(__file__)),
            "expected_open_bank_sha256": dict(EXPECTED_BANK_SHA256),
            "candidate_rows": {split: len(rows) for split, rows in data.items()},
            "open_bank_rows_read": {name: len(rows) for name, rows in panels.items()},
            "candidate_contact_rows": {split: len(ids) for split, ids in contacted.items()},
            "candidate_contact_ids": contacted, "comparisons": comparisons,
            "identity_fields": sorted(ID_FIELDS | ID_LIST_FIELDS), "identity_matching": "exact values across identity field types",
            "normalized_exact_policy": "NFKC, casefold, whitespace collapse; full user-turn list and individual user turns; digits/punctuation retained",
            "pilot_text_contact_policy": guard.POLICY, "system_and_assistant_text_checked": False,
            "candidate_files_modified": False, "filtering_performed": False,
            "scope": "Only the three explicitly OPEN named banks; no other or sealed datasets inspected",
            "semantic_contamination_certificate": False, "training_permission": False,
            "evaluation_permission": False, "quality_claim": False, "release_permitted": False}


def write_report(path: Path, report: dict) -> None:
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe audit output path")
    require(path.parent.is_dir(), "audit output parent must already exist")
    with path.open("xb") as stream:
        stream.write(canonical(report) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and (args.out.exists() or args.out.is_symlink()):
        parser.error("--out must be a new file; existing evidence is never overwritten")
    try:
        report = audit_candidate(args.candidate)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
        report = {"schema": 1, "record": "BINDING_SIGNAL_OPEN_EXCLUSION_AUDIT", "status": "BLOCKED_ON_INPUT_INTEGRITY",
                  "reasons": [str(exc)], "candidate_files_modified": False, "filtering_performed": False,
                  "training_permission": False, "evaluation_permission": False, "quality_claim": False, "release_permitted": False}
    if args.out is not None:
        write_report(args.out, report)
        print(json.dumps({"status": report["status"], "output": str(args.out),
                          "candidate_contact_rows": report.get("candidate_contact_rows")}))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 2 if report["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
