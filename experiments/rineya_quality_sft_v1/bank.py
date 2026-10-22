"""Read-only consumer of an exact reviewed dataset-owner export.

No generator or admission authority lives here. Header/ancestry and separately
attributable, hash-pinned review receipts are checked BEFORE either payload is
opened. Hashes bind declarations; they do not authenticate a reviewer by magic.
The real exporter must supply the attributable review records itself.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import unicodedata

from . import integrity as I
from .integrity import decode, digest, regular
from . import EXPERIMENT
from .contract import DATASET_OWNER, DEV_COUNTS, TRAIN_COUNTS

REVIEW_KINDS = {"rights", "gold", "partition", "shortcut"}
DIMENSIONS = {"solving", "harder_transfer", "grounding_answerability", "output_contract", "communication"}
SCORERS = {"exact", "number", "json_exact", "fixed_list", "semantic_review"}
BLOCKED_ANCESTRY = {"EXCLUDED", "TEST_ONLY", "SEALED", "CONFIRMATION", "PROTECTED", "SPENT", "ANY_READ"}
BLOCKED_ROLE_WORDS = {"EXCLUDED", "TEST", "SEALED", "CONFIRMATION", "PROTECTED", "SPENT", "CAL", "CALIBRATION"}


def _sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Explicit nonempty {name} required")
    return value


def _safe_input(path, base):
    path = Path(path)
    if not path.is_absolute():
        path = base / path
    # The custody helper inspects lexical ancestors/reparse attributes before
    # any resolve operation could hide a junction or other input alias.
    path = I.check_input_path(path)
    names = [p.casefold() for p in path.parts]
    if any("sealed" in p or "confirmation" in p or "protected" in p or p == ".codex" for p in names):
        raise ValueError("Protected/.codex input paths are forbidden")
    return path.absolute()


def _ancestry(value, fixture):
    if not isinstance(value, list) or not value:
        raise ValueError("Explicit complete ancestry declarations required")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Structured ancestry declarations required")
        role = _text(item.get("role"), "ancestry role")
        # Check the role independently of an eligible-looking status. A new
        # receipt cannot upgrade calibration/test controls or blocked roots.
        words = re.findall(r"[A-Z0-9]+", unicodedata.normalize("NFKC", role).upper())
        if (item.get("protected") is not False
                or item.get("status") in BLOCKED_ANCESTRY
                or set(words) & BLOCKED_ROLE_WORDS
                or any(words[i:i + 2] == ["ANY", "READ"] for i in range(len(words) - 1))
                or (not fixture and "FIXTURE" in words)):
            raise ValueError("Excluded/test/protected ancestry cannot be admitted")
        allowed = {"ADMITTED", "SOURCE_VERIFIED"} | ({"FIXTURE_ONLY"} if fixture else set())
        if item.get("status") not in allowed or not _sha(item.get("sha256")):
            raise ValueError("Ancestry must be explicitly eligible and hash-pinned")


def _bounded_bytes(path, limit):
    target = regular(path)
    if target.stat().st_size > limit:
        raise ValueError("Input exceeds bounded reader limit")
    with target.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Consumed input exceeds bounded reader limit")
    return raw


def _pinned_bytes(path, expected, limit):
    """Bounded bytes actually decoded must match the reviewed raw-file hash."""
    target = regular(path)
    # Retain the independent pre-read custody check, then bind the actual read.
    # The second comparison detects a change between that check and consumption.
    if _bounded_hash(target, limit) != expected:
        raise ValueError("Pinned input exceeds bound or differs from reviewed exact bytes")
    raw = _bounded_bytes(target, limit)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Consumed bytes differ from reviewed exact bytes or bound")
    return raw


def _bounded_hash(path, limit):
    # Hashing has the same growth-resistant cap as the subsequent parsed read.
    return hashlib.sha256(_bounded_bytes(path, limit)).hexdigest()


def validate_review_principals(reviews):
    """One immutable exporter; no review author may be that export author."""
    exporters = set()
    for review in reviews:
        exporter = _text(review.get("exporter_id"), "export author identity")
        reviewer = _text(review.get("reviewer_id"), "review author identity")
        if reviewer in {exporter, DATASET_OWNER["thread_id"]}:
            raise ValueError("Export author cannot independently review the export")
        exporters.add(exporter)
    if len(exporters) != 1:
        raise ValueError("All four review receipts must identify the same export author")


def export_binding(manifest):
    """Subject reviewed before outcomes; no receipt/content-path circular hash."""
    return digest({"schema": manifest["schema"], "experiment": manifest["experiment"],
                   "fixture_only": manifest["fixture_only"], "admission": manifest["admission"],
                   "owner": manifest["owner"], "export_id": manifest["export_id"],
                   "files": {s: {k: manifest["files"][s][k] for k in ("sha256", "rows")} for s in ("train", "dev")},
                   "ancestry": manifest["metadata"]["ancestry"]})


def owner_request_template():
    """Metadata-only schema description; never creates an admitted bank."""
    return {"schema": "RQSFT_OWNER_BUNDLE_V1", "experiment": EXPERIMENT,
            "owner": dict(DATASET_OWNER), "export_id": "OWNER_ASSIGN_EXACT_EXPORT_ID",
            "fixture_only": False, "admission": "PENDING_OWNER_EXPORT",
            "files": {"train": {"path": "train.jsonl", "sha256": "ACTUAL_SHA256", "rows": 200},
                      "dev": {"path": "dev.jsonl", "sha256": "ACTUAL_SHA256", "rows": 512}},
            "metadata": {"ancestry": [{"status": "ADMITTED", "role": "open_train_or_dev",
                                        "protected": False, "sha256": "ACTUAL_PARENT_SHA256"}],
                         "reviews": [{"kind": k, "path": k + "-review.json", "sha256": "ACTUAL_RECEIPT_SHA256",
                                      "reviewer_id": "ACTUAL_INDEPENDENT_REVIEWER", "reviewer_role": "independent",
                                      "status": "APPROVED"} for k in sorted(REVIEW_KINDS)]},
            "review_manifest_sha256": "export_binding(manifest)",
            "receipt_schema": {"schema": "RQSFT_OWNER_REVIEW_V1", "kind": "ONE_REVIEW_KIND",
                               "owner_export_binding_sha256": "export_binding(manifest)", "status": "APPROVED",
                               "reviewer_id": "ACTUAL_REVIEWER", "reviewer_role": "independent",
                               "exporter_id": "ACTUAL_EXPORT_AUTHOR",
                               "attribution": {"principal": "ACTUAL_REVIEWER", "record_reference": "ACTUAL_REVIEW_RECORD"},
                               "checks": {"ONE_REVIEW_KIND": True}},
            "row_schema": {"id": "unique", "parent_id": "complete family group", "parent_size": 1,
                           "pair_id": None, "split": "train|dev", "category": "registered quota stratum",
                           "family": "underlying base family", "scenario_id": "scenario graph", "lexicon_id": "lexicon pool",
                           "admission": "ADMITTED", "ancestry": "same eligible structured ancestry contract",
                           "messages": "full role/content history ending final assistant target",
                           "final_target_verified": True, "empty_target_verified": False,
                           "evaluation": {"dimension": "declared quality dimension", "scorer": "declared supported scorer",
                                          "answer": "independently reviewed canonical gold", "answerable": None, "critical": False}},
            "train_counts": TRAIN_COUNTS, "dev_counts": DEV_COUNTS,
            "required_grounded_train_pairs": 40,
            "pair_contract": "Each grounded TRAIN pair has exactly one answerable and one missing-evidence view; boolean-tagged DEV grounding_answerability rows also require complete balanced pairs",
            "request_only": True, "native_permission": False}


def _header(manifest, base, allow_fixture):
    if (not isinstance(manifest, dict) or manifest.get("schema") != "RQSFT_OWNER_BUNDLE_V1"
            or manifest.get("experiment") != EXPERIMENT or type(manifest.get("fixture_only")) is not bool):
        raise ValueError("Exact RQSFT owner bundle header required before payload access")
    if manifest.get("request_only", False) is not False:
        raise ValueError("A request/template is never an actual admitted export")
    fixture = manifest["fixture_only"]
    if fixture and not allow_fixture:
        raise ValueError("Fixture-only bundle cannot authorize real native work")
    if manifest.get("admission") != ("FIXTURE_ONLY" if fixture else "ADMITTED"):
        raise ValueError("No excluded/test-only/unadmitted bundle may be consumed")
    if manifest.get("owner") != DATASET_OWNER:
        raise ValueError("Bundle must be the exact existing dataset-owner export")
    _text(manifest.get("export_id"), "owner export identity")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"train", "dev"}:
        raise ValueError("Exact TRAIN/DEV file declarations required")
    paths = {}
    for split, count in (("train", 200), ("dev", 512)):
        item = files[split]
        if (not isinstance(item, dict) or not _sha(item.get("sha256"))
                or type(item.get("rows")) is not int or item["rows"] != count):
            raise ValueError("Owner export must declare exactly 200 TRAIN and 512 DEV")
        paths[split] = _safe_input(_text(item.get("path"), "payload path"), base)
        if paths[split].suffix.casefold() != ".jsonl":
            raise ValueError("Explicit JSONL payload files required")
    if paths["train"] == paths["dev"]:
        raise ValueError("TRAIN/DEV cannot use the same payload")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Owner review/ancestry metadata object required")
    _ancestry(metadata.get("ancestry"), fixture)
    binding = export_binding(manifest)
    if manifest.get("review_manifest_sha256") != binding:
        raise ValueError("Owner review subject binding differs from export metadata")
    reviews = metadata.get("reviews", [])
    if fixture:
        if reviews:
            raise ValueError("Fixture bundles must not invent real approvals")
        return fixture, paths, binding, []
    if (not isinstance(reviews, list) or len(reviews) != 4 or any(not isinstance(r, dict) for r in reviews)
            or {r.get("kind") for r in reviews} != REVIEW_KINDS):
        raise ValueError("Four pinned independent rights/gold/partition/shortcut receipts required")
    validated = []
    for review in reviews:
        if (review.get("status") != "APPROVED" or review.get("reviewer_role") != "independent"
                or not _sha(review.get("sha256")) or not review.get("reviewer_id")):
            raise ValueError("Declared review metadata incomplete")
        path = _safe_input(_text(review.get("path"), "review receipt path"), base)
        if path.suffix.casefold() != ".json" or path in paths.values():
            raise ValueError("Review must be a distinct metadata JSON receipt")
        receipt = decode(_pinned_bytes(path, review["sha256"], 1024 * 1024).decode("utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("Independent review receipt must be an object")
        reviewer = receipt.get("reviewer_id")
        attribution = receipt.get("attribution", {})
        if (receipt.get("schema") != "RQSFT_OWNER_REVIEW_V1" or receipt.get("kind") != review["kind"]
                or receipt.get("owner_export_binding_sha256") != binding or receipt.get("status") != "APPROVED"
                or reviewer != review["reviewer_id"] or receipt.get("reviewer_role") != "independent"
                or not receipt.get("exporter_id") or reviewer == receipt["exporter_id"]
                or reviewer == DATASET_OWNER["thread_id"] or attribution.get("principal") != reviewer
                or not attribution.get("record_reference") or receipt.get("checks", {}).get(review["kind"]) is not True):
            raise ValueError("Receipt contents do not establish independently attributable bound review")
        validated.append({"kind": review["kind"], "sha256": review["sha256"], "reviewer_id": reviewer,
                          "exporter_id": receipt["exporter_id"],
                          "attribution": attribution, "subject_sha256": binding})
    validate_review_principals(validated)
    return fixture, paths, binding, validated


def normalized_prompt(row):
    return " ".join(unicodedata.normalize("NFKC", "\n".join(
        m["role"] + ":" + m["content"] for m in row["messages"][:-1])).casefold().split())


def validate_rows(banks, fixture):
    if set(banks) != {"train", "dev"}:
        raise ValueError("Complete TRAIN/DEV banks required")
    seen, parent_rows, pair_rows = {}, {}, {}
    for split, quota in (("train", TRAIN_COUNTS), ("dev", DEV_COUNTS)):
        rows = banks[split]
        if not isinstance(rows, list) or len(rows) != sum(quota.values()) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("All 200 TRAIN/512 DEV rows must survive unchanged")
        if Counter(r.get("category") for r in rows) != Counter(quota):
            raise ValueError("Registered category quotas underfilled/overfilled")
        for row in rows:
            for field in ("id", "parent_id", "family", "scenario_id", "lexicon_id"):
                _text(row.get(field), field)
            if row.get("split") != split or row.get("admission") != ("FIXTURE_ONLY" if fixture else "ADMITTED"):
                raise ValueError("Row admission/split differs from reviewed export")
            _ancestry(row.get("ancestry"), fixture)
            if type(row.get("parent_size")) is not int or row["parent_size"] < 1:
                raise ValueError("Explicit complete parent size required")
            if "pair_id" not in row:
                raise ValueError("Explicit nullable pair identity required")
            if row.get("pair_id") is not None:
                _text(row["pair_id"], "pair identity")
                pair_rows.setdefault((split, row["pair_id"]), []).append(row)
            parent_rows.setdefault((split, row["parent_id"]), []).append(row)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2 or any(not isinstance(m, dict) for m in messages):
                raise ValueError("Full conversation plus final target required")
            start = 1 if messages[0].get("role") == "system" else 0
            expected = (["system"] if start else []) + ["user" if i % 2 == 0 else "assistant" for i in range(len(messages) - start)]
            if [m.get("role") for m in messages] != expected or messages[-1].get("role") != "assistant":
                raise ValueError("History must alternate user/assistant and end with final assistant")
            for i, message in enumerate(messages):
                if (set(message) != {"role", "content"} or not isinstance(message["content"], str)
                        or any(t in message["content"] for t in ("<|", "<think>", "</think>"))):
                    raise ValueError("Canonical role/content history without embedded control markers required")
                if i < len(messages) - 1 and not message["content"].strip():
                    raise ValueError("Missing history content cannot be silently omitted")
            final = messages[-1]["content"]
            if (row.get("final_target_verified") is not True or final != final.strip()
                    or (final == "" and row.get("empty_target_verified") is not True)
                    or (final != "" and row.get("empty_target_verified") is True)
                    or type(row.get("empty_target_verified")) is not bool):
                raise ValueError("Missing/unverified target is not intentional verified empty target")
            evaluation = row.get("evaluation")
            if (not isinstance(evaluation, dict) or evaluation.get("dimension") not in DIMENSIONS
                    or evaluation.get("scorer") not in SCORERS or not isinstance(evaluation.get("answer"), str)
                    or "answerable" not in evaluation
                    or evaluation.get("answerable") is not None and type(evaluation.get("answerable")) is not bool
                    or type(evaluation.get("critical")) is not bool):
                raise ValueError("Explicit independent evaluation contract required; no target fallback")
            keys = [("id", row["id"]), ("parent", row["parent_id"]), ("family", row["family"]),
                    ("scenario", row["scenario_id"]), ("lexicon", row["lexicon_id"]),
                    ("normalized", normalized_prompt(row)), ("digitblind", re.sub(r"\d+", "#", normalized_prompt(row)))]
            if row.get("pair_id") is not None:
                keys.append(("pair", row["pair_id"]))
            for key in keys:
                if key in seen and (seen[key] != split or key[0] == "id"):
                    raise ValueError(f"Prohibited duplicate/cross-split {key[0]} overlap")
                seen[key] = split
    for rows in parent_rows.values():
        if any(r["parent_size"] != len(rows) for r in rows):
            raise ValueError("Incomplete parent group; no pair splitting or silent refill")
    if any(len(rows) != 2 for rows in pair_rows.values()):
        raise ValueError("Every declared answerability pair requires exactly two views")
    for pair in pair_rows.values():
        if any(len({r[field] for r in pair}) != 1
               for field in ("parent_id", "family", "scenario_id", "lexicon_id")):
            raise ValueError("Evidence pair must share its underlying parent/family/scenario/lexicon")
    grounded = [r for r in banks["train"] if r["category"] == "grounded"]
    if any(r["pair_id"] is None for r in grounded) or len({r["pair_id"] for r in grounded}) != 40:
        raise ValueError("All 80 grounded TRAIN rows must form exactly 40 complete missing-evidence pairs")
    required_pairs = {("train", r["pair_id"]) for r in grounded}
    for row in banks["dev"]:
        if row["evaluation"]["dimension"] == "grounding_answerability" and type(row["evaluation"]["answerable"]) is bool:
            if row["pair_id"] is None:
                raise ValueError("Boolean-tagged DEV answerability requires a complete evidence pair")
            required_pairs.add(("dev", row["pair_id"]))
    for split, pair_id in required_pairs:
        pair = pair_rows[(split, pair_id)]
        if ({r["evaluation"]["answerable"] for r in pair} != {True, False}
                or any(type(r["evaluation"]["answerable"]) is not bool for r in pair)
                or any(r["evaluation"]["dimension"] != "grounding_answerability" for r in pair)
                or (split == "train" and any(r["category"] != "grounded" for r in pair))):
            raise ValueError("Evidence pairs need one answerable and one missing-evidence view")
    return {"counts": {s: len(r) for s, r in banks.items()},
            "grounded_train_pairs": 40,
            "family_scenario_lexicon_parent_pair_normalized_digitblind_overlap": "PASS",
            "semantic_shortcut_review": "FIXTURE_ONLY" if fixture else "OWNER_PINNED_INDEPENDENT_REVIEW"}


def read_owner_bundle(manifest_path, allow_fixture=False):
    manifest_path = _safe_input(manifest_path, Path.cwd())
    manifest_raw = _bounded_bytes(manifest_path, 1024 * 1024)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    manifest = decode(manifest_raw.decode("utf-8"))
    fixture, paths, binding, reviews = _header(manifest, manifest_path.parent, allow_fixture)
    # No payload stat/hash/read occurs until all real review receipts above pass.
    banks = {}
    for split, path in paths.items():
        raw = _pinned_bytes(path, manifest["files"][split]["sha256"], 16 * 1024 * 1024)
        lines = raw.decode("utf-8").split("\n")
        # One ordinary terminal newline is allowed. Interior/extra blank records
        # never disappear from the reviewed physical export; CRLF also works.
        if lines[-1] == "":
            lines.pop()
        if len(lines) != manifest["files"][split]["rows"] or any(not line.strip() for line in lines):
            raise ValueError("Physical JSONL records must exactly cover every declared row without blanks")
        rows = [decode(line) for line in lines]
        if any(not isinstance(r, dict) for r in rows):
            raise ValueError("Every payload row must be an object")
        banks[split] = rows
    audit = validate_rows(banks, fixture)
    if _bounded_hash(manifest_path, 1024 * 1024) != manifest_sha256:
        raise ValueError("Owner manifest changed during consumption")
    return {**banks, "fixture_only": fixture, "owner": manifest["owner"],
            "manifest": manifest, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha256,
            "export_binding_sha256": binding, "reviews": reviews, "audit": audit}
