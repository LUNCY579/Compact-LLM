"""Offline Phase-A screening. Reads scored JSON, never weights or CHECK predictions during CAL.

Receipt consistency and deterministic selection are not native qualification, chronology
proof, external-scorer verification, noninferiority, or authorization to execute a model.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

from score import load_run, summary
from soup import ARMS, ASSETS, EXPERIMENT, PINS, read_json, require, sha256

PROTOCOL_SHA256 = "3a1c7c205170b716f0f5c56608ae03a92f7a6b5720407c983f52b7b732393c7f"
INHERITED_CODE = {
    "soup.py": "b6e15252851a66691aa0711592470c5942108edf5e116de25b3f4e7226376b40",
    "score.py": "6ff474c6ffe031090f132c5cce5aae801a4dcc720adcf8349365aee02bb427db",
}
NEW_ARMS = tuple(k for k in ARMS if not k.startswith("local_"))
CANDIDATES = {"sparse_opallo_50": "titan_sparse", "titan_opallo_50": "titan"}
DOMAINS = ("transition", "arithmetic", "short_format", "answerable_retrieval",
           "unanswerable_abstention", "long_context")
RUNTIME_KEYS = ("runtime_sha256", "scorer_sha256", "decoding_sha256")
CONTRACT_KEYS = (*RUNTIME_KEYS, "qualification_receipt_sha256", "ancestry_audit_sha256")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "unbound SHA-256")
    return value


def count(value, maximum, minimum=1):
    require(type(value) is int and minimum <= value <= maximum, "unbound/out-of-range count floor")
    return value


def read_receipt(spec, root):
    require(isinstance(spec, dict) and isinstance(spec.get("path"), str) and spec["path"],
            "unbound receipt path")
    path = root / spec["path"]
    expected = hash_value(spec.get("sha256"))
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected, "receipt hash mismatch")
    obj = read_json(path)
    require(sha256(path) == expected, "receipt changed while reading")
    return obj


def load_freeze(path):
    before = sha256(path)
    f = read_json(path)
    require(sha256(path) == before, "freeze changed while reading")
    require(f.get("schema") == 1 and f.get("experiment") == EXPERIMENT, "wrong freeze schema/experiment")
    here = Path(__file__).resolve().parent
    require(f.get("protocol_sha256") == PROTOCOL_SHA256 == sha256(here / "protocol.json"),
            "registered protocol changed; amendment required")
    code = {**INHERITED_CODE, "screen.py": sha256(Path(__file__))}
    require(f.get("code_sha256") == code, "unbound/changed implementation")
    require(all(sha256(here / k) == v for k, v in INHERITED_CODE.items()), "inherited code changed")
    for key in CONTRACT_KEYS:
        hash_value(f.get(key))
    # These two hashes bind external reviews; this program does not certify their content.
    require(f.get("tokenizer_sha256") == ASSETS["tokenizer.json"], "tokenizer pin mismatch")
    require(f.get("asset_bundle_sha256") == digest(ASSETS), "tokenizer/template/stop sidecars mismatch")
    artifacts = f.get("checkpoint_sha256", {})
    require(set(artifacts) == set(PINS) | set(NEW_ARMS), "exact nine configurations required")
    require(all(artifacts.get(k) == v for k, v in PINS.items()), "published endpoint pin mismatch")
    receipts = f.get("build_receipts", {})
    require(set(receipts) == set(NEW_ARMS), "five build receipts required")
    for arm in NEW_ARMS:
        hash_value(artifacts[arm])
        r = read_receipt(receipts[arm], path.parent)
        require(r.get("experiment") == EXPERIMENT and r.get("arm") == arm, "wrong build identity")
        require(r.get("status") == "MATERIALIZED_NOT_EVALUATED", "incomplete build")
        require(r.get("pair") == list(ARMS[arm]) and r.get("alpha_right") == 0.5,
                "mixture coefficients/source order changed")
        require(r.get("stored_dtype") == "float32" and r.get("weight_sha256") == artifacts[arm],
                "build output hash/dtype mismatch")
        require(r.get("sources_rehashed_after_build") is True and r.get("optimizer_steps") == 0,
                "unverified sources or training in build")
        sources = {"supa", *ARMS[arm][:2]}
        require(set(r.get("files", {})) == sources, "build source set mismatch")
        for source in sources:
            files = r["files"][source]
            require(files.get("model.safetensors") == PINS[source], "build source pin mismatch")
            require(all(files.get(k) == v for k, v in ASSETS.items()), "build sidecar mismatch")
    banks = f.get("banks", {})
    require(set(banks) == {"CAL", "CHECK"}, "CAL and CHECK must both be frozen")
    parents, ids = {}, {}
    for split, bank in banks.items():
        hash_value(bank.get("dataset_sha256"))
        require(isinstance(bank.get("suite_id"), str) and bank["suite_id"], "unbound suite")
        rows = bank.get("row_manifest")
        require(isinstance(rows, list) and 1 <= len(rows) <= 256, "bank must contain 1..256 manifest rows")
        for row in rows:
            require(set(row) == {"id", "parent_id", "domain", "answerable"},
                    "manifest permits identities only, no prompts/gold/predictions")
            require(all(isinstance(row[k], str) and row[k].strip() for k in ("id", "parent_id")),
                    "explicit row/parent IDs required")
            require(row["domain"] in DOMAINS and type(row["answerable"]) is bool, "invalid row role")
            if row["domain"] in ("answerable_retrieval", "unanswerable_abstention"):
                require(row["answerable"] == (row["domain"] == "answerable_retrieval"), "answerability mismatch")
        ids[split], parents[split] = {r["id"] for r in rows}, {r["parent_id"] for r in rows}
        require(len(ids[split]) == len(parents[split]) == len(rows),
                "mini screen requires unique rows and one canonical view per parent")
        require({r["domain"] for r in rows} == set(DOMAINS), "all six domains required")
        sizes = {d: sum(r["domain"] == d for r in rows) for d in DOMAINS}
        floors = bank.get("minimum_source_correct", {})
        require(set(floors) == set(DOMAINS), "six source competence floors required")
        for d in DOMAINS:
            count(floors[d], sizes[d])
        donors, shortcuts = bank.get("donor_min_correct", {}), bank.get("shortcut_correct", {})
        require(set(donors) == {"titan", "titan_sparse", "opallo"}, "donor floors required")
        require(set(shortcuts) == {"transition", "arithmetic"}, "frozen shortcut scores required")
        for donor, d in (("titan", "transition"), ("titan_sparse", "transition"), ("opallo", "arithmetic")):
            count(shortcuts[d], sizes[d], minimum=0)
            count(donors[donor], sizes[d])
            require(donors[donor] > shortcuts[d], "donor floor must beat frozen shortcut score")
    require(not parents["CAL"] & parents["CHECK"] and not ids["CAL"] & ids["CHECK"], "CAL/CHECK overlap")
    require(banks["CAL"]["dataset_sha256"] != banks["CHECK"]["dataset_sha256"], "same dataset in both roles")
    return f, before


def load_bundle(directory, expected, f, freeze_sha, split):
    require({p.name for p in directory.glob("*.json")} == {k + ".json" for k in expected},
            "missing/extra configuration files; no intersections or extra sweep arms")
    bank = f["banks"][split]
    manifest = {r["id"]: r for r in bank["row_manifest"]}
    result, hashes = {}, {}
    for arm in sorted(expected):
        path = directory / (arm + ".json")
        before = sha256(path)
        meta, rows = load_run(path)
        require(sha256(path) == before, "run changed while reading")
        required = {"experiment": EXPERIMENT, "arm_id": arm, "split": split,
                    "suite_id": bank["suite_id"], "dataset_sha256": bank["dataset_sha256"],
                    "checkpoint_sha256": f["checkpoint_sha256"][arm], "freeze_sha256": freeze_sha,
                    "evaluation_dtype": "float32", "tokenizer_sha256": f["tokenizer_sha256"],
                    "asset_bundle_sha256": f["asset_bundle_sha256"],
                    **{k: f[k] for k in CONTRACT_KEYS}}
        require(all(meta.get(k) == v for k, v in required.items()), "run identity/runtime/freeze mismatch")
        require(set(rows) == set(manifest), "run rows differ from frozen bank")
        for key, row in rows.items():
            require(all(row.get(k) == v and type(row.get(k)) is type(v) for k, v in manifest[key].items()),
                    "run row differs from frozen parent/domain/answerability")
            require(type(row.get("abstained")) is bool, "explicit boolean abstained required")
            require(not (row["answerable"] and row["abstained"] and row["correct"]), "contradictory scoring")
        result[arm], hashes[arm] = rows, before
    return result, hashes


def domain_pairs(a, b):
    return {d: summary([(a[k]["correct"], b[k]["correct"]) for k in sorted(a) if a[k]["domain"] == d])
            for d in DOMAINS}


def correct(rows, domain):
    return sum(r["correct"] for r in rows.values() if r["domain"] == domain)


def false_abstentions(rows):
    return {d: sum(r["answerable"] and r["abstained"] for r in rows.values() if r["domain"] == d)
            for d in DOMAINS}


def assess(arm, bundle, bank):
    source, candidate, donor = bundle["supa"], bundle[arm], CANDIDATES[arm]
    retention = domain_pairs(source, candidate)
    require(all(retention[d]["reference_correct"] >= bank["minimum_source_correct"][d] for d in DOMAINS),
            "insufficient source-correct denominator; no vacuous retention pass")
    reasons, ratios = [], []
    for name, domain in ((donor, "transition"), ("opallo", "arithmetic")):
        ref, value = correct(bundle[name], domain), correct(candidate, domain)
        if ref < bank["donor_min_correct"][name]:
            reasons.append("donor_competence:" + name)
        if ref == 0 or 20 * value < 19 * ref:
            reasons.append("acquisition_below_95pct:" + domain)
        ratios.append(Fraction(value, ref) if ref else Fraction(0))
    for d, stats in retention.items():
        # Protect source-correct items in acquisition domains too; gains never cancel losses.
        if stats["source_correct_lost"] > stats["reference_correct"] // 20:
            reasons.append("source_loss_over_5pct:" + d)
    source_fa, candidate_fa = false_abstentions(source), false_abstentions(candidate)
    if sum(candidate_fa.values()) > sum(source_fa.values()):
        reasons.append("increased_false_abstention")
    ratio = sum(ratios) / 2
    return {"eligible": not reasons, "reasons": reasons, "retention": retention,
            "source_false_abstentions": source_fa, "candidate_false_abstentions": candidate_fa,
            "total_source_correct_losses": sum(x["source_correct_lost"] for x in retention.values()),
            "mean_donor_normalized_acquisition": str(ratio)}


def execute(freeze, directory, phase, lock=None, lock_sha256=None):
    require(phase in ("CAL", "CHECK"), "unknown phase")
    f, freeze_sha = load_freeze(freeze)
    selected = None
    if phase == "CHECK":
        require(lock is not None and sha256(lock) == hash_value(lock_sha256), "missing/changed externally sealed CAL lock")
        saved = read_json(lock)
        require(sha256(lock) == lock_sha256, "CAL lock changed while reading")
        require(saved.get("experiment") == EXPERIMENT and saved.get("phase") == "CAL"
                and saved.get("freeze_sha256") == freeze_sha, "wrong CAL lock")
        selected = saved.get("selected_candidate")
        require(selected in CANDIDATES and saved.get("assessments", {}).get(selected, {}).get("eligible") is True,
                "no eligible locked candidate; CHECK stays closed")
    else:
        require(lock is None and lock_sha256 is None, "CAL never accepts a CHECK/selection lock")
    expected = set(PINS) | (set(NEW_ARMS) if phase == "CAL" else {selected})
    bundle, hashes = load_bundle(directory, expected, f, freeze_sha, phase)
    targets = tuple(CANDIDATES) if phase == "CAL" else (selected,)
    assessments = {arm: assess(arm, bundle, f["banks"][phase]) for arm in targets}
    if phase == "CAL":
        eligible = [arm for arm, result in assessments.items() if result["eligible"]]
        selected = max(eligible, key=lambda arm: (-assessments[arm]["total_source_correct_losses"],
                        Fraction(assessments[arm]["mean_donor_normalized_acquisition"]),
                        arm == "sparse_opallo_50"), default=None)
    comparisons = {}
    for arm in targets:
        refs = set(PINS)
        if phase == "CAL":
            refs |= {"supa_opallo_50", "supa_sparse_50" if arm == "sparse_opallo_50" else "supa_titan_50"}
        comparisons[arm] = {ref: domain_pairs(bundle[ref], bundle[arm]) for ref in sorted(refs)}
    require(sha256(freeze) == freeze_sha, "freeze changed during report")
    require(all(sha256(directory / (arm + ".json")) == value for arm, value in hashes.items()), "runs changed during report")
    return {"experiment": EXPERIMENT, "phase": phase, "freeze_sha256": freeze_sha,
            "selected_candidate": selected, "assessments": assessments, "comparisons": comparisons,
            "run_sha256": hashes, "cal_lock_sha256": lock_sha256, "configuration_count": len(bundle),
            "scored_row_count": sum(len(x) for x in bundle.values()),
            "status": "OFFLINE_EXPLORATORY_SCREEN_NOT_NATIVE_QUALIFICATION",
            "native_execution_authorized": False, "training_authorized": False,
            "quality_improvement_established": False, "release_permitted": False,
            "external_scoring_verified": False, "checkpoint_bytes_reverified_here": False,
            "ancestry_completeness_verified_here": False, "no_prior_CHECK_access_proved": False,
            "attenuation_controls_available_in_this_split": phase == "CAL"}


def write_new(path, result):
    # Exclusive create: refuse an existing lock/report. Preserve a partial file on I/O failure.
    with path.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("CAL", "CHECK"))
    p.add_argument("--freeze", required=True, type=Path)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--lock", type=Path)
    p.add_argument("--lock-sha256")
    a = p.parse_args()
    try:
        result = execute(a.freeze, a.runs, a.phase, a.lock, a.lock_sha256)
        write_new(a.output, result)
        print(json.dumps({"status": result["status"], "selected_candidate": result["selected_candidate"],
                          "output_sha256": sha256(a.output), "native_execution_authorized": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc), "native_execution_authorized": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
