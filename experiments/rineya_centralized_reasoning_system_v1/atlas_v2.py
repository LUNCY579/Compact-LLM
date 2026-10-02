"""Versioned schema/binding and descriptive estimates; no native attestation.

This module does not launch, read model assets, grant qualification or promote.
Receipt bytes can bind a declared measurement to a row; they cannot attest that
a model really executed. Actual truth and eligibility belong to reviewed runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re

VERSION = "CRS-LEARNING-INSTRUMENTS-v2"
REQUIRED_ROW_FIELDS = frozenset({
    "schema_version", "study", "recipe", "checkpoint_sha256", "source_checkpoint_sha256", "evidence_class",
    "family", "composition_depth", "train_draw_seed", "order_seed", "eval_family_seed", "initialization_seed",
    "unique_parents", "presentations", "processed_tokens", "supervised_tokens", "updates",
    "worker_seconds", "peak_ram_bytes", "peak_vram_bytes", "optimizer", "update_scope",
    "data_sha256", "scorer_sha256", "qualification_sha256", "measurement_receipt_sha256", "code_commit", "exposure",
    "acquisition", "transfer", "retention", "calibration", "failures", "stop_reason"})
METRICS = ("acquisition", "transfer", "retention", "calibration")
COUNTS = ("unique_parents", "presentations", "processed_tokens", "supervised_tokens", "updates", "failures")
HASHES = ("checkpoint_sha256", "source_checkpoint_sha256", "data_sha256", "scorer_sha256", "qualification_sha256", "measurement_receipt_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return type(value) is str and bool(value.strip()) and value == value.strip()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json_finite(value, depth=0):
    _require(depth <= 16, "metadata nesting ceiling")
    if type(value) is dict:
        _require(all(type(k) is str for k in value), "metadata JSON keys")
        for child in value.values():
            _json_finite(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _json_finite(child, depth + 1)
    elif type(value) is float:
        _require(math.isfinite(value), "nonfinite metadata")
    else:
        _require(value is None or type(value) in (str, int, bool), "metadata JSON types")


def _canonical(value):
    _json_finite(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def validate_row(row):
    """Validate declared schema only; never attest execution/qualification."""
    _require(type(row) is dict and set(row) == REQUIRED_ROW_FIELDS, "complete version2 atlas row required")
    _json_finite(row)
    _require(type(row["schema_version"]) is int and row["schema_version"] == 2, "atlas row version2 required")
    _require(row["evidence_class"] in {"NATIVE_EXPLORATORY", "NATIVE_CONFIRMATION", "UNMEASURED"}, "unregistered evidence class")
    for name in COUNTS:
        _require(type(row[name]) is int and row[name] >= 0, "nonnegative actual integer: " + name)
    _require(row["unique_parents"] <= row["presentations"], "unique parents exceed presentations")
    _require(row["updates"] <= row["presentations"], "updates exceed presentations")
    _require(row["presentations"] == 0 or (row["unique_parents"] > 0 and row["processed_tokens"] >= row["presentations"]), "presented examples need actual parent/token exposure")
    _require(row["updates"] == 0 or row["supervised_tokens"] >= row["updates"], "supervised adaptation updates need target exposure")
    _require(row["supervised_tokens"] <= row["processed_tokens"], "supervised tokens exceed processed tokens")
    for name in ("worker_seconds", "peak_ram_bytes", "peak_vram_bytes"):
        _require(row[name] is None or (_finite(row[name]) and row[name] >= 0), "invalid resource: " + name)
    for name in METRICS:
        metric = row[name]
        if metric is not None:
            _require(type(metric) is dict and set(metric) == {"estimate", "interval95", "scope"}, "metric schema")
            interval = metric["interval95"]
            _require(type(interval) is list and len(interval) == 2 and _text(metric["scope"]), "metric interval/scope")
            values = [metric["estimate"], *interval]
            _require(all(_finite(v) and 0 <= v <= 1 for v in values) and interval[0] <= metric["estimate"] <= interval[1], "proportion estimate/interval inconsistent")
    if row["evidence_class"] == "UNMEASURED":
        _require(all(row[m] is None for m in METRICS), "unmeasured row cannot contain measured metrics")
        return row
    for name in ("study", "recipe", "family", "update_scope", "exposure", "stop_reason"):
        _require(_text(row[name]), "explicit native metadata required: " + name)
    _require(type(row["composition_depth"]) is int and row["composition_depth"] >= 1, "positive explicit composition depth required")
    for name in ("train_draw_seed", "order_seed", "eval_family_seed", "initialization_seed"):
        _require(type(row[name]) is int and row[name] >= 0, "explicit nonnegative seed required: " + name)
    for name in HASHES:
        _require(type(row[name]) is str and re.fullmatch(r"[a-f0-9]{64}", row[name]) is not None, "SHA256 binding required: " + name)
    _require(type(row["code_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", row["code_commit"]) is not None, "exact code commit required")
    _require(row["acquisition"] is not None and row["worker_seconds"] is not None, "native row lacks acquisition/cost")
    _require(type(row["optimizer"]) is dict and bool(row["optimizer"]), "explicit recipe required")
    return row


def measurement_binding(row, receipt_bytes, *, expected_gain):
    """Associate exact bytes and declared fields; not execution attestation.

    Envelope: {schema_version:2,kind:'CRS_LA_MEASUREMENT',payload:{row:...
    excluding measurement_receipt_sha256,gain:...}}. Caller assertions of status
    or a hash-shaped string cannot substitute for the actual bound bytes.
    """
    validate_row(row)
    _require(type(receipt_bytes) is bytes and len(receipt_bytes) <= 262144, "bounded measurement receipt bytes required")
    _require(hashlib.sha256(receipt_bytes).hexdigest() == row["measurement_receipt_sha256"], "measurement receipt byte hash mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate measurement receipt key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite receipt constant: " + value)
    value = json.loads(receipt_bytes, object_pairs_hook=unique, parse_constant=bad)
    _json_finite(value)
    _require(type(value) is dict and set(value) == {"schema_version", "kind", "payload"}
             and type(value["schema_version"]) is int and value["schema_version"] == 2
             and value["kind"] == "CRS_LA_MEASUREMENT", "measurement receipt schema")
    payload = value["payload"]
    _require(type(payload) is dict and set(payload) == {"row", "gain"}, "measurement payload schema")
    expected = {k: v for k, v in row.items() if k != "measurement_receipt_sha256"}
    _require(_canonical(payload["row"]) == _canonical(expected) and payload["gain"] == expected_gain and
             (expected_gain is None or _finite(payload["gain"])), "measurement/dose/gain binding mismatch")
    return {"bytes_bound": True, "truth_attested": False, "qualification_granted": False}


def efficiency(gain, row, *, measurement_receipt=None):
    validate_row(row)
    _require(gain is None or (_finite(gain) and -1 <= gain <= 1), "gain must be an accuracy proportion difference")
    _require(gain is None or row["evidence_class"] != "UNMEASURED", "unmeasured row cannot establish numeric efficiency")
    binding = None
    if gain is not None:
        binding = measurement_binding(row, measurement_receipt, expected_gain=gain)
    ratios = {name: gain / row[name] if gain is not None and row[name] not in (None, 0) else None
              for name in ("unique_parents", "processed_tokens", "supervised_tokens", "updates", "worker_seconds")}
    return {"ratios": ratios, "measurement_binding": binding, "native_truth_attested": False,
            "promotion_allowed": False, "scope": "declared bound measurements only; no universal frontier or preservation proof"}


def paired_parent_interval(records, *, expected_cases=None, resamples=2000, seed=20261001):
    _require(type(records) is list and bool(records), "nonempty paired records required")
    _require(type(resamples) is int and 100 <= resamples <= 10000 and type(seed) is int and seed >= 0, "bounded resamples and explicit seed")
    parents, cases, variants = {}, set(), set()
    required = {"case_id", "variant_id", "parent_id", "family", "B", "C"}
    for row in records:
        _require(type(row) is dict and set(row) == required, "paired version2 record schema")
        _require(all(_text(row[k]) for k in ("case_id", "variant_id", "parent_id", "family")), "paired identities required")
        _require(row["case_id"] not in cases and (row["parent_id"], row["variant_id"]) not in variants, "duplicate case or parent variant")
        cases.add(row["case_id"]); variants.add((row["parent_id"], row["variant_id"]))
        for arm in ("B", "C"):
            _require(type(row[arm]) is bool or (_finite(row[arm]) and 0 <= row[arm] <= 1), "finite paired score in [0,1]")
        parent = parents.setdefault(row["parent_id"], {"family": row["family"], "deltas": []})
        _require(parent["family"] == row["family"], "parent changes family")
        parent["deltas"].append(float(row["C"]) - float(row["B"]))
        if expected_cases is not None:
            _require(type(expected_cases) is dict and row["case_id"] in expected_cases, "unregistered paired case")
            _require(expected_cases[row["case_id"]] == {k: row[k] for k in ("parent_id", "variant_id", "family")}, "paired bank identity mismatch")
    if expected_cases is not None:
        _require(set(expected_cases) == cases, "missing or extra scheduled paired cases")
    strata = {}
    for parent in parents.values():
        strata.setdefault(parent["family"], []).append(sum(parent["deltas"]) / len(parent["deltas"]))
    count = len(parents)
    observed = sum(sum(v) for v in strata.values()) / count
    rng = random.Random(seed)
    samples = sorted(sum(sum(rng.choice(v) for _ in v) for v in strata.values()) / count for _ in range(resamples))
    return {"delta": observed, "parent_count": count, "variant_count": len(records), "family_count": len(strata),
            "interval95": [samples[int(.025 * resamples)], samples[int(.975 * resamples) - 1]],
            "seed": seed, "resamples": resamples, "membership_bound": expected_cases is not None,
            "scope": "descriptive whole-parent bootstrap conditional on supplied family mix; no seed-population or untouched-confirmation inference",
            "few_clusters_warning": any(len(v) < 10 for v in strata.values()), "noninferiority_established": False,
            "native_truth_attested": False, "automatic_promotion": False}


def skill_matrices(checkpoints, source, incumbent=None):
    _require(type(checkpoints) is dict and type(source) is dict, "fixed skill panels required")
    panels = [("source", source), *checkpoints.items()]
    if incumbent is not None:
        panels.append(("incumbent", incumbent))
    for label, panel in panels:
        _require(type(panel) is dict and set(panel) == set(source), "skill membership changed: " + str(label))
        for skill, items in panel.items():
            _require(_text(skill) and type(items) is dict and set(items) == set(source[skill]), "fixed item membership required")
            _require(all(_text(k) and type(v) is bool for k, v in items.items()), "explicit item IDs and strict semantic booleans required")
    performance, forgetting = {}, {}
    for checkpoint, panel in checkpoints.items():
        _require(_text(checkpoint), "checkpoint label required")
        performance[checkpoint], forgetting[checkpoint] = {}, {}
        for skill, items in panel.items():
            n = len(items); src = source[skill]; correct = sum(src.values())
            performance[checkpoint][skill] = sum(items.values()) / n if n else None
            inc = None if incumbent is None else incumbent[skill]
            forgetting[checkpoint][skill] = {
                "scheduled_items": n, "source_losses": sum(src[k] and not items[k] for k in items),
                "source_repairs": sum(not src[k] and items[k] for k in items), "source_correct_denominator": correct,
                "source_correct_retention": sum(src[k] and items[k] for k in items) / correct if correct else None,
                "incumbent_status": "UNAVAILABLE" if inc is None else "DECLARED_PANEL",
                "incumbent_losses": None if inc is None else sum(inc[k] and not items[k] for k in items),
                "incumbent_correct_denominator": None if inc is None else sum(inc.values()),
                "incumbent_correct_retention": None if inc is None or not sum(inc.values()) else sum(inc[k] and items[k] for k in items) / sum(inc.values()),
                "backward_transfer_vs_source": (sum(items.values()) - sum(src.values())) / n if n else None,
                "noninferiority_established": False}
    worst = min((r["backward_transfer_vs_source"] for rows in forgetting.values() for r in rows.values()
                 if r["backward_transfer_vs_source"] is not None), default=None)
    return {"performance": performance, "forgetting": forgetting, "worst_backward_transfer": worst,
            "incumbent_available": incumbent is not None, "native_truth_attested": False,
            "identity_binding": "source/incumbent/checkpoint identities require external immutable run receipts"}
