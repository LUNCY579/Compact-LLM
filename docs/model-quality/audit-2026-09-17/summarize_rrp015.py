"""CPU-only, create-only audit of RRP-015; never imports a model runtime.

Rows and identities are checked before an exploratory gate can be called valid.
Missing/incompatible evidence is unavailable, never an observed zero or a pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SELECT_SHA = "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3"
WEIGHTS = {
    "supa": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "opallo": "e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b",
}
FLOORS = {"ledger_replay": 30, "prerequisite_plan": 10, "register_machine": 9}
LABELS = ("supa", "opallo", "control", "rehearsal")
PANEL_SHA = "5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path)
    if not path.is_file():
        return None, {"path": str(path), "status": "MISSING"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value, {"path": str(path), "status": "PRESENT", "sha256": sha(path)}
    except (OSError, ValueError) as exc:
        return None, {"path": str(path), "status": "INVALID", "error": str(exc)}


def safe_summary(function, value):
    try:
        return function(value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "INVALID", "issues": [f"{type(exc).__name__}: {exc}"]}


def sealed_manifest_valid(value):
    if not value or value.get("status") != "COMPLETE":
        return False
    body = {k: v for k, v in value.items() if k != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return (hashlib.sha256(encoded).hexdigest() == value.get("sha256")
            and len(value.get("entries", [])) == 220
            and len({e.get("name") for e in value["entries"]}) == 220)


def cp(k, n):
    def tail(x, count):
        return sum(math.comb(n, i) * x**i * (1 - x)**(n-i) for i in range(count, n+1))
    def bound(count, target):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo+hi)/2
            if tail(mid, count) < target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0.0 if k == 0 else bound(k, .025),
            1.0 if k == n else bound(k+1, .975)]


def process_summary(value, expected_rows=None):
    issues, families = [], {}
    if value.get("views_sha256") != SELECT_SHA or value.get("instrument") != "select-views-001":
        issues.append("wrong selection instrument identity")
    if value.get("instrument_admits_selection") is not True:
        issues.append("instrument is not admitted for selection")
    rows = value.get("predictions", [])
    ids = [r.get("id") for r in rows]
    if len(rows) != 384 or len(set(ids)) != 384 or value.get("distinct_parents") != 384:
        issues.append("selection output is incomplete or has duplicate identities")
    if expected_rows is not None:
        expected = {r["id"]: (r["family"], r["reference"]) for r in expected_rows}
        if {r.get("id"): (r.get("family"), r.get("reference")) for r in rows} != expected:
            issues.append("prediction identities/references differ from frozen selection views")
    counts = Counter(r.get("family") for r in rows)
    if counts != {f: 128 for f in FLOORS}:
        issues.append("selection family denominators differ from 128 each")
    for family, floor in FLOORS.items():
        group = [r for r in rows if r.get("family") == family]
        k, strict, terminated = 0, 0, 0
        for row in group:
            text = row.get("prediction")
            if not isinstance(text, str) or type(row.get("terminated")) is not bool:
                issues.append(f"malformed prediction in {family}")
                continue
            try:
                canonical = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
                correct = canonical == row["reference"]
            except (ValueError, TypeError):
                correct = False
            exact = text == row.get("reference")
            if row.get("canonical") != correct or row.get("strict") != exact:
                issues.append(f"stored scoring differs from raw prediction in {family}")
            k += correct
            strict += exact
            terminated += row["terminated"]
        interval = cp(k, len(group)) if group else None
        published = value.get("per_family", {}).get(family, {})
        if published.get("binding_floor_count") != floor or published.get("n") != 128:
            issues.append(f"published floor/denominator differs for {family}")
        if published.get("canonical_correct") != k:
            issues.append(f"aggregate differs for {family}")
        families[family] = {"rows": len(group), "canonical_correct": k,
                            "strict_correct": strict, "terminated": terminated,
                            "floor_count": floor, "floor_denominator": 128,
                            "cp95": interval,
                            "clears_floor": interval[0] > floor/128 if interval else None}
    return {"status": "VALID" if not issues else "INCOMPATIBLE", "issues": sorted(set(issues)),
            "rows": len(rows), "per_family": families,
            "canonical_correct": sum(f["canonical_correct"] for f in families.values()),
            "terminated": sum(f["terminated"] for f in families.values()),
            "all_families_clear": all(f["clears_floor"] is True for f in families.values()),
            "scorer_limitation": "Historical canonical correctness does not require EOS; termination is separate."}


def retention_summary(value):
    issues, policies = [], {}
    if value.get("probe_rows") != 150 or value.get("probe_sha256") != PANEL_SHA:
        issues.append("arithmetic panel identity or denominator differs")
    for policy, expected in (("permissive", 47), ("strict", 41)):
        item = value.get("by_bare_policy", {}).get(policy)
        if not isinstance(item, dict):
            issues.append(f"missing {policy} policy")
            continue
        source, correct, gained, lost = (item.get(k) for k in (
            "source_correct_total", "arm_correct_total", "arm_correct_outside_source", "lost_total"))
        if any(type(k) is not int for k in (source, correct, gained, lost)):
            issues.append(f"invalid {policy} counts")
            continue
        retained = source-lost
        if source != expected or correct != retained+gained or min(retained, gained, lost) < 0:
            issues.append(f"inconsistent {policy} source-paired arithmetic counts")
        classes = item.get("per_class", {})
        if sum(c.get("rows", 0) for c in classes.values()) != 150:
            issues.append(f"incomplete {policy} classes")
        policies[policy] = {"correct": correct, "rows": 150, "source_correct": source,
                            "source_retained": retained, "gained_vs_source": gained,
                            "lost_vs_source": lost,
                            "terminated": sum(c.get("terminated", 0) for c in classes.values()),
                            "legacy_class_monitor_status": item.get("status")}
    return {"status": "VALID" if not issues else "INCOMPATIBLE", "issues": issues,
            "by_policy": policies,
            "note": "Retained=source_correct-lost; union-reference retained/gained fields are not source retention."}


def long_summary(value, source=None):
    issues = []
    predictions, correct = value.get("predictions", {}), value.get("per_row_correct", {})
    if (value.get("rows_total") != 256 or len(predictions) != 256 or len(correct) != 256
            or set(predictions) != set(correct) or value.get("rows_skipped_context_overflow") != 0):
        issues.append("long-context output incomplete or contains refused rows")
    if any(type(x) is not bool for x in correct.values()) or sum(correct.values()) != value.get("all256_hits"):
        issues.append("long-context correctness counts disagree")
    unknown = lambda r: r.get("prediction", "").strip() == "UNKNOWN"
    unanswerable = {i for i in predictions if i.endswith("-U")}
    if len(unanswerable) != 64:
        issues.append("unanswerable identities differ from 64 expected suite variants")
    paired = None
    if source:
        prior = source.get("per_row_correct", {})
        keys = ("implementation_digest", "max_context", "max_new_tokens", "numerics")
        if set(prior) != set(correct) or any(source.get(k) != value.get(k) for k in keys):
            issues.append("historical long-context source is incompatible; no paired claim")
        elif len(prior) == 256 and source.get("rows_skipped_context_overflow") == 0:
            paired = {"source_correct": sum(prior.values()),
                      "retained": sum(prior[i] and correct[i] for i in prior),
                      "gained": sum(not prior[i] and correct[i] for i in prior),
                      "lost": sum(prior[i] and not correct[i] for i in prior)}
    return {"status": "VALID_DIAGNOSTIC" if not issues else "INCOMPATIBLE", "issues": issues,
            "correct": value.get("all256_hits"), "rows": value.get("rows_total"),
            "terminated": sum(r.get("terminated") is True for r in predictions.values()),
            "literal_UNKNOWN_unanswerable": sum(unknown(predictions[i]) for i in unanswerable),
            "literal_false_abstention_answerable": sum(unknown(r) for i, r in predictions.items() if i not in unanswerable),
            "legacy_substring_UNKNOWN_unanswerable": value.get("says_unknown_on_unanswerable"),
            "paired_vs_historical_source": paired,
            "admissible_use": "REPORT_ONLY; barred from selection, stopping on efficacy, and confirmation",
            "semantic_abstention_review": "NOT_PERFORMED"}


def training_summary(records, arm):
    directory = records / f"{arm}-journal" / "runs"
    attempts = []
    for run in sorted(directory.iterdir()) if directory.is_dir() else []:
        files, values = {}, {}
        for name in ("binding", "qualification", "result", "run_summary", "supervisor", "failure"):
            values[name], files[name] = artifact(run / f"{name}.json")
        binding = (values["binding"] or {}).get("binding", {})
        q, result, supervisor = values["qualification"] or {}, values["result"] or {}, values["supervisor"] or {}
        valid = (q.get("status") == "PASS" and q.get("optimizer_steps") == 0
                 and q.get("binding") == binding and result.get("binding") == binding
                 and binding.get("source", {}).get("assets", {}).get("model.safetensors") == WEIGHTS["supa"]
                 and result.get("complete") is True and supervisor.get("returncode") == 0
                 and values["run_summary"] is not None and values["failure"] is None
                 and not values["run_summary"].get("budget_after", {}).get("budget_breached", True))
        attempts.append({"run": run.name, "status": "COMPLETE_VERIFIED" if valid else "FAILED_OR_INCOMPLETE",
                         "artifacts": files, "binding": binding, "progress": result.get("progress"),
                         "qualification": {k: q.get(k) for k in ("status", "optimizer_steps", "qualified_training_shape_tokens")},
                         "supervisor": values["supervisor"], "failure": values["failure"],
                         "budget_after": (values["run_summary"] or {}).get("budget_after")})
    return {"status": "COMPLETE_VERIFIED" if any(a["status"] == "COMPLETE_VERIFIED" for a in attempts) else "MISSING_OR_INCOMPLETE",
            "attempts": attempts}


def model_identity(value, root, records, name, *, retention_uses_hf=False):
    directory = Path(value.get("model_dir", ""))
    if not directory.is_absolute():
        directory = root / directory
    weight = directory / "model.safetensors"
    identity = {"model_directory": str(directory), "weight_sha256": None,
                "source_weight_sha256": WEIGHTS.get(name, WEIGHTS["supa"]),
                "training_dtype": None if name in WEIGHTS else "float32",
                "candidate_export_path": "BF16-rounded weights evaluated with FP32 DenseReference",
                "native_BF16_serving_qualified": False, "issues": []}
    if weight.is_file():
        identity["weight_sha256"] = sha(weight)
        with weight.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if size > 64 * 2**20:
                raise ValueError("unreasonable safetensors header")
            header = json.loads(handle.read(size))
        identity["stored_tensor_dtypes"] = sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})
    if name in WEIGHTS:
        if identity["weight_sha256"] != WEIGHTS[name]:
            identity["issues"].append("published model bytes missing or hash differs")
    else:
        receipts = {}
        for kind in ("hf", "research"):
            receipt, meta = artifact(records / f"{name}-{kind}" / "MATERIALIZATION.json")
            custody, custody_meta = artifact(records / f"{name}-{kind}" / "CUSTODY.json")
            receipts[kind] = {"artifact": meta, "receipt": receipt,
                              "custody_artifact": custody_meta, "custody": custody}
        identity["materializations"] = receipts
        checkpoint = value.get("checkpoint")
        if checkpoint:
            manifest_path = Path(checkpoint["path"]) / "manifest.json"
            manifest, meta = artifact(manifest_path)
            identity["checkpoint_manifest"] = meta
            if (not sealed_manifest_valid(manifest) or meta.get("sha256") != checkpoint.get("manifest_sha256")
                    or manifest.get("binding", {}).get("source", {}).get("assets", {}).get("model.safetensors") != WEIGHTS["supa"]):
                identity["issues"].append("checkpoint/source binding is missing or incompatible")
            if identity["weight_sha256"] != WEIGHTS["supa"]:
                identity["issues"].append("checkpoint overlay source is not exact Supa")
        else:
            receipt = receipts["hf"]["receipt"] or {}
            manifest_path = Path(receipt.get("checkpoint", "")) / "manifest.json"
            manifest, meta = artifact(manifest_path)
            identity["checkpoint_manifest"] = meta
            if (not sealed_manifest_valid(manifest) or meta.get("sha256") != receipt.get("checkpoint_manifest_sha256")
                    or receipt.get("source_weight_sha256") != WEIGHTS["supa"]
                    or manifest.get("binding", {}).get("source", {}).get("assets", {}).get("model.safetensors") != WEIGHTS["supa"]
                    or not receipt.get("out_weight_sha256")
                    or (identity["weight_sha256"] is not None and identity["weight_sha256"] != receipt["out_weight_sha256"])):
                identity["issues"].append("materialized model/checkpoint/source linkage is missing or incompatible")
            identity["custody"] = "LIVE_BYTES" if weight.is_file() else "RECEIPT_ONLY_DERIVED_BYTES_ABSENT"
        expected_manifest = identity.get("checkpoint_manifest", {}).get("sha256")
        for kind, entry in receipts.items():
            if kind == "research" and retention_uses_hf:
                continue
            receipt = entry["receipt"] or {}
            custody = entry["custody"] or {}
            if (receipt.get("source_weight_sha256") != WEIGHTS["supa"]
                    or receipt.get("checkpoint_manifest_sha256") != expected_manifest
                    or receipt.get("tensors_applied") != 220 or not receipt.get("out_weight_sha256")):
                identity["issues"].append(f"{kind} materialization has no verified common checkpoint linkage")
            if (custody.get("source_binding_verified_against_checkpoint") is not True
                    or custody.get("materialization_sha256") != entry["artifact"].get("sha256")
                    or custody.get("checkpoint_manifest_sha256") != expected_manifest
                    or custody.get("source_weight_sha256") != WEIGHTS["supa"]
                    or custody.get("output_weight_sha256") != receipt.get("out_weight_sha256")
                    or custody.get("stored_dtype_inventory") != (["BF16"] if kind == "hf" else ["F32"])
                    or custody.get("evaluation_dtype") != "float32"):
                identity["issues"].append(f"{kind} source/dtype custody is absent or inconsistent")
        identity["checkpoint_binding"] = manifest.get("binding") if manifest else None
        identity["checkpoint_progress"] = manifest.get("progress") if manifest else None
        identity["candidate_export_weight_sha256"] = (receipts["hf"]["receipt"] or {}).get("out_weight_sha256")
    identity["status"] = "VERIFIED" if not identity["issues"] else "UNVERIFIED"
    return identity


def collect(evidence, records):
    root = evidence.resolve().parents[3]
    preparation, prep_meta = artifact(records / "PREPARATION.json")
    view_path = records.parent / "select-views-001" / "SELECT_VIEWS.jsonl"
    views = None
    instrument_issues = []
    if view_path.is_file() and sha(view_path) == SELECT_SHA:
        views = [json.loads(line) for line in view_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        instrument_issues.append("frozen selection views unavailable or hash differs")
    source_long, source_long_meta = artifact(evidence / "LONGCTX_supa.json")
    result = {"record": "RRP015_CPU_RESULTS_AUDIT", "created_utc": datetime.now(timezone.utc).isoformat(),
              "model_calls": 0, "broad_improvement_proven": False, "release_permission": False,
              "preparation": {"artifact": prep_meta, "receipt": preparation},
              "instrument_issues": instrument_issues, "historical_long_context_source": source_long_meta,
              "models": {}, "evaluation_attempts": []}
    process_values = {}
    for name in LABELS:
        value, meta = artifact(evidence / f"PROCESS_rrp015_{name}.json")
        model = {"process_artifact": meta, "process": None, "identity": None}
        if value:
            process_values[name] = value
            model["process"] = safe_summary(lambda x: process_summary(x, views), value)
            try:
                prior_retention, _ = artifact(evidence / f"RETENTION_PROBE_rrp015_{name}.json")
                uses_hf = bool(prior_retention and str(prior_retention.get("target", "")).startswith("hf:"))
                model["identity"] = model_identity(value, root, records, name, retention_uses_hf=uses_hf)
            except (OSError, ValueError, KeyError) as exc:
                model["identity"] = {"status": "UNVERIFIED", "issues": [str(exc)]}
        if name not in WEIGHTS:
            model["training"] = training_summary(records, name)
            for kind, prefix, summarize in (("retention", "RETENTION_PROBE", retention_summary),
                                             ("long_context", "LONGCTX", lambda x: long_summary(x, source_long))):
                raw, info = artifact(evidence / f"{prefix}_rrp015_{name}.json")
                model[f"{kind}_artifact"] = info
                model[kind] = safe_summary(summarize, raw) if raw else None
                if raw:
                    declared = raw.get("target") if kind == "retention" else raw.get("model_directory")
                    hf_retention = kind == "retention" and isinstance(declared, str) and declared.startswith("hf:")
                    target = Path(declared[3:] if hf_retention else declared) if declared else None
                    if target and not target.is_absolute():
                        target = root / target
                    expected = records / f"{name}-{'research' if kind == 'retention' and not hf_retention else 'hf'}"
                    if not target or target.resolve() != expected.resolve():
                        model[kind]["issues"].append("measurement target differs from the bound materialization")
                        model[kind]["status"] = "INCOMPATIBLE"
                    if hf_retention:
                        observed = raw.get("hf_candidate_identity") or {}
                        bound = ((model.get("identity") or {}).get("materializations") or {}).get("hf") or {}
                        receipt = bound.get("receipt") or {}
                        if (observed.get("source_weight_sha256") != WEIGHTS["supa"]
                                or observed.get("weight_sha256") != receipt.get("out_weight_sha256")
                                or observed.get("materialization_sha256") != (bound.get("artifact") or {}).get("sha256")
                                or observed.get("checkpoint_manifest_sha256") != receipt.get("checkpoint_manifest_sha256")
                                or Path(observed.get("directory", "")).resolve() != expected.resolve()):
                            model[kind]["issues"].append("HF retention identity is not the bound candidate export")
                            model[kind]["status"] = "INCOMPATIBLE"
        result["models"][name] = model
    source = process_values.get("supa")
    for name in LABELS:
        model = result["models"][name]
        compatible = bool(source and model["process"] and model["process"]["status"] == "VALID"
                          and result["models"]["supa"]["process"]["status"] == "VALID")
        if compatible:
            for key in ("views_sha256", "implementation_digest", "runtime", "numerics", "max_new_tokens"):
                if not source.get(key) or source.get(key) != process_values[name].get(key):
                    compatible = False
        model["compatible_with_complete_source_process"] = compatible
        if name not in WEIGHTS:
            panel = ((model.get("retention") or {}).get("by_policy") or {}).get("permissive")
            observed = None if not panel or not model["process"] or "all_families_clear" not in model["process"] else (
                panel["correct"] >= 62 and panel["source_retained"] >= 37 and model["process"]["all_families_clear"])
            eligible = (compatible and not instrument_issues and model["identity"]["status"] == "VERIFIED"
                        and result["models"]["supa"].get("identity", {}).get("status") == "VERIFIED"
                        and model["training"]["status"] == "COMPLETE_VERIFIED"
                        and model.get("retention", {}).get("status") == "VALID") if model["process"] and model.get("retention") else False
            model["gate_v2"] = {"observed_conjunction": observed,
                                 "validated_exploratory_result": observed if eligible else None,
                                 "evidence_complete_and_compatible": bool(eligible),
                                 "broad_preservation_or_confirmation": False}
    for directory in sorted(records.glob("eval-*")):
        values, metas = {}, {}
        for item in ("probe_binding", "probe_terminal", "supervisor", "retention_predictions"):
            values[item], metas[item] = artifact(directory / f"{item}.json")
        probe_binding, terminal = values["probe_binding"] or {}, values["probe_terminal"] or {}
        declared_result = probe_binding.get("result")
        expected_prefix = {"process": "PROCESS", "retention": "RETENTION_PROBE",
                           "long_context": "LONGCTX"}.get(probe_binding.get("kind"))
        matches_result = False
        if declared_result and expected_prefix:
            result_path = Path(declared_result).resolve()
            expected_names = {f"{expected_prefix}_rrp015_{label}.json" for label in LABELS}
            matches_result = (result_path.name in expected_names
                              and result_path.parent == evidence.resolve()
                              and terminal.get("result") is not None
                              and Path(terminal["result"]).resolve() == result_path)
        result["evaluation_attempts"].append({"attempt": directory.name, "artifacts": metas,
            "binding": values["probe_binding"], "terminal": values["probe_terminal"],
            "supervisor": values["supervisor"],
            "retained_prediction_rows": (values["retention_predictions"] or {}).get("rows"),
            "kind_label_and_result_path_match": matches_result,
            "complete_supervised": bool(values["probe_terminal"] and values["supervisor"]
                and values["probe_binding"]
                and values["probe_terminal"].get("complete") is True
                and values["probe_terminal"].get("script_unchanged") is True
                and values["probe_terminal"].get("script_sha256_after") == values["probe_binding"].get("script_sha256")
                and values["supervisor"].get("returncode") == 0)})
    verified_evaluations = {Path(a["binding"]["result"]).name: a for a in result["evaluation_attempts"]
                            if a["complete_supervised"] and a["kind_label_and_result_path_match"]}
    for name in ("control", "rehearsal"):
        model = result["models"][name]
        expected_outputs = [f"PROCESS_rrp015_{name}.json", "PROCESS_rrp015_supa.json",
                            f"RETENTION_PROBE_rrp015_{name}.json"]
        complete_supervision = all(key in verified_evaluations for key in expected_outputs)
        retention_attempt = verified_evaluations.get(expected_outputs[-1], {})
        complete_supervision &= retention_attempt.get("retained_prediction_rows") == 150
        model["gate_v2"]["complete_supervised_evaluation"] = complete_supervision
        # The evaluation's checkpoint must be the same complete state that the
        # qualified, supervised trainer recorded, rather than another valid state.
        checkpoint_binding = (model.get("identity") or {}).get("checkpoint_binding")
        checkpoint_progress = (model.get("identity") or {}).get("checkpoint_progress")
        linked_training = checkpoint_binding is not None and any(
            a["status"] == "COMPLETE_VERIFIED" and a["binding"] == checkpoint_binding
            and a["progress"] == checkpoint_progress for a in model["training"]["attempts"])
        model["gate_v2"]["training_checkpoint_binding_matches"] = linked_training
        if not complete_supervision or not linked_training:
            model["gate_v2"]["validated_exploratory_result"] = None
            model["gate_v2"]["evidence_complete_and_compatible"] = False
    result["session_resources"] = {
        "observed_supervisor_wall_seconds": sum((a.get("supervisor") or {}).get("wall_seconds", 0)
            for a in result["evaluation_attempts"]) + sum((a.get("supervisor") or {}).get("wall_seconds", 0)
            for name in ("control", "rehearsal") for a in result["models"][name]["training"]["attempts"]),
        "registered_model_work_seconds_ceiling": 8400,
        "storage_amendment": "18 GiB retained high-water; 22 GiB cumulative writes; compare with custody receipts, not final directory size",
        "disk_and_total_owned_work_ceiling_verified": False,
        "note": "Missing terminal receipts leave resource accounting incomplete; zeros are not inferred for absent attempts."}
    result["limitations"] = ["One training draw; exploratory within-generator skill acquisition only.",
        "Gate v2 requires arithmetic improvement and retention plus acquisition; it is not comprehensive preservation.",
        "The inherited permissive arithmetic policy remains pending review; strict results are reported alongside it.",
        "Long-context and abstention are report-only spent instruments and must not select a candidate.",
        "Materialization/source hashes establish byte linkage, not independent cryptographic proof of model execution.",
        "Historical retention outputs contain only 24 examples; complete generator returns are preserved separately when available."]
    return result


def markdown(result):
    lines = ["# RRP-015 results audit", "", "Exploratory evidence only. No broad improvement or release claim.", "",
             "| Model | Acquisition | Families clearing floors | Terminated | Arithmetic (permissive) | Source retained | Gate v2 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, model in result["models"].items():
        p = model.get("process")
        if p and "canonical_correct" not in p:
            p = None
        arithmetic = ((model.get("retention") or {}).get("by_policy") or {}).get("permissive", {})
        gate = model.get("gate_v2", {}).get("validated_exploratory_result")
        acquisition = f"{p['canonical_correct']}/{p['rows']} ({p['status']})" if p else "MISSING/INVALID"
        lines.append(f"| {name} | {acquisition} | "
                     f"{sum(f['clears_floor'] is True for f in p['per_family'].values()) if p else 'unavailable'} | "
                     f"{p['terminated'] if p else 'unavailable'} | {arithmetic.get('correct', 'unavailable')} | "
                     f"{arithmetic.get('source_retained', 'unavailable')} | {'PASS' if gate is True else 'FAIL' if gate is False else 'UNAVAILABLE'} |")
    for name, model in result["models"].items():
        lines += ["", f"## {name}", "", f"Source compatibility: {model['compatible_with_complete_source_process']}."]
        if model.get("process") and "per_family" in model["process"]:
            for family, item in model["process"]["per_family"].items():
                lines.append(f"- {family}: {item['canonical_correct']}/{item['rows']}; floor {item['floor_count']}/128; "
                             f"CP95 {item['cp95']}; clears={item['clears_floor']}.")
            lines.append(f"- Process audit: {model['process']['status']}; {model['process']['issues']}.")
        identity = model.get("identity") or {}
        lines.append(f"- Identity: {identity.get('status', 'MISSING')}; weight SHA {identity.get('weight_sha256')}; "
                     f"checkpoint manifest SHA {identity.get('checkpoint_manifest', {}).get('sha256')}; issues {identity.get('issues', [])}.")
        lines.append("- Precision: source/export BF16 rounded weights, FP32 DenseReference evaluation; no native BF16 serving qualification.")
        if name not in WEIGHTS:
            lines.append(f"- Arithmetic, both policies: {json.dumps(model.get('retention'), sort_keys=True)}")
            lines.append(f"- Long context, report only: {json.dumps(model.get('long_context'), sort_keys=True)}")
            lines.append(f"- Training status: {model['training']['status']}; {len(model['training']['attempts'])} retained attempts.")
    lines += ["", "## Limits", "", *[f"- {x}" for x in result["limitations"]], "",
              "Full artifact hashes, attempt failures, qualification/source bindings, and budget receipts are in SUMMARY.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir or args.records / "summary-001"
    if out.exists():
        raise FileExistsError(f"create-only summary already exists: {out}")
    result = collect(args.evidence_dir, args.records)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "SUMMARY.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (out / "SUMMARY.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown(result))
    print(json.dumps({"summary": str(out), "model_calls": 0, "broad_improvement_proven": False}))


if __name__ == "__main__":
    main()
