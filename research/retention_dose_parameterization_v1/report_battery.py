"""Offline, fail-closed report of the fixed DOSEINT-001 battery (stdlib only).

All 16 execution legs and cross-arm identities are gated before endpoints are
extracted. A failure returns BLOCKED_ON_COMPARABILITY without endpoint numbers.
The only dynamically loaded scorer is the pure, torch-free retention.py module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.retention_dose_parameterization_v1 import paired, run_probe

ARMS = ("a100", "a025", "a050", "a075")
CANDIDATES = ARMS[1:]
ALPHAS = {"a100": 1.0, "a025": 0.25, "a050": 0.5, "a075": 0.75}
LEGS = {"dose": "process", "select": "process", "retention": "retention", "long_context": "long_context"}
DOSE_BANDS = {"a100": (284, 324), "a025": (220, 285), "a050": (260, 310), "a075": (280, 320)}
SELECT_BANDS = {"a100": (270, 330), "a025": (210, 285), "a050": (250, 310), "a075": (270, 320)}
RETAINED_BANDS = {"a100": (24, 36), "a025": (40, 47), "a050": (37, 44), "a075": (31, 40)}
LIMITATIONS = [
    "Exploratory development evidence; no release, promotion, model selection, or general-intelligence claim.",
    "The 2,697-row dual-bank-disjoint corpus changes both dose and draw from historical 3,072-row D4; historical direct contrasts remain blocked.",
    "Each bank is reported separately. No pooled acquisition score, bank offset, cross-bank paired test, or transported historical ranking.",
    "All four scalar arms share one training draw. The cross-draw 20-view floor does not apply to their deterministic paired contrasts.",
    "Minimum detectable net here means the exact significance threshold at observed pair discordance, not a prospective power calculation. The absolute best case is six discordant wins.",
    "A nonsignificant contrast does not establish equivalence. Fixed engineering targets are not equivalence or noninferiority tests.",
    "Primary references_retained uses the same 47 registered permissive-source IDs. Strict scoring has only 41 correct source references; its own reference denominator is also reported.",
    "Long-context endpoints use a spent report-only instrument. Literal UNKNOWN reproduces the frozen script's normalized substring test and is not semantic abstention.",
    "Nominal and Holm-adjusted p-values are descriptive. Holm families contain the three alpha-versus-full contrasts separately for each endpoint.",
]


def require(condition, reason):
    if not condition:
        raise paired.ComparabilityError(reason)


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_rows(rows, label):
    require(isinstance(rows, list), f"{label}: rows must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"{label}: row must be an object")
        key = row.get("id")
        require(isinstance(key, str) and key and key not in result, f"{label}: duplicate/missing ID")
        result[key] = row
    require(bool(result), f"{label}: empty rows")
    return result


def jsonl(path):
    return unique_rows([json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
                        if line.strip()], str(path))


def additional_path(binding, name):
    before = binding.get("additional_input_sha256_before")
    require(isinstance(before, dict) and before == binding.get("additional_input_sha256_after"),
            "additional instrument identities absent or changed")
    matches = [Path(path) for path in before if Path(path).name == name]
    require(len(matches) == 1, f"expected exactly one bound {name}")
    path = matches[0]
    require(sha(path) == before[str(path)], f"additional input hash mismatch: {name}")
    return path


def load_pure_retention():
    path = run_probe.RAB / "retention.py"
    spec = importlib.util.spec_from_file_location("doseint_offline_retention", path)
    require(spec is not None and spec.loader is not None, "pure retention scorer unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require("torch" not in sys.modules, "offline report unexpectedly imported torch")
    return module


def retention_endpoints(raw, binding, scorer):
    """Re-score all predictions and reproduce both legacy policies before use."""
    rows = jsonl(binding["views_path"])
    require(len(rows) == 150, "retention requires the complete 150-row panel")
    identity = raw.get("hf_candidate_identity")
    require(isinstance(identity, dict) and identity.get("weight_sha256") == binding["weight_sha256"],
            "retention's recorded HF weight identity differs from execution custody")
    audit_path = additional_path(binding, "PROBE_LABEL_AUDIT.json")
    frozen_path = additional_path(binding, "retention.json")
    require(Path(raw["reference_receipt"]).resolve() == frozen_path.resolve(), "retention source receipt differs")
    audit, frozen = read(audit_path), read(frozen_path)
    saved_path = Path(binding["retention_predictions_path"])
    require(sha(saved_path) == binding["retention_predictions_sha256"], "retention predictions hash differs")
    saved = read(saved_path)
    require(saved.get("scope") == "UNCHANGED_LEGACY_GENERATOR_RETURN" and saved.get("scorer_modified") is False
            and saved.get("rows") == 150, "retention generator-return receipt differs")
    predictions = unique_rows(saved["predictions"], "retention predictions")
    frozen_rows = unique_rows(frozen["rows"], "frozen source predictions")
    require(predictions.keys() == rows.keys() == frozen_rows.keys(), "retention/source row inventory differs")
    for key, prediction in predictions.items():
        require(type(prediction.get("terminated")) is bool and isinstance(prediction.get("text"), str)
                and prediction.get("reply_class") == rows[key]["reply_class"], "invalid retention prediction")
    registered = raw.get("registered_bare_policy")
    require(registered == "permissive", "registered primary policy changed")
    registered_ids = set(frozen["reference_ids_by_policy"][registered])
    require(len(registered_ids) == 47 and registered_ids <= rows.keys(), "registered source references must be 47 IDs")
    policies, maps = {}, {}
    for policy in ("permissive", "strict"):
        overlay = scorer.build_overlay(audit, policy)
        targets = overlay["targets"]
        require(targets.keys() == rows.keys(), "typed label overlay differs from complete panel")
        source, candidate = {}, {}
        for key in rows:
            original = frozen_rows[key]["source"]
            require(type(original.get("terminated")) is bool and isinstance(original.get("raw"), str),
                    "frozen source lacks a valid raw answer")
            source[key] = scorer.score_prediction(original["raw"], original["terminated"], targets[key])["correct"]
            prediction = predictions[key]
            candidate[key] = scorer.score_prediction(prediction["text"], prediction["terminated"], targets[key])["correct"]
        own_ids = {key for key, correct in source.items() if correct}
        declared_ids = frozen["reference_ids_by_policy"][policy]
        require(len(declared_ids) == len(set(declared_ids)) and own_ids == set(declared_ids),
                f"source rescoring does not reproduce frozen {policy} IDs")
        require(len(own_ids) == (47 if policy == "permissive" else 41), "source policy reference count changed")
        observed = raw["by_bare_policy"][policy]
        candidate_ids = {key for key, correct in candidate.items() if correct}
        lost_ids = own_ids - candidate_ids
        require(observed["arm_correct_total"] == len(candidate_ids)
                and observed["source_correct_total"] == len(own_ids)
                and observed["lost_total"] == len(lost_ids)
                and observed["reference_size"] == len(own_ids | candidate_ids),
                f"legacy {policy} totals do not reproduce full predictions")
        classes = {}
        for cls in sorted({row["reply_class"] for row in rows.values()}):
            ids = {key for key, row in rows.items() if row["reply_class"] == cls}
            require(len(ids) == 50, "retention class geometry changed")
            detail = observed["per_class"][cls]
            expected_lost = sorted(lost_ids & ids)
            require(detail["lost_ids"] == expected_lost and detail["lost"] == len(expected_lost)
                    and detail["arm_correct"] == sum(candidate[key] for key in ids)
                    and detail["terminated"] == sum(predictions[key]["terminated"] for key in ids),
                    f"legacy {policy}/{cls} counts do not reproduce predictions")
            classes[cls] = {"n": len(ids), "correct": sum(candidate[key] for key in ids),
                            "terminated": sum(predictions[key]["terminated"] for key in ids),
                            "own_policy_source_lost_ids": expected_lost}
        reference_map = {key: candidate[key] for key in sorted(registered_ids)}
        policies[policy] = {
            "panel_n": 150, "panel_correct": sum(candidate.values()),
            "registered_reference_n": 47, "references_retained": sum(reference_map.values()),
            "registered_source_lost_ids": sorted(key for key, correct in reference_map.items() if not correct),
            "own_policy_source_reference_n": len(own_ids),
            "own_policy_source_references_retained": len(own_ids - lost_ids),
            "own_policy_source_lost_ids": sorted(lost_ids), "per_class": classes,
        }
        maps[f"panel_{policy}"] = candidate
        maps[f"references_retained_{policy}"] = reference_map
        maps[f"ratio_proportion_{policy}"] = {key: candidate[key] for key in rows
                                               if rows[key]["reply_class"] == "ratio_proportion"}
    maps["termination"] = {key: prediction["terminated"] for key, prediction in predictions.items()}
    return {"registered_policy": registered, "policies": policies,
            "termination_n": 150, "terminated": sum(maps["termination"].values())}, maps


def normalize_long_context(value):
    return " ".join(str(value).strip().lower().replace(".", " ").split())


def long_context_endpoints(raw, binding):
    rows = jsonl(binding["views_path"])
    key = jsonl(additional_path(binding, "all_answer_key.jsonl"))
    predictions, declared = raw["predictions"], raw["per_row_correct"]
    require(len(rows) == 256 and rows.keys() == key.keys() == predictions.keys() == declared.keys(),
            "long-context requires all 256 prompt/key/prediction/score IDs")
    total, unknown, unanswerable = {}, {}, {}
    skipped = 0
    for row_id in rows:
        record, answer = predictions[row_id], key[row_id]
        require(record.get("id") == row_id and isinstance(record.get("prediction"), str)
                and type(declared[row_id]) is bool, "invalid long-context prediction")
        require(record.get("status") in ("ok", "skipped_context_overflow"), "unknown long-context status")
        skipped += record["status"] == "skipped_context_overflow"
        accepted = answer.get("accepted_answers") or [answer["answer"]]
        first = record["prediction"].splitlines()[0] if record["prediction"] else ""
        hit = normalize_long_context(first) in {normalize_long_context(value) for value in accepted}
        require(hit == declared[row_id], "long-context score does not reproduce frozen acceptance rule")
        total[row_id] = hit
        if answer["variant"] == "unanswerable":
            unanswerable[row_id] = hit
            unknown[row_id] = "unknown" in normalize_long_context(record["prediction"])
    require(bool(unknown) and raw["all256_hits"] == sum(total.values())
            and raw["unanswerable_rows"] == len(unknown)
            and raw["unanswerable_correct"] == sum(unanswerable.values())
            and raw["says_unknown_on_unanswerable"] == sum(unknown.values())
            and raw["rows_skipped_context_overflow"] == skipped,
            "long-context aggregate does not reproduce complete predictions")
    return {"n": len(total), "correct": sum(total.values()), "skipped": skipped,
            "unanswerable_n": len(unknown), "unanswerable_correct": sum(unanswerable.values()),
            "literal_unknown": sum(unknown.values())}, {
                "long_context_total": total, "long_context_unanswerable_correct": unanswerable,
                "long_context_literal_unknown": unknown}


def holm(contrasts):
    """Attach Holm p-values for exactly the three preregistered comparisons."""
    require(set(contrasts) == set(CANDIDATES), "Holm family must contain exactly three fixed arms")
    running = 0.0
    for index, (arm, value) in enumerate(sorted(contrasts.items(), key=lambda item: item[1]["exact_p"])):
        running = max(running, min(1.0, (3 - index) * value["exact_p"]))
        value["holm_adjusted_p"] = running
        value["holm_nominally_significant"] = running <= 0.05
        value["holm_family_size"] = 3
    return contrasts


def band(value, bounds):
    return {"observed": value, "lower": bounds[0], "upper": bounds[1],
            "status": "PASS" if bounds[0] <= value <= bounds[1] else "FAILED_PREDICTION"}


def build_report(root):
    root = Path(root).resolve()
    try:
        require("torch" not in sys.modules, "run the offline reporter in a fresh stdlib-only process")
        loaded, custody, acquisition = {}, {}, {}
        # No correctness values are consumed until every leg and inter-arm gate passes.
        for arm in ARMS:
            loaded[arm], custody[arm] = {}, {}
            for leg, kind in LEGS.items():
                directory = root / "eval" / f"{arm}-{leg}"
                raw_path, binding_path = directory / "result.json", directory / "PROBE_WEIGHT_BINDING.json"
                raw, binding, ids = paired.validate_probe_binding(raw_path, binding_path, kind)
                loaded[arm][leg] = (raw, binding, ids)
                custody[arm][leg] = {"binding_sha256": sha(binding_path), "raw_sha256": sha(raw_path),
                                     "weight_sha256": binding["weight_sha256"]}
                receipt = read(binding["arm_receipt_path"])
                require(receipt["alpha"] == ALPHAS[arm], f"wrong registered alpha: {arm}/{leg}")
        reference_identity = loaded["a100"]["dose"][1]["training_identity"]
        current_implementation = run_probe.implementation_digest()
        gate_hashes = set()
        for arm in ARMS:
            weight = loaded[arm]["dose"][1]["weight_sha256"]
            for leg in LEGS:
                raw, binding, ids = loaded[arm][leg]
                require(binding["weight_sha256"] == weight, f"different model bytes across {arm} battery")
                require(binding["training_identity"] == reference_identity, "arms are not one training draw/source")
                require(binding["implementation_digest"] == current_implementation,
                        "current offline scorer implementation differs from executed probes")
                baseline = loaded["a100"][leg][1]
                for field in ("views_sha256", "implementation_digest", "script_sha256", "wrapper_sha256",
                              "additional_input_sha256_before"):
                    require(binding.get(field) == baseline.get(field), f"{leg} cross-arm identity differs: {field}")
                require(binding["build_gate_sha256"] == sha(binding["build_gate_path"]), "build gate hash differs")
                gate_hashes.add(binding["build_gate_sha256"])
                if leg == "long_context":
                    for field in ("max_context", "max_new_tokens", "numerics", "transformers", "device"):
                        require(raw.get(field) == loaded["a100"][leg][0].get(field),
                                f"long-context execution differs: {field}")
        require(len(gate_hashes) == 1, "battery spans different build gates")
        gate = read(loaded["a100"]["dose"][1]["build_gate_path"])
        require(gate["registration_sha256"] == sha(root / "PREREG.md")
                and gate["amendment_sha256"] == sha(root / "AMENDMENT_01.md"), "registration/amendment identity differs")
        require(gate["training_rows"] == gate["training_parent_count"] == 2697,
                "registered dual899 training inventory changed")
        training_snapshot = None
        for arm in ARMS:
            receipt_binding = loaded[arm]["dose"][1]
            live_training = run_probe.training_snapshot(gate, (
                read(receipt_binding["arm_receipt_path"]), read(receipt_binding["materialization_path"])))
            require(live_training["processed_tokens"] == 380921 and live_training["updates"] == 193,
                    "arm is not the complete registered high-dose trajectory")
            if training_snapshot is None:
                training_snapshot = live_training
            require(live_training == training_snapshot, "arms have different verified training data or plans")
            for leg, kind in LEGS.items():
                raw, binding, _ = loaded[arm][leg]
                require(binding.get("training_snapshot_before") == binding.get("training_snapshot_after") == live_training,
                        f"{arm}/{leg}: saved training custody differs from verified trajectory")
                require(binding["script_sha256"] == gate["probe_script_sha256"][kind]
                        and binding["wrapper_sha256"] == gate["wrapper_sha256"],
                        f"{arm}/{leg}: probe code differs from prospective build gate")
                if leg in ("dose", "select"):
                    expected_instrument = "process-transition-probe-001" if leg == "dose" else "select-views-001"
                    require(raw["instrument"] == expected_instrument
                            and binding["views_sha256"] == gate["banks"][leg]["sha256"],
                            f"{arm}/{leg}: bank label does not identify the registered instrument")
        for bank in ("dose", "select"):
            for arm in CANDIDATES:
                left, right = root / "eval" / f"a100-{bank}", root / "eval" / f"{arm}-{bank}"
                contrast = paired.compare_acquisition(left / "result.json", right / "result.json",
                           reference_binding=left / "PROBE_WEIGHT_BINDING.json",
                           candidate_binding=right / "PROBE_WEIGHT_BINDING.json", relationship="same_training_draw")
                require(contrast["status"] == "COMPARABLE", f"{bank}/{arm}: {contrast.get('reasons')}")
                acquisition.setdefault(bank, {})[arm] = contrast
        scorer = load_pure_retention()
        endpoints, maps = {}, {}
        for arm in ARMS:
            endpoints[arm], maps[arm] = {"alpha": ALPHAS[arm]}, {}
            for bank in ("dose", "select"):
                raw, binding, ids = loaded[arm][bank]
                predictions = unique_rows(raw["predictions"], f"{arm}/{bank}")
                require(len(ids) == 384 and predictions.keys() == ids, "acquisition requires complete384 bank")
                correctness = {key: row["canonical"] for key, row in predictions.items()}
                require(all(type(value) is bool for value in correctness.values())
                        and raw["aggregate"]["canonical_correct"] == sum(correctness.values()),
                        "acquisition aggregate does not reproduce paired rows")
                endpoints[arm][bank] = {"n": 384, "correct": sum(correctness.values()), "views_sha256": binding["views_sha256"]}
            retention, retention_maps = retention_endpoints(*loaded[arm]["retention"][:2], scorer)
            long_context, long_context_maps = long_context_endpoints(*loaded[arm]["long_context"][:2])
            endpoints[arm].update(retention=retention, long_context=long_context)
            maps[arm].update(retention_maps, **long_context_maps)
        contrasts = {f"acquisition_{bank}": holm({arm: acquisition[bank][arm]["paired"] for arm in CANDIDATES})
                     for bank in ("dose", "select")}
        for endpoint in maps["a100"]:
            contrasts[endpoint] = holm({arm: paired.paired_boolean_summary(maps["a100"][endpoint], maps[arm][endpoint],
                                          relationship="same_training_draw") for arm in CANDIDATES})
        predictions = {}
        for arm, values in endpoints.items():
            registered = values["retention"]["policies"]["permissive"]
            primary = values["dose"]["correct"] >= 284 and registered["references_retained"] >= 40 and registered["panel_correct"] >= 62
            restoration = registered["per_class"]["ratio_proportion"]["correct"] >= 16 and values["retention"]["terminated"] == 150
            predictions[arm] = {
                "dose_acquisition_band": band(values["dose"]["correct"], DOSE_BANDS[arm]),
                "select_acquisition_band": band(values["select"]["correct"], SELECT_BANDS[arm]),
                "references_retained_band": band(registered["references_retained"], RETAINED_BANDS[arm]),
                "primary_conjunction": "PASS" if primary else "FAILED_PREDICTION",
                "ratio_and_termination_restored": "PASS" if restoration else "FAILED_PREDICTION",
                "primary_and_restoration": "PASS" if primary and restoration else "FAILED_PREDICTION",
            }
        order = ("a025", "a050", "a075", "a100")
        monotone_acquisition = {bank: all(endpoints[a][bank]["correct"] <= endpoints[b][bank]["correct"]
                                        for a, b in zip(order, order[1:])) for bank in ("dose", "select")}
        monotone_retention = all(endpoints[a]["retention"]["policies"]["permissive"]["references_retained"] >=
                                 endpoints[b]["retention"]["policies"]["permissive"]["references_retained"]
                                 for a, b in zip(order, order[1:]))
        require("torch" not in sys.modules, "offline report unexpectedly imported torch")
        return {"schema": 1, "status": "COMPLETE_COMPARABLE", "experiment": "DOSEINT-001",
                "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
                "training_identity": reference_identity, "build_gate_sha256": next(iter(gate_hashes)),
                "training_snapshot": training_snapshot,
                "implementation_digest": current_implementation,
                "registration_sha256": sha(root / "PREREG.md"), "amendment_01_sha256": sha(root / "AMENDMENT_01.md"),
                "reporter_sha256": sha(Path(__file__)), "custody": custody,
                "endpoints": endpoints, "paired_contrasts": contrasts, "predictions": predictions,
                "primary_prediction": "PASS" if any(predictions[a]["primary_conjunction"] == "PASS" for a in CANDIDATES) else "FAILED_PREDICTION",
                "acquisition_monotone": {bank: "PASS" if value else "FAILED_PREDICTION" for bank, value in monotone_acquisition.items()},
                "retention_nonincreasing": "PASS" if monotone_retention else "FAILED_PREDICTION",
                "limitations": LIMITATIONS, "quality_claim": False, "release_permitted": False, "selection_permitted": False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
        return {"schema": 1, "status": paired.BLOCKED, "reasons": [str(exc)],
                "quality_claim": False, "release_permitted": False, "selection_permitted": False}


def markdown(report):
    require(report["status"] == "COMPLETE_COMPARABLE", "cannot render endpoint report without all gates")
    lines = ["# DOSEINT-001 results", "", f"Primary fixed engineering prediction: **{report['primary_prediction']}**.",
             "All sixteen legs passed model-byte custody and comparability checks. All contrasts are within one training draw. The first table uses the registered permissive retention policy.", "",
             "| Alpha | Dose /384 | SELECT /384 | Retained /47 | Panel /150 | Ratio /50 | Terminated /150 | LC /256 | UNKNOWN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        e = report["endpoints"][arm]; r = e["retention"]["policies"]["permissive"]; lc = e["long_context"]
        lines.append(f"| {e['alpha']:.2f} | {e['dose']['correct']} | {e['select']['correct']} | {r['references_retained']} | {r['panel_correct']} | {r['per_class']['ratio_proportion']['correct']} | {e['retention']['terminated']} | {lc['correct']} | {lc['literal_unknown']}/{lc['unanswerable_n']} |")
    lines += ["", "## Fixed predictions", "", "| Alpha | Dose band | SELECT band | Retained band | Primary conjunction | Ratio + termination |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        p = report["predictions"][arm]
        lines.append(f"| {ALPHAS[arm]:.2f} | {p['dose_acquisition_band']['status']} | {p['select_acquisition_band']['status']} | {p['references_retained_band']['status']} | {p['primary_conjunction']} | {p['ratio_and_termination_restored']} |")
    lines += ["", f"Acquisition monotonicity: dose {report['acquisition_monotone']['dose']}; SELECT {report['acquisition_monotone']['select']}. Source retention nonincreasing: {report['retention_nonincreasing']}.",
              "", "## Both typed scoring policies", "", "| Alpha | Policy | Panel /150 | Correct on fixed 47 source IDs | Own-policy source retained | Ratio /50 |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for policy, values in report["endpoints"][arm]["retention"]["policies"].items():
            lines.append(f"| {ALPHAS[arm]:.2f} | {policy} | {values['panel_correct']} | {values['references_retained']} | {values['own_policy_source_references_retained']}/{values['own_policy_source_reference_n']} | {values['per_class']['ratio_proportion']['correct']} |")
    lines += ["", "## Every alpha-versus-full paired contrast", "", "Positive net favors the interpolant. Threshold is the minimum significant absolute net at this observed discordance.", "",
              "| Endpoint | Alpha | net | nd | exact p | Holm p | threshold |", "|---|---|---:|---:|---:|---:|---:|"]
    for endpoint, family in report["paired_contrasts"].items():
        for arm in CANDIDATES:
            c = family[arm]
            threshold = c["minimum_significant_abs_net_at_observed_nd"]
            lines.append(f"| {endpoint} | {ALPHAS[arm]:.2f} | {c['net']:+d} | {c['nd']} | {c['exact_p']:.6g} | {c['holm_adjusted_p']:.6g} | {threshold if threshold is not None else 'unattainable'} |")
    lines += ["", "## Limits", ""] + [f"- {limitation}" for limitation in report["limitations"]]
    lines += ["", "RESULTS.json contains both policies, all class counts, lost IDs, predictions, and the evidence hashes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    destinations = [args.root / "RESULTS.json", args.root / "RESULTS.md"]
    require(not any(path.exists() for path in destinations), "create-only result destination already exists")
    report = build_report(args.root)
    if report["status"] != "COMPLETE_COMPARABLE":
        print(json.dumps(report, indent=2))
        return 2
    rendered = markdown(report)
    with destinations[0].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    with destinations[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
    print(json.dumps({"status": report["status"], "results": str(destinations[0]), "report": str(destinations[1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
