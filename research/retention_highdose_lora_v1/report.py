"""Offline DOSELORA report: all custody gates before any endpoint disclosure.

Re-scores preserved generations only, using the prior audited paired reader and
pure retention/long-context functions. Does not import model runtimes or dispatch
workers. A prospective magnitude-control failure cannot become a scored result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.retention_dose_parameterization_v1 import paired, report_battery as prior, run_probe
from research.retention_highdose_lora_v1 import materialize

DEFAULT_ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora001-20260920")
DEFAULT_OLD_ROOT = DEFAULT_ROOT.parent / "doseint001-20260919"
NEW_ARMS = ("lora", "norm")
OLD_ARMS = ("a025", "a050", "a075", "a100")
LEGS = prior.LEGS
require, read, sha = prior.require, prior.read, prior.sha


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unseal(record):
    body = {key: value for key, value in record.items() if key != "sha256"}
    require(record.get("sha256") == digest(body), "sealed norm/checkpoint receipt hash differs")
    return body


def attempt_map(root, arms):
    path = root / "ATTEMPT_MAP.json"
    if not path.exists():
        return {}
    value = read(path)
    require(value.get("schema") == 1 and isinstance(value.get("replacements"), dict), "invalid attempt map")
    if tuple(arms) == NEW_ARMS:
        require(len(value["replacements"]) <= 1, "the new experiment permits only one inference retry in total")
    allowed = {f"{arm}-{leg}" for arm in arms for leg in LEGS}
    for key, relative in value["replacements"].items():
        require(key in allowed and isinstance(relative, str)
                and Path(relative).parts == ("eval", f"{key}-retry1"), "unregistered inference replacement")
    return value["replacements"]


def probe_cost_accounting(root, replacements):
    """Reconcile individual attempts; unobserved worker termination costs 900s.

    Attempt receipts live at posttrain/<eval-directory-basename>.json. A retry
    therefore writes <leg>-retry1.json and never replaces the original leg file.
    Aggregate TERMINAL totals are checked elsewhere, never added to these costs.
    """
    root = Path(root).resolve()
    require(len(replacements) <= 1, "more than one inference retry was requested")
    required = {f"{arm}-{leg}" for arm in NEW_ARMS for leg in LEGS}
    allowed = required | {f"{key}-retry1" for key in replacements}
    records, hashes, sources, successes = [], {}, {}, set()
    for path in sorted((root / "posttrain").glob("*.json")):
        if path.name in ("TERMINAL.json", "RECOVERY_COMPLETE.json", "PLAN.json", "RECOVERY_PLAN.json"):
            continue
        require(path.stem in allowed, "unexpected individual probe receipt in posttrain")
        record = read(path)
        key = record["leg"]
        require(key in required, "attempt receipt names an unregistered leg")
        directory = Path(record["out"]).resolve()
        require(directory.parent == root / "eval" and directory.name == path.stem,
                "individual attempt receipt points to a different eval directory")
        expected_name = f"{key}-retry1" if path.stem.endswith("-retry1") else key
        require(path.stem == expected_name, "retry receipt changed its leg identity")
        count, max_new = {"dose": (384, 128), "select": (384, 128),
                          "retention": (150, 24), "long_context": (256, 24)}[key.split("-", 1)[1]]
        require(record["reserved_rows"] == count and record["reserved_generation_tokens"] == count * max_new,
                "probe row/token reservation differs from its fixed instrument")
        require(type(record.get("returncode")) is int, "attempt lacks an observed foreground exit code")
        relative = path.relative_to(root).as_posix()
        hashes[relative] = sha(path); sources[relative] = hashes[relative]
        charged = 900.0
        supervisor_path = directory / "worker/supervisor.json"
        if supervisor_path.is_file():
            supervisor = read(supervisor_path)
            sources[supervisor_path.relative_to(root).as_posix()] = sha(supervisor_path)
            observed = supervisor.get("wall_seconds")
            if (type(supervisor.get("returncode")) is int and type(observed) in (int, float)
                    and math.isfinite(observed) and observed >= 0
                    and supervisor.get("status") in ("EXITED", "TIMEOUT_OR_INTERRUPTED")):
                charged = float(observed)
        require(type(record.get("worker_seconds_charged")) in (int, float)
                and math.isfinite(record["worker_seconds_charged"])
                and math.isclose(record["worker_seconds_charged"], charged, rel_tol=0, abs_tol=1e-6),
                "probe time omits observed failure time or its unobserved 900-second reservation")
        success = record["returncode"] == 0
        if success:
            require(key not in successes, "multiple successful attempts for the same leg")
            successes.add(key)
        records.append({"receipt": relative, "leg": key, "out": str(directory),
                        "returncode": record["returncode"], "worker_seconds_charged": charged,
                        "reserved_rows": count, "reserved_generation_tokens": count * max_new})
    require(len(records) == 8 + len(replacements) <= 9 and successes == required,
            "final cost inventory does not contain exactly eight successful legs and the allowed failed attempt")
    failed = [record for record in records if record["returncode"] != 0]
    require(len(failed) == len(replacements) and {record["leg"] for record in failed} == set(replacements),
            "failed attempt inventory differs from the explicit retry map")
    for record in records:
        if record["returncode"] == 0:
            expected = root / replacements[record["leg"]] if record["leg"] in replacements else root / "eval" / record["leg"]
            require(Path(record["out"]) == expected.resolve(), "final success differs from the selected fixed attempt")
    seconds = sum(record["worker_seconds_charged"] for record in records)
    rows = sum(record["reserved_rows"] for record in records)
    tokens = sum(record["reserved_generation_tokens"] for record in records)
    require(seconds <= 8100 and rows <= 3000 and tokens <= 270000, "registered all-attempt inference budget exceeded")
    return {"attempt_count": len(records), "successful_attempts": 8, "failed_attempts": len(failed),
            "probe_worker_seconds": seconds, "probe_rows_reserved": rows, "probe_generation_tokens_reserved": tokens,
            "attempt_receipt_sha256": hashes, "cost_source_sha256": sources, "individual_attempts": records}


def completion_receipt(root, replacements):
    """Accept original completion, or one separately recorded failed-leg recovery.

    Recovery schema: schema=1, status=COMPLETE_EIGHT_PROBES,
    prior_terminal_sha256, attempt_map_sha256, final_binding_sha256 (all eight
    leg keys), cost_accounting (the exact probe_cost_accounting return value).
    Preserve the original TERMINAL.json and every failed attempt unchanged.
    """
    root = Path(root).resolve()
    original_path = root / "posttrain/TERMINAL.json"
    original = read(original_path)
    if original.get("status") == "COMPLETE_EIGHT_PROBES":
        require(not replacements, "an originally complete battery cannot select a retry")
        terminal, terminal_path = original, original_path
    else:
        require(original.get("status") == "FAILED_OR_BLOCKED" and len(replacements) == 1,
                "new battery is incomplete and has no single explicit inference recovery")
        terminal_path = root / "posttrain/RECOVERY_COMPLETE.json"
        terminal = read(terminal_path)
        require(terminal.get("schema") == 1 and terminal.get("status") == "COMPLETE_EIGHT_PROBES"
                and terminal.get("prior_terminal_sha256") == sha(original_path)
                and terminal.get("attempt_map_sha256") == sha(root / "ATTEMPT_MAP.json"),
                "recovery completion is not bound to the preserved original failure and attempt map")
        for record in original.get("attempts", []):
            source = root / "posttrain" / f"{Path(record['out']).name}.json"
            require(read(source) == record, "an original attempt receipt changed during recovery")
    cost = probe_cost_accounting(root, replacements)
    final_bindings = {}
    for arm in NEW_ARMS:
        for leg in LEGS:
            key = f"{arm}-{leg}"
            directory = prior.resolve_attempt(root, key, replacements)
            path = directory / "PROBE_WEIGHT_BINDING.json"
            binding = read(path)
            require(binding.get("status") == "COMPLETE" and binding.get("complete") is True
                    and binding.get("returncode") == 0, "final binding is incomplete")
            final_bindings[key] = sha(path)
    if terminal_path != original_path:
        require(terminal.get("final_binding_sha256") == final_bindings and terminal.get("cost_accounting") == cost,
                "recovery completion does not reproduce all eight final bindings and individual attempt costs")
    else:
        require(math.isclose(terminal["probe_worker_seconds"], cost["probe_worker_seconds"], rel_tol=0, abs_tol=1e-6)
                and terminal["probe_rows_reserved"] == cost["probe_rows_reserved"]
                and terminal["probe_generation_tokens_reserved"] == cost["probe_generation_tokens_reserved"],
                "original terminal totals differ from individual attempt receipts")
    return {"path": str(terminal_path), "sha256": sha(terminal_path), "prior_terminal_sha256": sha(original_path),
            "final_binding_sha256": final_bindings, "cost_accounting": cost}


def verify_training_supervision(run, binding, norm, *receipts):
    """Recheck late worker failures and every settled journal event, not labels."""
    identity = materialize._supervised_success(Path(run).absolute(), binding)
    require("torch" not in sys.modules, "supervision validation imported a model runtime")
    for record in (norm, *receipts):
        require(all(record.get(key) == value for key, value in identity.items()),
                "training supervisor or sealed journal identity differs from norm/arm custody")
    return identity


def holm(family):
    """Descriptive Holm values over the exact preregistered family provided."""
    require(bool(family), "empty multiplicity family")
    maximum = 0.0
    for index, (_, contrast) in enumerate(sorted(family.items(), key=lambda item: item[1]["exact_p"])):
        maximum = max(maximum, min(1.0, (len(family) - index) * contrast["exact_p"]))
        contrast.update(holm_adjusted_p=maximum, holm_family_size=len(family), holm_nominally_significant=maximum <= 0.05)
    return family


def verify_norm(root, receipts, old_receipt):
    """Verify sealed magnitude custody and actual rounded-norm arithmetic."""
    path = root / "NORM_MATCH.json"
    norm = unseal(read(path))
    require(norm.get("schema") == 1 and norm.get("record") == "DOSELORA_NORM_MATCH"
            and norm.get("experiment") == "DOSELORA-001" and norm.get("status") == "PASS", "norm construction is not PASS")
    beta, lora_norm, native_norm = norm["beta"], norm["lora_dense_delta_norm"], norm["native_dense_delta_norm"]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (beta, lora_norm, native_norm))
            and 0 <= beta <= 2 and lora_norm >= 0 and native_norm > 0 and beta == lora_norm / native_norm,
            "norm ratio does not reproduce the registered coefficient")
    require(norm["beta_admissible_interval"] == [0, 2] and norm["relative_match_tolerance"] == 0.005
            and norm["dense_updated_tensor_count"] == 220 and norm["projection_count"] == 140
            and norm["trained_norm_count"] == 80 and norm["lora_rank"] == 16 and norm["lora_alpha"] == 32,
            "norm-matching scope/tolerance changed")
    require(len(norm["dense_updated_tensor_names"]) == len(set(norm["dense_updated_tensor_names"])) == 220,
            "norm tensor-name inventory differs")
    expected_dense_names = {entry["name"] for entry in old_receipt["checkpoint_entries"]}
    require(set(norm["dense_updated_tensor_names"]) == expected_dense_names, "norm scope differs from native 220-tensor inventory")
    lora, scalar = receipts["lora"], receipts["norm"]
    require(lora["parameterization"] == "lora" and scalar["parameterization"] == "native_scalar_norm_match",
            "new arm parameterizations are reversed or unregistered")
    require(lora["alpha"] == 1 and scalar["alpha"] == beta and lora["norm_matched_scalar_beta"] == beta,
            "new arm coefficient differs from the frozen norm formula")
    require(lora["dense_delta_norm"] == lora_norm and scalar["target_lora_dense_delta_norm"] == lora_norm,
            "arm norm targets differ")
    measured = scalar["dense_delta_norm"]
    require(type(measured) in (int, float) and math.isfinite(measured) and measured >= 0, "invalid actual scalar norm")
    relative = abs(measured - lora_norm) / lora_norm if lora_norm else (0.0 if measured == 0 else math.inf)
    require(math.isfinite(relative) and relative <= 0.005 and relative == scalar["relative_norm_discrepancy"]
            and scalar["relative_norm_tolerance"] == 0.005 and scalar["coefficient_search_performed"] is False,
            "actual rounded scalar norm does not satisfy the registered match")
    require(scalar["norm_record_sha256"] == sha(path) and Path(scalar["norm_record_path"]).resolve() == path.resolve(),
            "scalar was bound to a different norm record")
    verified = unseal(read(root / "arms/norm/NORM_MATCH_VERIFICATION.json"))
    require(verified["status"] == "PASS" and verified["norm_record_sha256"] == sha(path)
            and verified["relative_norm_discrepancy"] == relative and verified["dense_delta_norm"] == measured,
            "scalar norm-verification receipt differs")
    for arm, key in (("lora", "lora_checkpoint"), ("norm", "native_checkpoint")):
        receipt, identity = receipts[arm], norm[key]
        checkpoint = Path(receipt["checkpoint"])
        require(Path(identity["checkpoint"]).resolve() == checkpoint.resolve()
                and sha(checkpoint / "manifest.json") == identity["checkpoint_manifest_sha256"] == receipt["checkpoint_manifest_sha256"],
                "norm/arm checkpoint identity differs")
        manifest = unseal(read(checkpoint / "manifest.json"))
        require(manifest["status"] == "COMPLETE" and manifest["binding"] == receipt["checkpoint_binding"]
                and digest(manifest["binding"]) == receipt["checkpoint_binding_sha256"] == identity["checkpoint_binding_sha256"]
                and manifest["progress"] == identity["checkpoint_progress"] == receipt["checkpoint_progress"],
                "norm/arm checkpoint binding or progress differs")
        require({entry["name"]: entry["sha256"] for entry in manifest["entries"]} == identity["tensor_sha256"],
                "norm receipt was computed from a different checkpoint tensor inventory")
    require(scalar["checkpoint_manifest_sha256"] == old_receipt["checkpoint_manifest_sha256"],
            "matched scalar does not use the authoritative native training draw")
    require(lora["checkpoint_binding"]["parent_data_contract_sha256"] == scalar["checkpoint_binding"]["data_contract_sha256"],
            "LoRA training is not derived from the native control data")
    require(norm["source_assets_sha256"] == lora["source_assets_sha256"] == scalar["source_assets_sha256"] == old_receipt["source_assets_sha256"],
            "norm-matching source assets differ")
    require(norm["lora_weight_sha256"] == lora["arm_weight_sha256"]
            and norm["lora_arm_receipt_sha256"] == sha(root / "arms/lora/ARM.json")
            and norm["lora_materialization_sha256"] == sha(root / "arms/lora/MATERIALIZATION.json"),
            "norm record names a different LoRA artifact")
    training_path = Path(lora["training_result_path"])
    supervision = verify_training_supervision(training_path.parent, lora["checkpoint_binding"], norm,
                                              lora, read(root / "arms/lora/MATERIALIZATION.json"))
    require(sha(training_path) == lora["training_result_sha256"] == norm["training_result_sha256"], "training-result custody differs")
    training = read(training_path)
    require(training["status"] == "COMPLETE" and training["complete"] is True
            and training["binding"] == lora["checkpoint_binding"]
            and training["checkpoint_manifest_sha256"] == lora["checkpoint_manifest_sha256"], "LoRA training is not complete and bound")
    merge_path = Path(training["merge_qualification_path"])
    require(sha(merge_path) == training["merge_qualification_sha256"] == lora["merge_qualification_sha256"] == norm["merge_qualification_sha256"],
            "merge-qualification custody differs")
    merge = read(merge_path)
    require(merge["status"] == "PASS" and merge["complete"] is True and merge["binding"] == lora["checkpoint_binding"]
            and merge["checkpoint_manifest_sha256"] == lora["checkpoint_manifest_sha256"]
            and merge["atol"] == merge["rtol"] == 2e-4 and merge["argmax_equality_required"] is True
            and merge["original_modules_restored"] is True and len(merge["dense_tensor_sha256"]) == 220
            and len(merge["inputs"]) == 3 and all(value["passed"] is True and value["tolerance_violations"] == value["argmax_mismatches"] == 0
                                                 for value in merge["inputs"]), "merged deployment qualification is not a complete PASS")
    require(set(merge["dense_tensor_sha256"]) == expected_dense_names
            and norm["merged_tensor_sha256"] == lora["merged_tensor_sha256"] == merge["dense_tensor_sha256"],
            "published LoRA tensor hashes differ from the qualified CPU merge")
    return {"record_sha256": sha(path), "beta": beta, "coefficient_kind": "extrapolation" if beta > 1 else "interpolation",
            "lora_dense_delta_norm": lora_norm, "native_dense_delta_norm": native_norm,
            "actual_scalar_dense_delta_norm": measured, "relative_norm_discrepancy": relative,
            "training_result_sha256": sha(training_path), "merge_qualification_sha256": sha(merge_path),
            "training_supervision": supervision}


def extract(raws, scorer):
    endpoints, maps = {}, {}
    for arm, legs in raws.items():
        endpoints[arm], maps[arm] = {}, {}
        for bank in ("dose", "select"):
            raw, binding, ids = legs[bank]
            rows = prior.unique_rows(raw["predictions"], f"{arm}/{bank}")
            require(len(ids) == 384 and rows.keys() == ids, "acquisition bank is incomplete")
            values = {key: row["canonical"] for key, row in rows.items()}
            require(all(type(value) is bool for value in values.values())
                    and sum(values.values()) == raw["aggregate"]["canonical_correct"], "acquisition aggregate does not reproduce rows")
            endpoints[arm][bank] = {"n": 384, "correct": sum(values.values()), "views_sha256": binding["views_sha256"]}
        endpoints[arm]["retention"], retention_maps = prior.retention_endpoints(*legs["retention"][:2], scorer)
        endpoints[arm]["long_context"], lc_maps = prior.long_context_endpoints(*legs["long_context"][:2])
        maps[arm].update(retention_maps, **lc_maps)
    return endpoints, maps


def prediction_verdicts(lora, norm, primary_acquisition):
    """Fixed engineering criteria, independent of post-hoc multiplicity choices."""
    retention = lora["retention"]["policies"]["permissive"]
    retention_targets = retention["references_retained"] >= 40 and retention["panel_correct"] >= 62
    engineering = retention_targets and all(lora[bank]["correct"] >= 284 for bank in ("dose", "select"))
    strong = retention_targets and retention["references_retained"] >= norm["retention"]["policies"]["permissive"]["references_retained"]
    strong = strong and all(primary_acquisition[bank]["net"] >= 20 and primary_acquisition[bank]["exact_p"] <= 0.05
                            for bank in ("dose", "select"))
    status = lambda passed: "PASS" if passed else "FAILED_PREDICTION"
    return {"acquisition_dose": prior.band(lora["dose"]["correct"], (260, 325)),
            "acquisition_select": prior.band(lora["select"]["correct"], (260, 325)),
            "references_retained": prior.band(retention["references_retained"], (37, 45)),
            "arithmetic_panel": prior.band(retention["panel_correct"], (55, 80)),
            "long_context": prior.band(lora["long_context"]["correct"], (145, 170)),
            "literal_unknown": prior.band(lora["long_context"]["literal_unknown"], (0, 16)),
            "primary_engineering_conjunction": status(engineering), "strong_beyond_magnitude_rule": status(strong),
            "ratio_at_least_16": status(retention["per_class"]["ratio_proportion"]["correct"] >= 16),
            "termination_150": status(lora["retention"]["terminated"] == 150)}


def build_report(root, old_root=DEFAULT_OLD_ROOT):
    root, old_root = Path(root).resolve(), Path(old_root).resolve()
    try:
        require("torch" not in sys.modules, "report must run in a fresh stdlib-only process")
        replacements = {"new": attempt_map(root, NEW_ARMS), "old": attempt_map(old_root, OLD_ARMS)}
        completion = completion_receipt(root, replacements["new"])
        old_result = read(old_root / "RESULTS.json")
        require(old_result["status"] == "COMPLETE_COMPARABLE", "authoritative native study is incomplete")
        gates = {"lora": root / "BUILD_GATE_LORA_FINAL.json", "norm": root / "BUILD_GATE_NATIVE_NORM.json"}
        loaded, directories, receipts, custody, snapshots = {}, {}, {}, {}, {}
        interpreter_context = None
        implementation = run_probe.implementation_digest()
        for arm in (*NEW_ARMS, *OLD_ARMS):
            base = root if arm in NEW_ARMS else old_root
            mapping = replacements["new" if arm in NEW_ARMS else "old"]
            loaded[arm], directories[arm], custody[arm] = {}, {}, {}
            for leg, kind in LEGS.items():
                directory = prior.resolve_attempt(base, f"{arm}-{leg}", mapping)
                directories[arm][leg] = directory
                raw, binding, ids = paired.validate_probe_binding(directory / "result.json", directory / "PROBE_WEIGHT_BINDING.json", kind)
                loaded[arm][leg] = (raw, binding, ids)
                precheck = read(directory / "PRECHECK.json")
                require(precheck.get("status") == "PRECHECK_PASSED" and precheck.get("kind") == kind,
                        "probe precheck execution identity is absent")
                command = precheck.get("command")
                require(isinstance(command, list) and len(command) > 1 and isinstance(command[0], str),
                        "probe precheck lacks its pinned interpreter command")
                context = (str(Path(command[0]).resolve()), str(Path(precheck["cwd"]).resolve()))
                if interpreter_context is None:
                    interpreter_context = context
                require(context == interpreter_context, "probes were dispatched with different interpreters/cwds")
                custody[arm][leg] = {"directory": str(directory), "binding_sha256": sha(directory / "PROBE_WEIGHT_BINDING.json"),
                                     "result_sha256": sha(directory / "result.json"), "weight_sha256": binding["weight_sha256"]}
                require(binding["implementation_digest"] == implementation, "executed probes differ from the current pure scorer")
                if arm in OLD_ARMS:
                    require(custody[arm][leg]["binding_sha256"] == old_result["custody"][arm][leg]["binding_sha256"],
                            "old arm is not the authoritative previously reported attempt")
            binding = loaded[arm]["dose"][1]
            receipts[arm] = read(binding["arm_receipt_path"])
            require(receipts[arm]["stored_dtype"] == "float32", "comparator was not exported as FP32")
            gate_path = gates[arm] if arm in NEW_ARMS else Path(binding["build_gate_path"])
            gate = read(gate_path)
            if arm == "lora":
                require(receipts[arm]["checkpoint_binding"]["execution_code"] == gate["training_execution_identity"],
                        "LoRA executable binding differs from the prospective final gate")
            require(gate["training_rows"] == gate["training_parent_count"] == 2697, "matched training row geometry changed")
            live = run_probe.training_snapshot(gate, (receipts[arm], read(binding["materialization_path"])))
            snapshots[arm] = live
            require(live["processed_tokens"] == 380921 and live["supervised_tokens"] == 37090 and live["updates"] == 193,
                    "candidate is not the complete matched training trajectory")
            require(receipts[arm]["checkpoint_binding"]["checkout"] == gate["pilot_checkout"], "training executable checkout differs from its gate")
            for leg, kind in LEGS.items():
                raw, current, _ = loaded[arm][leg]
                require(Path(current["build_gate_path"]).resolve() == gate_path.resolve()
                        and current["build_gate_sha256"] == sha(gate_path), "probe used a different prospective gate")
                require(current["weight_sha256"] == binding["weight_sha256"]
                        and current["training_identity"] == binding["training_identity"], "model/training identity changed within an arm")
                require(current["training_snapshot_before"] == current["training_snapshot_after"] == live,
                        "saved training custody differs from the current verified plan/data")
                require(current["script_sha256"] == gate["probe_script_sha256"][kind]
                        and current["wrapper_sha256"] == gate["wrapper_sha256"], "probe code differs from its frozen gate")
                if leg in ("dose", "select"):
                    instrument = "process-transition-probe-001" if leg == "dose" else "select-views-001"
                    require(raw["instrument"] == instrument and raw["max_new_tokens"] == 128
                            and current["views_sha256"] == gate["banks"][leg]["sha256"], "acquisition bank/settings mislabeled")
            print(f"Validated custody: {arm}", flush=True)
        require(read(gates["lora"])["registration_sha256"] == sha(root / "PREREG.md"), "new registration changed")
        native_identity = loaded["a100"]["dose"][1]["training_identity"]
        require(all(loaded[arm]["dose"][1]["training_identity"] == native_identity for arm in ("norm", *OLD_ARMS)),
                "native scalar comparators are not one authoritative training draw")
        require(loaded["lora"]["dose"][1]["training_identity"]["checkpoint_manifest_sha256"] != native_identity["checkpoint_manifest_sha256"],
                "LoRA was mislabeled as the native training draw")
        data_names = ("microfit-train.jsonl", "ARM_BUILD.json")
        native_files = {name: sha(Path(snapshots["a100"]["prepared"]) / name) for name in data_names}
        for arm in loaded:
            require({name: sha(Path(snapshots[arm]["prepared"]) / name) for name in data_names} == native_files,
                    "comparators do not use byte-identical training payloads")
            require(receipts[arm]["source_assets_sha256"] == receipts["a100"]["source_assets_sha256"], "source asset identities differ")
            for leg in LEGS:
                raw, binding, ids = loaded[arm][leg]
                reference_raw, reference_binding, reference_ids = loaded["a100"][leg]
                require(ids == reference_ids, "paired row inventories differ")
                for key in ("views_sha256", "implementation_digest", "script_sha256", "wrapper_sha256", "additional_input_sha256_before"):
                    require(binding[key] == reference_binding[key], f"cross-arm comparability differs: {leg}/{key}")
                fields = ("device", "runtime", "numerics", "max_new_tokens") if leg in ("dose", "select") else (
                    ("device", "transformers", "numerics", "max_context", "max_new_tokens") if leg == "long_context" else
                    ("device", "registered_bare_policy", "frozen_reference_sizes", "probe_rows"))
                for key in fields:
                    require(key in raw and key in reference_raw and raw[key] == reference_raw[key], f"execution setting differs: {leg}/{key}")
        norm = verify_norm(root, receipts, receipts["a100"])
        acquisition = {}
        for comparator in ("norm", *OLD_ARMS):
            acquisition[comparator] = {}
            for bank in ("dose", "select"):
                reference, candidate = directories[comparator][bank], directories["lora"][bank]
                contrast = paired.compare_acquisition(reference / "result.json", candidate / "result.json",
                            reference_binding=reference / "PROBE_WEIGHT_BINDING.json", candidate_binding=candidate / "PROBE_WEIGHT_BINDING.json",
                            relationship="cross_training_draw")
                require(contrast["status"] == "COMPARABLE", f"{comparator}/{bank}: {contrast.get('reasons')}")
                acquisition[comparator][bank] = contrast["paired"]
        holm({bank: acquisition["norm"][bank] for bank in ("dose", "select")})
        holm({f"{arm}-{bank}": acquisition[arm][bank] for arm in OLD_ARMS for bank in ("dose", "select")})
        endpoints, maps = extract(loaded, prior.load_pure_retention())
        secondary = {comparator: {endpoint: paired.paired_boolean_summary(maps[comparator][endpoint], maps["lora"][endpoint],
                                                  relationship="cross_training_draw") for endpoint in maps["lora"]}
                     for comparator in ("norm", *OLD_ARMS)}
        predictions = prediction_verdicts(endpoints["lora"], endpoints["norm"], acquisition["norm"])
        require("torch" not in sys.modules, "offline report imported a model runtime")
        return {"schema": 1, "experiment": "DOSELORA-001", "status": "COMPLETE_COMPARABLE",
                "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "old_root": str(old_root),
                "registration_sha256": sha(root / "PREREG.md"), "posttrain_terminal_sha256": sha(root / "posttrain/TERMINAL.json"),
                "verified_completion": completion,
                "old_results_sha256": sha(old_root / "RESULTS.json"), "reporter_sha256": sha(Path(__file__)),
                "implementation_digest": implementation, "norm_match": norm, "training_snapshots": snapshots,
                "probe_interpreter_and_cwd": interpreter_context,
                "attempt_replacements": replacements, "custody": custody, "endpoints": endpoints,
                "acquisition_contrasts_lora_minus_comparator": acquisition,
                "secondary_contrasts_lora_minus_comparator": secondary, "predictions": predictions,
                "multiplicity": {"primary_acquisition": "two LoRA-versus-norm banks; Holm family size 2",
                                 "exploratory_acquisition": "four prior native alpha arms by two banks; Holm family size 8",
                                 "retention_and_long_context": "all nominal paired p-values reported descriptively; no multiplicity-based efficacy claim"},
                "limitations": ["The practical LoRA regime uses peak LR 1e-4 versus native 1e-5. This is not a parameterization-only causal experiment.",
                    "All LoRA-versus-native comparisons cross training draws: the independent absolute net floor is 20 acquisition views, in addition to exact paired p. A one-view difference is unresolved.",
                    "The primary and beyond-magnitude rules are fixed finite engineering criteria. An equal/higher observed retention count does not establish equivalence or noninferiority.",
                    "The source-answer reference is the same 47 permissive-source IDs; strict source correctness is 41/47 and is reported separately. Exact thresholds depend on each observed discordance; best case is six wins.",
                    "Both banks remain separate. No pooled score, inferred bank offset, transferred ranking, or unseen population claim is made.",
                    "Long-context and literal UNKNOWN are spent report-only endpoints. UNKNOWN is the existing normalized substring criterion, not a claim of repaired semantic abstention.",
                    "Runtime gates compare recorded Python/Torch and Transformers/numerics fields and the pinned interpreter/cwd. Historical retention records do not independently record library-version fingerprints; their frozen script/scorer and execution-path custody are retained.",
                    "Failure of a prediction or matched-magnitude advantage does not establish a capacity limit. No automatic model selection, promotion, or release is licensed."],
                "quality_claim": False, "selection_permitted": False, "release_permitted": False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
        return {"schema": 1, "experiment": "DOSELORA-001", "status": paired.BLOCKED, "reasons": [str(exc)],
                "quality_claim": False, "selection_permitted": False, "release_permitted": False}


def markdown(result):
    lines = ["# DOSELORA-001 results", "", f"Primary engineering prediction: **{result['predictions']['primary_engineering_conjunction']}**.",
             f"Stronger matched-magnitude rule: **{result['predictions']['strong_beyond_magnitude_rule']}**.", "",
             f"Matched scalar beta: {result['norm_match']['beta']:.8g} ({result['norm_match']['coefficient_kind']}); actual norm discrepancy {100*result['norm_match']['relative_norm_discrepancy']:.5g}%.", "",
             "| Arm | Dose /384 | SELECT /384 | Retained /47 | Panel /150 | Ratio /50 | Termination /150 | LC /256 | UNKNOWN /64 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in (*NEW_ARMS, *OLD_ARMS):
        e = result["endpoints"][arm]; r = e["retention"]["policies"]["permissive"]; lc = e["long_context"]
        lines.append(f"| {arm} | {e['dose']['correct']} | {e['select']['correct']} | {r['references_retained']} | {r['panel_correct']} | {r['per_class']['ratio_proportion']['correct']} | {e['retention']['terminated']} | {lc['correct']} | {lc['literal_unknown']} |")
    lines += ["", "## Fixed LoRA predictions", "", "| Prediction | Outcome |", "|---|---|"]
    for name, value in result["predictions"].items():
        outcome = f"{value['status']} ({value['observed']} in [{value['lower']}, {value['upper']}])" if isinstance(value, dict) else value
        lines.append(f"| {name} | {outcome} |")
    lines += ["", "## Acquisition contrasts", "", "Positive net favors LoRA. The exact discordance threshold and the independent 20-view cross-draw floor are separate requirements.", "",
              "| Comparator | Bank | net | nd | exact p | Holm p | exact threshold | 20-view floor |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for comparator, banks in result["acquisition_contrasts_lora_minus_comparator"].items():
        for bank, c in banks.items():
            threshold = c["minimum_significant_abs_net_at_observed_nd"]
            lines.append(f"| {comparator} | {bank} | {c['net']:+d} | {c['nd']} | {c['exact_p']:.6g} | {c['holm_adjusted_p']:.6g} | {threshold if threshold is not None else 'unattainable'} | {'cleared' if c['cross_training_draw_floor_cleared'] else 'unresolved'} |")
    lines += ["", "## Strict and permissive retention", "", "| Arm | Policy | Panel /150 | Correct on fixed47 | Own-policy source retained |", "|---|---|---:|---:|---:|"]
    for arm, e in result["endpoints"].items():
        for policy, r in e["retention"]["policies"].items():
            lines.append(f"| {arm} | {policy} | {r['panel_correct']} | {r['references_retained']} | {r['own_policy_source_references_retained']}/{r['own_policy_source_reference_n']} |")
    lines += ["", "## Secondary paired contrasts", "", "Nominal descriptive p-values; all endpoint-specific thresholds are included.", "",
              "| Comparator | Endpoint | net | nd | exact p | exact threshold |", "|---|---|---:|---:|---:|---:|"]
    for comparator, endpoints in result["secondary_contrasts_lora_minus_comparator"].items():
        for endpoint, c in endpoints.items():
            threshold = c["minimum_significant_abs_net_at_observed_nd"]
            lines.append(f"| {comparator} | {endpoint} | {c['net']:+d} | {c['nd']} | {c['exact_p']:.6g} | {threshold if threshold is not None else 'unattainable'} |")
    lines += ["", "## Interpretation limits", ""] + [f"- {value}" for value in result["limitations"]] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    args = parser.parse_args()
    paths = (args.root / "RESULTS.json", args.root / "RESULTS.md")
    require(not any(path.exists() for path in paths), "report outputs are create-only")
    result = build_report(args.root, args.old_root)
    if result["status"] != "COMPLETE_COMPARABLE":
        print(json.dumps(result, indent=2)); return 2
    rendered = markdown(result)
    with paths[0].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    with paths[1].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
    print(json.dumps({"status": result["status"], "outputs": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
