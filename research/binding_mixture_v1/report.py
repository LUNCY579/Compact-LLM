"""Offline mixed-candidate report; custody failures disclose no endpoints.

The control and mixture have different registered training payloads. Evaluation
code and inputs must match exactly. Baseline scoring is reused without renewing
its consumed execution grant or changing its canonical/EOS/decoder rules.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_mixture_v1 import binding_common as bc, materialize, run_probe
from research.binding_signal_eval_v1 import score as bind_score
from research.retention_highdose_lora_resume_v1 import paired, report as control_report
from research.retention_dose_parameterization_v1 import report_battery as prior

ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-mixture001-20260920")
CONTROL = ROOT.parent / "highdose-lora001-resume-20260920"
BASELINE = ROOT.parent / "binding-dev-baseline001-20260920"
LEGS = {**prior.LEGS, "binding": "binding"}
RESERVATIONS = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24),
                "long_context": (256, 24), "binding": (384, 24)}
require = run_probe.require


def read(path):
    return run_probe.read(Path(path))


def sha(path):
    return run_probe.sha(Path(path))


def verify_prelaunch(root):
    """Recheck the prospective executable/artifact freeze used at dispatch."""
    path = root / "PRELAUNCH_BINDING.json"; frozen = read(path)
    require(frozen.get("status") == "PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION", "prelaunch freeze did not pass")
    require({"report.py", "run_probe.py", "binding_common.py", "binding_probe.py"} <= set(frozen["code_files_sha256"]),
            "prelaunch freeze omitted a probe/report executable")
    for name, expected in frozen["code_files_sha256"].items():
        target = (Path(__file__).parent / name).resolve()
        require(target.parent == Path(__file__).parent and sha(target) == expected, "probe/report code changed after prelaunch freeze")
    for name, expected in frozen["artifact_files_sha256"].items():
        target = (root / name).resolve()
        require(target.is_relative_to(root) and sha(target) == expected, "registered build artifact changed after prelaunch freeze")
    for name, key in (("SESSION_ENVELOPE.json", "session_envelope_sha256"), ("PLAN.json", "plan_sha256"),
                      ("GRANT.json", "grant_sha256"), ("PREREG.md", "registration_sha256"), ("BUILD_GATE.json", "build_gate_sha256")):
        require(sha(root / name) == frozen[key], "frozen prelaunch input changed")
    return sha(path)


def attempt_map(root):
    path = Path(root) / "ATTEMPT_MAP.json"
    if not path.exists():
        return {}
    record = read(path)
    require(record.get("schema") == 1 and isinstance(record.get("replacements"), dict), "invalid mixture attempt map")
    values = record["replacements"]
    require(len(values) <= 1, "only ONE total inference retry is registered across all five legs")
    for key, value in values.items():
        require(key in {"mixed-" + leg for leg in LEGS} and isinstance(value, str) and
                Path(value).parts == ("eval", key + "-retry1"), "unregistered mixed replacement")
    return values


def resolve_attempt(root, key, replacements):
    if key != "mixed-binding":
        return prior.resolve_attempt(root, key, replacements)
    first = Path(root) / "eval" / key
    if key not in replacements:
        return first
    directory = Path(root) / replacements[key]
    binding = read(directory / "BINDING.json")
    bc.require_retry_failure(Path(root), binding["before"])
    previous = read(first / "BINDING.json")
    require(previous["grant_sha256"] == binding["grant_sha256"] and
            previous["training_before"] == binding["training_before"], "BIND retry changed its grant/training identity")
    for key in ("arm", "build_gate", "baseline_plan", "baseline_plan_sha256", "grant"):
        require(previous["arguments"][key] == binding["arguments"][key], "BIND retry changed its frozen arguments")
    return directory


def probe_cost_accounting(root, replacements, envelope):
    root = Path(root).resolve()
    required = {"mixed-" + leg for leg in LEGS}
    allowed = required | {key + "-retry1" for key in replacements}
    records, hashes, successes = [], {}, set()
    excluded = {"PLAN.json", "TERMINAL.json", "RECOVERY_PLAN.json", "RECOVERY_COMPLETE.json"}
    for path in sorted((root / "posttrain").glob("*.json")):
        if path.name in excluded:
            continue
        require(path.stem in allowed, "unexpected mixture probe attempt receipt")
        record = read(path); key = record["leg"]
        require(key in required, "attempt receipt has an unregistered leg")
        directory = Path(record["out"]).resolve()
        require(directory == root / "eval" / path.stem and
                path.stem in (key, key + "-retry1"), "attempt output/receipt identity differs")
        n, maximum = RESERVATIONS[key.removeprefix("mixed-")]
        require(record["reserved_rows"] == n and record["reserved_generation_tokens"] == n * maximum and
                type(record["returncode"]) is int, "attempt reservation or terminal code differs")
        supervisor_path = directory / ("supervisor.json" if key == "mixed-binding" else "worker/supervisor.json")
        charged = 900.0
        if supervisor_path.is_file():
            supervisor = read(supervisor_path); observed = supervisor.get("wall_seconds")
            if (type(supervisor.get("returncode")) is int and type(observed) in (int, float) and
                    math.isfinite(observed) and observed >= 0 and supervisor.get("status") in ("EXITED", "TIMEOUT_OR_INTERRUPTED")):
                charged = float(observed)
            hashes[supervisor_path.relative_to(root).as_posix()] = sha(supervisor_path)
        require(type(record["worker_seconds_charged"]) in (int, float) and math.isfinite(record["worker_seconds_charged"]) and
                math.isclose(record["worker_seconds_charged"], charged, rel_tol=0, abs_tol=1e-6), "attempt cost omits observed failure or full unobserved reservation")
        if record["returncode"] == 0:
            require(key not in successes, "multiple successful attempts for one mixture leg")
            successes.add(key)
            expected = root / replacements[key] if key in replacements else root / "eval" / key
            require(directory == expected, "successful attempt is not the fixed selected attempt")
        hashes[path.relative_to(root).as_posix()] = sha(path)
        records.append({"leg": key, "out": str(directory), "returncode": record["returncode"],
                        "worker_seconds_charged": charged, "reserved_rows": n, "reserved_generation_tokens": n * maximum})
    failed = [r for r in records if r["returncode"] != 0]
    require(len(records) == 5 + len(replacements) and successes == required and
            len(failed) == len(replacements) and {r["leg"] for r in failed} == set(replacements), "all-attempt inventory differs from five successes plus the sole explicit retry")
    totals = {"probe_worker_seconds": sum(r["worker_seconds_charged"] for r in records),
              "probe_rows_reserved": sum(r["reserved_rows"] for r in records),
              "probe_generation_tokens_reserved": sum(r["reserved_generation_tokens"] for r in records)}
    for key, limit in (("probe_worker_seconds", "probe_worker_seconds_all_attempts"),
                       ("probe_rows_reserved", "probe_rows_reserved_all_attempts"),
                       ("probe_generation_tokens_reserved", "probe_generation_tokens_reserved_all_attempts")):
        require(totals[key] <= envelope[limit], "global five-leg inference envelope exceeded")
    require(len(records) <= envelope["probe_workers_all_attempts"] and len(failed) <= envelope["probe_retries_total"], "global worker/retry envelope exceeded")
    return {**totals, "attempt_count": len(records), "successful_attempts": 5,
            "failed_attempts": len(failed), "individual_attempts": records, "cost_source_sha256": hashes}


def completion_receipt(root, replacements, envelope):
    original_path = root / "posttrain/TERMINAL.json"; original = read(original_path)
    path, terminal = original_path, original
    if original.get("status") != "COMPLETE_FIVE_PROBES":
        require(original.get("status") == "FAILED_OR_BLOCKED" and len(replacements) == 1, "mixture battery incomplete")
        path = root / "posttrain/RECOVERY_COMPLETE.json"; terminal = read(path)
        require(terminal.get("prior_terminal_sha256") == sha(original_path) and
                terminal.get("attempt_map_sha256") == sha(root / "ATTEMPT_MAP.json"), "recovery is not bound to preserved failure and attempt map")
        require(isinstance(original.get("attempts"), list) and bool(original["attempts"]), "original failure terminal omitted its attempts")
        for record in original["attempts"]:
            source = root / "posttrain" / (Path(record["out"]).name + ".json")
            require(read(source) == record, "an original attempt receipt changed during recovery")
    else:
        require(not replacements, "completed battery cannot replace a successful leg")
    require(terminal.get("status") == "COMPLETE_FIVE_PROBES", "five-leg completion absent")
    costs = probe_cost_accounting(root, replacements, envelope)
    for key in ("probe_worker_seconds", "probe_rows_reserved", "probe_generation_tokens_reserved"):
        require(type(terminal.get(key)) in (int, float) and math.isclose(terminal[key], costs[key], rel_tol=0, abs_tol=1e-6), "completion totals disagree with individual attempts")
    bindings = {}
    for leg in LEGS:
        directory = resolve_attempt(root, "mixed-" + leg, replacements)
        filename = "BINDING.json" if leg == "binding" else "PROBE_WEIGHT_BINDING.json"
        binding = read(directory / filename)
        require(binding.get("status") == "COMPLETE" and binding.get("complete") is True, "final leg binding incomplete")
        bindings["mixed-" + leg] = sha(directory / filename)
    if path != original_path:
        require(terminal.get("final_binding_sha256") == bindings and terminal.get("cost_accounting") == costs,
                "recovery final bindings or individual attempt costs differ")
    elif "final_binding_sha256" in terminal:
        require(terminal["final_binding_sha256"] == bindings, "completion final binding identities differ")
    return {"path": str(path), "sha256": sha(path), "prior_terminal_sha256": sha(original_path),
            "final_binding_sha256": bindings, "cost_accounting": costs}


def check_standard_pair(candidate, control, archive):
    """Only scientific inputs/code/execution settings are paired across arms."""
    for leg in prior.LEGS:
        raw, binding, ids = candidate[leg]; other, old, old_ids = control[leg]
        require(ids == old_ids, "standard paired row inventories differ")
        for key in ("views_sha256", "implementation_digest", "script_sha256", "wrapper_sha256"):
            require(binding[key] == old[key], f"standard evaluation identity differs: {leg}/{key}")
        require(archive.normalize_additional_inputs(binding["additional_input_sha256_before"]) ==
                archive.normalize_additional_inputs(old["additional_input_sha256_before"]), "standard additional inputs differ")
        fields = (("device", "runtime", "numerics", "max_new_tokens") if leg in ("dose", "select") else
                  (("device", "transformers", "numerics", "max_context", "max_new_tokens") if leg == "long_context" else
                   ("device", "registered_bare_policy", "frozen_reference_sizes", "probe_rows")))
        for key in fields:
            require(key in raw and key in other and raw[key] == other[key], f"standard execution setting differs: {leg}/{key}")


def binding_inputs(root, gate, candidate_directory, control_weight, control_root=CONTROL):
    """Revalidate both old baseline workers and new worker before any scoring."""
    scores_path = BASELINE / "BASELINE_SCORES.json"; score_plan_path = BASELINE / "SCORE_PLAN.json"
    require(sha(scores_path) == gate["binding_baseline_scores_sha256"] and
            sha(score_plan_path) == gate["binding_score_plan_sha256"], "prospectively frozen binding baseline changed")
    frozen = read(scores_path); config = read(score_plan_path)
    require(set(config) == bind_score.SCORE_KEYS and config["record"] == "BINDING_DEV_SCORE_PLAN" and
            frozen["score_plan_sha256"] == sha(score_plan_path), "binding scoring plan identity differs")
    paths = {key: bc.absolute(config[key + "_path"], file=True) for key in ("baseline_plan", "grant", "gold", "generator")}
    for key, path in paths.items():
        require(sha(path) == config[key + "_sha256"], "frozen binding scoring input changed")
    require(sha(Path(bind_score.__file__)) == config["scorer_sha256"] == frozen["scorer_sha256"], "frozen binding scorer changed")
    base = bc.baseline.validate_plan(read(paths["baseline_plan"]))
    require(base["models"]["lora"]["weight_sha256"] == control_weight and
            Path(base["models"]["lora"]["path"]).resolve() == Path(control_root).resolve() / "arms/lora", "binding control differs from standard control")
    bc.baseline.validate_grant(paths["baseline_plan"], read(paths["grant"]))
    inputs = bc.baseline.validate_inputs(bc.baseline.read_jsonl(base["input_path"]))
    require(sha(base["input_path"]) == base["input_sha256"] == frozen["input_sha256"], "frozen binding input differs")
    gold = bind_score.validate_gold(inputs, bc.baseline.read_jsonl(paths["gold"]))
    bundles, custody = {}, {}
    for role in ("source", "lora"):
        bundles[role], custody[role] = bind_score.verify_worker(base, paths["baseline_plan"], paths["grant"], role)
        require(custody[role] == frozen["custody"][role], "baseline successful worker differs from reported frozen custody")
    bundles["mixed"], candidate_binding, candidate_base = bc.verify_binding(candidate_directory)
    require(candidate_base == base and bundles["mixed"]["eos_ids"] == bundles["source"]["eos_ids"] == bundles["lora"]["eos_ids"], "binding EOS or frozen generation plan differs")
    for role, bundle in bundles.items():
        require(bundle["numerics"] == bc.baseline.NUMERICS and len(bundle["predictions"]) == 384, "binding generation comparability differs")
    custody["mixed"] = {"binding_sha256": sha(candidate_directory / "BINDING.json"),
                         "generations_sha256": sha(candidate_directory / "GENERATIONS.json")}
    spec = importlib.util.spec_from_file_location("mixture_frozen_binding_policy_decoder", paths["generator"])
    require(spec is not None and spec.loader is not None, "frozen independent decoder unavailable")
    decoder = importlib.util.module_from_spec(spec); sys.modules[spec.name] = decoder; spec.loader.exec_module(decoder)
    require("torch" not in sys.modules, "offline decoder imported a model runtime")
    return inputs, gold, bundles, custody, frozen, decoder.decode_policy_input


def score_binding(inputs, gold, bundles, frozen, decode):
    endpoints, maps, roster = {}, {}, {}
    for role, bundle in bundles.items():
        endpoints[role], maps[role], roster[role] = bind_score.score_rows(inputs, gold, bundle["predictions"], decode)
    for role in ("source", "lora"):
        require(endpoints[role] == frozen["endpoints"][role] and maps[role] == frozen["world_maps"][role] and
                roster[role] == frozen["roster_person_world_any_maps"][role], "rescoring does not reproduce frozen baseline endpoints")
    contrasts = {policy: {name: paired.paired_boolean_summary(maps["lora"][policy][name], values,
                  relationship="cross_training_draw") for name, values in endpoints_.items()}
                 for policy, endpoints_ in maps["mixed"].items()}
    for policy in contrasts:
        control_report.holm({category: contrasts[policy][category] for category in bind_score.UNKNOWN_CATEGORIES})
    return {"endpoints": endpoints, "world_maps": maps, "world_contrasts_mixed_minus_control": contrasts,
            "roster_person_world_any_maps": roster,
            "roster_person_world_any_mixed_minus_control": paired.paired_boolean_summary(roster["lora"], roster["mixed"], relationship="cross_training_draw"),
            "normalization": frozen["normalization"], "paired_unit": "32 independent parent worlds; each category requires all three views"}


def prediction_verdicts(candidate, binding, acquisition, criteria):
    """Criteria JSON supplies every prediction bound and decision threshold."""
    retention = candidate["retention"]["policies"]["permissive"]
    fresh = binding["endpoints"]["mixed"]
    observed = {"dose": candidate["dose"]["correct"], "select": candidate["select"]["correct"],
                "references_retained": retention["references_retained"], "arithmetic_panel": retention["panel_correct"],
                "long_context_report_only": candidate["long_context"]["correct"],
                "literal_unknown_report_only": candidate["long_context"]["literal_unknown"],
                "false_unknown_on_positive": fresh["totals"]["false_unknown_on_positive"],
                **{category: fresh["categories"][category]["canonical_complete_exact"] for category in bind_score.CATEGORIES}}
    bands = {key: prior.band(observed[key], bounds) for key, bounds in criteria["prediction_bands"].items()}
    binding_names = (*bind_score.CATEGORIES, "false_unknown_on_positive")
    p = criteria["primary_engineering"]
    require(p["all_binding_row_bands_required"] is True, "engineering conjunction changed its binding requirement")
    engineering = (all(observed[bank] >= p["each_acquisition_at_least"] for bank in ("dose", "select")) and
                   observed["references_retained"] >= p["fixed47_retained_at_least"] and observed["arithmetic_panel"] >= p["arithmetic150_at_least"] and
                   all(bands[key]["status"] == "PASS" for key in binding_names))
    rule = criteria["binding_efficacy"]
    require(rule["primary_categories"] == list(bind_score.UNKNOWN_CATEGORIES) and rule["holm_family_size"] == 2 and
            rule["positive_net_required"] is True and rule["positive_control_and_false_unknown_bands_required"] is True,
            "binding efficacy family/positive-control rule changed")
    primary = binding["world_contrasts_mixed_minus_control"]["canonical_complete_exact"]
    controls_pass = all(bands[key]["status"] == "PASS" for key in ("person_answerable", "role_answerable", "false_unknown_on_positive"))
    efficacy = controls_pass and any(primary[cat]["net"] > 0 and primary[cat]["holm_adjusted_p"] <= rule["adjusted_exact_p_at_most"] for cat in rule["primary_categories"])
    acquisition_rule = criteria["acquisition_comparison"]
    cost = {bank: {"net": contrast["net"], "minimum": acquisition_rule["cost_engineering_net_minimum"],
                    "status": "PASS" if contrast["net"] >= acquisition_rule["cost_engineering_net_minimum"] else "FAILED_PREDICTION",
                    "establishes_preservation": False} for bank, contrast in acquisition.items()}
    return {"bands": bands, "primary_engineering_conjunction": "PASS" if engineering else "FAILED_PREDICTION",
            "binding_efficacy": "ESTABLISHED_EXPLORATORY_DEV_GAIN" if efficacy else "NOT_ESTABLISHED",
            "binding_positive_controls_pass": controls_pass, "acquisition_cost_engineering": cost,
            "long_context_used_for_selection_or_confirmation": False}


def build_report(root=ROOT, control_root=CONTROL):
    root, control_root = Path(root).resolve(), Path(control_root).resolve()
    try:
        require("torch" not in sys.modules, "offline report requires a fresh process without model runtimes")
        gate_path = root / "BUILD_GATE.json"; gate = read(gate_path)
        require(str(gate.get("status", "")).startswith("PASS"), "mixed build gate is not PASS")
        prelaunch_sha = verify_prelaunch(root)
        for filename, field in (("PREREG.md", "registration_sha256"), ("CRITERIA.json", "criteria_sha256"), ("SESSION_ENVELOPE.json", "session_envelope_sha256")):
            require(sha(root / filename) == gate[field], "mixed prospective registration changed")
        criteria, envelope = read(root / "CRITERIA.json"), read(root / "SESSION_ENVELOPE.json")
        require(criteria["experiment"] == "BINDING-MIXTURE-001" and criteria["route"] == "NEW_SKILL_ACQUISITION_NOT_RESTORATION", "unregistered claim route")
        archive = paired.SourceArchive.load(root / "SOURCE_RELOCATION.json", gate["source_relocation_sha256"])
        require(archive.archive == run_probe.PILOT.resolve() and archive.original == control_report.run_probe.PILOT.resolve(), "clone A/B source relocation differs")
        replacements = attempt_map(root); completion = completion_receipt(root, replacements, envelope)
        control_result = read(control_root / "RESULTS.json")
        require(control_result["status"] == "COMPLETE_COMPARABLE" and sha(control_root / "RESULTS.json") == gate["control_results_sha256"], "frozen high-dose LoRA result changed")
        require(sha(control_report.__file__) == control_result["reporter_sha256"], "frozen control report/scoring helper changed")
        old_replacements = control_report.attempt_map(control_root, control_report.NEW_ARMS)
        loaded, custody, snapshots, receipts = {}, {}, {}, {}
        implementation = run_probe.implementation_digest(); context = None
        for role in ("mixed", "lora"):
            base = root if role == "mixed" else control_root
            mapping = replacements if role == "mixed" else old_replacements
            loaded[role], custody[role] = {}, {}
            for leg, kind in prior.LEGS.items():
                directory = resolve_attempt(base, role + "-" + leg, mapping)
                raw, binding, ids = paired.validate_probe_binding(directory / "result.json", directory / "PROBE_WEIGHT_BINDING.json", kind, source_archive=archive)
                loaded[role][leg] = raw, binding, ids
                precheck = read(directory / "PRECHECK.json"); command = precheck.get("command")
                require(precheck.get("status") == "PRECHECK_PASSED" and precheck.get("kind") == kind and isinstance(command, list) and len(command) >= 2,
                        "standard probe precheck identity absent")
                observed = (str(Path(command[0]).resolve()), archive.canonical_cwd(precheck["cwd"], command[0]))
                context = observed if context is None else context
                require(observed == context and binding["implementation_digest"] == implementation, "standard runtime/scorer context differs")
                custody[role][leg] = {"directory": str(directory), "binding_sha256": sha(directory / "PROBE_WEIGHT_BINDING.json"),
                                       "result_sha256": sha(directory / "result.json"), "weight_sha256": binding["weight_sha256"]}
                if role == "lora":
                    require(custody[role][leg]["binding_sha256"] == control_result["custody"]["lora"][leg]["binding_sha256"], "control is not its authoritative successful attempt")
            bound = loaded[role]["dose"][1]
            require(Path(bound["arm_dir"]).resolve() == base / "arms" / role, "standard leg points to an unregistered arm directory")
            receipt = receipts[role] = read(bound["arm_receipt_path"])
            current_gate_path = gate_path if role == "mixed" else control_root / "BUILD_GATE_LORA_FINAL.json"
            current_gate = read(current_gate_path)
            snapshot_fn = run_probe.arm_snapshot if role == "mixed" else control_report.run_probe.arm_snapshot
            live = snapshot_fn(Path(bound["arm_dir"]), current_gate); snapshots[role] = live["training"]
            require(receipt["stored_dtype"] == "float32" and receipt["checkpoint_binding"]["checkout"] == current_gate["pilot_checkout"], "arm precision/training checkout differs")
            for leg, kind in prior.LEGS.items():
                raw, current, _ = loaded[role][leg]
                require(Path(current["build_gate_path"]).resolve() == current_gate_path and current["build_gate_sha256"] == sha(current_gate_path), "standard leg used another build gate")
                require(current["weight_sha256"] == live["weight_sha256"] and current["training_identity"] == bound["training_identity"] and
                        current["training_snapshot_before"] == current["training_snapshot_after"] == live["training"], "within-arm model/training custody differs")
                require(current["script_sha256"] == current_gate["probe_script_sha256"][kind] and current["wrapper_sha256"] == current_gate["wrapper_sha256"], "standard probe code differs from its gate")
                if leg in ("dose", "select"):
                    require(raw["instrument"] == ("process-transition-probe-001" if leg == "dose" else "select-views-001") and raw["max_new_tokens"] == 128 and
                            current["views_sha256"] == current_gate["banks"][leg]["sha256"], "acquisition instrument/settings mislabeled")
        require(receipts["mixed"]["source_assets_sha256"] == receipts["lora"]["source_assets_sha256"], "arms have different starting source assets")
        require(snapshots["mixed"]["updates"] == 296 and snapshots["lora"]["updates"] == 193 and
                snapshots["mixed"]["plan_sha256"] != snapshots["lora"]["plan_sha256"], "different registered training trajectories were collapsed")
        check_standard_pair(loaded["mixed"], loaded["lora"], archive)
        arm = receipts["mixed"]
        supervision = materialize._supervised_success(Path(arm["training_result_path"]).parent, arm["checkpoint_binding"])
        for name in ("ARM.json", "MATERIALIZATION.json"):
            receipt = read(root / "arms/mixed" / name)
            require(all(receipt.get(key) == value for key, value in supervision.items()) and
                    sha(receipt["training_result_path"]) == receipt["training_result_sha256"] and
                    sha(receipt["merge_qualification_path"]) == receipt["merge_qualification_sha256"], "mixed training/merge supervision changed")
        directory = resolve_attempt(root, "mixed-binding", replacements)
        binding_data = binding_inputs(root, gate, directory, loaded["lora"]["dose"][1]["weight_sha256"], control_root)
        archive.revalidate()
        # All five candidate legs and all control custody are now established.
        endpoints, maps = control_report.extract(loaded, control_report.load_pure_retention())
        acquisition = {}
        for bank in ("dose", "select"):
            left, right = loaded["lora"][bank], loaded["mixed"][bank]
            acquisition[bank] = paired.paired_boolean_summary(paired._rows(left[0], left[2]), paired._rows(right[0], right[2]), relationship="cross_training_draw")
        control_report.holm(acquisition)
        for value in acquisition.values():
            floor = criteria["acquisition_comparison"]["absolute_net_floor"]
            value.update(cross_training_draw_abs_net_floor=floor, cross_training_draw_floor_cleared=abs(value["net"]) >= floor,
                         structural_claim_eligible=abs(value["net"]) >= floor and value["holm_adjusted_p"] <= .05)
        inputs, gold, bundles, binding_custody, frozen, decode = binding_data
        binding = score_binding(inputs, gold, bundles, frozen, decode)
        secondary = {name: paired.paired_boolean_summary(maps["lora"][name], values, relationship="cross_training_draw") for name, values in maps["mixed"].items()}
        verdicts = prediction_verdicts(endpoints["mixed"], binding, acquisition, criteria)
        archive.revalidate()
        require(verify_prelaunch(root) == prelaunch_sha, "prelaunch freeze changed during report")
        require("torch" not in sys.modules, "offline report imported a model runtime")
        return {"schema": 1, "experiment": "BINDING-MIXTURE-001", "status": "COMPLETE_COMPARABLE", "created_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root), "control_root": str(control_root), "build_gate_sha256": sha(gate_path), "criteria_sha256": sha(root / "CRITERIA.json"),
                "reporter_sha256": sha(Path(__file__)), "prelaunch_binding_sha256": prelaunch_sha,
                "verified_completion": completion, "training_snapshots": snapshots, "training_supervision": supervision,
                "source_relocation_sha256": archive.sha256, "custody": custody, "binding_custody": binding_custody, "endpoints": endpoints, "binding": binding,
                "acquisition_contrasts_mixed_minus_control": acquisition, "secondary_contrasts_mixed_minus_control": secondary, "predictions": verdicts,
                "multiplicity": {"acquisition": "separate Holm(2) family for dose and SELECT", "binding": "separate Holm(2) UNKNOWN-category family; canonical + EOS is primary; strict is sensitivity", "secondary": "nominal descriptive paired p-values"},
                "limitations": ["Exploratory DEV new-skill acquisition; no restoration claim or independent confirmation.",
                    "Training payloads and update/token counts differ deliberately. This compares practical mixture regimes, not an isolated signal coefficient.",
                    "Binding paired units are 32 worlds, never 384 independent paraphrases; every category-world requires all three views correct.",
                    "The acquisition cost allowance net >= -20 is engineering only. Significant losses remain damage; nonsignificance is not equivalence or preservation.",
                    "The fixed 47 permissive-source reference IDs remain fixed; strict policy has 41 own-source successes and is reported separately.",
                    "Long context is spent report-only material. Its literal UNKNOWN substring metric does not establish semantic abstention repair.",
                    "Reserved-bank coverage and old long-context stopword fingerprints remain unverified. Human data review was NOT_PERFORMED.",
                    "No automatic selection, registry promotion, quality claim, or release permission follows any endpoint."],
                "quality_claim": False, "selection_permitted": False, "release_permitted": False, "human_data_review": "NOT_PERFORMED"}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, subprocess.SubprocessError) as exc:
        return {"schema": 1, "experiment": "BINDING-MIXTURE-001", "status": paired.BLOCKED, "reasons": [str(exc)],
                "quality_claim": False, "selection_permitted": False, "release_permitted": False}


def markdown(result):
    lines = ["# BINDING-MIXTURE-001", "", "Status: **" + result["status"] + "**.", ""]
    if result["status"] != "COMPLETE_COMPARABLE":
        return "\n".join(lines + result["reasons"]) + "\n"
    p = result["predictions"]
    lines += [f"Primary engineering conjunction: **{p['primary_engineering_conjunction']}**. Binding efficacy: **{p['binding_efficacy']}**.", "",
              "| Arm | Dose /384 | SELECT /384 | Retained /47 | Panel /150 | LC /256 (report only) | UNKNOWN /64 (report only) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for role, v in result["endpoints"].items():
        r = v["retention"]["policies"]["permissive"]
        lines.append(f"| {role} | {v['dose']['correct']} | {v['select']['correct']} | {r['references_retained']} | {r['panel_correct']} | {v['long_context']['correct']} | {v['long_context']['literal_unknown']} |")
    lines += ["", "| Arm | Retention policy | Fixed cohort retained /47 | Panel /150 | Own-policy source references | Terminated /150 |",
              "|---|---|---:|---:|---:|---:|"]
    for role, values in result["endpoints"].items():
        retention = values["retention"]
        for policy, value in retention["policies"].items():
            lines.append(f"| {role} | {policy} | {value['references_retained']} | {value['panel_correct']} | {value['own_policy_source_reference_n']} | {retention['terminated']} |")
    lines += ["", "| Model | BIND category | Canonical text /96 | Strict text /96 | Canonical + EOS /96 | Strict + EOS /96 | Terminated /96 |", "|---|---|---:|---:|---:|---:|---:|"]
    for role, value in result["binding"]["endpoints"].items():
        for category, v in value["categories"].items():
            lines.append(f"| {role} | {category} | {v['canonical_text_exact']} | {v['strict_text_exact']} | {v['canonical_complete_exact']} | {v['strict_complete_exact']} | {v['terminated']} |")
    lines += ["", "| Model | BIND terminated /384 | False exact UNKNOWN on positive /192 | Exact roster person on unanswerable /192 | Any exact roster person /32 worlds |",
              "|---|---:|---:|---:|---:|"]
    for role, value in result["binding"]["endpoints"].items():
        v = value["totals"]
        lines.append(f"| {role} | {v['terminated']} | {v['false_unknown_on_positive']} | {v['exact_roster_person_on_unanswerable']} | {value['roster_person_world_any_count']} |")
    lines += ["", "| Contrast (mixed minus control) | net | nd | Exact p | Holm p | Minimum significant absolute net at observed nd |", "|---|---:|---:|---:|---:|---:|"]
    contrasts = dict(result["acquisition_contrasts_mixed_minus_control"])
    for policy, values in result["binding"]["world_contrasts_mixed_minus_control"].items():
        contrasts.update({"BIND " + policy + "/" + key: value for key, value in values.items()})
    contrasts.update(result["secondary_contrasts_mixed_minus_control"])
    for name, v in contrasts.items():
        lines.append(f"| {name} | {v['net']:+d} | {v['nd']} | {v['exact_p']:.8g} | {v.get('holm_adjusted_p', 'descriptive')} | {v['minimum_significant_abs_net_at_observed_nd']} |")
    lines += ["", "Binding rows are descriptive; the statistical unit is 32 parent worlds.", "", "| Prediction | Observed | Registered band | Verdict |", "|---|---:|---|---|"]
    for name, v in p["bands"].items():
        lines.append(f"| {name} | {v['observed']} | {v['lower']}–{v['upper']} | {v['status']} |")
    lines += ["", *["- " + value for value in result["limitations"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--control-root", type=Path, default=CONTROL)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(); output = args.out or args.root / "RESULTS.json"
    require(not output.exists() and not output.with_suffix(".md").exists(), "report outputs are create-only")
    result = build_report(args.root, args.control_root)
    run_probe.write(output, result)
    with output.with_suffix(".md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(result))
    print({"status": result["status"], "out": str(output)})
    return 0 if result["status"] == "COMPLETE_COMPARABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
