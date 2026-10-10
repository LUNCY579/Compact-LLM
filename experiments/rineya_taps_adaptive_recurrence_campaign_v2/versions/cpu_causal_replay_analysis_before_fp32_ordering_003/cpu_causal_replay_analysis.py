"""Source-bound causal replay analysis; standard library, no native imports.

All task cells and execution proofs are required before primary summaries.
Canonical answer PREFIX likelihood and accepted-EOS mass remain separate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent.parent
LANE = "CPU_CAUSAL_REPLAY_FP32"
STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
CUSTOM = "b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d"
QWEN = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
EOS = (151643, 151645)
ALPHAS = (1., 1.01, .99)
GIB = 1024**3
DESIGN = "f8b962447f403105a564ec071bbfa89fd3e40843fcfef6800b7ec72a22412492"
DESIGN_4500 = "d586b68c0e43f8cbccaee0593be9356072a3987b66793ccea6f25abd736c6657"
DEADLINE_CONTRACTS = {3600: ("CPU_CAUSAL_REPLAY_DESIGN_DRAFT_002.md", DESIGN),
                      4500: ("CPU_CAUSAL_REPLAY_DESIGN_DRAFT_003.md", DESIGN_4500)}
PACKET = "021ea720f4b644d0e0fff0b975ab460d80e6c02cb401400fd0be58a0ba4cb359"
SOURCE_PINS = {"cpu_h3_native.py": "f242676e0ec0f019dd94ac27e5f3cb5eb8f3e311beb64cc0355102492396eaca",
               "sensitivity.py": "5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d",
               "CPU_CAUSAL_REPLAY_DESIGN_DRAFT_002.md": DESIGN,
               "CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json": PACKET,
               "bulk_bank.py": "3242439231064fc81991e157c0da62833f4e4223ec71d60cf9ffd2851815b5eb",
               "open_bank.py": "25db8ad7b8bd4166188eccab4be73c9a430f696610dea5b8c83b380883f9d42a",
               "runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json": "566e6d6e64f258ff4deaa28a9909841a451fb96cf939ca44533726c59b2a7ff0",
               "runs/causal-replay-DEV-export-001/DEVELOPMENT_TOKEN_BINDING.json": "6529b48b26264123fafc9f81d67cb6cc5a957fe30a0ab4853440790c56bfdd53",
               "runs/causal-replay-DEV-export-001/MANIFEST.json": "278242ad25488a5f7896957aab4e9269dfdf96babe21abf64bc8167838514302"}
EXPECTED_EVENTS = {"VALUE_FORWARD_STARTED": 18, "VALUE_FORWARD_COMPLETED": 17,
                   "VALUE_FORWARD_FAILED": 1, "REPLAY_FORWARD_STARTED": 1905,
                   "REPLAY_FORWARD_COMPLETED": 1905,
                   "CACHE_REJECTION_STARTED": 2, "CACHE_REJECTED": 2}
EXPECTED_COUNTS = {"cases": 192, "parents": 96, "answer_targets": 440,
                   "EOS_targets": 192, "model_API_attempts": 1925,
                   "full_forward_attempts": 1923, "completed_forwards": 1922,
                   "state_adjoints": 635, "completed_prefix_positions": 252696,
                   "cache_rejections": 2, "expected_partial": 1}
FEATURE_KEYS = {"case_id", "progress", "fluctuation", "balance", "input_ids_sha256",
                "teacher_conditioned", "target_independent", "preceding_gold_tokens",
                "preceding_gold_count", "phase", "prefix_kind", "source_prefix_sha256",
                "frozen_before_loss", "loss_or_gradient_predictors", "freeze_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def mapping(value):
    return value if isinstance(value, dict) else {}


def _resources(sample, where, errors, initial=False, parent=False):
    sample = mapping(sample)
    rss, host = sample.get("rss_bytes"), sample.get("host_available_bytes")
    if not numeric(rss) or rss < 0 or rss > 6*GIB or not numeric(host) or host < (math.ceil(7.4*GIB) if initial else GIB):
        errors.append(where+": CPU RSS/host reserve proof missing or violated")
    peak = sample.get("peak_rss_bytes")
    if peak is not None and (not numeric(peak) or not 0 <= peak <= 6*GIB):
        errors.append(where+": lifetime CPU peak RSS violated")
    if not parent and sample.get("device") != "cpu":
        errors.append(where+": CPU resource device proof missing")


def _rms(row, where, errors):
    records = row.get("RMS", [])
    if not isinstance(records, list) or len(records) != 1:
        errors.append(where+": actual block RMS proof missing")
        return
    record = mapping(records[0])
    normal, controlled, ratio = (record.get(k) for k in ("normal_rms", "controlled_rms", "rms_ratio_to_normal"))
    if not all(numeric(v) and v >= 0 for v in (normal, controlled, ratio)) or not .25 <= ratio <= 4:
        errors.append(where+": actual block RMS gate violated")
    elif not same_number(ratio, 1. if normal == controlled == 0 else controlled/max(normal, 1e-12)):
        errors.append(where+": actual RMS ratio inconsistent")


def same_number(a, b):
    return numeric(a) and numeric(b) and abs(a-b) <= 1e-9*(1+abs(a)+abs(b))


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def rank(values):
    return [1+sum(other < value for other in values)+(sum(other == value for other in values)-1)/2
            for value in values]


def correlation(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    a, b = math.fsum(x)/len(x), math.fsum(y)/len(y)
    denominator = math.sqrt(math.fsum((v-a)**2 for v in x)*math.fsum((v-b)**2 for v in y))
    return math.fsum((u-a)*(v-b) for u, v in zip(x, y))/denominator if denominator else None


def _number(value, where, errors):
    if not numeric(value):
        errors.append(where+": missing/nonfinite numeric scalar")
        return None
    return float(value)


def _triplet(result, where, errors, eos=False):
    """Recompute, never trust saved gate or case aggregation."""
    if not isinstance(result, dict):
        errors.append(where+": missing target result")
        return None
    values = []
    for key, alpha, gradients in zip(("nominal", "plus", "minus"), ALPHAS, (1, 0, 0)):
        row = result.get(key, {})
        if not isinstance(row, dict):
            row = {}
        loss = _number(row.get("loss"), where+"/"+key+" loss", errors)
        if row.get("alpha") != alpha or row.get("state_gradient_evaluations") != gradients:
            errors.append(where+"/"+key+": alpha/adjoint contract differs")
        if row.get("anchor_counts") != [1, 1, 1]:
            errors.append(where+"/"+key+": anchors not one-shot")
        _rms(row, where+"/"+key, errors)
        if eos:
            eot = _number(row.get("canonical_EOT_CE"), where+"/"+key+" EOT", errors)
            if loss is not None and eot is not None and loss > eot+1e-5:
                errors.append(where+"/"+key+": EOS mass loss exceeds single EOT CE")
        values.append(loss)
    adjoint = _number(mapping(result.get("nominal")).get("sensitivity"), where+" adjoint", errors)
    if adjoint is None or any(value is None for value in values):
        return None
    fd = (values[1]-values[2])/.02
    error, tolerance = abs(adjoint-fd), .001+.03*abs(adjoint)
    if error > tolerance:
        errors.append(where+": target FD gate failed")
    if not same_number(result.get("finite_difference"), fd):
        errors.append(where+": saved target FD differs")
    if not same_number(result.get("absolute_error"), error) or not same_number(result.get("registered_tolerance"), tolerance):
        errors.append(where+": saved target gate fields differ")
    if result.get("full_model_calls") != 3 or result.get("state_adjoints") != 1:
        errors.append(where+": target call/adjoint counts differ")
    return {"loss": dict(zip(("nominal", "plus", "minus"), values)),
            "adjoint": adjoint, "finite_difference": fd, "absolute_error": error,
            "tolerance": tolerance,
            "canonical_EOT_CE": {key: result[key].get("canonical_EOT_CE") for key in ("nominal", "plus", "minus")} if eos else None}


def _proofs(bundle, errors):
    receipt, terminal = mapping(bundle.get("receipt")), mapping(bundle.get("terminal"))
    admission, review = mapping(bundle.get("admission")), mapping(bundle.get("review"))
    registration, qualification = mapping(bundle.get("registration")), mapping(bundle.get("qualification"))
    forecast = mapping(bundle.get("forecast"))
    deadline = admission.get("hard_deadline_seconds")
    contract = DEADLINE_CONTRACTS.get(deadline) if type(deadline) is int else None
    if contract is None:
        errors.append("unreviewed parent deadline contract")
        deadline = 3600  # arithmetic fallback only; failed gate prevents a primary
        contract = DEADLINE_CONTRACTS[3600]
    design_name, design_sha = contract
    if admission.get("finalization_reserve_seconds") != 180 or admission.get("native_finalization_reserve_seconds") != 1800:
        errors.append("unchanged internal/global reserves missing")
    if deadline == 4500:
        if bundle.get("io_checks", {}).get("4500_human_scope") is not True:
            errors.append("4500 requires a new exact direct-human permit proof")
        if registration.get("parent_deadline_seconds") != 4500 or registration.get("body_deadline_seconds") != 4320 or registration.get("internal_reserve_seconds") != 180:
            errors.append("4500/4320 registered budget proof differs")
    required = [
        (admission.get("status") == "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH", "READY admission absent"),
        (receipt.get("status") == "CPU_CAUSAL_REPLAY_COMPLETE_SURROGATE_ONLY", "worker not complete"),
        (terminal.get("status") == "WORKER_TERMINAL" and terminal.get("exit_code") == 0, "terminal not successful"),
        (receipt.get("lane") == registration.get("lane") == admission.get("lane") == LANE, "CPU lane differs"),
        (qualification.get("status") == "PASS_CPU_CAUSAL_REPLAY_MECHANICS", "native qualification absent"),
        (qualification.get("device") == registration.get("device") == admission.get("device") == "cpu", "CPU device proof absent"),
        (registration.get("dtype") == admission.get("dtype") == "float32", "FP32 proof absent"),
        (registration.get("threads") == 4, "CPU thread proof absent"),
        (receipt.get("loaded_custom_code_sha256") == CUSTOM, "custom head code differs/missing"),
        (receipt.get("parameter_gradients") == qualification.get("parameter_gradients") == "NONE", "frozen parameter gradient proof absent"),
        (receipt.get("trainable_parameters") == 0 and receipt.get("only_state_leaf_grad") is True, "state-only frozen-gradient proof absent"),
        (receipt.get("training") is False and receipt.get("optimizer_steps") == 0, "training/optimizer proof absent"),
        (receipt.get("generation_pool_access") is False and receipt.get("native_test") is False, "pool/TEST firewall proof absent"),
        (receipt.get("source_unchanged") is True and qualification.get("source_unchanged") is True, "unchanged source proof absent"),
        (receipt.get("checkpoint_files_unchanged") is True and qualification.get("checkpoint_files_unchanged") is True, "unchanged asset proof absent"),
        (qualification.get("maximum_prefix_tokens") == 256 and qualification.get("cache_rejections") == 2 and qualification.get("forced_restoration") is True, "shape/cache/restoration proof absent"),
        (qualification.get("qualified_state_adjoints") == 3 and qualification.get("qualified_vector_pairs") == 3, "three qualification adjoints/vectors absent"),
        (review.get("status") == "PASS_CPU_CAUSAL_REPLAY_DISPATCH", "dispatch review absent"),
        (forecast.get("admitted") is True and forecast.get("device") == "cpu", "fresh CPU forecast not admitted"),
    ]
    for condition, message in required:
        if not condition:
            errors.append(message)
    prefix = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/"
    dispatch = mapping(admission.get("dispatch_hashes"))
    for name, expected in SOURCE_PINS.items():
        if name == "CPU_CAUSAL_REPLAY_DESIGN_DRAFT_002.md":
            name, expected = design_name, design_sha
        if dispatch.get(prefix+name) != expected:
            errors.append("mandatory admitted source/data pin differs: "+name)
    worker = registration.get("worker_sha256")
    if not isinstance(worker, str) or len(worker) != 64 or dispatch.get(prefix+"cpu_causal_replay_native.py") != worker:
        errors.append("actual worker/admission source pin missing")
    if registration.get("design_sha256") != design_sha or registration.get("metadata_packet_sha256") != PACKET or registration.get("CPU_helper_sha256") != SOURCE_PINS["cpu_h3_native.py"] or registration.get("math_sha256") != SOURCE_PINS["sensitivity.py"]:
        errors.append("actual helper/math/design/data lineage missing")
    for name, values in (("receipt", [receipt.get(k) for k in ("loaded_state_sha256", "before_state_sha256", "after_state_sha256")]),
                         ("qualification", [qualification.get(k) for k in ("before_state_sha256", "after_state_sha256")])):
        if any(value != STATE for value in values):
            errors.append(name+": missing or different actual frozen state")
    runtime = mapping(registration.get("runtime"))
    for key, expected in (("python", "3.11.15"), ("torch", "2.12.1+cu126"), ("transformers", "5.14.1")):
        if runtime.get(key) != expected:
            errors.append("runtime "+key+" proof differs/missing")
    if runtime.get("qwen_sha256", runtime.get("qwen_implementation_sha256", runtime.get("Qwen_implementation_sha256"))) != QWEN:
        errors.append("runtime Qwen implementation proof absent")
    _resources(mapping(bundle.get("preflight")).get("resources"), "preweight preflight", errors, initial=True)
    for name in ("after_load_resources", "final_resources", "final_resources_on_stop"):
        _resources(receipt.get(name), "receipt "+name, errors)
    _resources(qualification.get("resources"), "qualification resources", errors)
    parent_samples = bundle.get("resources", [])
    if not parent_samples:
        errors.append("parent resource samples absent")
    for sample in parent_samples:
        _resources(sample, "parent resource sample", errors, parent=True)
    for key in ("admission_review_registration", "population_metadata", "frozen_source_closure", "source_snapshot", "qualification_vectors"):
        if bundle.get("io_checks", {}).get(key) is not True:
            errors.append(key+": nonmissing source binding proof required")
    if not numeric(terminal.get("native_worker_seconds")) or not 0 < terminal["native_worker_seconds"] <= deadline:
        errors.append("actual native worker cost missing/outside ceiling")
    for label, ledger in (("raw", mapping(bundle.get("ledger"))), ("receipt", mapping(receipt.get("compute_ledger")))):
        for key, expected in EXPECTED_COUNTS.items():
            if ledger.get(key) != expected:
                errors.append(label+" compute ledger "+key+" differs/missing")
    if forecast.get("factor") != 1.5 or forecast.get("remaining_task_targets") != 632 or forecast.get("remaining_task_forwards") != 1896 or forecast.get("internal_reserve_seconds") != 180 or forecast.get("parent_deadline_seconds") != deadline:
        errors.append("complete fresh CPU forecast contract missing")
    if deadline == 4500 and forecast.get("body_deadline_seconds") != 4320:
        errors.append("4500 forecast body reserve proof differs")
    maximum = [e for e in bundle.get("events", []) if e.get("event") == "REPLAY_FORWARD_COMPLETED" and e.get("label") in ("qualification_maximum256_answer", "qualification_maximum256_EOS")]
    timing_contract = Counter((e.get("label"), e.get("alpha"), e.get("state_gradient_evaluations")) for e in maximum)
    expected_timing = Counter((label, alpha, grad) for label in ("qualification_maximum256_answer", "qualification_maximum256_EOS") for alpha, grad in zip(ALPHAS, (1, 0, 0)))
    if len(maximum) != 6 or timing_contract != expected_timing or any(not numeric(e.get("elapsed_seconds")) or e["elapsed_seconds"] <= 0 for e in maximum):
        errors.append("fresh scalar/EOS maximum CPU timing matrix missing")
    else:
        gradient = max(e["elapsed_seconds"] for e in maximum if e.get("state_gradient_evaluations") == 1)
        value = max(e["elapsed_seconds"] for e in maximum if e.get("state_gradient_evaluations") == 0)
        remaining = 632*(gradient+2*value)*1.5
        if not same_number(forecast.get("maximum_whole_forward_adjoint_seconds"), gradient) or not same_number(forecast.get("maximum_value_forward_seconds"), value) or not same_number(forecast.get("remaining_forecast_seconds"), remaining) or not numeric(forecast.get("elapsed_seconds")):
            errors.append("fresh CPU complete forecast arithmetic differs")
        elif forecast["elapsed_seconds"]+remaining+180 > deadline:
            errors.append("complete fresh CPU matrix does not fit")
    parity = qualification.get("parity", [])
    parity = parity if isinstance(parity, list) else []
    combinations = {(p.get("label"), p.get("alpha"), p.get("extra")) for p in parity if isinstance(p, dict)}
    expected = {(label, alpha, extra) for label in ("representative", "maximum256")
                for alpha, extra in ((1., 0), (.99, 1), (1., 1), (1.01, 1))}
    if len(parity) != 8 or combinations != expected or any(p.get("exact_equal") is not True or not p.get("reference_sha256") or p.get("reference_sha256") != p.get("diagnostic_sha256") for p in parity):
        errors.append("eight exact value parity proofs differ/missing")
    scalar = qualification.get("scalar_CE_FD", [])
    scalar = scalar if isinstance(scalar, list) else []
    if len(scalar) != 2:
        errors.append("two scalar CE native FD checks missing")
    for index, result in enumerate(scalar):
        _triplet(result, "qualification scalar "+str(index), errors)
    _triplet(qualification.get("EOS_mass_FD"), "qualification EOS mass", errors, eos=True)


def _journal(events, errors):
    counts = Counter(e.get("event") for e in events)
    if dict(counts) != EXPECTED_EVENTS:
        errors.append("exact attempted/completed event counts differ")
    complete = [e for e in events if e.get("event") in ("VALUE_FORWARD_COMPLETED", "REPLAY_FORWARD_COMPLETED")]
    if any(type(e.get("prefix_tokens")) is not int for e in complete):
        errors.append("completed prefix positions missing")
    positions = sum(e.get("prefix_tokens", 0) for e in complete if type(e.get("prefix_tokens")) is int)
    adjoints = sum(e.get("state_gradient_evaluations", 0) for e in complete if type(e.get("state_gradient_evaluations")) is int)
    if positions != 252696 or adjoints != 635:
        errors.append("252696 positions/635 adjoints differ")
    for event in complete:
        _resources(event.get("resources"), "completed forward resources", errors)
    failures = [e for e in events if e.get("event", "").endswith("FAILED")]
    if len(failures) != 1 or failures[0].get("error") != "RuntimeError: EXPECTED_CAUSAL_SUFFIX_EXCEPTION" or failures[0].get("partial_compute") != "UNKNOWN":
        errors.append("one exact intentional partial required")
    caches = [e for e in events if e.get("event") == "CACHE_REJECTION_STARTED"]
    if {(e.get("use_cache"), e.get("cache_object_present")) for e in caches} != {(True, False), (False, True)}:
        errors.append("two cache variants missing")
    if any(e.get("model_compute_started") is not False or e.get("wrapper_forward_count") != 0 for e in events if e.get("event") == "CACHE_REJECTED"):
        errors.append("cache refusal entered model")
    return {"events": dict(counts), "API_attempts": sum(counts[k] for k in ("VALUE_FORWARD_STARTED", "REPLAY_FORWARD_STARTED", "CACHE_REJECTION_STARTED")),
            "completed_forwards": len(complete), "state_adjoints": adjoints,
            "completed_prefix_positions": positions, "partial_positions": None}


def _direction_summary(cases):
    neutral = [x for x in cases if x["neutral"]]
    uncertain = [x for x in cases if not x["neutral"] and x["numerically_uncertain"]]
    reliable = [x for x in cases if not x["neutral"] and not x["numerically_uncertain"]]
    return {"cases": len(cases), "neutral_cases": len(neutral), "nonneutral_cases": len(cases)-len(neutral),
            "numerically_uncertain_nonneutral_cases": len(uncertain), "reliable_sign_denominator": len(reliable),
            "coherence_agreements": sum(x["coherence_direction"] == x["useful_direction"] for x in reliable),
            "constant_increase_agreements": sum(x["useful_direction"] == 1 for x in reliable),
            "constant_decrease_agreements": sum(x["useful_direction"] == -1 for x in reliable),
            "spearman_all_cases_balance_negative_derivative": correlation(rank([x["balance"] for x in cases]), rank([-x["mean_answer_derivative"] for x in cases]))}


def _population_summaries(cases):
    parents = defaultdict(list)
    for case in cases:
        parents[case["parent_id"]].append(case)
    parent_rows = []
    for parent, variants in sorted(parents.items()):
        reliable = [v for v in variants if not v["neutral"] and not v["numerically_uncertain"]]
        parent_rows.append({"parent_id": parent, "category": variants[0]["category"],
                           "case_ids": [v["case_id"] for v in variants], "variant_rows": variants,
                           "mean_answer_loss": {a: math.fsum(v["mean_answer_loss"][a] for v in variants)/len(variants) for a in ("nominal", "plus", "minus")},
                           "mean_answer_derivative": math.fsum(v["mean_answer_derivative"] for v in variants)/len(variants),
                           "reliable_cases": len(reliable),
                           "conditional_coherence_agreement_mean": math.fsum(v["coherence_direction"] == v["useful_direction"] for v in reliable)/len(reliable) if reliable else None,
                           "conditional_constant_increase_agreement_mean": math.fsum(v["useful_direction"] == 1 for v in reliable)/len(reliable) if reliable else None,
                           "conditional_constant_decrease_agreement_mean": math.fsum(v["useful_direction"] == -1 for v in reliable)/len(reliable) if reliable else None,
                           "opposite_reliable_variant_directions": len(reliable) == 2 and reliable[0]["useful_direction"] != reliable[1]["useful_direction"],
                           "any_plus_loss_increase": any(v["answer_loss_delta_plus"] > 0 for v in variants),
                           "any_minus_loss_increase": any(v["answer_loss_delta_minus"] > 0 for v in variants),
                           "pair_kind": variants[0]["pair_kind"]})
    strata = {}
    for field in ("category", "difficulty_stratum", "answerability", "source_status"):
        values = defaultdict(list)
        for case in cases:
            values[str(case[field])].append(case)
        strata[field] = {key: _direction_summary(value) for key, value in sorted(values.items())}
    by_category = defaultdict(list)
    for row in parent_rows:
        by_category[row["category"]].append(row)
    category_means = {category: {a: math.fsum(p["mean_answer_loss"][a] for p in ps)/len(ps) for a in ("nominal", "plus", "minus")}
                      for category, ps in sorted(by_category.items())}
    population = {a: math.fsum(v[a] for v in category_means.values())/len(category_means) for a in ("nominal", "plus", "minus")}
    return {"parents": parent_rows, "parent_count": len(parent_rows), "shared_generator_categories": len(by_category),
            "category_mean_answer_loss": category_means, "equal_category_parent_mean_answer_loss": population,
            "strata": strata, "any_variant_loss_increase_parent_counts": {"plus": sum(p["any_plus_loss_increase"] for p in parent_rows), "minus": sum(p["any_minus_loss_increase"] for p in parent_rows)},
            "opposite_reliable_direction_parents": sum(p["opposite_reliable_variant_directions"] for p in parent_rows)}


def analyze_bundle(bundle):
    """Pure in-memory analysis; trusted IO checks are supplied only by read_run."""
    errors = list(bundle.get("io_errors", []))
    _proofs(bundle, errors)
    events = bundle.get("events", [])
    journal = _journal(events, errors)
    population, targets = bundle.get("population", []), bundle.get("targets", [])
    features, outcomes = bundle.get("features", []), bundle.get("outcomes", [])
    pop = {p.get("case_id"): p for p in population}
    if len(population) != 192 or len(pop) != 192:
        errors.append("all192 unique population cases required")
    bank_rows = bundle.get("bank", [])
    bank = {row.get("case_id", row.get("id")): row for row in bank_rows}
    if len(bank_rows) != 192 or set(bank) != set(pop) or any(row.get("split") != "DEVELOPMENT" for row in bank_rows):
        errors.append("frozen DEV-only bank metadata proof absent")
    parents = Counter(p.get("parent_id") for p in population)
    categories = Counter(p.get("category") for p in population)
    if len(parents) != 96 or set(parents.values()) != {2} or len(categories) != 6 or set(categories.values()) != {32}:
        errors.append("96 paired parents/six32-case categories required")
    expected_cells = {}
    for case, plan in pop.items():
        answer = plan.get("canonical_answer_token_ids", [])
        ts = plan.get("answer_targets", [])
        if not answer or len(answer) != len(ts):
            errors.append(str(case)+": frozen answer sequence missing")
            continue
        prompt = ts[0].get("prefix_token_ids", [])
        for index, target in enumerate(ts):
            prefix = prompt+answer[:index]
            if target.get("answer_position") != index or target.get("target_token_id") != answer[index] or target.get("prefix_token_ids") != prefix or target.get("prefix_sha256") != digest(prefix):
                errors.append(str(case)+": frozen causal answer construction differs")
            expected_cells[(case, "answer", index)] = {"prefix": prefix, "hash": digest(prefix), "target": answer[index], "preceding": index}
        eos_prefix = prompt+answer
        if plan.get("EOS", {}).get("prefix_token_ids") != eos_prefix or plan.get("EOS", {}).get("prefix_sha256") != digest(eos_prefix):
            errors.append(str(case)+": frozen EOS prefix differs")
        expected_cells[(case, "EOS", len(answer))] = {"prefix": eos_prefix, "hash": digest(eos_prefix), "target": EOS, "preceding": len(answer)}
    if len(expected_cells) != 632 or sum(key[1] == "answer" for key in expected_cells) != 440:
        errors.append("frozen440 answer/192 EOS targets required")
    target_map = {}
    for target in targets:
        key = (target.get("case_id"), target.get("kind"), target.get("answer_position"))
        if key in target_map:
            errors.append("duplicate target cell "+str(key))
        target_map[key] = target
    if len(targets) != 632 or set(target_map) != set(expected_cells):
        errors.append("exact632 target cell coverage differs")
    feature_map = {f.get("case_id"): f for f in features}
    outcome_map = {o.get("case_id"): o for o in outcomes}
    if len(features) != 192 or set(feature_map) != set(pop) or len(outcomes) != 192 or set(outcome_map) != set(pop):
        errors.append("exact192 feature/outcome cells required")
    event_cells = defaultdict(list)
    task_indices = []
    for event in events:
        if event.get("event") in ("REPLAY_FORWARD_STARTED", "REPLAY_FORWARD_COMPLETED") and event.get("case_id") is not None:
            event_cells[(event.get("case_id"), event.get("kind"), event.get("target_index"))].append(event)
    task_indices = [index for index, event in enumerate(events) if event.get("case_id") is not None]
    qualification_indices = [index for index, event in enumerate(events) if event.get("case_id") is None]
    if not task_indices or not qualification_indices or max(qualification_indices) >= min(task_indices):
        errors.append("qualification/cache/restoration must finish before first task")
    if set(event_cells) != set(expected_cells):
        errors.append("task event cell coverage differs")
    parsed = {}
    for key, expected in expected_cells.items():
        target = target_map.get(key, {})
        where = "target "+str(key)
        if target.get("prefix_sha256") != expected["hash"] or target.get("prefix_tokens") != len(expected["prefix"]) or not 1 <= len(expected["prefix"]) <= 256:
            errors.append(where+": prefix proof differs")
        if target.get("preceding_gold_count") != expected["preceding"] or target.get("teacher_conditioned") is not (expected["preceding"] > 0):
            errors.append(where+": teacher-prefix scope differs")
        if target.get("current_future_target_positions_in_prefix") is not False:
            errors.append(where+": current/future target firewall absent")
        if key[1] == "answer" and target.get("target_token_id") != expected["target"]:
            errors.append(where+": current target differs")
        if key[1] == "EOS" and tuple(sorted(target.get("accepted_EOS_ids", []))) != EOS:
            errors.append(where+": accepted EOS set differs")
        cell_events = event_cells.get(key, [])
        expected_order = [(event, alpha) for alpha in ALPHAS for event in ("REPLAY_FORWARD_STARTED", "REPLAY_FORWARD_COMPLETED")]
        if [(e.get("event"), e.get("alpha")) for e in cell_events] != expected_order:
            errors.append(where+": chronological START/COMPLETE alpha sequence differs")
        for event in cell_events:
            if event.get("prefix_ids_sha256") != expected["hash"] or event.get("prefix_tokens") != len(expected["prefix"]) or event.get("preceding_gold_count") != expected["preceding"] or event.get("teacher_conditioned") is not (expected["preceding"] > 0) or event.get("current_future_target_positions_in_prefix") is not False:
                errors.append(where+": event prefix/firewall differs")
            if event.get("event") == "REPLAY_FORWARD_COMPLETED" and event.get("state_gradient_evaluations") != int(event.get("alpha") == 1.):
                errors.append(where+": unit-only adjoint differs")
            if event.get("event") == "REPLAY_FORWARD_COMPLETED":
                branch = {1.: "nominal", 1.01: "plus", .99: "minus"}.get(event.get("alpha"))
                saved = target.get("result", {}).get(branch, {}) if branch else {}
                if event.get("scalar_result") != saved:
                    errors.append(where+": event and target scalar results differ")
        result = _triplet(target.get("result"), where, errors, eos=key[1] == "EOS")
        if result:
            parsed[key] = result
    cases = []
    for case, plan in sorted(pop.items()):
        n = len(plan.get("canonical_answer_token_ids", []))
        answers = [parsed.get((case, "answer", j)) for j in range(n)]
        eos = parsed.get((case, "EOS", n))
        if not n or any(answer is None for answer in answers) or eos is None:
            continue
        feature, stored = feature_map.get(case, {}), outcome_map.get(case, {})
        forbidden = set(feature)-FEATURE_KEYS
        if forbidden or feature.get("phase") != "INPUT_ONLY" or feature.get("prefix_kind") != "INPUT_ONLY" or feature.get("target_independent") is not True or feature.get("frozen_before_loss") is not True or feature.get("loss_or_gradient_predictors") is not False or feature.get("teacher_conditioned") is not False or feature.get("preceding_gold_count") != 0:
            errors.append(case+": input-only predictor firewall differs")
        base_hash = expected_cells[(case, "answer", 0)]["hash"]
        if feature.get("input_ids_sha256") != base_hash or feature.get("source_prefix_sha256") != base_hash:
            errors.append(case+": predictor source prefix differs")
        without_hash = {key: value for key, value in feature.items() if key != "freeze_sha256"}
        try:
            if feature.get("freeze_sha256") != digest(without_hash):
                errors.append(case+": frozen predictor hash differs")
        except (ValueError, TypeError):
            errors.append(case+": invalid frozen predictor scalars")
        balance = _number(feature.get("balance"), case+" balance", errors)
        first = target_map[(case, "answer", 0)].get("result", {}).get("nominal", {})
        if balance is None or not -1 <= balance <= 1 or not same_number(balance, first.get("balance")):
            errors.append(case+": first-input balance differs")
            balance = 0.
        mean_loss = {a: math.fsum(answer["loss"][a] for answer in answers)/n for a in ("nominal", "plus", "minus")}
        derivative = math.fsum(answer["adjoint"] for answer in answers)/n
        fd = (mean_loss["plus"]-mean_loss["minus"])/.02
        error, tolerance = abs(derivative-fd), .001+.03*abs(derivative)
        if error > tolerance:
            errors.append(case+": mean FD cancellation gate failed")
        for a in mean_loss:
            if not same_number(stored.get("mean_answer_CE", {}).get(a), mean_loss[a]) or not same_number(stored.get("sum_answer_CE", {}).get(a), mean_loss[a]*n):
                errors.append(case+": stored answer mean/sum differs")
        if stored.get("answer_target_count") != n or not same_number(stored.get("mean_answer_sensitivity"), derivative) or not same_number(stored.get("sum_answer_sensitivity"), derivative*n) or not same_number(stored.get("mean_level_FD", {}).get("finite_difference"), fd):
            errors.append(case+": stored mean derivative/FD differs")
        if not same_number(stored.get("mean_level_FD", {}).get("absolute_error"), error) or not same_number(stored.get("mean_level_FD", {}).get("registered_tolerance"), tolerance):
            errors.append(case+": stored mean gate differs")
        if stored.get("input_only_predictor") != feature:
            errors.append(case+": case predictor differs from frozen feature")
        # Stored EOS is a duplicate convenience record, never the source of truth.
        _triplet(stored.get("EOS_secondary"), case+" saved EOS", errors, eos=True)
        if stored.get("EOS_secondary") != target_map[(case, "EOS", n)].get("result"):
            errors.append(case+": stored EOS differs from target journal")
        metadata = {key: stored.get(key, plan.get(key)) for key in ("parent_id", "category", "difficulty_stratum", "answerability", "source_status", "variant_id", "pair_kind")}
        if any(value is None for value in metadata.values()) or any(metadata[k] != bank.get(case, {}).get(k) for k in metadata):
            errors.append(case+": paired/stratum metadata missing or different")
        cases.append({"case_id": case, **metadata, "answer_target_count": n, "balance": balance,
                      "mean_answer_loss": mean_loss, "mean_answer_derivative": derivative,
                      "mean_FD": fd, "mean_FD_error": error, "mean_FD_tolerance": tolerance,
                      "sum_answer_loss": {a: mean_loss[a]*n for a in mean_loss}, "sum_answer_derivative": derivative*n,
                      "neutral": abs(derivative) <= .001,
                      "numerically_uncertain": abs(derivative) > .001 and sign(derivative) != sign(fd),
                      "useful_direction": 0 if abs(derivative) <= .001 else -sign(derivative),
                      "coherence_direction": sign(balance),
                      "answer_loss_delta_plus": mean_loss["plus"]-mean_loss["nominal"],
                      "answer_loss_delta_minus": mean_loss["minus"]-mean_loss["nominal"],
                      "EOS_secondary": eos})
    if len(cases) != 192:
        errors.append("all192 reconstructed case means required")
    report = {"status": "INCOMPLETE_NO_PRIMARY_MECHANISM_COMPARISON", "errors": sorted(set(errors)),
              "primary": None, "EOS_secondary": None, "canonical_prefix_sum_descriptive": None,
              "semantic_efficacy": None, "bulk_validation_primary": None,
              "counts": journal, "population_cases": len(population), "case_rows": len(outcomes),
              "target_rows": len(targets), "feature_rows": len(features),
              "native_worker_seconds": bundle.get("terminal", {}).get("native_worker_seconds"),
              "source_bindings": bundle.get("source_bindings", {}),
              "scope": "Exposed DEV canonical answer PREFIX likelihood; EOS separate; no fit, validation, generation or GPU pooling."}
    if not errors:
        report["status"] = "COMPLETE_CPU_CAUSAL_PREFIX_SURROGATE_ONLY"
        report["primary"] = {"case_rows": cases, "direction": _direction_summary(cases),
                             "population": _population_summaries(cases),
                             "inference": "Descriptive exposed paired worlds/shared six generators; no confirmatory inference, fitted rule or automated advancement."}
        report["EOS_secondary"] = {"case_rows": [{"case_id": case["case_id"], **case["EOS_secondary"]} for case in cases],
                                   "mean_mass_loss": {a: math.fsum(case["EOS_secondary"]["loss"][a] for case in cases)/192 for a in ("nominal", "plus", "minus")},
                                   "loss_increase_case_counts": {a: sum(case["EOS_secondary"]["loss"][a] > case["EOS_secondary"]["loss"]["nominal"] for case in cases) for a in ("plus", "minus")},
                                   "interpretation": "Accepted-stop mass after supplied canonical answer; not on-policy termination. No primary composite."}
        report["canonical_prefix_sum_descriptive"] = {a: math.fsum(case["sum_answer_loss"][a] for case in cases)/192 for a in ("nominal", "plus", "minus")}
    return report


def _read_npy(blob):
    if not blob.startswith(b"\x93NUMPY") or len(blob) < 10:
        raise ValueError("not a native NPY vector")
    major = blob[6]
    width, offset = (2, 10) if major == 1 else (4, 12)
    size = int.from_bytes(blob[8:8+width], "little")
    header = ast.literal_eval(blob[offset:offset+size].decode("latin1"))
    shape = header["shape"]
    if header["descr"] != "<f4" or header["fortran_order"] is not False or len(shape) != 3 or shape[0] != 1 or not 1 <= shape[1] <= 256 or shape[2] != 1024:
        raise ValueError("qualification vector is not bounded CPU FP32 state")
    raw = blob[offset+size:]
    if len(raw) != math.prod(shape)*4:
        raise ValueError("qualification vector byte/shape mismatch")
    return shape, raw


def _vectors(run, qualification):
    results = qualification.get("scalar_CE_FD", [])+[qualification.get("EOS_mass_FD", {})]
    if len(results) != 3:
        return False
    for result in results:
        vector = result.get("nominal", {}).get("qualification_vectors")
        if not isinstance(vector, dict):
            return False
        path = Path(vector.get("path", "")).resolve()
        if (run/"result").resolve() not in path.parents or not path.is_file() or sha(path) != vector.get("file_sha256"):
            return False
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != {"G.npy", "d2.npy"} or any(x.file_size > 2*1024**2 for x in archive.infolist()):
                return False
            shape, g = _read_npy(archive.read("G.npy"))
            other_shape, d = _read_npy(archive.read("d2.npy"))
        if shape != other_shape or list(shape) != vector.get("shape") or hashlib.sha256(g).hexdigest() != vector.get("G_sha256") or hashlib.sha256(d).hexdigest() != vector.get("d2_sha256"):
            return False
        products = [a[0]*b[0] for a, b in zip(struct.iter_unpack("<f", g), struct.iter_unpack("<f", d))]
        if any(not math.isfinite(x) for x in products):
            return False
        dot, total = math.fsum(products), math.fsum(abs(x) for x in products)
        scalar = result.get("nominal", {}).get("sensitivity")
        if not numeric(scalar) or abs(dot-scalar) > 1e-7+5e-6*total:
            return False
    return True


def read_run(run):
    """Read terminal artifacts only; data/binding checks are generated here."""
    run = Path(run).resolve()
    names = {"admission": run/"admission.json", "review": run/"review.json", "terminal": run/"terminal_receipt.json",
             "registration": run/"result/registration.json", "receipt": run/"result/receipt.json",
             "qualification": run/"result/qualification.json", "forecast": run/"result/matrix_forecast.json",
             "ledger": run/"result/compute_ledger.json", "population": run/"result/population.json",
             "events": run/"result/attempts.jsonl", "targets": run/"result/targets.jsonl",
             "outcomes": run/"result/outcomes.jsonl", "features": run/"result/features.jsonl",
             "source_manifest": run/"source_manifest.json", "resources": run/"resources.jsonl",
             "preflight": run/"result/preflight.json"}
    bundle = {"io_errors": [], "source_bindings": {}, "io_checks": {}}
    for key, path in names.items():
        try:
            text = path.read_text(encoding="utf-8")
            bundle[key] = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
            bundle["source_bindings"][key] = {"path": str(path), "sha256": sha(path)}
        except (OSError, ValueError) as error:
            bundle["io_errors"].append(key+": "+type(error).__name__)
            bundle[key] = [] if key in ("population", "events", "targets", "outcomes", "features") else {}
    a, r, reg = bundle["admission"], bundle["review"], bundle["registration"]
    actual = bundle["source_bindings"].get("admission", {}).get("sha256")
    bundle["io_checks"]["admission_review_registration"] = bool(actual) and r.get("admission_sha256") == reg.get("admission_sha256") == actual
    if a.get("hard_deadline_seconds") == 4500:
        try:
            grant_path = Path(a.get("human_extension_path", "")).resolve()
            if grant_path != (PACKAGE/"HUMAN_CAUSAL_REPLAY_EXTENDED_ALLOWANCE_001.json").resolve():
                raise ValueError("not the distinct4500 allowance path")
            grant = json.loads(grant_path.read_text(encoding="utf-8"))
            grant_sha = sha(grant_path)
            expected = {"status": "EXPLICIT_ADDITIONAL_CAUSAL_REPLAY_ATTEMPT_AUTHORIZED", "id": "CAUSAL-REPLAY-EXTENDED-001",
                        "lane": LANE, "attempts": 1, "max_native_worker_seconds": 4500,
                        "model_API_attempts": 1925, "state_adjoints": 635, "cases": 192,
                        "maximum_prefix_tokens": 256, "parent_run": "native-cpu-causal-replay-002",
                        "training": False, "full_bulk_matrix_retry": False,
                        "worker_sha256": reg.get("worker_sha256")}
            bundle["io_checks"]["4500_human_scope"] = a.get("requires_additional_human_authorization") is True and grant_sha == a.get("human_extension_sha256") == bundle["terminal"].get("human_extension_sha256") and all(grant.get(key) == value for key, value in expected.items()) and grant_sha != "5397a0b51cdc62891c00c0f6a2147f5e53fafe5e277759b71d28ce3b93948063"
            bundle["source_bindings"]["4500_allowance"] = {"path": str(grant_path), "sha256": grant_sha}
        except (OSError, ValueError, TypeError):
            bundle["io_checks"]["4500_human_scope"] = False
    try:
        packet = PACKAGE/"CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json"
        metadata = json.loads(packet.read_text(encoding="utf-8"))
        bundle["io_checks"]["population_metadata"] = sha(packet) == reg.get("metadata_packet_sha256") and bundle["population"] == metadata["rows"]
        bank_path = PACKAGE/"runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json"
        bundle["bank"] = json.loads(bank_path.read_text(encoding="utf-8"))
        if sha(bank_path) != SOURCE_PINS["runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json"]:
            bundle["io_errors"].append("DEV bank hash differs")
        source_hashes = a.get("dispatch_hashes", {})
        allowed = bool(source_hashes)
        for relative, expected in source_hashes.items():
            path = (ROOT/relative).resolve()
            safe = PACKAGE.resolve() in path.parents or (path.suffix == ".py" and any(path.is_relative_to(ROOT/folder) for folder in ("experiments/rineya_taps_adaptive_recurrence_v1", "experiments/anchor_routed_middle_recurrence_v1/src", "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"))) or relative in ("Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl", "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl")
            if not safe or sha(path) != expected:
                allowed = False
        bundle["io_checks"]["frozen_source_closure"] = allowed
        snapshot = mapping(bundle.get("source_manifest"))
        snapshot_hashes = mapping(snapshot.get("sha256"))
        prefix = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/"
        required_python = [prefix+name for name in ("cpu_causal_replay_native.py", "cpu_h3_native.py", "sensitivity.py", "bulk_bank.py", "open_bank.py")]
        bundle["io_checks"]["source_snapshot"] = snapshot.get("head") == "7ece5cead285ba65c7d73ab57c0327fae2dc40bb" and all(snapshot_hashes.get(name) == source_hashes.get(name) and name in source_hashes for name in required_python)
        bundle["io_checks"]["qualification_vectors"] = _vectors(run, bundle["qualification"])
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, struct.error, zipfile.BadZipFile):
        bundle["io_errors"].append("source/data/vector binding proof unavailable")
    return bundle


def analyze(run):
    return analyze_bundle(read_run(run))


def main(run, out):
    report = analyze(run)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report["analyzer_sha256"] = sha(__file__)
    (out/"REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# CPU causal replay evidence", "", report["status"], "",
             "Primary is per-case mean canonical answer PREFIX loss/direction; EOS is separate.", ""]
    if report["primary"]:
        direction = report["primary"]["direction"]
        lines += [f"All {direction['cases']} cases retained: {direction['neutral_cases']} neutral, "
                  f"{direction['numerically_uncertain_nonneutral_cases']} numerically uncertain; "
                  f"{direction['reliable_sign_denominator']} reliable directional cases.", "",
                  f"Coherence/constant increase/constant decrease agreements: "
                  f"{direction['coherence_agreements']}/{direction['constant_increase_agreements']}/{direction['constant_decrease_agreements']}. "
                  f"Exploratory tied-rank Spearman={direction['spearman_all_cases_balance_negative_derivative']}.", ""]
    else:
        lines += ["Complete matrix/proof gate failed; no survivor primary comparison.", ""]+["- "+error for error in report["errors"]]
    lines += ["", "Canonical prefix likelihood is not semantic correctness or a terminated-response probability. "
              "Shared generators/exposed paired worlds support no fitted rule, confirmation, automated advancement or speed claim."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
