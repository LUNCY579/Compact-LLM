"""Terminal-only stdlib fresh replication reader; no model/Torch/tokenizer.

Complete raw native and provenance checks precede reading offline labels.
The fixed primary can fail; no secondary contrast replaces it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time

V3 = Path(__file__).resolve().parent
CAMPAIGN = V3.parent
ROOT = CAMPAIGN.parents[1]
RUN = V3 / "runs/native-fresh-open-replication-001"
DATA = V3 / "data/fresh_open_replication_001"
ADMISSION = V3 / "fresh_admission_v2.json"
PLAN = V3 / "FRESH_ANALYSIS_PLAN_DRAFT_001.md"
STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
CODE = "b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d"
GUARD = "EVIDENCE_CONSISTENCY_GUARD"
CONFIDENT = "CONFIDENT_RECURRENCE"
UNCERTAIN = "UNCERTAIN_RECURRENCE"
PATHS = ("K0", "UNIT", CONFIDENT, UNCERTAIN, GUARD, "HEADER_CUE", "RANDOM17", "RANDOM29", "PRIMARY_REPEAT")
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def nonnegative(value): return finite(value) and value >= 0
def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else None
    except (TypeError, ValueError): return None
def check(condition, message, errors):
    if not condition: errors.append(message)


def review_receipt_path(admission):
    if admission.get("review_receipt") != "reviews/fresh_dispatch_review_002.json":
        raise PermissionError("exact V2 independent dispatch review path required")
    return (V3 / admission["review_receipt"]).resolve(strict=True)


def raw_gate(population, outcomes, forwards, receipt, qualification, forecast, routes):
    from .opposed_gate_policy import decide, header_cue
    errors = []
    check(receipt.get("status") == "FRESH_K01_RAW_MATRIX_COMPLETE_NOT_ANALYZED" and receipt.get("complete") is True,
          "worker not complete", errors)
    by = {x.get("case_id"): x for x in population}
    cells = {(x.get("case_id"), x.get("path")): x for x in outcomes}
    check(len(population) == len(by) == 456 and Counter(x.get("cohort") for x in population) == {"SCIENTIFIC": 432, "PRESERVATION": 24},
          "complete456 prompt population required", errors)
    check(len(outcomes) == len(cells) == 4104 and set(cells) == {(x, p) for x in by for p in PATHS},
          "exact456x9 unique raw cells required", errors)
    forbidden = {"gold", "scorer", "family", "parent_id", "answerability", "source_status", "category"}
    check(all(not forbidden.intersection(x) for x in population), "runtime population label contamination", errors)
    starts = [x for x in forwards if x.get("event") in ("FORWARD_STARTED", "CACHE_REJECTION_STARTED")]
    done = [x for x in forwards if x.get("event") == "FORWARD_COMPLETED"]
    failed = [x for x in forwards if x.get("event") == "FORWARD_FAILED"]
    caches = [x for x in forwards if x.get("event") == "CACHE_REJECTED"]
    terminal = done + failed + caches
    check(all(nonnegative(x.get("parent_elapsed_seconds")) and timestamp(x.get("timestamp_utc")) is not None for x in forwards),
          "finite nonnegative native event clock/UTC required", errors)
    if all(nonnegative(x.get("parent_elapsed_seconds")) and timestamp(x.get("timestamp_utc")) is not None for x in forwards):
        check(all(forwards[i]["parent_elapsed_seconds"] <= forwards[i + 1]["parent_elapsed_seconds"] and
                  timestamp(forwards[i]["timestamp_utc"]) <= timestamp(forwards[i + 1]["timestamp_utc"]) for i in range(len(forwards) - 1)),
              "native journal clocks/UTC out of order", errors)
    check(all(nonnegative(x.get("forward_seconds", x.get("elapsed_seconds"))) for x in terminal),
          "finite nonnegative measured forward/cache/partial elapsed required", errors)
    check(len(starts) <= 34596 and len(starts) == len(terminal) and
          sorted(x.get("API_attempt", -1) for x in starts) == list(range(1, len(starts) + 1)) and
          len({x.get("API_attempt") for x in terminal}) == len(terminal), "global start/terminal API coverage", errors)
    start_by = {x.get("API_attempt"): x for x in starts}
    for x in terminal:
        s = start_by.get(x.get("API_attempt"), {})
        check(s.get("backend") == x.get("backend") and s.get("local_API_attempt") == x.get("local_API_attempt") and
              s.get("parent_elapsed_seconds", math.inf) <= x.get("parent_elapsed_seconds", -math.inf),
              "global terminal backend/order mismatch", errors)
    check(len(failed) == 2 and {x.get("error") for x in failed} == {
        "RuntimeError: EXPECTED_FRESH_PREFIX_EXCEPTION", "RuntimeError: EXPECTED_FRESH_DRIVER_EXCEPTION"}
        and all(x.get("partial_compute") == "UNKNOWN" and x.get("phase") != "TASK" for x in failed),
        "two declared qualification partials only", errors)
    cache_starts = [x for x in starts if x.get("event") == "CACHE_REJECTION_STARTED"]
    check(len(caches) == len(cache_starts) == 2 and {(x.get("use_cache"), x.get("cache_object_present")) for x in cache_starts} ==
          {(True, False), (False, True)} and all(x.get("model_compute_started") is False for x in caches), "both precompute cache variants", errors)
    task = [x for x in done if x.get("phase") == "TASK"]
    qual = [x for x in terminal if x.get("phase") != "TASK"]
    check(len(qual) <= 852 and qualification.get("status") == "PASS_FRESH_K01_NATIVE_MECHANICS" and
          qualification.get("max_driver_API_attempts") == 9 and qualification.get("minimal_probability_native_parity") is True and
          qualification.get("forced_exception_restoration") is True, "fresh complete qualification required", errors)
    prefix_counts = qualification.get("prefix_counts", {})
    check(prefix_counts == {"model_API_attempts": 67, "full_forward_attempts": 65, "completed_forwards": 64,
                           "expected_partial": 1, "cache_rejections": 2, "completed_prefix_positions": 4921, "state_adjoints": 0},
          "exact67 prefix qualification proof missing", errors)
    prefix_done = [x for x in done if x.get("backend") == "prefix"]
    max_done = [x for x in done if x.get("phase") == "QUALIFICATION_MAX_DRIVER"]
    max_failed = [x for x in failed if x.get("phase") == "QUALIFICATION_MAX_DRIVER"]
    check(len(prefix_done) == 64 and sum(x.get("prefix_tokens", 0) for x in prefix_done) == 4921 and
          len(max_done) == 8 and len(max_failed) == 1 and all(x.get("prefix_tokens") == 357 for x in max_done + max_failed),
          "actual prefix/max qualification call shape coverage differs", errors)
    parity = qualification.get("prefix_parity", [])
    check(len(parity) == 9 and all(len(x.get("hashes", [])) == 7 and
          x["hashes"][0] == x["hashes"][1] == x["hashes"][2] == x["hashes"][6] and
          x["hashes"][3] == x["hashes"][4] == x["hashes"][5] for x in parity), "native prefix parity hashes missing", errors)
    full_parity = qualification.get("full_generation_parity", [])
    check(len(full_parity) == 8 and len({x.get("case_id") for x in full_parity}) == 8 and
          all(x.get("score_status") == "NOT_COLLECTED_NOT_SCORED" and isinstance(x.get("K0_tokens"), list) and
              isinstance(x.get("unit_tokens"), list) and 1 <= len(x["K0_tokens"]) <= 24 and 1 <= len(x["unit_tokens"]) <= 24 for x in full_parity),
          "eight full SMOKE rollout parity proofs missing", errors)
    check(qualification.get("max_driver_proof", {}).get("forced_exception_restoration") is True and
          len(qualification.get("actual_input_policy_mapping", [])) == 8, "driver restoration/native policy mapping proof missing", errors)
    for x in (receipt, qualification):
        check(x.get("before_state_sha256") == x.get("after_state_sha256") == STATE and x.get("parameter_gradients") == "NONE" and
              x.get("source_unchanged") is True, "resident/source/gradient preservation missing", errors)
    check(forecast.get("admitted") is True and forecast.get("task_API_cap") == 33744 and forecast.get("task_rows") == 4104 and
          forecast.get("forward_factor") == 1.5 and forecast.get("internal_reserve_seconds") == 1800 and
          forecast.get("parent_deadline_seconds") == 15600 and
          all(nonnegative(forecast.get(k)) for k in ("remaining_forward_seconds", "remaining_nonforward_seconds", "remaining_forecast_seconds",
              "elapsed_seconds", "nonforward_per_row_bound_seconds", "max_observed_warm_forward_seconds")) and
          forecast.get("max_observed_warm_forward_seconds", 0) > 0 and
          math.isclose(forecast["remaining_forward_seconds"], 33744 * 1.5 * forecast["max_observed_warm_forward_seconds"], rel_tol=1e-12, abs_tol=1e-9) and
          math.isclose(forecast["remaining_nonforward_seconds"], 4104 * forecast["nonforward_per_row_bound_seconds"], rel_tol=1e-12, abs_tol=1e-9) and
          math.isclose(forecast["remaining_forecast_seconds"], forecast["remaining_forward_seconds"] + forecast["remaining_nonforward_seconds"], rel_tol=1e-12, abs_tol=1e-9) and
          sum(forecast.get(k, math.inf) for k in ("remaining_forward_seconds", "remaining_nonforward_seconds", "elapsed_seconds")) + 1800 <= 15600,
          "complete all-control forecast proof missing", errors)
    if task:
        check(max(x.get("parent_elapsed_seconds", math.inf) for x in qual) <= forecast.get("elapsed_seconds", -1) <=
              min(x.get("parent_elapsed_seconds", -1) for x in task), "qualification/forecast must precede all task calls", errors)
    resources = [x.get("resources") for x in done if isinstance(x.get("resources"), dict)]
    resources += [x[k] for x in (receipt, qualification) for k in ("after_load_resources", "final_resources", "final_resources_on_stop", "resources") if isinstance(x.get(k), dict)]
    for r in resources:
        check(all(nonnegative(r.get(k)) for k in ("gpu_peak_allocated_bytes", "gpu_free_bytes", "rss_bytes", "host_available_bytes")) and
              r.get("gpu_peak_allocated_bytes", math.inf) <= 5 * GIB and r.get("gpu_free_bytes", -1) >= GIB and
              r.get("rss_bytes", math.inf) <= 6 * GIB and r.get("host_available_bytes", -1) >= GIB, "recorded resource ceiling/reserve violation", errors)
    check(len([x for x in done if isinstance(x.get("resources"), dict)]) == len(done), "missing complete-forward resource samples", errors)
    expected_counts = {"model_API_attempts": len(starts), "completed_forwards": len(done), "full_forward_attempts": len(starts) - len(caches),
                       "failed_forwards": len(failed), "cache_rejections": len(caches), "completed_prefix_positions": sum(x.get("prefix_tokens", 0) for x in done)}
    check(receipt.get("counts") == expected_counts, "raw actual counters disagree with worker", errors)
    task_by = {x.get("API_attempt"): x for x in task}; used = []
    for (cid, path), r in cells.items():
        p = by.get(cid, {}); ids = p.get("input_token_ids", []); tokens = r.get("token_ids", [])
        if not ids or not isinstance(tokens, list) or not 1 <= len(tokens) <= 8:
            errors.append("nonempty original/token cap schema required"); continue
        basic = (all(type(x) is int and 0 <= x < 151936 for x in ids + tokens) and type(r.get("K")) is int and r["K"] in (0, 1) and
                 r.get("alpha") == 1 and r.get("generation_cap") == 8 and r.get("state_adjoints") == r.get("gold_calls") == 0)
        check(basic, "raw token/K/alpha/generation/target contract differs", errors)
        if not basic: continue
        check(all(nonnegative(r.get(k)) for k in ("wall_seconds", "physical_forward_seconds", "probability_seconds",
                                                  "prefill_seconds", "decode_seconds")), "finite nonnegative raw case cost required", errors)
        check(r.get("cohort") == p.get("cohort") and r.get("prefix_ids_sha256") == p.get("prefix_ids_sha256") == digest(ids) and
              r.get("physical_run") is True and r.get("safety_stopped") is False and r.get("termination") in ("eos", "token_cap"),
              "raw original-prefix/cohort/safety contract differs", errors)
        check(r.get("eos_emitted") == (r.get("termination") == "eos") and r.get("truncated") == (r.get("termination") == "token_cap") and
              ((tokens[-1] in (151643, 151645)) == r.get("eos_emitted")), "literal EOS/cap flags differ", errors)
        pilot = path in (CONFIDENT, UNCERTAIN)
        check(r.get("pilot_used") == pilot and r.get("logical_first_reused") == (pilot and r.get("K") == 0), "pilot branch metadata differs", errors)
        original = [(ids + tokens[:j], r.get("K"), "GENERATION") for j in range(len(tokens))]
        if pilot:
            original = [(ids, 0, "PILOT")] + (original if r.get("K") == 1 else original[1:])
            check(finite(r.get("pilot_probability")) and 0 <= r["pilot_probability"] <= 1, "finite actual original-pilot probability missing", errors)
        else: check(r.get("pilot_probability") is None, "direct control has invented pilot probability", errors)
        if path in (CONFIDENT, UNCERTAIN) and finite(r.get("pilot_probability")):
            decision = decide(path, p["raw_prompt"], float(r["pilot_probability"]))
        elif path in (GUARD, "PRIMARY_REPEAT"): decision = decide(GUARD, p["raw_prompt"])
        elif path == "HEADER_CUE": decision = header_cue(p["raw_prompt"])
        else: decision = None
        if decision is not None:
            check((r.get("K"), r.get("recognized"), r.get("certificate")) ==
                  (decision.k, decision.recognized, decision.certificate), "raw prompt-only policy route/certificate differs", errors)
        if path in ("K0", "UNIT"):
            check(r.get("K") == int(path == "UNIT"), "fixed baseline route differs", errors)
        calls = r.get("physical_API_attempts", []); used.extend(calls)
        check(len(calls) == len(original) == r.get("full_model_calls") and r.get("extra_layer_applications") == 2 * r.get("K", -1) * len(tokens),
              "own-path physical calls/extra layers differ", errors)
        for api, (prefix, k, stage) in zip(calls, original):
            e = task_by.get(api, {})
            check(e.get("case_id") == cid and e.get("path") == path and e.get("prefix_tokens") == len(prefix) and
                  e.get("prefix_ids_sha256") == digest(prefix) and e.get("K") == k and e.get("stage") == stage and e.get("depth") == 28 + 2 * k,
                  "actual global call prefix/stage/K/depth lineage differs", errors)
        check(r.get("completed_token_positions") == sum(len(x[0]) for x in original), "row positions disagree with prefixes", errors)
    check(len(used) == len(set(used)) and set(used) == set(task_by), "task rows do not partition physical task calls", errors)
    check(routes.get("rate") == 160 / 192 and len(routes.get("seed17", [])) == len(routes.get("seed29", [])) == 456 and
          routes.get("order_ids_sha256") == digest([x.get("case_id") for x in population]), "frozen random routing order/rate missing", errors)
    for seed in (17, 29):
        rng = random.Random(seed); expected = [int(rng.random() < 160 / 192) for _ in population]
        check(routes.get("seed" + str(seed)) == expected, "registered random stream differs", errors)
        for i, p in enumerate(population):
            check(cells.get((p.get("case_id"), "RANDOM" + str(seed)), {}).get("K") == expected[i], "actual random row route differs", errors)
    for cid in by:
        a, b, g, repeat = (cells.get((cid, p), {}) for p in (CONFIDENT, UNCERTAIN, GUARD, "PRIMARY_REPEAT"))
        check(type(a.get("K")) is int and type(b.get("K")) is int and
              a.get("pilot_probability") == b.get("pilot_probability") and a.get("pilot_logits_sha256") == b.get("pilot_logits_sha256") and
              a.get("K", -2) + b.get("K", -2) == 1 and finite(a.get("pilot_probability")) and
              a.get("K") == int(a.get("pilot_probability", -1) >= .5), "actual independent confidence complementarity proof differs", errors)
        keys = ("token_ids", "response", "termination", "K", "recognized", "certificate", "pilot_probability", "selected_first_logits_sha256", "pilot_logits_sha256")
        check(all(g.get(k) == repeat.get(k) for k in keys), "actual deterministic primary repeat differs", errors)
    return errors, cells, expected_counts


def parent_means(labels, values):
    groups = defaultdict(list)
    for x in labels: groups[(x["family"], x["mechanism"], x["parent_id"])].append(values[x["case_id"]])
    return {key: sum(xs) / len(xs) for key, xs in groups.items()}


def stratified_mean(parents):
    mechanisms = defaultdict(list)
    for (family, mechanism, _), value in parents.items(): mechanisms[(family, mechanism)].append(value)
    families = defaultdict(list)
    for (family, _), values in mechanisms.items(): families[family].append(sum(values) / len(values))
    return sum(sum(values) / len(values) for values in families.values()) / len(families)


def paired_bootstrap(parents):
    strata = defaultdict(list)
    for (family, mechanism, parent), value in sorted(parents.items()): strata[(family, mechanism)].append(value)
    if len(strata) != 12 or any(len(x) != 18 for x in strata.values()): raise ValueError("exact12mechanisms/18parents required")
    rng = random.Random(730217); draws = []
    ordered = [strata[k] for k in sorted(strata)]
    for _ in range(20000):
        draws.append(sum(sum(xs[rng.randrange(18)] for _ in range(18)) / 18 for xs in ordered) / 12)
    draws.sort()
    return {"point": stratified_mean(parents), "lower": draws[499], "upper": draws[19499], "draws": 20000,
            "seed": 730217, "endpoint_indices": [499, 19499], "scope": "conditional randomized parent worlds within12 authored blocks; no source/human generalization"}


def analyze_scored(science, panel, cells, scorer, panel_scorer, solver=None, panel_solver=None, blind=None):
    by = {x["case_id"]: x for x in science}; panel_by = {x["case_id"]: x for x in panel}
    if (len(science) != len(by) or len(by) != 432 or len(panel) != len(panel_by) or len(panel_by) != 24):
        raise ValueError("complete unique offline labels required")
    parents = defaultdict(list)
    for x in science: parents[(x["family"], x["mechanism"], x["parent_id"])].append(x)
    blocks = Counter((f, m) for f, m, p in parents)
    family_blocks = Counter(f for f, m in blocks)
    if (len(parents) != 216 or len(blocks) != 12 or set(blocks.values()) != {18} or len(family_blocks) != 6
            or set(family_blocks.values()) != {2} or any(len(xs) != 2 or {x["variant_id"] for x in xs} != {0, 1} for xs in parents.values())):
        raise ValueError("exact216paired parents/six families/twelve18parent mechanisms required")
    grades = {(cid, p): scorer(cells[(cid, p)]["response"], label) for cid, label in by.items() for p in PATHS}
    scores = {p: {cid: float(grades[(cid, p)]["correct"]) for cid in by} for p in PATHS}
    scores["RANDOM_MEAN"] = {cid: (scores["RANDOM17"][cid] + scores["RANDOM29"][cid]) / 2 for cid in by}
    accuracy = {p: stratified_mean(parent_means(science, vals)) for p, vals in scores.items()}
    diff = {cid: scores[GUARD][cid] - scores["UNIT"][cid] for cid in by}
    effect = parent_means(science, diff); interval = paired_bootstrap(effect)
    contrasts, regressions = {}, []
    for baseline in ("K0", "UNIT"):
        gains = [cid for cid in by if scores[GUARD][cid] > scores[baseline][cid]]
        losses = [cid for cid in by if scores[GUARD][cid] < scores[baseline][cid]]
        family = {}
        for fam in sorted({x["family"] for x in science}):
            selected = [x for x in science if x["family"] == fam]
            family[fam] = sum(scores[GUARD][x["case_id"]] - scores[baseline][x["case_id"]] for x in selected) / len(selected)
        format_delta = sum(grades[(cid, GUARD)]["format_valid"] - grades[(cid, baseline)]["format_valid"] for cid in by)
        unsupported_delta = sum(grades[(cid, GUARD)]["unsupported_canonical_answer"] - grades[(cid, baseline)]["unsupported_canonical_answer"] for cid in by)
        false_delta = sum(grades[(cid, GUARD)]["false_UNKNOWN"] - grades[(cid, baseline)]["false_UNKNOWN"] for cid in by)
        harms = {
            "format_invalid": [cid for cid in by if grades[(cid, baseline)]["format_valid"] and not grades[(cid, GUARD)]["format_valid"]],
            "unsupported": [cid for cid in by if not grades[(cid, baseline)]["unsupported_canonical_answer"] and grades[(cid, GUARD)]["unsupported_canonical_answer"]],
            "false_UNKNOWN": [cid for cid in by if not grades[(cid, baseline)]["false_UNKNOWN"] and grades[(cid, GUARD)]["false_UNKNOWN"]],
        }
        family_endpoints = {}
        for fam in family:
            ids = [x["case_id"] for x in science if x["family"] == fam]
            family_endpoints[fam] = {"denominator": len(ids), "correctness_delta": family[fam],
                "format_count_delta": sum(grades[(cid, GUARD)]["format_valid"] - grades[(cid, baseline)]["format_valid"] for cid in ids),
                "unsupported_count_delta": sum(grades[(cid, GUARD)]["unsupported_canonical_answer"] - grades[(cid, baseline)]["unsupported_canonical_answer"] for cid in ids),
                "false_UNKNOWN_count_delta": sum(grades[(cid, GUARD)]["false_UNKNOWN"] - grades[(cid, baseline)]["false_UNKNOWN"] for cid in ids),
                "case_losses": [cid for cid in losses if cid in ids]}
        contrasts[baseline] = {"gains": gains, "losses": losses, "baseline_correct": sum(scores[baseline].values()),
                               "retained_correct": sum(scores[baseline][cid] * scores[GUARD][cid] for cid in by),
                               "denominator": 432, "family_delta": family, "format_valid_count_delta": format_delta,
                               "unsupported_count_delta": unsupported_delta, "false_UNKNOWN_count_delta": false_delta,
                               "new_case_endpoint_harms": harms, "family_endpoints": family_endpoints}
        if losses or any(harms.values()) or min(family.values()) < 0 or format_delta < 0 or unsupported_delta > 0 or false_delta > 0: regressions.append(baseline)
    panel_grades = {(cid, p): panel_scorer(cells[(cid, p)]["response"], label["gold"]) for cid, label in panel_by.items() for p in PATHS}
    panel_losses = {b: [cid for cid in panel_by if panel_grades[(cid, b)]["correct"] and not panel_grades[(cid, GUARD)]["correct"]] for b in ("K0", "UNIT")}
    eligible = interval["point"] >= .05 and interval["lower"] > 0 and not regressions and not any(panel_losses.values())
    endpoint_counts = {}
    for p in PATHS:
        endpoint_counts[p] = {"strict_correct": sum(scores[p].values()), "denominator": 432,
            "format_valid": sum(grades[(cid, p)]["format_valid"] for cid in by),
            "normalized_UNKNOWN": sum(grades[(cid, p)].get("canonical_output") == "UNKNOWN" for cid in by),
            "literal_UNKNOWN": sum(cells[(cid, p)]["response"] == "UNKNOWN" for cid in by),
            "stripped_uppercase_UNKNOWN": sum(cells[(cid, p)]["response"].strip() == "UNKNOWN" for cid in by),
            "false_UNKNOWN": sum(grades[(cid, p)]["false_UNKNOWN"] for cid in by),
            "unsupported": sum(grades[(cid, p)]["unsupported_canonical_answer"] for cid in by),
            "EOS": sum(cells[(cid, p)]["eos_emitted"] for cid in by), "cap_stops": sum(cells[(cid, p)]["truncated"] for cid in by)}
    oracle = {cid: max(scores["K0"][cid], scores["UNIT"][cid]) for cid in by}
    oracle_accuracy = stratified_mean(parent_means(science, oracle)); headroom = oracle_accuracy - accuracy["UNIT"]
    floors = {}
    if solver:
        began = time.perf_counter(); values = {}
        for cid, x in by.items():
            answer = solver(x["prompt"])
            values[cid] = float(scorer(answer["answer"] if isinstance(answer, dict) else str(answer), x)["correct"])
        floors["independent_full_solver"] = {"accuracy": stratified_mean(parent_means(science, values)), "CPU_seconds": time.perf_counter() - began, "neural_reasoning": False}
    if blind:
        predictions = {cid: blind(x["prompt"]) for cid, x in by.items()}
        for name in sorted(set.intersection(*(set(x) for x in predictions.values()))):
            values = {cid: float(scorer(predictions[cid][name], by[cid])["correct"]) for cid in by}
            floors[name] = {"accuracy": stratified_mean(parent_means(science, values)), "bank_scope": "scientific only"}
    secondary = {}
    for p in (*PATHS[:8], "RANDOM_MEAN"):
        secondary[p] = {}
        for base in ("K0", "UNIT"):
            secondary[p][base] = {"paired_delta": stratified_mean(parent_means(science, {cid: scores[p][cid] - scores[base][cid] for cid in by})),
                "gains": [cid for cid in by if scores[p][cid] > scores[base][cid]],
                "losses": [cid for cid in by if scores[p][cid] < scores[base][cid]], "denominator": 432,
                "promotable": p == GUARD, "pvalue": None}
    groups = {}
    for recognized in (True, False):
        ids = [cid for cid in by if cells[(cid, GUARD)]["recognized"] is recognized]
        groups[str(recognized)] = {"items": len(ids), "guard_accuracy": None if not ids else sum(scores[GUARD][cid] for cid in ids) / len(ids)}
    panel_floor = None
    if panel_solver:
        started = time.perf_counter()
        panel_floor = {"correct": sum(panel_scorer(panel_solver(x["prompt"]), x["gold"])["correct"] for x in panel),
                       "denominator": 24, "CPU_seconds": time.perf_counter() - started, "neural_reasoning": False}
    return {"primary": {"policy": GUARD, "comparator": "UNIT_PRIOR_FROZEN_ALPHA1", "accuracy": accuracy[GUARD],
                        "delta": interval["point"], "conditional95_interval": interval, "practical_margin": .05,
                        "meets_margin": interval["point"] >= .05, "eligible_restricted_quality_result": eligible,
                        "preservation_regression_baselines": regressions, "copy_panel_loss_cases": panel_losses},
            "accuracy": accuracy, "primary_case_contrasts": contrasts, "secondary_contrasts": secondary,
            "recognized_groups": groups, "endpoints": endpoint_counts,
            "mechanism_primary_effects": {f + "/" + m: sum(v for (ff, mm, _), v in effect.items() if (ff, mm) == (f, m)) / 18 for f, m, _ in effect},
            "constant_binary_oracle": {"accuracy": oracle_accuracy, "deployable": False, "headroom_over_unit": headroom,
                "headroom_over_primary": oracle_accuracy - accuracy[GUARD],
                "capture_over_unit": None if headroom <= 1e-12 else (accuracy[GUARD] - accuracy["UNIT"]) / headroom,
                "scope": "measured complete K0/unit responses only; capture unclipped/NULL unlesspositive headroom"},
            "preservation_panel": {"denominator": 24, "accuracy": {p: sum(panel_grades[(cid, p)]["correct"] for cid in panel_by) / 24 for p in PATHS},
                "baseline_correct_losses": panel_losses, "measured_copy_floor": panel_floor,
                "floor": "visible exact copying;100% verified symbolic floor is not neural reasoning"},
            "ordinary_floors": floors, "secondary_policy_promotion": False, "repeats_independent_units": False,
            "limits": ["Authored twelve semantic blocks; conditional parent uncertainty, not independent human/source transfer.",
                       "No secondary pvalues or post-result primary selection. No production speed from instrumented timing."]}


def load_native(run_root):
    """No offline fresh labels are opened here, including on any failure."""
    run_root = Path(run_root).resolve(strict=True)
    if run_root != RUN.resolve(strict=True): raise PermissionError("exact first-time fresh native run required")
    terminal = read(run_root / "terminal_receipt.json")
    if terminal.get("status") != "WORKER_TERMINAL" or terminal.get("exit_code") != 0:
        raise RuntimeError("terminal failed/incomplete; no label access")
    admission = read(run_root / "admission.json")
    if sha(run_root / "admission.json") != sha(ADMISSION) or terminal.get("admission_sha256") != sha(ADMISSION):
        raise RuntimeError("exact copied fresh admission binding differs")
    if admission.get("status") != "READY_FOR_FRESH_K01_FP32_DISPATCH" or admission.get("primary_policy") != GUARD:
        raise RuntimeError("new fresh admission/primary differs")
    manifest = read(run_root / "source_manifest.json")["sha256"]
    dispatch = admission["dispatch_hashes"]
    if manifest != dispatch: raise RuntimeError("launch exact source manifest lacks admitted closure")
    required = (Path(__file__), PLAN, V3 / "fresh_native.py", V3 / "opposed_gate_generation.py", V3 / "opposed_gate_policy.py",
                V3 / "fresh_open_bank.py", V3 / "fresh_open_verifier.py", V3 / "short_response_panel.py")
    for path in required:
        relative = path.relative_to(ROOT).as_posix()
        if dispatch.get(relative) != sha(path): raise RuntimeError("actual reader/plan/scorer/native dependency membership differs:" + relative)
    for relative, expected in dispatch.items():
        path = (ROOT / relative).resolve(strict=True)
        if ROOT not in path.parents or sha(path) != expected: raise RuntimeError("current frozen dispatch bytes differ:" + relative)
    review_path = review_receipt_path(admission)
    review = read(run_root / "review.json")
    if V3 not in review_path.parents or sha(review_path) != sha(run_root / "review.json") or review.get("status") != "PASS_FRESH_K01_FP32_DISPATCH" or review.get("admission_sha256") != sha(ADMISSION):
        raise RuntimeError("exact independent dispatch review absent")
    result = run_root / "result"
    receipt, qualifier, forecast, registration, routes = (read(result / name) for name in
        ("receipt.json", "qualification.json", "matrix_forecast.json", "registration.json", "random_routes.json"))
    if receipt.get("dispatch_hashes") != dispatch or registration.get("admission_sha256") != sha(ADMISSION):
        raise RuntimeError("worker actual source/admission lineage differs")
    assets = admission["checkpoint_files"]
    if (len(assets) != 10 or qualifier.get("checkpoint_files") != assets or receipt.get("checkpoint_files_before") != assets
            or receipt.get("checkpoint_files_after") != assets or receipt.get("loaded_custom_code_sha256") != CODE):
        raise RuntimeError("actual10asset/customhead before-after identity absent")
    for name, expected in assets.items():
        if sha(Path(admission["source"]) / name) != expected: raise RuntimeError("canonical disk asset changed:" + name)
    runtime = receipt.get("runtime", {})
    expected_runtime = {"python": "3.11.15", "torch": "2.12.1+cu126", "transformers": "5.14.1", "CPU_threads": 4,
                        "dtype": "float32", "attention": "eager", "TF32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "qwen_implementation_sha256": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"}
    if any(runtime.get(k) != value for k, value in expected_runtime.items()) or registration.get("runtime") != runtime:
        raise RuntimeError("actual recorded qualified FP32runtime differs")
    initial = registration.get("initial_resources", {})
    if not all(nonnegative(initial.get(k)) for k in ("gpu_free_bytes", "host_available_bytes")) or initial.get("gpu_free_bytes", -1) < 6 * GIB or initial.get("host_available_bytes", -1) < 6 * GIB:
        raise RuntimeError("initial worker6GiB floors missing")
    parent_samples = rows(run_root / "resources.jsonl")
    if not parent_samples or any(not all(nonnegative(x.get(k)) for k in ("rss_bytes", "host_available_bytes", "worker_seconds", "elapsed_monotonic")) or
                                x.get("rss_bytes", math.inf) > 6 * GIB or x.get("host_available_bytes", -1) < GIB for x in parent_samples):
        raise RuntimeError("parent recorded RSS/running reserve violation")
    if any(parent_samples[i]["worker_seconds"] > parent_samples[i + 1]["worker_seconds"] or
           parent_samples[i]["elapsed_monotonic"] > parent_samples[i + 1]["elapsed_monotonic"] for i in range(len(parent_samples) - 1)):
        raise RuntimeError("parent sample clock order differs")
    inventory = terminal.get("gpu_initial_inventory", {})
    if not nonnegative(inventory.get("free_bytes")) or inventory.get("uuid") != admission.get("gpu_uuid") or inventory.get("driver_version") != admission.get("gpu_driver_version") or inventory.get("free_bytes", -1) < 6 * GIB:
        raise RuntimeError("initialGPUidentity/free floor missing")
    if (not all(nonnegative(terminal.get(k)) for k in ("native_worker_seconds", "supervisor_wall_seconds"))
            or not nonnegative(receipt.get("parent_elapsed_seconds")) or terminal.get("native_worker_seconds", math.inf) > 15600
            or receipt.get("parent_elapsed_seconds", math.inf) >= 13800
            or timestamp(terminal.get("started_utc")) is None or timestamp(terminal.get("ended_utc")) is None
            or timestamp(terminal["started_utc"]) > timestamp(terminal["ended_utc"])):
        raise RuntimeError("complete body/parent deadline exceeded")
    population, outcomes, forward_rows = read(result / "population.json"), rows(result / "outcomes.jsonl"), rows(result / "forwards.jsonl")
    validate_native_population(admission, population)
    errors, cells, counts = raw_gate(population, outcomes, forward_rows, receipt, qualifier, forecast, routes)
    if errors: raise RuntimeError("native proof failed: " + "; ".join(sorted(set(errors))))
    ledger = rows(CAMPAIGN / "BUDGET_LEDGER.jsonl")
    joined = [x for x in ledger if x.get("study_id") == admission["study_id"]]
    if (len(joined) != 2 or [x.get("event") for x in joined] != ["native_started", "native_completed"]
            or joined[1].get("worker_pid") != terminal.get("worker_pid")
            or joined[1].get("native_worker_seconds") != terminal.get("native_worker_seconds")
            or joined[1].get("admission_sha256") != sha(ADMISSION)):
        raise RuntimeError("sole study terminal/charged ledger join fails")
    if (not all(nonnegative(x.get("native_worker_seconds", 0)) for x in ledger if x.get("event") == "native_completed")
            or timestamp(joined[0].get("timestamp_utc")) is None or timestamp(joined[1].get("ended_utc")) is None
            or timestamp(joined[0]["timestamp_utc"]) > timestamp(joined[1]["ended_utc"])):
        raise RuntimeError("finite nonnegative charged ledger and ordered study times required")
    return {"admission": admission, "terminal": terminal, "receipt": receipt, "qualifier": qualifier,
            "forecast": forecast, "population": population, "outcomes": outcomes, "cells": cells,
            "forwards": forward_rows, "counts": counts, "parent_resources": parent_samples, "ledger": ledger}


def validate_native_population(admission, population):
    """Exact source export -> runtime projection, prior to any offline labels."""
    directory = DATA / "model_inputs"
    specs = (("scientific_bank", "SCIENTIFIC_PROMPTS.json"), ("scientific_token_binding", "SCIENTIFIC_TOKEN_BINDING.json"),
             ("preservation_bank", "PRESERVATION_PROMPTS.json"), ("preservation_token_binding", "PRESERVATION_TOKEN_BINDING.json"),
             ("prompt_export_manifest", "MODEL_EXPORT_MANIFEST.json"))
    packets, hashes = {}, {}
    for field, name in specs:
        path = Path(admission[field + "_path"]).resolve(strict=True)
        if path != (directory / name).resolve(strict=True) or sha(path) != admission[field + "_sha256"]:
            raise PermissionError("actual pinned prompt/token export differs:" + field)
        if admission["dispatch_hashes"].get(path.relative_to(ROOT).as_posix()) != sha(path):
            raise PermissionError("actual prompt/token export absent from native dispatch map")
        packets[field] = read(path); hashes[name] = sha(path)
    manifest = packets["prompt_export_manifest"]
    if (manifest.get("status") != "FROZEN_FRESH_OPEN_MODEL_EXPORT" or manifest.get("scientific_cases") != 432 or
            manifest.get("preservation_cases") != 24 or manifest.get("maximum_input_tokens") != 350 or
            manifest.get("maximum_forward_prefix_tokens") != 357 or manifest.get("labels_excluded") is not True or
            manifest.get("tokenizer_source") != admission["source"] or manifest.get("model_loads") != 0 or
            manifest.get("files") != {k: v for k, v in hashes.items() if k != "MODEL_EXPORT_MANIFEST.json"}):
        raise RuntimeError("pinned native export manifest differs")
    raw_keys = {"case_id", "raw_prompt", "raw_prompt_sha256"}
    token_keys = raw_keys | {"rendered_prompt", "rendered_prompt_sha256", "input_token_ids", "input_tokens"}
    expected = []
    for raw_key, token_key, size, cohort in (("scientific_bank", "scientific_token_binding", 432, "SCIENTIFIC"),
                                           ("preservation_bank", "preservation_token_binding", 24, "PRESERVATION")):
        raw, token = packets[raw_key], packets[token_key]
        if (set(token) != {"status", "canonical_source", "model_loads", "rows"} or
                token.get("status") != "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING" or token.get("canonical_source") != admission["source"] or
                token.get("model_loads") != 0 or len(raw) != size or len(token["rows"]) != size):
            raise RuntimeError("pinned original cohort count/tokenizer schema differs")
        by = {x["case_id"]: x for x in token["rows"]}
        if len(by) != size or len({x["case_id"] for x in raw}) != size: raise RuntimeError("pinned cohort duplicates")
        for x in raw:
            t = by.get(x.get("case_id"), {}); ids = t.get("input_token_ids", [])
            if (set(x) != raw_keys or set(t) != token_keys or x["raw_prompt"] != t["raw_prompt"] or
                    x["raw_prompt_sha256"] != t["raw_prompt_sha256"] or
                    hashlib.sha256(x["raw_prompt"].encode("utf-8")).hexdigest() != x["raw_prompt_sha256"] or
                    hashlib.sha256(t["rendered_prompt"].encode("utf-8")).hexdigest() != t["rendered_prompt_sha256"] or
                    not 1 <= len(ids) <= 350 or t["input_tokens"] != len(ids) or any(type(i) is not int or not 0 <= i < 151936 for i in ids)):
                raise RuntimeError("pinned raw/rendered/token/hash contract differs")
            expected.append({**x, **{k: t[k] for k in token_keys - raw_keys}, "cohort": cohort, "prefix_ids_sha256": digest(ids)})
    if len({x["case_id"] for x in expected}) != 456 or population != expected:
        raise RuntimeError("runtime population detached from exact ordered456 source inputs")
    return expected


def read_offline_labels(admission):
    """Called only after load_native succeeds; no discovery or fallback."""
    specs = (("scientific_labels", "SCIENTIFIC_LABELS.json"), ("preservation_labels", "PRESERVATION_LABELS.json"),
             ("corpus_manifest", "CORPUS_MANIFEST.json"), ("cpu_audit", "CPU_AUDIT.json"), ("token_shape_audit", "TOKEN_SHAPE_AUDIT.json"))
    values = {}
    for key, name in specs:
        path = Path(admission[key + "_path"]).resolve(strict=True)
        if path != (DATA / name).resolve(strict=True) or sha(path) != admission[key + "_sha256"]:
            raise PermissionError("exact offline label/CPU proof path/hash differs")
        values[key] = read(path)
    if values["cpu_audit"].get("errors") != []: raise RuntimeError("CPU independent verification audit failed")
    return values


def run(run_root, out):
    out = Path(out).resolve()
    if V3 / "analyses" not in out.parents: raise PermissionError("exclusive analysis must remain in V3analyses")
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "INCOMPLETE_NO_EFFICACY", "primary": None, "errors": [], "all_cases_required": True,
              "model_Torch_tokenizer_calls": 0, "fresh_labels_read": False, "secondary_policy_promotion": False}
    bindings = {}
    try:
        bundle = load_native(run_root)
        offline = read_offline_labels(bundle["admission"]); report["fresh_labels_read"] = True
        from .fresh_open_bank import score_response, blind_predictions
        from .fresh_open_verifier import verify_prompt
        from .short_response_panel import score_response as panel_score, solve_prompt as panel_solver
        science, panel = offline["scientific_labels"], offline["preservation_labels"]
        by = {x["case_id"]: x for x in science + panel}
        if set(by) != {x["case_id"] for x in bundle["population"]}: raise RuntimeError("offline labels lack exact raw native population")
        for x in bundle["population"]:
            if by[x["case_id"]]["prompt"] != x["raw_prompt"]: raise RuntimeError("offline label prompt/raw native prefix differs")
        for x in science:
            verified = verify_prompt(x["prompt"])
            if verified["answer"] != x["gold"] or verified["answer_kind"] != x["answer_kind"] or verified["source_status"] != x["source_status"]:
                raise RuntimeError("independent prompt-only solver disagrees with pinned scientific label")
        for x in panel:
            if panel_solver(x["prompt"]) != x["gold"]: raise RuntimeError("independent visible-copy panel control differs")
        result = analyze_scored(science, panel, bundle["cells"], score_response, panel_score, verify_prompt, panel_solver, blind_predictions)
        report.update(result, status="COMPLETE_CONDITIONAL_FRESH_REPLICATION", native_counts=bundle["counts"])
        report["costs"] = {"charged_native_worker_seconds": bundle["terminal"]["native_worker_seconds"],
            "worker_parent_elapsed_seconds": bundle["receipt"]["parent_elapsed_seconds"],
            "peak_GPU_allocated_bytes": max(x["resources"]["gpu_peak_allocated_bytes"] for x in bundle["forwards"] if isinstance(x.get("resources"), dict)),
            "peak_parent_RSS_bytes": max(x["rss_bytes"] for x in bundle["parent_resources"]),
            "partial_positions": None, "partial_compute": "UNKNOWN", "timing": "instrumented/descriptive, not production or matched-quality speed",
            "cumulative_native_seconds": sum(x.get("native_worker_seconds", 0.) for x in bundle["ledger"] if x.get("event") == "native_completed")}
        root = Path(run_root)
        for path in (root / "admission.json", root / "terminal_receipt.json", root / "review.json", root / "source_manifest.json"):
            bindings[str(path)] = sha(path)
        for name in ("receipt.json", "qualification.json", "registration.json", "matrix_forecast.json", "population.json", "random_routes.json", "outcomes.jsonl", "forwards.jsonl", "trajectories.jsonl"):
            bindings[str(root / "result" / name)] = sha(root / "result" / name)
        for key in ("scientific_labels", "preservation_labels", "corpus_manifest", "cpu_audit", "token_shape_audit"):
            bindings[bundle["admission"][key + "_path"]] = bundle["admission"][key + "_sha256"]
    except Exception as error:
        report.update(status="INCOMPLETE_NO_EFFICACY", primary=None, errors=[f"{type(error).__name__}: {error}"])
    bindings[str(Path(__file__))] = sha(Path(__file__)); bindings[str(PLAN)] = sha(PLAN)
    report["bindings"] = bindings
    with (out / "REPORT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); report = run(args.run, args.out)
    return 0 if not report["errors"] else 2


if __name__ == "__main__": raise SystemExit(main())
