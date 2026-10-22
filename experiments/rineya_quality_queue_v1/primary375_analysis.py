"""Unselected fixed-3.75 registration and paired-tally metadata prototypes.

Commitment hashes and correctness/refusal flags are supplied by a caller. Their
shape does not prove source custody, independent data, semantic correctness,
confidence eligibility, retention or cost. No file, model or bank is loaded.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

FAMILIES = ("simple_retrieval", "counting_arithmetic", "state_tracking")
CONTROLS = ("always_abstain", "never_abstain", "position", "operator_blind", "printed_number")
NO_ACTIONS = {key: False for key in (
    "native_execution_authorized", "protected_bank_access_authorized",
    "spending_authorized", "promotion_permitted", "release_permitted",
    "serving_change_authorized")}
_REG_KEYS = frozenset((
    "schema", "arms", "serving_bias", "families", "cal_rows", "check_rows",
    "max_input_tokens", "max_new_tokens", "retune", "practical_margin",
    "alpha", "confidence", "weighting", "multiplicity", "cluster_map_sha256",
    "component_design_review_sha256", "scorer", "controls", "control_floor_rule",
    "commitments", *NO_ACTIONS))
_CONF_KEYS = frozenset(("method", "sidedness", "repetitions", "seed", "quantile_rule",
                        "probability_convention", "source_sha256"))
_SCORER_KEYS = frozenset(("primary_tag", "domain", "scorer_sha256", "rubric_sha256", "review_sha256"))
_COMMIT_KEYS = frozenset(("source", "runtime", "ancestry", "data_review", "retention", "cost"))
_ENROLL_KEYS = frozenset(("case_id", "family", "cluster_id", "answerable"))
_OUTCOME_KEYS = frozenset(("case_id", "complete", "parent_correct", "candidate_correct",
                           "parent_abstained", "candidate_abstained"))


def _record(value, keys, label):
    if type(value) is not dict or len(value) != len(keys) or any(type(k) is not str for k in value) or set(value) != keys:
        raise ValueError(label + ": exact plain-dict keys required")


def _text(value, label, limit=128):
    if type(value) is not str or not 1 <= len(value) <= limit or not value.strip():
        raise ValueError(label + ": bounded nonempty plain string required")


def _hash(value, label):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + ": lowercase SHA256 required")


def _number(value, label, low, high):
    if type(value) not in (int, float) or not low < value <= high or not math.isfinite(value):
        raise ValueError(label + ": finite numeric bound required")


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + ": bounded plain integer required")


def validate_primary375_registration(record):
    """Validate one declaration format only; select no real analysis values."""
    _record(record, _REG_KEYS, "registration")
    _text(record["schema"], "schema")
    if record["schema"] != "RINEYA_FIXED375_ANALYSIS_DRAFT_V1":
        raise ValueError("registration schema")
    for key, expected in (("arms", ["parent_3.75", "matched_mlp_3.75"]),
                          ("families", list(FAMILIES)), ("controls", list(CONTROLS))):
        if (type(record[key]) is not list or len(record[key]) != len(expected)
                or any(type(x) is not str for x in record[key]) or record[key] != expected):
            raise ValueError(key + ": exact registered order and identities required")
    _number(record["serving_bias"], "serving_bias", 0, 8)
    if record["serving_bias"] != 3.75 or record["retune"] is not False:
        raise ValueError("fixed 3.75 without retuning required")
    for key, expected in (("cal_rows", 96), ("check_rows", 192),
                          ("max_input_tokens", 1024), ("max_new_tokens", 64)):
        _integer(record[key], key, expected, expected)
    _number(record["practical_margin"], "practical_margin", 0, 1)
    _number(record["alpha"], "alpha", 0, .5)
    if record["alpha"] == .5:
        raise ValueError("alpha must be strictly below .5")
    _text(record["weighting"], "weighting")
    if record["weighting"] != "EQUAL_FAMILY_CASE_WEIGHTED":
        raise ValueError("case-weighted family estimand required")
    _text(record["multiplicity"], "multiplicity", 2048)
    _text(record["control_floor_rule"], "control_floor_rule", 2048)
    for key in ("cluster_map_sha256", "component_design_review_sha256"):
        _hash(record[key], key)
    conf = record["confidence"]
    _record(conf, _CONF_KEYS, "confidence")
    _text(conf["method"], "confidence method")
    for key in ("sidedness", "quantile_rule", "probability_convention"):
        _text(conf[key], key)
    if conf["sidedness"] != "one_sided_lower" or conf["quantile_rule"] != "inverse_empirical_cdf":
        raise ValueError("explicit lower-tail direction and inverse-CDF rule required")
    if conf["probability_convention"] != "EXACT_SUPPLIED_BINARY_FLOAT_RATIO":
        raise ValueError("explicit probability convention required")
    _integer(conf["repetitions"], "repetitions", 1000, 50000)
    _integer(conf["seed"], "seed", 0, 2**63 - 1)
    _hash(conf["source_sha256"], "confidence source")
    scorer = record["scorer"]
    _record(scorer, _SCORER_KEYS, "scorer")
    _text(scorer["primary_tag"], "primary scorer tag")
    _text(scorer["domain"], "scorer domain")
    if scorer["domain"] not in ("raw_prediction", "extracted_first_line", "independent_semantic_adjudication"):
        raise ValueError("explicit scorer domain required")
    for key in ("scorer_sha256", "rubric_sha256", "review_sha256"):
        _hash(scorer[key], key)
    _record(record["commitments"], _COMMIT_KEYS, "commitments")
    for key, value in record["commitments"].items():
        _hash(value, key + " commitment")
    if any(record[key] is not False for key in NO_ACTIONS):
        raise ValueError("registration grants no execution or release authority")
    return {"status": "STRUCTURAL_REVIEW_ONLY", "method_selected_for_experiment": False,
            "scientific_independence_verified": False, "quality_improvement_established": False,
            "limits": "Declarations are not truth or independent design/data review; alpha-to-engine compatibility is external.",
            **NO_ACTIONS}


def _counts():
    return dict.fromkeys(("pairs", "parent_correct", "candidate_correct", "repairs", "losses",
                          "net", "parent_abstained", "candidate_abstained",
                          "new_refusals", "parent_wrong_nonrefusal", "candidate_wrong_nonrefusal"), 0)


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _proxy(answerable, unanswerable):
    return Fraction(27, 40) * (1 - Fraction(answerable["parent_correct"], answerable["pairs"])) + \
           Fraction(39, 40) * (1 - Fraction(unanswerable["parent_correct"], unanswerable["pairs"]))


def primary375_paired_tally(enrollment, outcomes):
    """Tally complete CHECK pairs using externally supplied rubric flags only.

    Refusal and correctness axes stay separate, even if a supplied rubric permits
    them to overlap. This does not score text or call an overlap a semantic truth.
    Rejected or incomplete records remain caller-owned evidence; no subset is used.
    """
    if type(enrollment) is not list or len(enrollment) != 192:
        raise ValueError("exact 192-case CHECK enrollment required")
    if type(outcomes) is not list or len(outcomes) != 192:
        raise ValueError("every enrolled case needs exactly one complete paired outcome")
    enrolled, clusters = {}, {}
    for row in enrollment:
        _record(row, _ENROLL_KEYS, "enrollment")
        for key in ("case_id", "family", "cluster_id"):
            _text(row[key], key)
        if row["family"] not in FAMILIES or type(row["answerable"]) is not bool:
            raise ValueError("registered family and exact answerability boolean required")
        case_id, cluster, family = row["case_id"], row["cluster_id"], row["family"]
        if case_id in enrolled:
            raise ValueError("duplicate enrolled case")
        if cluster in clusters and clusters[cluster] != family:
            raise ValueError("cluster spans families; this design cannot assert independent strata")
        enrolled[case_id] = row
        clusters[cluster] = family
    groups = {f: {"answerable": _counts(), "unanswerable": _counts()} for f in FAMILIES}
    paired = {}
    for row in outcomes:
        _record(row, _OUTCOME_KEYS, "outcome")
        _text(row["case_id"], "case_id")
        if any(type(row[key]) is not bool for key in _OUTCOME_KEYS - {"case_id"}) or row["complete"] is not True:
            raise ValueError("complete pairs and exact boolean flags required")
        if row["case_id"] in paired or row["case_id"] not in enrolled:
            raise ValueError("duplicate or unmatched paired outcome")
        paired[row["case_id"]] = row
    if set(paired) != set(enrolled):
        raise ValueError("missing paired outcome; no filtering or denominator shrink")
    for case_id in sorted(enrolled):
        entry, result = enrolled[case_id], paired[case_id]
        bucket = groups[entry["family"]]["answerable" if entry["answerable"] else "unanswerable"]
        p, c = result["parent_correct"], result["candidate_correct"]
        pa, ca = result["parent_abstained"], result["candidate_abstained"]
        bucket["pairs"] += 1
        bucket["parent_correct"] += int(p)
        bucket["candidate_correct"] += int(c)
        bucket["repairs"] += int(c and not p)
        bucket["losses"] += int(p and not c)
        bucket["net"] += int(c) - int(p)
        bucket["parent_abstained"] += int(pa)
        bucket["candidate_abstained"] += int(ca)
        bucket["new_refusals"] += int(ca and not pa)
        bucket["parent_wrong_nonrefusal"] += int(not p and not pa)
        bucket["candidate_wrong_nonrefusal"] += int(not c and not ca)
    if any(g["answerable"]["pairs"] != 48 or g["unanswerable"]["pairs"] != 16 for g in groups.values()):
        raise ValueError("each CHECK family requires exactly48A/16U; no pooled replacement")
    totals = {axis: {key: sum(g[axis][key] for g in groups.values()) for key in _counts()}
              for axis in ("answerable", "unanswerable")}
    parent_proxy = _proxy(totals["answerable"], totals["unanswerable"])
    candidate_a = dict(totals["answerable"], parent_correct=totals["answerable"]["candidate_correct"])
    candidate_u = dict(totals["unanswerable"], parent_correct=totals["unanswerable"]["candidate_correct"])
    candidate_proxy = _proxy(candidate_a, candidate_u)
    return {"status": "COMPLETE_PAIRED_METADATA_TALLY_ONLY", "families": groups, "totals": totals,
            "answerable_delta": _fraction(Fraction(totals["answerable"]["net"], 144)),
            "parent_proxy": _fraction(parent_proxy), "candidate_proxy": _fraction(candidate_proxy),
            "proxy_improvement": _fraction(parent_proxy - candidate_proxy),
            "observed_guards": {
                "complete_pairs": True,
                "no_family_net_answerable_loss": all(g["answerable"]["net"] >= 0 for g in groups.values()),
                "no_new_answerable_refusals": totals["answerable"]["new_refusals"] == 0,
                "no_unanswerable_correct_to_wrong": totals["unanswerable"]["losses"] == 0,
                "inherited_proxy_improvement": candidate_proxy < parent_proxy},
            "confidence_eligibility": "NOT_ESTABLISHED",
            "quality_improvement_established": False, "scientific_independence_verified": False,
            "limits": "External flags/identities only; no confidence, control-floor, retention, cost, semantic or native admission.",
            **NO_ACTIONS}
