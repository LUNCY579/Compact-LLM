"""Unselected paired source-preservation diagnostic over supplied metadata.

No text is graded and no files, models or banks are read. IDs and judgement flags
are caller declarations, not evidence of ancestry, correctness or native origin.
The finite-pair diagnostic never establishes quality advancement or permission.
"""
from __future__ import annotations

_ENROLL = frozenset(("case_id", "source_id", "family_id", "context_bin", "answerable",
                     "baseline_strict_correct", "baseline_semantic_correct"))
_CELL = frozenset(("source_id", "family_id", "context_bin", "answerable"))
_PAIR = frozenset(("case_id", "status", "baseline", "candidate"))
_FLAGS = ("strict_correct", "semantic_correct", "refused", "justified_refusal", "unsupported_answer",
          "terminated", "truncated")
_VIEW = frozenset((*_FLAGS, "output_tokens"))
_UNRESOLVED = ("RUN_FAILED", "ADJUDICATION_UNRESOLVED", "NOT_MEASURED")
_ACTIONS = ("native_execution_authorized", "protected_bank_access_authorized",
            "spending_authorized", "promotion_permitted", "release_permitted",
            "serving_change_authorized")


def _record(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ValueError(label + ": exact plain-dict keys required")


def _text(value, label):
    if type(value) is not str or not 1 <= len(value) <= 128 or not value.strip():
        raise ValueError(label + ": bounded nonempty plain string required")


def _view(value):
    _record(value, _VIEW, "judgement")
    if any(type(value[k]) is not bool for k in _FLAGS):
        raise ValueError("judgements require exact boolean flags")
    if type(value["output_tokens"]) is not int or not 0 <= value["output_tokens"] <= 1048576:
        raise ValueError("output_tokens requires a bounded plain integer")
    if value["terminated"] and value["truncated"]:
        raise ValueError("natural termination and truncation cannot both be true")


def _length():
    return {"count": 0, "tokens": 0, "minimum": None, "maximum": None}


def _bucket():
    counts = dict.fromkeys(("expected_pairs", "complete_pairs", "missing_pairs",
                           "run_failed_pairs", "unresolved_pairs", "unmeasured_pairs",
                           "new_answerable_refusals", "justified_refusal_losses", "new_unsupported_answers",
                           "natural_termination_losses", "new_truncations"), 0)
    for arm in ("baseline", "candidate"):
        counts.update(dict.fromkeys((arm + "_false_abstentions", arm + "_justified_refusals",
                                     arm + "_unsupported_answers", arm + "_natural_terminations",
                                     arm + "_truncations"), 0))
    for axis in ("strict", "semantic"):
        counts.update(dict.fromkeys(("baseline_" + axis + "_correct",
                                     "candidate_" + axis + "_correct",
                                     axis + "_repairs", axis + "_losses", axis + "_kept",
                                     "frozen_" + axis + "_successes",
                                     "unknown_" + axis + "_baseline_memberships",
                                     "unresolved_" + axis + "_source_successes"), 0))
    counts["length"] = {arm: {axis: {outcome: _length() for outcome in ("correct", "wrong")}
                                     for axis in ("strict", "semantic")}
                        for arm in ("baseline", "candidate")}
    return counts


def _tally(bucket, pair, entry):
    bucket["expected_pairs"] += 1
    answerable = entry["answerable"]
    for axis in ("strict", "semantic"):
        known = entry["baseline_" + axis + "_correct"]
        if known is None:
            bucket["unknown_" + axis + "_baseline_memberships"] += 1
        else:
            bucket["frozen_" + axis + "_successes"] += int(known)
            if known and (pair is None or pair["status"] != "COMPLETE"):
                bucket["unresolved_" + axis + "_source_successes"] += 1
    if pair is None:
        bucket["missing_pairs"] += 1
        return []
    if pair["status"] != "COMPLETE":
        field = {"RUN_FAILED": "run_failed_pairs",
                 "ADJUDICATION_UNRESOLVED": "unresolved_pairs",
                 "NOT_MEASURED": "unmeasured_pairs"}[pair["status"]]
        bucket[field] += 1
        return []
    bucket["complete_pairs"] += 1
    p, c = pair["baseline"], pair["candidate"]
    damage = []
    for arm, view in (("baseline", p), ("candidate", c)):
        bucket[arm + "_false_abstentions"] += int(answerable and view["refused"])
        bucket[arm + "_justified_refusals"] += int(view["justified_refusal"])
        bucket[arm + "_unsupported_answers"] += int(view["unsupported_answer"])
        bucket[arm + "_natural_terminations"] += int(view["terminated"])
        bucket[arm + "_truncations"] += int(view["truncated"])
    for axis in ("strict", "semantic"):
        key = axis + "_correct"
        bucket["baseline_" + key] += int(p[key])
        bucket["candidate_" + key] += int(c[key])
        bucket[axis + "_repairs"] += int(c[key] and not p[key])
        bucket[axis + "_kept"] += int(c[key] and p[key])
        loss = p[key] and not c[key]
        bucket[axis + "_losses"] += int(loss)
        if loss:
            damage.append(axis + "_correct_to_wrong")
        for arm, view in (("baseline", p), ("candidate", c)):
            length = bucket["length"][arm][axis]["correct" if view[key] else "wrong"]
            n = view["output_tokens"]
            length["count"] += 1
            length["tokens"] += n
            length["minimum"] = n if length["minimum"] is None else min(length["minimum"], n)
            length["maximum"] = n if length["maximum"] is None else max(length["maximum"], n)
    events = {
        "new_answerable_refusals": answerable and c["refused"] and not p["refused"],
        "justified_refusal_losses": p["justified_refusal"] and not c["justified_refusal"],
        "new_unsupported_answers": c["unsupported_answer"] and not p["unsupported_answer"],
        "natural_termination_losses": p["terminated"] and not c["terminated"],
        "new_truncations": c["truncated"] and not p["truncated"],
    }
    for key, occurred in events.items():
        bucket[key] += int(occurred)
        if occurred:
            damage.append(key)
    return damage


def source_preservation_tally(enrollment, outcomes, required_cells):
    """Expose paired losses without a pooled repair offset or subset acceptance.

    Up to 16384 explicit enrolled cases and supplied outcomes are supported. This
    object-shape bound is not a native phase allowance or measured resource fit.
    Required source/family/context/answerability cells must be supplied separately
    from observed outcomes. Baseline correctness membership is frozen before
    candidate outcomes and is retained for missing or failed pairs. Null membership
    stays unknown rather than being filled from later outcomes. Missing
    pairs and failed/unresolved/unmeasured records remain in the enrolled totals;
    no judgement is invented for them. Correctness, literal refusal, unsupported
    answers and natural termination are independent supplied axes. Token lengths
    must already follow the caller's prospectively defined EOS convention.
    """
    if type(enrollment) is not list or not 1 <= len(enrollment) <= 16384:
        raise ValueError("bounded nonempty enrollment required")
    if type(outcomes) is not list or len(outcomes) > 16384:
        raise ValueError("bounded outcome list required")
    if type(required_cells) is not list or not 1 <= len(required_cells) <= 16384:
        raise ValueError("explicit bounded required cells needed independently of outcomes")
    cells = set()
    for row in required_cells:
        _record(row, _CELL, "required cell")
        for key in ("source_id", "family_id", "context_bin"):
            _text(row[key], key)
        if type(row["answerable"]) is not bool:
            raise ValueError("required cell answerability must be boolean")
        key = tuple(row[k] for k in ("source_id", "family_id", "context_bin", "answerable"))
        if key in cells:
            raise ValueError("duplicate required cell")
        cells.add(key)
    enrolled, paired = {}, {}
    for row in enrollment:
        _record(row, _ENROLL, "enrollment")
        for key in ("case_id", "source_id", "family_id", "context_bin"):
            _text(row[key], key)
        if type(row["answerable"]) is not bool or row["case_id"] in enrolled:
            raise ValueError("exact answerability and unique enrolled case required")
        for axis in ("strict", "semantic"):
            known = row["baseline_" + axis + "_correct"]
            if known is not None and type(known) is not bool:
                raise ValueError("frozen baseline membership requires boolean or null")
        cell = tuple(row[k] for k in ("source_id", "family_id", "context_bin", "answerable"))
        if cell not in cells:
            raise ValueError("enrollment cell absent from the declared required set")
        enrolled[row["case_id"]] = row
    for row in outcomes:
        _record(row, _PAIR, "pair")
        _text(row["case_id"], "case_id")
        if row["case_id"] not in enrolled or row["case_id"] in paired:
            raise ValueError("unmatched or duplicate paired outcome")
        if type(row["status"]) is not str or row["status"] not in ("COMPLETE", *_UNRESOLVED):
            raise ValueError("explicit complete or unresolved status required")
        if row["status"] == "COMPLETE":
            _view(row["baseline"])
            _view(row["candidate"])
            for axis in ("strict", "semantic"):
                known = enrolled[row["case_id"]]["baseline_" + axis + "_correct"]
                if known is not None and row["baseline"][axis + "_correct"] is not known:
                    raise ValueError("paired baseline differs from frozen correctness membership")
            for view in (row["baseline"], row["candidate"]):
                if view["justified_refusal"] and (enrolled[row["case_id"]]["answerable"] or not view["refused"]):
                    raise ValueError("justified refusal requires unanswerable enrollment and a refusal")
        elif row["baseline"] is not None or row["candidate"] is not None:
            raise ValueError("unresolved pair requires null judgements; preserve partial evidence externally")
        paired[row["case_id"]] = row
    sources, total, damage = {}, _bucket(), {}
    for source_id, family_id, context_bin, answerable in sorted(cells):
        context = sources.setdefault(source_id, {}).setdefault(family_id, {}).setdefault(context_bin, {})
        context["answerable" if answerable else "unanswerable"] = _bucket()
    unresolved_ids = []
    unresolved_success_ids = {axis: [] for axis in ("strict", "semantic")}
    unknown_membership_ids = {axis: [] for axis in ("strict", "semantic")}
    for case_id in sorted(enrolled):
        entry, pair = enrolled[case_id], paired.get(case_id)
        source = sources.setdefault(entry["source_id"], {})
        family = source.setdefault(entry["family_id"], {})
        context = family.setdefault(entry["context_bin"], {})
        bucket = context.setdefault("answerable" if entry["answerable"] else "unanswerable", _bucket())
        _tally(bucket, pair, entry)
        flags = _tally(total, pair, entry)
        if flags:
            damage[case_id] = flags
        if pair is None or pair["status"] != "COMPLETE":
            unresolved_ids.append(case_id)
            for axis in ("strict", "semantic"):
                if entry["baseline_" + axis + "_correct"] is True:
                    unresolved_success_ids[axis].append(case_id)
        for axis in ("strict", "semantic"):
            if entry["baseline_" + axis + "_correct"] is None:
                unknown_membership_ids[axis].append(case_id)
    empty_cells, empty_source_correct = [], []
    for source_id, family_id, context_bin, answerable in sorted(cells):
        bucket = sources[source_id][family_id][context_bin]["answerable" if answerable else "unanswerable"]
        cell = {"source_id": source_id, "family_id": family_id,
                "context_bin": context_bin, "answerable": answerable}
        if bucket["expected_pairs"] == 0:
            empty_cells.append(cell)
        else:
            for axis in ("strict", "semantic"):
                if (bucket["frozen_" + axis + "_successes"] == 0
                        and bucket["unknown_" + axis + "_baseline_memberships"] == 0):
                    empty_source_correct.append({**cell, "axis": axis})
    observed = ("OBSERVED_DAMAGE" if damage else "UNRESOLVED_PAIRS" if unresolved_ids
                else "UNKNOWN_BASELINE_MEMBERSHIP" if any(total["unknown_" + axis + "_baseline_memberships"]
                                                          for axis in ("strict", "semantic"))
                else "UNMEASURED_REQUIRED_CELLS" if empty_cells
                else "UNMEASURED_SOURCE_CORRECT_CELLS" if empty_source_correct
                else "NO_OBSERVED_DAMAGE_IN_SUPPLIED_PAIRS")
    return {"status": "PAIRED_SOURCE_METADATA_DIAGNOSTIC_ONLY", "sources": sources,
            "totals": total, "observed_preservation": observed,
            "damage_case_ids": damage, "unresolved_case_ids": unresolved_ids,
            "unresolved_source_success_case_ids": unresolved_success_ids,
            "unknown_baseline_membership_case_ids": unknown_membership_ids,
            "empty_required_cells": empty_cells, "empty_source_correct_cells": empty_source_correct,
            "quality_advancement": "HOLD", "method_selected_for_experiment": False,
            "quality_improvement_established": False, "scientific_independence_verified": False,
            "limits": "Supplied flags and IDs only. Zero observed damage is not a confidence bound, "
                      "proof of ancestral independence, model measurement or general preservation. "
                      "No PRSC, MLP or RSD thresholds or acceptance procedures are transferred.",
            **dict.fromkeys(_ACTIONS, False)}
