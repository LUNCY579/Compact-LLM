"""Explicit independent-gold scoring; no model calls or target-forward inputs.

Natural abstention/communication use two independent blinded reviews bound to
the exact prediction bytes. Unknown semantic results remain None, never zero.
The A3 correction (research/reasoning_posttraining_v1/rrp005/A3_RUNBOOK.md)
is why canonical UNKNOWN and semantic abstention are separate measurements.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re

from .integrity import digest

DIMENSIONS = {"solving", "harder_transfer", "grounding_answerability", "output_contract", "communication"}
SCORERS = {"exact", "number", "json_exact", "fixed_list", "semantic_review"}
ERROR_CATEGORIES = {"format_only", "unnecessary_refusal", "unsupported_answer", "execution",
                    "evidence_binding", "missingknowledge"}
EVIDENCE_KINDS = {"format_only": {"contract_check"}, "unnecessary_refusal": {"answerability_review"},
    "unsupported_answer": {"grounding_review"}, "execution": {"verified_trace"},
    "evidence_binding": {"evidence_location_review"}, "missingknowledge": {"independent_knowledge_probe"}}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
ADJUDICATION_RUBRIC = {
    "experiment": "RQSFT-001", "version": 1,
    "review_process": {"minimum_independent_reviewers": 2, "blinded": True,
        "model_role_visible": False, "binding": "Exact row ID and UTF-8 prediction SHA256",
        "disagreement": "Semantic disagreement stays unresolved; missing judgments never become false or zero."},
    "semantic_correctness": "Judge the original complete input against explicit independently verified evaluation gold. Grounded answers must follow the supplied evidence and answerability contract. Do not use the final training target as a gold fallback.",
    "semantic_abstention": "Judge whether the response declines to assert an answer because the supplied evidence is insufficient. Natural language abstention can satisfy this judgment. Literal UNKNOWN is a separate exact-string count and cannot establish semantic awareness. A disclaimer alongside an unsupported asserted answer does not automatically count as abstention.",
    "communication": {"dimensions": ["usefulness", "clarity", "requested_conciseness"],
        "rating_minimum": 0, "rating_maximum": 4,
        "anchors": {"0": "Unusable, misleading, or fails to address the request.",
            "1": "Major problems in usefulness, clarity, or the requested conciseness.",
            "2": "Partly useful but with material communication problems.",
            "3": "Useful and clear; mostly follows the requested conciseness.",
            "4": "Fully useful, clear, and follows the requested conciseness."},
        "aggregation": "Mean independently supplied ratings per case; paired macro-family deltas. Missing ratings block advancement."},
    "error_diagnostics": {"categories": sorted(ERROR_CATEGORIES),
        "evidence_kinds": {key: sorted(values) for key, values in sorted(EVIDENCE_KINDS.items())},
        "requirement": "Two independent attributable reviews plus category-appropriate artifact hashes and rationale. Never infer missing knowledge, execution, or evidence-binding mechanisms from an unexplained wrong string."},
}
RUBRIC_SHA256 = digest(ADJUDICATION_RUBRIC)


def prediction_hash(prediction):
    if not isinstance(prediction, str):
        raise ValueError("Prediction must be the original captured string")
    return hashlib.sha256(prediction.encode("utf-8")).hexdigest()


def _json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def decimal(value):
        try:
            result = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("Invalid JSON decimal") from error
        if not result.is_finite():
            raise ValueError("Nonfinite JSON decimal")
        return result
    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
        elif isinstance(value, Decimal) and not value.is_finite():
            raise ValueError("Nonfinite JSON decimal")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON float")
    result = json.loads(text, object_pairs_hook=pairs, parse_float=decimal,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    finite(result)
    return result


def _typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _number(text):
    text = text.strip()
    if not NUMBER.fullmatch(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _reviews(row, prediction, adjudication):
    """Return only independently attributable and exact-byte-bound reviews."""
    if adjudication is None:
        return [], ["missing_blinded_adjudication"]
    if not isinstance(adjudication, dict):
        raise ValueError("Adjudication must be a prediction-bound object")
    if (adjudication.get("row_id") != row["id"]
            or adjudication.get("prediction_sha256") != prediction_hash(prediction)):
        raise ValueError("Adjudication row/prediction hash mismatch")
    if (adjudication.get("rubric_sha256") != RUBRIC_SHA256
            or row.get("evaluation", {}).get("rubric_sha256", RUBRIC_SHA256) != RUBRIC_SHA256):
        raise ValueError("Adjudication must use the prospectively registered public rubric")
    reviews = adjudication.get("reviews", [])
    if adjudication.get("status") != "RESOLVED" or not isinstance(reviews, list) or len(reviews) < 2:
        return [], ["unresolved_blinded_adjudication"]
    identifiers = []
    for review in reviews:
        if (not isinstance(review, dict) or not isinstance(review.get("reviewer"), str)
                or not review["reviewer"].strip() or review.get("independent") is not True
                or review.get("blinded") is not True or review.get("model_role_visible") is not False
                or "model_role" in review):
            raise ValueError("Two independent blinded reviewers without model-role visibility required")
        identifiers.append(review["reviewer"].strip())
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Repeated reviewer cannot establish independent adjudication")
    return reviews, []


def _consensus(reviews, field):
    values = [r.get(field) for r in reviews]
    if not values or any(type(v) is not bool for v in values) or len(set(values)) != 1:
        return None
    return values[0]


def _contract_format(prediction, spec, parsed, default):
    contract = spec.get("contract")
    if contract is None:
        return default
    if not isinstance(contract, dict) or not isinstance(contract.get("kind"), str):
        raise ValueError("Declared output contract needs an explicit kind")
    kind = contract["kind"]
    if kind in {"none", "free_text", "natural"}:
        return None
    if kind in {"exact", "literal"}:
        return prediction == contract.get("value", spec["answer"])
    if kind in {"unknown", "literal_unknown"}:
        return prediction == "UNKNOWN"
    if kind in {"number", "numeric"}:
        return _number(prediction) is not None
    if kind in {"boxed_number", "boxed_int"}:
        match = re.fullmatch(r"\\boxed\{([^{}]+)\}", prediction)
        value = _number(match[1]) if match else None
        return value is not None and (kind != "boxed_int" or value == value.to_integral_value())
    if kind in {"json", "json_object", "json_array"}:
        try:
            value = _json(prediction)
        except (ValueError, TypeError):
            return False
        if kind == "json_object" and not isinstance(value, dict):
            return False
        if kind == "json_array" and not isinstance(value, list):
            return False
        shape = contract.get("validshape")
        if shape is not None:
            if not isinstance(shape, dict) or set(shape) - {"keys", "length"}:
                raise ValueError("Unsupported declared JSON shape; review a new evaluator")
            if "keys" in shape and (not isinstance(value, dict) or set(value) != set(shape["keys"])):
                return False
            if "length" in shape and (not isinstance(value, (dict, list)) or len(value) != shape["length"]):
                return False
        return True
    if kind in {"fixed_list", "lines"}:
        separator = contract.get("separator", "\n")
        if not isinstance(separator, str) or not separator:
            raise ValueError("Fixed-list separator must be nonempty")
        expected = contract.get("length")
        return bool(prediction) and (expected is None or len(prediction.split(separator)) == expected)
    raise ValueError("Unsupported output contract; do not silently weaken it")


def score_prediction(row, prediction, adjudication=None):
    """Score evaluation.answer only. Training assistant targets are never gold fallback."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        raise ValueError("Explicit row identity required")
    spec = row.get("evaluation")
    if (not isinstance(spec, dict) or spec.get("dimension") not in DIMENSIONS
            or spec.get("scorer") not in SCORERS or not isinstance(spec.get("answer"), str)
            or type(spec.get("critical")) is not bool or "answerable" not in spec
            or spec.get("answerable") is not None and type(spec["answerable"]) is not bool):
        raise ValueError("Explicit independent gold, dimension, scorer and answerability required")
    if spec.get("rubric_sha256", RUBRIC_SHA256) != RUBRIC_SHA256:
        raise ValueError("Row rubric differs from the prospectively registered public rubric")
    if not isinstance(prediction, str):
        return {"id": row["id"], "status": "INCOMPLETE", "correct": None, "strict_format": None,
                "literal_unknown": None, "semantic_abstention": None, "communication_rating": None,
                "unresolved": ["missing_prediction"], "error_categories": [], "error_evidence": {}}
    answer, scorer = spec["answer"], spec["scorer"]
    correct = default_format = parsed = None
    if scorer == "exact":
        correct = prediction.strip() == answer.strip()
        default_format = None  # Wrong content is not an inferred format failure.
    elif scorer == "number":
        gold = _number(answer)
        if gold is None:
            raise ValueError("Independent numeric gold is invalid")
        parsed = _number(prediction)
        boxed = re.fullmatch(r"\s*\\boxed\{([^{}]+)\}\s*", prediction)
        if parsed is None and boxed:
            parsed = _number(boxed[1])
        correct = parsed == gold if parsed is not None else None
        default_format = _number(prediction) is not None
    elif scorer in {"json_exact", "fixed_list"}:
        gold = _json(answer)
        if scorer == "fixed_list" and not isinstance(gold, list):
            raise ValueError("Independent fixed-list gold must be a JSON list")
        parsed_ok = False
        try:
            parsed = _json(prediction)
            parsed_ok = True
            default_format = isinstance(parsed, list) if scorer == "fixed_list" else True
        except (ValueError, TypeError):
            default_format = False
            fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", prediction, flags=re.DOTALL)
            if fenced:
                try:
                    parsed = _json(fenced[1])
                    parsed_ok = True
                except (ValueError, TypeError):
                    pass
            if scorer == "fixed_list" and spec.get("contract", {}).get("kind") == "lines":
                separator = spec["contract"].get("separator", "\n")
                if not isinstance(separator, str) or not separator:
                    raise ValueError("Fixed-list separator must be nonempty")
                parsed, parsed_ok = prediction.split(separator), True
        correct = _typed_equal(parsed, gold) if parsed_ok else None
    strict_format = _contract_format(prediction, spec, parsed, default_format)
    needs_review = (scorer == "semantic_review" or spec["dimension"] == "communication"
                    or spec.get("answerable") is False or correct is None
                    or scorer == "exact" and correct is not True)
    reviews, unresolved = _reviews(row, prediction, adjudication) if adjudication is not None or needs_review else ([], [])
    reviewed_correct = _consensus(reviews, "semantic_correct")
    if scorer == "semantic_review" or correct is None or spec.get("answerable") is False or scorer == "exact" and correct is not True:
        correct = reviewed_correct
    semantic_abstention = _consensus(reviews, "semantic_abstention")
    # A parsed, non-UNKNOWN mathematical/structured answer is an explicit answer.
    # Canonical UNKNOWN never stands in for reviewed semantic abstention.
    if semantic_abstention is None and not needs_review and prediction.strip() != "UNKNOWN":
        semantic_abstention = False
    ratings = [r.get("communication_rating") for r in reviews]
    rating = None
    if spec["dimension"] == "communication":
        if ratings and all(type(v) in {int, float} and math.isfinite(v) and 0 <= v <= 4 for v in ratings):
            rating = sum(ratings) / len(ratings)
        else:
            unresolved.append("missing_or_invalid_communication_ratings")
    if correct is None:
        unresolved.append("unresolved_semantic_correctness")
    if needs_review and semantic_abstention is None:
        unresolved.append("unresolved_semantic_abstention")
    evidence = {}
    for review in reviews:
        for category, item in review.get("error_evidence", {}).items():
            if (category not in ERROR_CATEGORIES or not isinstance(item, dict)
                    or item.get("kind") not in EVIDENCE_KINDS.get(category, set())
                    or not isinstance(item.get("rationale"), str) or not item["rationale"].strip()
                    or not SHA.fullmatch(str(item.get("artifact_sha256", "")))):
                raise ValueError("Error mechanisms require attributable artifact-bound evidence")
            evidence.setdefault(category, []).append(dict(item, reviewer=review["reviewer"]))
    categories = []
    for category, items in evidence.items():
        if len({item["reviewer"] for item in items}) < 2:
            unresolved.append("unresolved_error_category:" + category)
            continue
        if category == "format_only" and not (correct is True and strict_format is False):
            raise ValueError("Format-only error requires semantic correctness and failed declared format")
        if category == "unnecessary_refusal" and not (spec.get("answerable") is True and semantic_abstention is True):
            raise ValueError("Unnecessary refusal needs reviewed refusal on an answerable input")
        if category == "unsupported_answer" and not (spec.get("answerable") is False and semantic_abstention is False):
            raise ValueError("Unsupported answer needs reviewed evidence/answerability failure")
        categories.append(category)
    return {"id": row["id"], "dimension": spec["dimension"], "status": "INCOMPLETE" if unresolved else "COMPLETE",
            "prediction_sha256": prediction_hash(prediction), "correct": correct, "strict_format": strict_format,
            "literal_unknown": prediction == "UNKNOWN", "normalized_unknown": prediction.strip() == "UNKNOWN",
            "semantic_abstention": semantic_abstention, "communication_rating": rating,
            "critical": spec["critical"], "answerable": spec.get("answerable"),
            "error_categories": sorted(categories), "error_evidence": evidence, "unresolved": sorted(set(unresolved)),
            "diagnostic_status": "EVIDENCE_BOUND" if categories else "UNASSIGNED_NO_MECHANISM_CLAIM"}


def evaluation_messages(row):
    """Full original history, excluding only the final assistant training target."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Complete history and explicit final assistant target required")
    return [dict(message) for message in messages[:-1]]


def shortcut_controls(row):
    """Fixed input-only controls. Never consult evaluation gold or final target."""
    messages = evaluation_messages(row)
    users = [message["content"] for message in messages if message.get("role") == "user"]
    if not users or any(not isinstance(value, str) for value in users):
        raise ValueError("Input-only controls require original user text")
    text = users[-1]
    words = text.split()
    labels = re.findall(r"(?:answer|result|label)\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return {"constant_answer": "0", "constant_refusal": "UNKNOWN", "copy_last_user": text,
            "position_first": words[0] if words else "", "position_last": words[-1] if words else "",
            "label_cue": labels[-1].strip() if labels else "UNKNOWN"}


def score_shortcut_controls(rows, adjudications=None):
    """Input-only shortcut audit; unresolved natural outputs require real reviews.

    This generates no data or model output. All control choices are fixed above;
    it does not choose a favorable constant using the evaluation gold.
    """
    adjudications = adjudications or {}
    grouped = {}
    for row in rows:
        for control, prediction in shortcut_controls(row).items():
            score = score_prediction(row, prediction, adjudications.get(control, {}).get(row["id"]))
            grouped.setdefault(control, []).append(score)
    result = {}
    for control, scores in grouped.items():
        resolved = all(score["status"] == "COMPLETE" for score in scores)
        result[control] = {"n": len(scores), "status": "COMPLETE" if resolved else "INCOMPLETE",
            "correctness": sum(score["correct"] for score in scores) / len(scores) if scores and resolved else None,
            "unresolved_ids": [score["id"] for score in scores if score["status"] != "COMPLETE"],
            "literal_unknown_count": sum(score["literal_unknown"] is True for score in scores),
            "semantic_abstention_count": (sum(score["semantic_abstention"] for score in scores)
                if all(type(score["semantic_abstention"]) is bool for score in scores) else None),
            "scores": scores}
    return {"status": "INCOMPLETE" if any(v["status"] != "COMPLETE" for v in result.values()) else "COMPLETE",
            "controls": result, "model_calls": 0, "control_choices_use_gold": False,
            "capability_evidence": False, "promotion_allowed": False}
