from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from typing import Any

from ..utils import normalize_casefold, normalize_ws


@dataclass
class ScoreResult:
    correct_strict: bool
    correct_lenient: bool
    answer_type: str
    expected_normalized: Any
    predicted_strict_normalized: Any
    predicted_lenient_normalized: Any
    scorer: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CURRENCY = "$£€¥"
_UNIT_ALIASES = {
    "hrs": "hour",
    "hr": "hour",
    "hours": "hour",
    "hour": "hour",
    "mins": "minute",
    "min": "minute",
    "minutes": "minute",
    "minute": "minute",
    "secs": "second",
    "sec": "second",
    "seconds": "second",
    "second": "second",
    "kilograms": "kg",
    "kilogram": "kg",
    "grams": "g",
    "gram": "g",
    "meters": "m",
    "meter": "m",
    "kilometers": "km",
    "kilometer": "km",
    "liters": "l",
    "liter": "l",
    "milliliters": "ml",
    "milliliter": "ml",
    "newtons": "n",
    "newton": "n",
    "bolts": "bolt",
    "bolt": "bolt",
    "items": "item",
    "item": "item",
    "units": "unit",
    "unit": "unit",
}


def _strip_wrapping(value: Any) -> str:
    text = normalize_ws(value)
    text = re.sub(r"^\s*(?:final\s+answer|answer|result|output)\s*[:=]\s*", "", text, flags=re.I)
    text = text.strip().strip("`\"")
    text = re.sub(r"[.!?]+$", "", text).strip()
    return text


def _extract_number_and_unit(value: Any) -> tuple[float | None, str | None, str]:
    text = _strip_wrapping(value)
    canonical = text.replace("−", "-").replace(",", "")
    canonical = canonical.replace("×", "*").replace("÷", "/")
    match = re.search(rf"([{re.escape(_CURRENCY)}]?)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?)\s*(%|[A-Za-z°²³/]+)?", canonical)
    if not match:
        return None, None, text
    currency, numeric_text, unit = match.groups()
    try:
        if "/" in numeric_text:
            number = float(Fraction(numeric_text.replace(" ", "")))
        else:
            number = float(numeric_text)
    except Exception:
        number = None
    normalized_unit = unit.casefold() if unit else None
    if normalized_unit:
        normalized_unit = _UNIT_ALIASES.get(normalized_unit, normalized_unit)
    if currency:
        normalized_unit = currency
    return number, normalized_unit, text


def _numbers_close(left: float | None, right: float | None, abs_tol: float, rel_tol: float) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)


def _parse_fraction(value: Any) -> Fraction | None:
    text = _strip_wrapping(value).replace("−", "-").replace(",", "")
    text = text.replace("\\frac", "frac")
    latex = re.search(r"frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}", text)
    if latex:
        try:
            return Fraction(int(latex.group(1)), int(latex.group(2)))
        except Exception:
            return None
    mixed = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed:
        whole, numerator, denominator = map(int, mixed.groups())
        sign = -1 if whole < 0 else 1
        return Fraction(whole, 1) + sign * Fraction(numerator, denominator)
    match = re.search(r"([+-]?\d+)\s*/\s*([+-]?\d+)", text)
    if match:
        try:
            return Fraction(int(match.group(1)), int(match.group(2)))
        except Exception:
            return None
    number, _, _ = _extract_number_and_unit(text)
    if number is not None:
        return Fraction(str(number))
    return None


def _parse_boolean(value: Any) -> bool | None:
    text = _strip_wrapping(value).casefold()
    if text in {"true", "yes", "1", "on"}:
        return True
    if text in {"false", "no", "0", "off"}:
        return False
    match = re.search(r"\b(true|false|yes|no)\b", text)
    if match:
        return match.group(1) in {"true", "yes"}
    return None


def _parse_time(value: Any) -> tuple[int, int] | None:
    text = _strip_wrapping(value).casefold()
    match = re.search(r"\b([01]?\d|2[0-3])\s*:\s*([0-5]\d)\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(1[0-2]|0?[1-9])\s*:\s*([0-5]\d)\s*(am|pm)\b", text)
    if match:
        hour = int(match.group(1)) % 12
        if match.group(3) == "pm":
            hour += 12
        return hour, int(match.group(2))
    return None


def _parse_equation(value: Any) -> tuple[str, str, float] | None:
    text = _strip_wrapping(value).replace("−", "-")
    match = re.search(r"\b([A-Za-z])\s*(=|>=|<=|>|<|≥|≤)\s*([+-]?\d+(?:\.\d+)?)\b", text)
    if not match:
        return None
    variable, op, number = match.groups()
    op = {"≥": ">=", "≤": "<="}.get(op, op)
    return variable.casefold(), op, float(number)


def _parse_set(value: Any) -> tuple[str, ...] | None:
    text = _strip_wrapping(value)
    match = re.search(r"\{(.*?)\}", text)
    if not match:
        return None
    items = [normalize_casefold(item) for item in match.group(1).split(",") if normalize_ws(item)]
    return tuple(sorted(items))


def _parse_state(value: Any) -> dict[str, str] | None:
    text = _strip_wrapping(value)
    pairs = re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_.+-]+)", text)
    if not pairs:
        return None
    return {key.casefold(): val.casefold() for key, val in pairs}


def _parse_statistic(value: Any) -> dict[str, float] | None:
    text = _strip_wrapping(value).replace("−", "-").replace(",", "")
    result: dict[str, float] = {}
    for key in ["mean", "range", "median", "mode"]:
        match = re.search(rf"\b{key}\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
        if match:
            result[key] = float(match.group(1))
    return result or None


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    text = _strip_wrapping(value)
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_text(value: Any, *, preserve_case: bool = False) -> str:
    text = _strip_wrapping(value)
    text = re.sub(r"\s+", " ", text)
    return text if preserve_case else text.casefold()


def _normalized_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _UNIT_ALIASES.get(unit.casefold(), unit.casefold())


def _score_numeric(
    expected: Any,
    predicted: Any,
    *,
    require_unit: bool,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, bool, Any, Any, dict[str, Any]]:
    exp_num, exp_unit, _ = _extract_number_and_unit(expected)
    pred_num, pred_unit, _ = _extract_number_and_unit(predicted)
    number_ok = _numbers_close(exp_num, pred_num, abs_tol, rel_tol)
    exp_unit = _normalized_unit(exp_unit)
    pred_unit = _normalized_unit(pred_unit)
    unit_ok = exp_unit == pred_unit if exp_unit is not None else True
    strict = number_ok and (unit_ok if require_unit else True)
    lenient = number_ok
    return strict, lenient, {"number": exp_num, "unit": exp_unit}, {"number": pred_num, "unit": pred_unit}, {
        "number_match": number_ok,
        "unit_match": unit_ok,
    }


def score_candidate(
    expected: Any,
    predicted: Any,
    answer_type: str,
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> tuple[bool, bool, Any, Any, str, dict[str, Any]]:
    kind = answer_type.casefold()
    if predicted is None:
        return False, False, expected, None, "missing", {"reason": "no predicted answer"}

    if kind in {"integer", "number", "decimal", "money", "measurement", "quantity", "rate", "table_result", "program_output"}:
        require_unit = kind in {"money", "measurement", "quantity", "rate"}
        strict, lenient, exp_norm, pred_norm, details = _score_numeric(
            expected,
            predicted,
            require_unit=require_unit,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        if kind == "program_output":
            strict = lenient = strict or _normalize_text(expected, preserve_case=True) == _normalize_text(predicted, preserve_case=True)
        return strict, lenient, exp_norm, pred_norm, "numeric", details

    if kind in {"fraction", "probability"}:
        exp = _parse_fraction(expected)
        pred = _parse_fraction(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, str(exp) if exp else None, str(pred) if pred else None, "fraction", {}

    if kind == "boolean":
        exp = _parse_boolean(expected)
        pred = _parse_boolean(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "boolean", {}

    if kind == "time":
        exp = _parse_time(expected)
        pred = _parse_time(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "time", {}

    if kind in {"equation", "inequality"}:
        exp = _parse_equation(expected)
        pred = _parse_equation(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "equation", {}

    if kind == "set":
        exp = _parse_set(expected)
        pred = _parse_set(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "set", {}

    if kind == "state":
        exp = _parse_state(expected)
        pred = _parse_state(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "state_assignments", {}

    if kind == "statistic":
        exp = _parse_statistic(expected)
        pred = _parse_statistic(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "statistic_fields", {}

    if kind in {"json", "json_object", "ordered_collection", "unordered_collection"}:
        exp = _parse_jsonish(expected)
        pred = _parse_jsonish(predicted)
        if kind == "unordered_collection" and isinstance(exp, list) and isinstance(pred, list):
            strict = sorted(map(str, exp)) == sorted(map(str, pred))
        else:
            strict = exp == pred
        return strict, strict, exp, pred, "json", {}

    if kind in {"date", "datetime"}:
        def parse_date(value: Any) -> str | None:
            text = _strip_wrapping(value)
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"]:
                try:
                    return datetime.strptime(text, fmt).date().isoformat()
                except ValueError:
                    continue
            return None

        exp = parse_date(expected)
        pred = parse_date(predicted)
        correct = exp is not None and pred is not None and exp == pred
        return correct, correct, exp, pred, "date", {}

    preserve_case = kind in {"string", "program_output"}
    exp = _normalize_text(expected, preserve_case=preserve_case)
    pred = _normalize_text(predicted, preserve_case=preserve_case)
    strict = exp == pred
    lenient = normalize_casefold(exp) == normalize_casefold(pred)
    return strict, lenient, exp, pred, "text", {}


def score_answer(
    expected: Any,
    strict_candidate: Any,
    lenient_candidate: Any,
    answer_type: str,
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> ScoreResult:
    strict_ok, _, exp_norm, strict_norm, scorer, strict_details = score_candidate(
        expected,
        strict_candidate,
        answer_type,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    _, lenient_ok, _, lenient_norm, _, lenient_details = score_candidate(
        expected,
        lenient_candidate,
        answer_type,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return ScoreResult(
        correct_strict=bool(strict_ok),
        correct_lenient=bool(lenient_ok),
        answer_type=answer_type,
        expected_normalized=exp_norm,
        predicted_strict_normalized=strict_norm,
        predicted_lenient_normalized=lenient_norm,
        scorer=scorer,
        details={"strict": strict_details, "lenient": lenient_details},
    )


def token_f1(expected: Any, predicted: Any) -> float:
    expected_tokens = re.findall(r"[A-Za-z0-9_.:+-]+", normalize_casefold(expected))
    predicted_tokens = re.findall(r"[A-Za-z0-9_.:+-]+", normalize_casefold(predicted))
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0
    from collections import Counter

    exp_counts = Counter(expected_tokens)
    pred_counts = Counter(predicted_tokens)
    overlap = sum((exp_counts & pred_counts).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalized_exact(expected: Any, predicted: Any) -> bool:
    return normalize_casefold(expected) == normalize_casefold(predicted)


def state_equivalence(expected: Any, predicted: Any, *, f1_threshold: float = 0.9) -> dict[str, Any]:
    exact = normalized_exact(expected, predicted)
    f1 = token_f1(expected, predicted)
    exp_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", normalize_casefold(expected).replace(",", ""))
    pred_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", normalize_casefold(predicted).replace(",", ""))
    numeric_consistent = exp_numbers == pred_numbers if exp_numbers else True
    lenient = exact or (f1 >= f1_threshold and numeric_consistent)
    return {
        "exact": exact,
        "token_f1": f1,
        "numeric_signature_expected": exp_numbers,
        "numeric_signature_predicted": pred_numbers,
        "numeric_consistent": numeric_consistent,
        "lenient_equivalent": lenient,
    }
