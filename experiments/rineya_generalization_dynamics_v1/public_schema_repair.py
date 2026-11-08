"""Pure, prospective public-schema serialization helper; no truth or model IO.

public_schema has exactly field_name (one nonempty string), value_type:"string",
nullable (bool), schema_provenance:"public_request", scalar_stringification (bool),
allow_bare_token (bool). All options are explicit. The frozen extraction profile
sets scalar_stringification=True and allow_bare_token=False. A caller may enable
the separately explicit bare-token rule prospectively, never from target flags.

Completed JSON objects are always returned byte-for-byte, including wrong keys,
wrong types/values, whitespace and order. Arrays, malformed/nonfinite/duplicate
JSON, multi-line or unquoted explanatory responses, empty responses and incomplete
generations are preserved. Quoted JSON strings are one complete value: their
decoded whitespace, escaped newlines and Unicode are preserved exactly when wrapped.
For numbers/booleans, stringification retains the entire original stripped JSON
lexeme (e.g. 1E+03 stays "1E+03"); it is not a numerical canonicalization. Literal
null is wrapped only when the PUBLIC schema allows null. UNKNOWN never becomes null.

Optional bare_token uses single_ascii_token_v1: 1-128 ASCII characters, initial
letter/_ and remaining letters/digits/_/./- only. Numeric-looking malformed JSON
and colon/equals explanation labels are excluded. It does not extract
from explanations, remove fences, split fields, solve tasks, inspect records,
repair spelling, normalize case, or choose a value. JSON-looking malformed text,
JSON quotes/brackets/braces and nonfinite JSON constants are never salvaged.

extract_declared_schema accepts only the source's anchored final Query/Ask
request grammar, or an exact retention instruction followed by Return only
{"key":"..."} with a string result. Null is allowed only by the validated
"N### code, or null if missing" request. Source-record Return/Reply phrases cannot
produce, change or make a request schema ambiguous. Multiple actual requests,
absent requests and multi-field/non-placeholder schemas return None.
No target, row ID, category, expected-value diagnostic, correctness flag or private
metadata is an input. Caller public provenance is a declaration, not verified review.
This helper cannot authorize deployment or establish learned/native quality gains.
"""
from __future__ import annotations

import json
import math
import re

from .common import Blocked, digest, require

PROFILE = "public_single_string_v1"
BARE_TOKEN_RULE = "single_ascii_token_v1"
MAX_RESPONSE_CHARS = 16384
_SCHEMA_FIELDS = {"field_name", "value_type", "nullable", "schema_provenance",
                  "scalar_stringification", "allow_bare_token"}
_BARE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{0,127}\Z", re.ASCII)
_SUBJECT = r"N[1-9][0-9]{2}"
_QUERY_BODY = (_SUBJECT + r" (?:code greatest revision|code, or null if missing|"
               r"fields north,south|route then destination then code|field (?:code|shipment|measurement))")
_QUERY_REQUEST = re.compile(r'\AQuery: (?P<body>' + _QUERY_BODY + r')\. Return (?P<shape>\{[^{}\r\n]*\})\.\Z')
_ASK_REQUEST = re.compile(r'\AAsk (?P<body>' + _QUERY_BODY + r')\. Reply (?P<shape>\{[^{}\r\n]*\})\.\Z')
_RETENTION_INSTRUCTIONS = {
    "Compute a+b.", "Compute a-b.", "Compute a*b.", "Compute a+b-c.",
    "Select minimum.", "Select maximum.", "Select median.", "Select second smallest.",
    "If value>=threshold use yes else no.", "If value<threshold use yes else no.",
    "If value>threshold use yes else no.", "If value<=threshold use yes else no.",
    "Compute left AND right.", "Compute left OR right.", "Compute left XOR right.", "Compute NOT(left AND right).",
    "Select status at greatest revision.", "Select status at smallest revision.", "Select status at middle revision.",
    "Uppercase text.", "Lowercase text.", "Reverse text preserving case.", "Take final three text characters.",
    "Look up selected key in table.", "Multiply quantity by factor.",
}
_RETENTION_REQUEST = re.compile(r'\A(?P<instruction>.+) Return only (?P<shape>\{[^{}\r\n]*\}) with a string result\.\Z')
_RETENTION_DEV_BODY = re.compile(r'\AOperation=[a-z_]+; values=\{[^\r\n]*\}\Z')
_RETENTION_TRAIN_RULE = re.compile(r'\ARule:[a-z_]+\Z')
_RETENTION_TRAIN_DATA = re.compile(r'\AData:\{[^\r\n]*\}\Z')


class _InvalidJSON(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise _InvalidJSON("nonfinite JSON constant")


def _finite(value):
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _parse(text):
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    if not _finite(value):
        raise _InvalidJSON("nonfinite JSON number")
    return value


def _schema(value):
    require(isinstance(value, dict) and set(value) == _SCHEMA_FIELDS,
            "Public schema must declare exactly one field, type and all frozen options")
    field = value["field_name"]
    require(isinstance(field, str) and 1 <= len(field) <= 128
            and bool(field.strip()) and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in field),
            "A nonempty single public field name is required")
    require(value["value_type"] == "string", "Only explicitly requested string values are supported")
    require(value["schema_provenance"] == "public_request", "Target-derived/private schemas are forbidden")
    for name in ("nullable", "scalar_stringification", "allow_bare_token"):
        require(type(value[name]) is bool, f"Public {name} must be an explicit boolean")
    return dict(value)


def _result(original, transformed, schema, reason, preservation=None):
    return {"profile": PROFILE, "applied": transformed != original, "reason": reason,
            "original_response": original, "transformed_response": transformed,
            "public_schema": dict(schema), "public_schema_sha256": digest(schema),
            "value_preservation": preservation, "truth_used": False,
            "native_quality_claim": False, "learned_capability_claim": False,
            "schema_provenance_verified_by_this_helper": False,
            "authorizes_deployment": False, "permits_model_promotion": False,
            "evidence_kind": "offline_mechanical_serialization_rule"}


def repair_completed_response(response: str, public_schema: dict, *, completed_generation: bool) -> dict:
    """Apply an explicit public contract to a whole completed response, without truth.

    Unsupported schema or input types raise Blocked. All unsupported response forms
    are returned unchanged with a reason. No extraction, model call or file IO occurs.
    """
    schema = _schema(public_schema)
    require(isinstance(response, str), "Response must be the original raw string")
    require(type(completed_generation) is bool, "Completion provenance must be an explicit boolean")
    if not completed_generation:
        return _result(response, response, schema, "INCOMPLETE_GENERATION")
    if len(response) > MAX_RESPONSE_CHARS:
        return _result(response, response, schema, "RESPONSE_LIMIT_PRESERVED")
    text = response.strip()
    if not text:
        return _result(response, response, schema, "EMPTY_RESPONSE")
    try:
        value = _parse(text)
    except _InvalidJSON:
        return _result(response, response, schema, "NONFINITE_OR_DUPLICATE_JSON_PRESERVED")
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError, OverflowError):
        if "\n" in text or "\r" in text:
            return _result(response, response, schema, "MULTILINE_RESPONSE_PRESERVED")
        if not schema["allow_bare_token"]:
            return _result(response, response, schema, "MALFORMED_OR_BARE_TOKEN_DISABLED")
        if _BARE_TOKEN.fullmatch(text) is None:
            return _result(response, response, schema, "MALFORMED_OR_EXPLANATORY_RESPONSE_PRESERVED")
        transformed = json.dumps({schema["field_name"]: text}, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False)
        return _result(response, transformed, schema, "WRAPPED_EXPLICIT_BARE_TOKEN", "whole_opaque_token_unchanged")
    if isinstance(value, dict):
        return _result(response, response, schema, "JSON_OBJECT_BYTE_PRESERVED")
    if isinstance(value, list):
        return _result(response, response, schema, "JSON_ARRAY_PRESERVED")
    if "\n" in text or "\r" in text:
        return _result(response, response, schema, "MULTILINE_RESPONSE_PRESERVED")
    if type(value) is str:
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            return _result(response, response, schema, "MALFORMED_UNICODE_PRESERVED")
        wrapped, reason, preservation = value, "WRAPPED_JSON_STRING", "decoded_json_string_unchanged"
    elif value is None:
        if not schema["nullable"]:
            return _result(response, response, schema, "NULL_NOT_PUBLICLY_ALLOWED")
        wrapped, reason, preservation = None, "WRAPPED_PUBLIC_LITERAL_NULL", "literal_json_null_unchanged"
    elif type(value) in {int, float, bool}:
        # Keep booleans distinct from int: True never becomes "1" and false never "0".
        if not schema["scalar_stringification"]:
            return _result(response, response, schema, "SCALAR_STRINGIFICATION_DISABLED")
        wrapped, reason, preservation = text, "WRAPPED_JSON_SCALAR_LEXEME", "entire_json_scalar_lexical_spelling_unchanged"
    else:
        return _result(response, response, schema, "UNSUPPORTED_JSON_VALUE_PRESERVED")
    transformed = json.dumps({schema["field_name"]: wrapped}, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
    return _result(response, transformed, schema, reason, preservation)


def _request(line):
    """Recognize only source-defined public request lines, excluding record prose."""
    for pattern in (_QUERY_REQUEST, _ASK_REQUEST):
        match = pattern.fullmatch(line)
        if match is not None:
            return match["shape"], match["body"].endswith(" code, or null if missing"), "evidence"
    match = _RETENTION_REQUEST.fullmatch(line)
    if match is not None and match["instruction"] in _RETENTION_INSTRUCTIONS:
        return match["shape"], False, "retention"
    return None


def extract_declared_schema(public_prompt: str) -> dict | None:
    """Extract only a frozen terminal PUBLIC declaration, never a target schema.

    Multi-field, multiple actual requests, absent/inferred/unrecognized declarations
    return None. Optional lexical bare-token salvage remains disabled by default.
    Record prose does not participate in either schema extraction or null routing.
    """
    if not isinstance(public_prompt, str) or not public_prompt.strip():
        return None
    lines = public_prompt.rstrip().splitlines()
    requests = [(index, request) for index, line in enumerate(lines)
                if (request := _request(line)) is not None]
    if len(requests) != 1 or requests[0][0] != len(lines) - 1:
        return None
    shape_text, nullable, request_kind = requests[0][1]
    if request_kind == "retention":
        dev_body = len(lines) >= 2 and _RETENTION_DEV_BODY.fullmatch(lines[-2]) is not None
        train_body = (len(lines) >= 3 and _RETENTION_TRAIN_RULE.fullmatch(lines[-3]) is not None
                      and _RETENTION_TRAIN_DATA.fullmatch(lines[-2]) is not None)
        if not (dev_body or train_body):
            return None
    try:
        shape = _parse(shape_text)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return None
    if not isinstance(shape, dict) or len(shape) != 1:
        return None
    field, placeholder = next(iter(shape.items()))
    if type(placeholder) is not str or placeholder != "...":
        return None
    try:
        return _schema({"field_name": field, "value_type": "string", "nullable": nullable,
                        "schema_provenance": "public_request", "scalar_stringification": True,
                        "allow_bare_token": False})
    except Blocked:
        return None
