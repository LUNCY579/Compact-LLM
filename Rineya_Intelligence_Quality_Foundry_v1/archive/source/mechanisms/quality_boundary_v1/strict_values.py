"""Candidate shared metadata boundary; integration requires independent review."""
import hashlib
import json
import math
import re

VERSION = "rineya.strict_metadata_values.v1"


class BoundaryError(ValueError):
    pass


def validate_json_value(value):
    """Exact JSON types, finite numbers and Unicode scalar text only."""
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise BoundaryError("nonfinite JSON number")
        return
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise BoundaryError("surrogate or invalid Unicode scalar text") from error
        return
    if type(value) is list:
        for item in value:
            validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise BoundaryError("JSON object key must be exact string")
            validate_json_value(key)
            validate_json_value(item)
        return
    raise BoundaryError("unsupported JSON value type: " + type(value).__name__)


def strict_decode(raw):
    if type(raw) not in (bytes, str):
        raise BoundaryError("exact bytes or string JSON source required")
    if type(raw) is bytes:
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise BoundaryError("invalid UTF8 JSON source") from error
    validate_json_value(raw)
    if raw.startswith("\ufeff"):
        raise BoundaryError("JSON BOM is not part of the frozen source encoding")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BoundaryError("duplicate JSON object key")
            result[key] = value
        return result
    def floating(text):
        result = float(text)
        if not math.isfinite(result):
            raise BoundaryError("JSON number overflow/nonfinite value")
        return result
    def constant(text):
        raise BoundaryError("nonstandard JSON numeric constant")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_float=floating, parse_constant=constant)
    except json.JSONDecodeError as error:
        raise BoundaryError("invalid JSON syntax") from error
    validate_json_value(value)
    return value


def canonical_bytes(value):
    validate_json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def exact_identity(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def exact_integer(value, *, minimum=None, maximum=None):
    if type(value) is not int:
        raise BoundaryError("exact integer required; bool/float aliases forbidden")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise BoundaryError("integer outside frozen bounds")
    return value


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_sha256(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BoundaryError("exact64 lowercase hexadecimal characters required")
    return value
