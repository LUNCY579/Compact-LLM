"""Public JSON request boundary. CPU fixture; no evaluator imports or file APIs."""
import json
import sys

from .core import ProcedureSnapshot, run_case


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def request(raw):
    if len(raw) > 524288:
        raise ValueError("public request byte limit")
    payload = json.loads(raw, object_pairs_hook=unique_object,
                         parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if type(payload) is not dict or set(payload) != {"case", "arm", "snapshot"}:
        raise ValueError("unknown/evaluator request fields rejected")
    snapshot = ProcedureSnapshot(payload["snapshot"]) if payload["snapshot"] is not None else None
    return run_case(payload["case"], arm=payload["arm"], snapshot=snapshot)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        raw = sys.stdin.buffer.read(524289)
        output = request(raw)
    except (ValueError, TypeError, KeyError) as exc:
        output = {"error": f"PUBLIC_BOUNDARY_REJECTED: {exc}", "stop_reason": "FAILED"}
    sys.stdout.write(json.dumps(output, ensure_ascii=False, allow_nan=False))
    return 0 if output.get("error") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
