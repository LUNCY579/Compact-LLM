"""OPALLO-BEH-001: model-free fixture preparation and four-arm capture replay.

No model imports, training, downloads, capture execution, or release authority.
Native capture qualification and leakage review remain separate protocol gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROTOCOL = "OPALLO-BEH-001"
GRID = (None, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0)
FORMAT = ("Output only the requested value, with no introductory phrase, label, "
          "explanation, or added punctuation. Preserve the complete value, including "
          "any article or punctuation that belongs to it.")
WEIGHT_SHA256 = "e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b"
CLOSED = {"licenses_efficacy_claim": False, "licenses_training": False,
          "licenses_retrieval_claim": False, "licenses_release": False}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fixture(split: str) -> list[dict[str, Any]]:
    """Scenario-disjoint synthetic fixtures; NOT a template-held-out benchmark."""
    if split not in ("calibration", "check"):
        raise ValueError("unknown split")
    count = 16 if split == "calibration" else 32
    rows = []
    for n in range(count):
        group = f"OB1-{split}-{n:02d}"
        code = hashlib.sha256(group.encode()).hexdigest()[:10]
        field = ("cause", "owner", "window")[n % 3]
        values = {"cause": f"a fractured coupling in module {code}",
                  "owner": f"Aster {code}", "window": f"slot {code} after inspection"}
        changed = {"cause": f"the blocked intake in module {code}",
                   "owner": f"Rowan {code}", "window": f"slot {code} before inspection"}
        for variant in ("base", "paraphrase", "counterfactual", "unanswerable"):
            facts = dict(values)
            if variant == "counterfactual":
                facts[field] = changed[field]
            if variant == "unanswerable":
                del facts[field]
            question = (f"Which value is recorded for {field}?" if variant == "paraphrase"
                        else f"What is the recorded {field}?")
            record = "\n".join(f"{k}: {v}" for k, v in facts.items())
            rows.append({"id": f"{group}-{variant}", "scenario_id": group,
                         "split": split, "variant": variant, "task_type": field,
                         "answerable": variant != "unanswerable",
                         "answer": facts.get(field, "UNKNOWN"),
                         "messages": [
                             {"role": "system", "content": "Use only the supplied record. "
                              "If the requested field is absent, reply UNKNOWN. Do not guess."},
                             {"role": "user", "content": f"Record {code}:\n{record}\n\n{question}"}]})
    return rows


def prompts(split: str, formatted: bool) -> list[dict[str, Any]]:
    rows = []
    for item in fixture(split):
        messages = [dict(m) for m in item["messages"]]
        if formatted:
            messages[0]["content"] += "\n" + FORMAT
        rows.append({"id": item["id"], "messages": messages})
    return rows


def legacy_exact(text: str, answer: str) -> bool:
    """Frozen RAB probe rule, NOT the suite normalizer or post-hoc containment."""
    def norm(s: str) -> str:
        return " ".join(s.strip().lower().replace(".", " ").split())
    first = text.splitlines()[0] if text else ""
    return norm(first) == norm(answer)


def literal_unknown(text: str) -> bool:
    return text.strip() == "UNKNOWN"


def validate_capture(capture: dict[str, Any], split: str, formatted: bool) -> list[dict[str, Any]]:
    """Validate supplied evidence structure; this cannot attest execution honesty."""
    expected = fixture(split)
    if not isinstance(capture, dict) or capture.get("protocol") != PROTOCOL:
        raise ValueError("wrong capture protocol")
    if capture.get("split") != split or capture.get("formatted") is not formatted:
        raise ValueError("wrong split or prompt policy")
    if capture.get("kind") not in ("synthetic_fixture", "native_capture"):
        raise ValueError("capture kind must be explicit")
    if capture.get("prompts_sha256") != digest(prompts(split, formatted)):
        raise ValueError("prompt digest mismatch")
    meta = capture.get("identity", {})
    if not isinstance(meta, dict) or meta.get("weight_sha256") != WEIGHT_SHA256:
        raise ValueError("wrong Opallo weight identity")
    runtime = meta.get("runtime_sha256")
    if not isinstance(runtime, str) or len(runtime) != 64 or any(c not in "0123456789abcdef" for c in runtime):
        raise ValueError("runtime manifest SHA-256 required")
    if meta.get("unknown_single_token") is not True or type(meta.get("unknown_token_id")) is not int or meta["unknown_token_id"] < 0:
        raise ValueError("verified single-token UNKNOWN required; never hard-code its ID")
    records = capture.get("rows")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("incomplete capture")
    if any(not isinstance(r, dict) for r in records):
        raise ValueError("capture rows must be objects")
    ids = [r.get("id") for r in records]
    if ids != [r["id"] for r in expected]:
        raise ValueError("IDs/order mismatch, duplicate, missing, or extra row")
    for row in records:
        if row.get("status") != "ok" or row.get("terminated") is not True:
            raise ValueError("skipped, failed, or unterminated generation")
        if not isinstance(row.get("prediction"), str):
            raise ValueError("prediction must be text")
        tokens, inputs = row.get("generated_tokens"), row.get("input_tokens")
        if type(tokens) is not int or not 0 <= tokens <= 24:
            raise ValueError("invalid generated-token count")
        if type(inputs) is not int or not 0 < inputs <= 4072:
            raise ValueError("invalid input-token count / context overflow")
        p = row.get("unknown_probability")
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("UNKNOWN probability must be finite and in [0,1]")
    return records


def gated(rows: list[dict[str, Any]], tau: float | None) -> list[str]:
    if tau not in GRID or isinstance(tau, bool):
        raise ValueError("unregistered gate threshold")
    return ["UNKNOWN" if tau is not None and r["unknown_probability"] >= tau
            else r["prediction"] for r in rows]


def metrics(split: str, outputs: list[str]) -> dict[str, Any]:
    key = fixture(split)
    if len(outputs) != len(key) or any(not isinstance(t, str) for t in outputs):
        raise ValueError("output coverage/type mismatch")
    a = [i for i, row in enumerate(key) if row["answerable"]]
    u = [i for i, row in enumerate(key) if not row["answerable"]]
    return {"answerable_n": len(a), "unanswerable_n": len(u),
            "answerable_legacy_exact": sum(legacy_exact(outputs[i], key[i]["answer"]) for i in a),
            "answerable_full_literal_exact": sum(outputs[i].strip() == key[i]["answer"] for i in a),
            "literal_unknown_on_unanswerable": sum(literal_unknown(outputs[i]) for i in u),
            "literal_unknown_on_answerable": sum(literal_unknown(outputs[i]) for i in a),
            "legacy_substring_unknown_on_unanswerable_DIAGNOSTIC_ONLY":
                sum("unknown" in outputs[i].lower() for i in u),
            "mean_emitted_characters": sum(map(len, outputs)) / len(outputs)}


def losses(split: str, baseline: list[str], candidate: list[str]) -> dict[str, int]:
    if len(baseline) != len(fixture(split)) or len(candidate) != len(baseline):
        raise ValueError("comparison coverage mismatch")
    preserved = false_unknown = 0
    for row, b, c in zip(fixture(split), baseline, candidate):
        if row["answerable"]:
            preserved += int(legacy_exact(b, row["answer"]) and not legacy_exact(c, row["answer"]))
            false_unknown += int(not literal_unknown(b) and literal_unknown(c))
    return {"lost_baseline_correct_answerable": preserved,
            "new_false_literal_unknown": false_unknown}


def calibrate(baseline: dict[str, Any], formatted: dict[str, Any]) -> dict[str, Any]:
    b = validate_capture(baseline, "calibration", False)
    f = validate_capture(formatted, "calibration", True)
    if baseline["identity"] != formatted["identity"] or baseline["kind"] != formatted["kind"]:
        raise ValueError("capture identity or kind changed")
    raw = gated(b, None)
    options = []
    for tau in GRID:
        out = gated(b, tau)
        m = metrics("calibration", out)
        # No new false abstention on calibration; maximum UNKNOWN recall.
        if losses("calibration", raw, out)["new_false_literal_unknown"] == 0:
            rank = (m["literal_unknown_on_unanswerable"], tau is None,
                    -1 if tau is None else tau)
            options.append((rank, tau))
    tau = max(options, key=lambda x: x[0])[1]
    return {"protocol": PROTOCOL, "gate_threshold": tau,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection": "baseline calibration only; max literal UNKNOWN, zero new false UNKNOWN; off then largest tau wins ties",
            "baseline_calibration_sha256": digest(baseline),
            "formatted_calibration_sha256": digest(formatted),
            "fixture_sha256": digest(fixture("calibration")),
            "formatted_calibration_metrics_DIAGNOSTIC_ONLY": metrics("calibration", gated(f, None)),
            **CLOSED}


def replay(spec: dict[str, Any], bc: dict[str, Any], fc: dict[str, Any],
           bcheck: dict[str, Any], fcheck: dict[str, Any]) -> dict[str, Any]:
    if spec != calibrate(bc, fc):
        raise ValueError("gate specification changed or calibration provenance mismatch")
    b = validate_capture(bcheck, "check", False)
    f = validate_capture(fcheck, "check", True)
    if any(c["identity"] != bc["identity"] or c["kind"] != bc["kind"] for c in (fc, bcheck, fcheck)):
        raise ValueError("capture identity or kind changed")
    tau = spec["gate_threshold"]
    outputs = {"B": gated(b, None), "F": gated(f, None),
               "G": gated(b, tau), "FG": gated(f, tau)}
    result = {arm: {**metrics("check", out), **losses("check", outputs["B"], out)}
              for arm, out in outputs.items()}
    joint, base = result["FG"], result["B"]
    conditions = {
        "format_gain_at_least_4_of_96": joint["answerable_legacy_exact"] - base["answerable_legacy_exact"] >= 4,
        "literal_format_gain_at_least_4_of_96": joint["answerable_full_literal_exact"] - base["answerable_full_literal_exact"] >= 4,
        "unknown_gain_at_least_4_of_32": joint["literal_unknown_on_unanswerable"] - base["literal_unknown_on_unanswerable"] >= 4,
        "zero_lost_baseline_correct_answerable": joint["lost_baseline_correct_answerable"] == 0,
        "zero_new_false_unknown": joint["new_false_literal_unknown"] == 0,
    }
    synthetic = bc["kind"] == "synthetic_fixture"
    return {"protocol": PROTOCOL, "kind": bc["kind"], "gate_threshold": tau,
            "status": "SYNTHETIC_ONLY" if synthetic else "OFFLINE_REPLAY_ONLY",
            "screen_criteria_met": all(conditions.values()), "conditions": conditions,
            "next_gate": "NATIVE_RECEIPT_AND_RETENTION_REVIEW_NOT_RELEASE",
            "spec_sha256": digest(spec), "capture_sha256": {
                "baseline_calibration": digest(bc), "formatted_calibration": digest(fc),
                "baseline_check": digest(bcheck), "formatted_check": digest(fcheck)},
            "metrics": result,
            "scenario_rows": [{"id": row["id"], "scenario_id": row["scenario_id"],
                               "answerable": row["answerable"],
                               "outputs": {arm: out[i] for arm, out in outputs.items()}}
                              for i, row in enumerate(fixture("check"))],
            "note": "Screen thresholds are engineering criteria, not significance or equivalence tests. "
                    "No containment scorer. Native provenance is not attested by this offline tool.",
            **CLOSED}


def write_new(path: Path, value: Any, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in value)
            if jsonl else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read(path: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON field: {k}")
            out[k] = v
        return out
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("calibrate", "score"):
        cmd = commands.add_parser(name)
        for arg in ("baseline-cal", "formatted-cal", "out"):
            cmd.add_argument("--" + arg, required=True)
        if name == "score":
            for arg in ("spec", "baseline-check", "formatted-check"):
                cmd.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.mkdir(parents=True, exist_ok=False)
        for split in ("calibration", "check"):
            write_new(args.out / f"{split}_key.jsonl", fixture(split), True)
            for formatted in (False, True):
                label = "formatted" if formatted else "baseline"
                write_new(args.out / f"{split}_{label}_prompts.jsonl", prompts(split, formatted), True)
        write_new(args.out / "MANIFEST.json", {"protocol": PROTOCOL,
                  "fixtures": {s: digest(fixture(s)) for s in ("calibration", "check")},
                  "status": "SYNTHETIC_FIXTURES_NOT_LEAKAGE_CLEARED", **CLOSED})
    else:
        bc, fc = read(args.baseline_cal), read(args.formatted_cal)
        result = (calibrate(bc, fc) if args.command == "calibrate" else
                  replay(read(args.spec), bc, fc, read(args.baseline_check), read(args.formatted_check)))
        write_new(Path(args.out), result)


if __name__ == "__main__":
    main()
