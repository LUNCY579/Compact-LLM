from __future__ import annotations

"""Compare code-side and training-side recovery of Phase 9 procedural transfer.

The report separates three effects:
1) historical legacy retrieval,
2) the v2.1 query-aware retrieval policy with *unchanged* weights,
3) the recovered student under the same v2.1 retrieval policy.

This prevents a retrieval fix from being misreported as a learned-skill gain and
prevents a training gain from being hidden by the old collision-heavy prompt.
"""

import argparse
import json
from pathlib import Path
from typing import Any

PRESERVATION_METRICS = (
    "supported_answer_correctness",
    "correct_abstention_rate",
    "conflict_safe_handling_rate",
    "evidence_faithfulness",
    "social_usable_rate",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected summary object: {path}")
    return value


def real(summary: dict[str, Any]) -> dict[str, Any]:
    packet = (summary.get("conditions") or {}).get("real_rag")
    if not isinstance(packet, dict):
        raise ValueError("Summary lacks conditions.real_rag")
    return packet


def value(packet: dict[str, Any], key: str) -> float | None:
    raw = packet.get(key)
    return None if raw is None else float(raw)


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def relative_loss(reference: float | None, candidate: float | None) -> float:
    if reference is None or candidate is None or reference <= 0:
        return 0.0
    return max(0.0, 1.0 - candidate / reference)


def compare(legacy: dict[str, Any], query: dict[str, Any], recovered: dict[str, Any],
            challenge_baseline: dict[str, Any] | None, challenge_recovered: dict[str, Any] | None) -> dict[str, Any]:
    lp, qp, rp = real(legacy), real(query), real(recovered)
    lproc = value(lp, "procedure_correctness")
    qproc = value(qp, "procedure_correctness")
    rproc = value(rp, "procedure_correctness")
    retrieval_gain = delta(lproc, qproc)
    training_gain = delta(qproc, rproc)
    combined_gain = delta(lproc, rproc)

    preservation = {}
    losses = []
    for metric in PRESERVATION_METRICS:
        before, after = value(qp, metric), value(rp, metric)
        loss = relative_loss(before, after)
        preservation[metric] = {"before": before, "after": after, "delta": delta(before, after), "relative_loss": loss}
        if before is not None and after is not None:
            losses.append(loss)
    preservation_cost = sum(losses) / len(losses) if losses else 0.0

    challenge = None
    challenge_training_gain = None
    if challenge_baseline is not None and challenge_recovered is not None:
        cb, cr = real(challenge_baseline), real(challenge_recovered)
        cbp, crp = value(cb, "procedure_correctness"), value(cr, "procedure_correctness")
        challenge_training_gain = delta(cbp, crp)
        challenge = {
            "baseline_procedure_correctness": cbp,
            "recovered_procedure_correctness": crp,
            "training_gain": challenge_training_gain,
        }

    gains = [g for g in (training_gain, challenge_training_gain) if g is not None]
    mean_training_gain = sum(max(0.0, g) for g in gains) / len(gains) if gains else 0.0
    opportunity_margin = mean_training_gain - preservation_cost
    invalid = []
    for name, summary in (("legacy_baseline", legacy), ("query_aware_baseline", query), ("recovered", recovered)):
        if not bool(summary.get("evaluation_valid", True)):
            invalid.append(name)
    zero_blocker_resolved = rproc is not None and rproc > 0.0
    if invalid:
        status = "INVALID"
    elif zero_blocker_resolved and opportunity_margin > 0:
        status = "PROMISING_RECOVERY"
    elif zero_blocker_resolved:
        status = "RECOVERY_WITH_PRESERVATION_COST"
    elif (retrieval_gain or 0.0) > 0:
        status = "RETRIEVAL_ONLY_GAIN; TRAINING_STILL_NEEDED"
    else:
        status = "PROCEDURAL_BLOCKER_REMAINS"

    return {
        "schema_version": "2.1",
        "status": status,
        "technical_invalidity": invalid,
        "original_phase9": {
            "legacy_procedure_correctness": lproc,
            "query_aware_same_weights_procedure_correctness": qproc,
            "recovered_query_aware_procedure_correctness": rproc,
            "retrieval_policy_gain": retrieval_gain,
            "training_gain": training_gain,
            "combined_gain": combined_gain,
            "zero_correct_procedure_blocker_resolved": zero_blocker_resolved,
        },
        "challenge_v21": challenge,
        "preservation": preservation,
        "mean_preservation_relative_loss": preservation_cost,
        "mean_positive_training_gain": mean_training_gain,
        "training_opportunity_margin": opportunity_margin,
        "interpretation": {
            "retrieval_policy_gain": "Gain from changing evidence selection while holding model weights fixed.",
            "training_gain": "Gain from procedural recovery training while holding query-aware retrieval fixed.",
            "opportunity_margin": "Mean positive procedural training gain minus mean relative preservation loss; comparison to zero is relative, not an absolute quality floor.",
        },
    }


def md(packet: dict[str, Any]) -> str:
    o = packet["original_phase9"]
    lines = [
        "# v2.1 Procedural Recovery Comparison",
        "",
        f"**Status:** `{packet['status']}`",
        "",
        "## Original Phase 9 dev decomposition",
        "",
        f"- Legacy Real-RAG procedure correctness: `{o['legacy_procedure_correctness']}`",
        f"- Same weights + query-aware v2.1 retrieval: `{o['query_aware_same_weights_procedure_correctness']}`",
        f"- Recovered weights + query-aware retrieval: `{o['recovered_query_aware_procedure_correctness']}`",
        f"- Retrieval-policy gain: `{o['retrieval_policy_gain']}`",
        f"- Training-only gain: `{o['training_gain']}`",
        f"- Combined gain vs historical legacy path: `{o['combined_gain']}`",
        f"- Zero-correct-procedure blocker resolved: `{o['zero_correct_procedure_blocker_resolved']}`",
        "",
        "## Preservation cost",
        "",
        "| Metric | Before | After | Relative loss |",
        "|---|---:|---:|---:|",
    ]
    for metric, row in packet["preservation"].items():
        lines.append(f"| `{metric}` | {row['before']} | {row['after']} | {row['relative_loss']:.6f} |")
    lines += [
        "",
        f"Mean preservation relative loss: `{packet['mean_preservation_relative_loss']}`",
        f"Mean positive procedural training gain: `{packet['mean_positive_training_gain']}`",
        f"Relative training opportunity margin: `{packet['training_opportunity_margin']}`",
    ]
    if packet.get("challenge_v21"):
        c = packet["challenge_v21"]
        lines += [
            "",
            "## v2.1 procedural challenge dev",
            "",
            f"- Baseline: `{c['baseline_procedure_correctness']}`",
            f"- Recovered: `{c['recovered_procedure_correctness']}`",
            f"- Training gain: `{c['training_gain']}`",
        ]
    lines += [
        "",
        "The report deliberately separates retrieval and learned-weight gains. A positive retrieval-only gain is useful engineering evidence, but it is not counted as learned procedural recovery.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-baseline", type=Path, required=True)
    p.add_argument("--query-aware-baseline", type=Path, required=True)
    p.add_argument("--recovered", type=Path, required=True)
    p.add_argument("--challenge-baseline", type=Path, default=None)
    p.add_argument("--challenge-recovered", type=Path, default=None)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()
    packet = compare(load(args.legacy_baseline), load(args.query_aware_baseline), load(args.recovered),
                     load(args.challenge_baseline) if args.challenge_baseline else None,
                     load(args.challenge_recovered) if args.challenge_recovered else None)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(md(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
