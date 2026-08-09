from __future__ import annotations

"""Select the v2.1.1 procedural recovery checkpoint on DEV only.

Candidate zero is the original student under query-aware retrieval.  Stage A,
B1, and B2 must *earn* their weight changes by improving the v2.1.1 challenge
more than they damage protected original Phase 9 behaviors.  The selector uses
relative opportunity margins rather than absolute quality floors and always
keeps candidate zero as a valid rollback.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTECTED_METRICS = (
    "procedure_correctness",
    "supported_answer_correctness",
    "evidence_faithfulness",
    "correct_abstention_rate",
    "conflict_safe_handling_rate",
    "social_usable_rate",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def real(summary: dict[str, Any]) -> dict[str, Any]:
    packet = (summary.get("conditions") or {}).get("real_rag")
    if not isinstance(packet, dict):
        raise ValueError("Summary lacks conditions.real_rag")
    return packet


def fvalue(packet: dict[str, Any], key: str) -> float | None:
    raw = packet.get(key)
    return None if raw is None else float(raw)


def rel_loss(reference: float | None, candidate: float | None) -> float:
    if reference is None or candidate is None or reference <= 0:
        return 0.0
    return max(0.0, 1.0 - candidate / reference)


def delta(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return candidate - reference


@dataclass(frozen=True)
class CandidateInput:
    name: str
    original_summary: dict[str, Any]
    challenge_summary: dict[str, Any]
    adapter: str


def weighted_cost(losses: dict[str, float], weights: dict[str, float]) -> float:
    active = [(metric, float(weights.get(metric, 1.0))) for metric in PROTECTED_METRICS if metric in losses]
    denom = sum(max(0.0, weight) for _metric, weight in active)
    if denom <= 0:
        return 0.0
    return sum(losses[metric] * max(0.0, weight) for metric, weight in active) / denom


def candidate_packet(
    candidate: CandidateInput,
    *,
    baseline_original: dict[str, Any],
    baseline_challenge: dict[str, Any],
    policy_weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    bo = real(baseline_original)
    bc = real(baseline_challenge)
    co = real(candidate.original_summary)
    cc = real(candidate.challenge_summary)

    losses: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for metric in PROTECTED_METRICS:
        before = fvalue(bo, metric)
        after = fvalue(co, metric)
        loss = rel_loss(before, after)
        losses[metric] = loss
        metrics[metric] = {
            "baseline": before,
            "candidate": after,
            "delta": delta(before, after),
            "relative_loss": loss,
        }

    challenge_before = fvalue(bc, "procedure_correctness")
    challenge_after = fvalue(cc, "procedure_correctness")
    challenge_gain = delta(challenge_before, challenge_after)
    positive_challenge_gain = max(0.0, challenge_gain or 0.0)

    margins: dict[str, Any] = {}
    for policy_name, weights in policy_weights.items():
        cost = weighted_cost(losses, weights)
        margins[policy_name] = {
            "weighted_preservation_cost": cost,
            "positive_challenge_gain": positive_challenge_gain,
            "opportunity_margin": positive_challenge_gain - cost,
        }

    invalid_reasons: list[str] = []
    for label, summary in (("original", candidate.original_summary), ("challenge", candidate.challenge_summary)):
        if not bool(summary.get("evaluation_valid", True)):
            invalid_reasons.append(f"{label}_evaluation_invalid")

    return {
        "name": candidate.name,
        "adapter": candidate.adapter,
        "technical_invalidity": invalid_reasons,
        "original_preservation": metrics,
        "challenge": {
            "baseline_procedure_correctness": challenge_before,
            "candidate_procedure_correctness": challenge_after,
            "training_gain": challenge_gain,
        },
        "policies": margins,
    }


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[str]:
    """Pareto frontier over protected original metrics + challenge procedure score."""
    def vector(packet: dict[str, Any]) -> list[float]:
        vals: list[float] = []
        for metric in PROTECTED_METRICS:
            raw = packet["original_preservation"][metric]["candidate"]
            vals.append(float(raw) if raw is not None else float("-inf"))
        raw = packet["challenge"]["candidate_procedure_correctness"]
        vals.append(float(raw) if raw is not None else float("-inf"))
        return vals

    valid = [c for c in candidates if not c["technical_invalidity"]]
    out: list[str] = []
    for i, a in enumerate(valid):
        va = vector(a)
        dominated = False
        for j, b in enumerate(valid):
            if i == j:
                continue
            vb = vector(b)
            if all(x >= y for x, y in zip(vb, va)) and any(x > y for x, y in zip(vb, va)):
                dominated = True
                break
        if not dominated:
            out.append(a["name"])
    return out


def select(
    *,
    config: dict[str, Any],
    baseline_original: dict[str, Any],
    baseline_challenge: dict[str, Any],
    baseline_adapter: str,
    trained: list[CandidateInput],
) -> dict[str, Any]:
    policy_cfg = ((config.get("selection") or {}).get("policies") or {})
    if not policy_cfg:
        raise ValueError("Selection config lacks selection.policies")
    official = str((config.get("selection") or {}).get("official_policy") or "preservation_guarded")
    if official not in policy_cfg:
        raise ValueError(f"Official selection policy is undefined: {official}")

    # Candidate zero is represented using the baseline twice, so its loss/gain is exactly zero.
    all_inputs = [CandidateInput("query_aware_only", baseline_original, baseline_challenge, baseline_adapter), *trained]
    packets = [
        candidate_packet(c, baseline_original=baseline_original, baseline_challenge=baseline_challenge, policy_weights=policy_cfg)
        for c in all_inputs
    ]

    selections: dict[str, str] = {}
    selection_evidence: dict[str, Any] = {}
    for policy_name in policy_cfg:
        eligible = [p for p in packets if not p["technical_invalidity"]]
        trained_eligible = [p for p in eligible if p["name"] != "query_aware_only" and p["policies"][policy_name]["opportunity_margin"] > 0]
        if trained_eligible:
            winner = max(
                trained_eligible,
                key=lambda p: (
                    p["policies"][policy_name]["opportunity_margin"],
                    p["challenge"]["candidate_procedure_correctness"] or -1.0,
                ),
            )
        else:
            winner = next(p for p in eligible if p["name"] == "query_aware_only")
        selections[policy_name] = winner["name"]
        selection_evidence[policy_name] = {
            "selected": winner["name"],
            "selected_margin": winner["policies"][policy_name]["opportunity_margin"],
            "reason": (
                "trained candidate earned positive preservation-adjusted challenge gain"
                if winner["name"] != "query_aware_only"
                else "no trained candidate earned positive preservation-adjusted opportunity; keep unchanged weights"
            ),
        }

    official_selected = selections[official]
    selected_packet = next(p for p in packets if p["name"] == official_selected)
    disagreement = len(set(selections.values())) > 1
    invalid = [p["name"] for p in packets if p["technical_invalidity"]]
    if invalid and official_selected == "query_aware_only":
        status = "QUERY_AWARE_ONLY_SELECTED_WITH_INVALID_TRAINED_CANDIDATES"
    elif disagreement:
        status = "POLICY_SENSITIVITY_REVIEW_REQUIRED"
    elif official_selected == "query_aware_only":
        status = "QUERY_AWARE_ONLY_SELECTED"
    else:
        status = "TRAINED_CANDIDATE_SELECTED"

    return {
        "schema_version": "2.1.1",
        "status": status,
        "official_policy": official,
        "selected_candidate": official_selected,
        "selected_adapter": selected_packet["adapter"],
        "policy_selections": selections,
        "policy_sensitivity_disagreement": disagreement,
        "pareto_frontier": pareto_frontier(packets),
        "candidate_zero_policy": "unchanged original student weights + query_aware_v21 retrieval remains a first-class candidate",
        "selection_evidence": selection_evidence,
        "candidates": packets,
        "interpretation": {
            "challenge_gain": "Learned gain on the new v2.1.1 procedural DEV challenge with retrieval held fixed.",
            "preservation_cost": "Weighted mean of nonnegative relative losses on the original query-aware DEV path; improvements cannot hide regressions.",
            "fallback": "When training does not earn its preservation cost, keep the original adapter/weights and rely on query-aware retrieval.",
        },
    }


def markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# v2.1.1 Procedural Recovery DEV Selection",
        "",
        f"**Status:** `{packet['status']}`",
        f"**Official policy:** `{packet['official_policy']}`",
        f"**Selected candidate:** `{packet['selected_candidate']}`",
        "",
        "Candidate zero is the unchanged student under `query_aware_v21`. Training is accepted only when residual challenge gain exceeds preservation cost.",
        "",
        "## Candidate comparison",
        "",
        "| Candidate | Original procedure | Supported | Faithfulness | Challenge procedure | Guarded cost | Guarded margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in packet["candidates"]:
        op = c["original_preservation"]
        policy = c["policies"].get("preservation_guarded") or next(iter(c["policies"].values()))
        lines.append(
            "| {name} | {proc} | {sup} | {faith} | {chall} | {cost:.6f} | {margin:.6f} |".format(
                name=c["name"],
                proc=op["procedure_correctness"]["candidate"],
                sup=op["supported_answer_correctness"]["candidate"],
                faith=op["evidence_faithfulness"]["candidate"],
                chall=c["challenge"]["candidate_procedure_correctness"],
                cost=policy["weighted_preservation_cost"],
                margin=policy["opportunity_margin"],
            )
        )
    lines += [
        "",
        f"Pareto frontier: `{', '.join(packet['pareto_frontier'])}`",
        f"Policy selections: `{packet['policy_selections']}`",
        f"Policy sensitivity disagreement: `{packet['policy_sensitivity_disagreement']}`",
        "",
        "If the official selection is `query_aware_only`, that is an intentional research result: the retrieval fix is worth keeping and the tested weight update did not justify its quality cost.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--baseline-original", type=Path, required=True)
    p.add_argument("--baseline-challenge", type=Path, required=True)
    p.add_argument("--baseline-adapter", default="")
    for name in ("stage-a", "stage-b1", "stage-b2"):
        p.add_argument(f"--{name}-original", type=Path, required=True)
        p.add_argument(f"--{name}-challenge", type=Path, required=True)
        p.add_argument(f"--{name}-adapter", required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()

    config = load_json(args.config)
    trained = []
    for cli, label in (("stage_a", "stage_a"), ("stage_b1", "stage_b1"), ("stage_b2", "stage_b2")):
        trained.append(CandidateInput(
            label,
            load_json(getattr(args, f"{cli}_original")),
            load_json(getattr(args, f"{cli}_challenge")),
            str(getattr(args, f"{cli}_adapter")),
        ))
    packet = select(
        config=config,
        baseline_original=load_json(args.baseline_original),
        baseline_challenge=load_json(args.baseline_challenge),
        baseline_adapter=args.baseline_adapter,
        trained=trained,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
