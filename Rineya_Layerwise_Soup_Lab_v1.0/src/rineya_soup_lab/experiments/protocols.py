from __future__ import annotations

from typing import Any

from ..candidates import CandidateManager
from ..run_context import RunContext
from ..utils import write_csv, write_json


def run_secondary_protocols(context: RunContext, manager: CandidateManager) -> dict[str, Any]:
    if not context.config.get("evaluation.run_secondary_protocols_on_finalists", True):
        return {"enabled": False, "protocols": {}}
    selection_protocol = str(context.config.get("evaluation.selection_protocol"))
    protocols = context.config.get("evaluation.protocols")
    enabled = [
        name
        for name, settings in protocols.items()
        if name != selection_protocol and settings.get("enabled", True) is not False
    ]
    if not enabled:
        result = {"enabled": True, "selection_protocol": selection_protocol, "protocols": {}}
        write_json(context.path("analysis", "protocols", "secondary_protocols.json"), result)
        return result
    aliases = context.state.payload.get("aliases") or {}
    finalist_aliases = list(context.config.get("experiments.finalists.include", []))
    rows: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for protocol in enabled:
        seen: set[str] = set()
        protocol_results = []
        for alias in finalist_aliases:
            if alias not in aliases:
                continue
            candidate_id = context.state.resolve_alias(alias)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            record = context.state.candidate(candidate_id)
            from ..model.layer_map import MergeRecipe

            recipe = MergeRecipe.from_dict(record["recipe"])
            result = manager.evaluate(
                recipe,
                protocol_name=protocol,
                split="dev",
                retain=bool(record.get("retained", False)),
            )
            protocol_results.append(result.to_record())
            rows.append(
                {
                    "protocol": protocol,
                    "alias": alias,
                    "candidate_id": candidate_id,
                    "objective": result.objective.score,
                    "mean_quality": result.evidence.summary.get("mean_quality_score"),
                    "worst_decile_quality": result.evidence.summary.get(
                        "worst_decile_mean_quality_score"
                    ),
                    "median_reply_tokens": result.evidence.summary.get("median_reply_tokens"),
                    "mean_reasoning_tokens": result.evidence.summary.get(
                        "mean_reasoning_tokens_estimate"
                    ),
                }
            )
        output[protocol] = protocol_results
    path = context.path("analysis", "protocols")
    path.mkdir(parents=True, exist_ok=True)
    write_csv(path / "secondary_protocol_scores.csv", rows)
    payload = {
        "enabled": True,
        "selection_protocol": selection_protocol,
        "secondary_results_are_descriptive_only": True,
        "protocols": output,
    }
    write_json(path / "secondary_protocols.json", payload)
    return payload
