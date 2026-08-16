from __future__ import annotations

from typing import Any

from .analysis.bootstrap import paired_bootstrap_objective_difference
from .candidates import CandidateManager, CandidateResult
from .errors import SealedTestError
from .model.checkpoint import checkpoint_fingerprint, checkpoint_fingerprint_digest
from .model.layer_map import MergeRecipe
from .run_context import RunContext
from .utils import read_json, sha256_file, utc_timestamp, write_json


def _comparison(
    context: RunContext, candidate: CandidateResult, reference: CandidateResult, seed_offset: int
) -> dict[str, Any]:
    stats = context.config.get("statistics")
    boot = paired_bootstrap_objective_difference(
        candidate.evidence.rows,
        reference.evidence.rows,
        context.config.get("objective"),
        samples=int(stats.get("bootstrap_samples", 2000)),
        confidence_level=float(stats.get("confidence_level", 0.95)),
        seed=int(stats.get("seed", 0)) + seed_offset,
    )
    accepted = (
        boot.observed_delta >= float(stats.get("minimum_objective_delta", 0.0))
        and boot.probability_improvement
        >= float(stats.get("minimum_accept_probability", 0.5))
    )
    return {"accepted": accepted, "bootstrap": boot.to_dict()}


def freeze_selection(context: RunContext, manager: CandidateManager) -> CandidateResult:
    path = context.path("selection", "selection_freeze.json")
    if path.exists():
        frozen = read_json(path)
        candidate_id = str(frozen["selected_candidate_id"])
        context.state.set_alias("selected", candidate_id)
        return manager.load_result(candidate_id)

    settings = context.config.get("experiments.finalists")
    include = list(settings.get("include", []))
    maximum = max(1, int(settings.get("maximum", 6)))
    available_by_alias: dict[str, CandidateResult] = {}
    available_unique: dict[str, CandidateResult] = {}
    aliases = context.state.payload.get("aliases") or {}
    for alias in include:
        if alias not in aliases:
            continue
        result = manager.load_result(alias)
        available_by_alias[alias] = result
        available_unique.setdefault(result.candidate_id, result)
    if not available_unique:
        raise RuntimeError("No configured finalists have development evidence.")
    available = sorted(
        available_unique.values(),
        key=lambda result: (result.objective.score, result.candidate_id),
        reverse=True,
    )[:maximum]
    available_ids = {result.candidate_id for result in available}
    raw_objective_winner = available[0]
    strategy = str(settings.get("selection_strategy", "robust_sequential"))
    tournament: list[dict[str, Any]] = []
    if strategy == "objective_max":
        selected = raw_objective_winner
    else:
        anchor = available_by_alias.get("anchor") or manager.load_result("anchor")
        greedy = available_by_alias.get("greedy") or manager.load_result("greedy")
        current = max((anchor, greedy), key=lambda result: result.objective.score)
        tournament.append(
            {
                "step": "best_endpoint",
                "candidate_id": current.candidate_id,
                "objective": current.objective.score,
                "accepted": True,
                "reason": "starting control",
            }
        )
        for index, alias in enumerate(settings.get("sequence", []), 1):
            if alias == "best_endpoint" or alias not in available_by_alias:
                continue
            candidate = available_by_alias[alias]
            if candidate.candidate_id not in available_ids:
                tournament.append(
                    {
                        "step": alias,
                        "candidate_id": candidate.candidate_id,
                        "objective": candidate.objective.score,
                        "accepted": False,
                        "reason": "excluded by finalists.maximum ranking cap",
                    }
                )
                continue
            if candidate.candidate_id == current.candidate_id:
                tournament.append(
                    {
                        "step": alias,
                        "candidate_id": candidate.candidate_id,
                        "objective": candidate.objective.score,
                        "accepted": True,
                        "reason": "same numerical candidate as current",
                    }
                )
                continue
            evidence = _comparison(context, candidate, current, 41000 + index * 131)
            tournament.append(
                {
                    "step": alias,
                    "candidate_id": candidate.candidate_id,
                    "reference_candidate_id": current.candidate_id,
                    "objective": candidate.objective.score,
                    **evidence,
                }
            )
            if evidence["accepted"]:
                current = candidate
        selected = current

    selected_model_path = manager.ensure_materialized(selected.candidate_id)
    selected_checkpoint_fingerprint = checkpoint_fingerprint(selected_model_path)
    selected_checkpoint_digest = checkpoint_fingerprint_digest(
        selected_checkpoint_fingerprint
    )
    context.state.set_alias("selected", selected.candidate_id)

    comparisons: list[dict[str, Any]] = []
    for reference_alias in ("anchor", "greedy", "global_best", "layerwise_seed"):
        if reference_alias not in aliases:
            continue
        reference = manager.load_result(reference_alias)
        if reference.candidate_id == selected.candidate_id:
            continue
        comparisons.append(
            {
                "reference_alias": reference_alias,
                "reference_candidate_id": reference.candidate_id,
                **_comparison(context, selected, reference, len(comparisons) * 317),
            }
        )
    audit = read_json(context.path("manifests", "dataset_split_audit.json"))
    finalist_records: dict[str, dict[str, Any]] = {
        result.candidate_id: result.to_record() for result in available
    }
    for control in (available_by_alias.get("anchor"), available_by_alias.get("greedy")):
        if control is not None:
            finalist_records.setdefault(control.candidate_id, control.to_record())
    freeze = {
        "schema_version": "1.0",
        "frozen_at": utc_timestamp(),
        "selection_strategy": strategy,
        "selected_candidate_id": selected.candidate_id,
        "selected_alias": selected.label,
        "selected_recipe": selected.recipe.to_dict(),
        "selected_checkpoint_fingerprint": selected_checkpoint_fingerprint,
        "selected_checkpoint_digest": selected_checkpoint_digest,
        "selected_development_objective": selected.objective.to_dict(),
        "raw_objective_winner_candidate_id": raw_objective_winner.candidate_id,
        "raw_objective_winner_score": raw_objective_winner.objective.score,
        "robust_tournament": tournament,
        "selection_protocol": context.config.get("evaluation.selection_protocol"),
        "evaluation_backend": context.config.get("evaluation.backend"),
        "development_dataset": audit.get("dev"),
        "sealed_test_dataset_fingerprint": audit.get("test"),
        "test_scores_were_not_used": True,
        "finalists": list(finalist_records.values()),
        "paired_comparisons": comparisons,
        "objective_policy": context.config.get("objective"),
        "source_compatibility_report_sha256": sha256_file(
            context.path("manifests", "compatibility.json")
        ),
    }
    write_json(path, freeze)
    context.state.payload["selection"] = {
        "freeze_path": str(path),
        "selected_candidate_id": selected.candidate_id,
        "frozen_at": freeze["frozen_at"],
    }
    context.state.save()
    return selected


def run_sealed_test(context: RunContext, manager: CandidateManager) -> CandidateResult | None:
    settings = context.config.get("sealed_test")
    if not settings.get("enabled", True):
        return None
    freeze_path = context.path("selection", "selection_freeze.json")
    if settings.get("require_selection_freeze", True) and not freeze_path.exists():
        raise SealedTestError("Selection must be frozen before sealed-test evaluation.")
    freeze = read_json(freeze_path)
    candidate_id = str(freeze["selected_candidate_id"])
    test_path = context.config.get("datasets.test")
    if not test_path:
        raise SealedTestError("sealed_test.enabled is true but datasets.test is not configured.")
    current_test_sha256 = sha256_file(test_path)
    frozen_test = freeze.get("sealed_test_dataset_fingerprint") or {}
    frozen_test_sha256 = frozen_test.get("sha256")
    if frozen_test_sha256 and current_test_sha256 != frozen_test_sha256:
        raise SealedTestError(
            "The sealed-test dataset changed after selection was frozen. Create a new run."
        )
    selected_model_path = manager.ensure_materialized(candidate_id)
    current_checkpoint_fingerprint = checkpoint_fingerprint(selected_model_path)
    current_checkpoint_digest = checkpoint_fingerprint_digest(current_checkpoint_fingerprint)
    frozen_checkpoint_digest = freeze.get("selected_checkpoint_digest")
    if frozen_checkpoint_digest and current_checkpoint_digest != frozen_checkpoint_digest:
        raise SealedTestError(
            "The selected checkpoint changed after selection was frozen. Create a new run or "
            "restore the frozen weights."
        )
    candidate_record = context.state.candidate(candidate_id)
    recipe = MergeRecipe.from_dict(candidate_record["recipe"])
    ledger_path = context.path("sealed", "access_ledger.json")
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        "schema_version": "1.0",
        "dataset_sha256": current_test_sha256,
        "accesses": [],
    }
    if str(ledger.get("dataset_sha256")) != current_test_sha256:
        raise SealedTestError(
            "The sealed-test access ledger belongs to a different dataset fingerprint."
        )
    completed = [entry for entry in ledger["accesses"] if entry.get("status") == "complete"]
    if completed and not settings.get("allow_repeat", False):
        last = completed[-1]
        if str(last.get("candidate_id")) != candidate_id:
            raise SealedTestError(
                "The sealed test was already completed for a different candidate. Reopening it "
                "would invalidate the selection protocol."
            )
        return manager.load_result(candidate_id, split="test")
    started = [entry for entry in ledger["accesses"] if entry.get("status") == "started"]
    if started:
        active = started[-1]
        if str(active.get("candidate_id")) != candidate_id:
            raise SealedTestError("An incomplete sealed-test access exists for another candidate.")
        access_id = str(active["access_id"])
    else:
        access_id = f"sealed_{len(ledger['accesses']) + 1:03d}"
        ledger["accesses"].append(
            {
                "access_id": access_id,
                "candidate_id": candidate_id,
                "started_at": utc_timestamp(),
                "status": "started",
                "selection_freeze_sha256": sha256_file(freeze_path),
                "selected_checkpoint_digest": current_checkpoint_digest,
                "dataset_sha256": current_test_sha256,
                "protocol": context.config.get("evaluation.selection_protocol"),
            }
        )
        write_json(ledger_path, ledger)
    try:
        result = manager.evaluate(
            recipe,
            split="test",
            alias="selected_tested",
            retain=True,
            resume=True,
        )
    except Exception as exc:
        for entry in ledger["accesses"]:
            if entry.get("access_id") == access_id:
                entry["last_error"] = f"{type(exc).__name__}: {exc}"
                entry["last_attempt_at"] = utc_timestamp()
        write_json(ledger_path, ledger)
        raise
    for entry in ledger["accesses"]:
        if entry.get("access_id") == access_id:
            entry.update(
                {
                    "status": "complete",
                    "completed_at": utc_timestamp(),
                    "evaluation_dir": str(result.evidence.output_dir),
                    "summary_sha256": sha256_file(result.evidence.output_dir / "summary.json"),
                    "objective": result.objective.to_dict(),
                }
            )
    write_json(ledger_path, ledger)
    write_json(
        context.path("sealed", "sealed_test_result.json"),
        {
            "selected_candidate_id": candidate_id,
            "selection_freeze_sha256": sha256_file(freeze_path),
            "result": result.to_record(),
            "interpretation": "Descriptive final evidence only; no coefficient was changed after this evaluation.",
        },
    )
    return result
