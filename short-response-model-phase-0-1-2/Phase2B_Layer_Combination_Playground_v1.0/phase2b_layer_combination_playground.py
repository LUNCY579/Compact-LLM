from __future__ import annotations

"""Independent Phase 2B layer-combination playground.

This experiment is intentionally outside the official phase pipeline. It builds
raw structural combinations from the full Phase 1 SFT reference, optionally
extends an already recovered Phase 2B endpoint, evaluates every variant with the
repository's existing evaluation harness, and writes evidence only under a
separate playground output directory.

It never edits phase gates, phase reports, model catalogs, or accepted frontier
artifacts.
"""

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PACKAGED_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGED_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGED_REPO_ROOT))

from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION, require_quality_v4, summary_quality_score
from scripts.common.quality_gate import assert_valid_reference, quality_index
from scripts.common.opportunity_evaluation import evaluate_opportunity, pareto_frontier, resource_reduction

PLAYGROUND_VERSION = "1.2"
DEFAULT_QUICK_DROPS = [26, 13, 14, 10, 6, 19]
DEFAULT_THOROUGH_DROPS = [24, 23, 14, 19, 3, 13, 10, 21, 22, 5]
SOURCE_FINGERPRINT_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class VariantSpec:
    """One model to build or evaluate."""

    name: str
    description: str
    source_model: Path
    additional_drops: tuple[int, ...]
    total_drops: tuple[int, ...]
    existing_model: bool = False


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        values = [int(part) for part in parts]
    else:
        values = [int(item) for item in value]
    if any(item < 0 for item in values):
        raise ValueError("Layer IDs must be non-negative integers.")
    return list(dict.fromkeys(values))


def json_read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def percent(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def relative_loss(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def relative_increase(reference: float, candidate: float) -> float:
    return max(0.0, candidate - reference) / max(abs(reference), 1e-12)


def max_category_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    ref = reference.get("category_quality_scores", reference.get("category_scores", {}))
    cand = candidate.get("category_quality_scores", candidate.get("category_scores", {}))
    losses = [1.0 if category not in cand else max(0.0, float(score) - float(cand[category])) for category, score in ref.items()]
    return max(losses, default=0.0)


def infer_repo_root(candidate: Path) -> Path:
    candidate = candidate.resolve()
    markers = (
        Path("scripts/common/depth_reduction.py"),
        Path("scripts/phase_1_dense_sft/evaluate_model.py"),
        Path("scripts/phase_2_layer_selection/evaluate_teacher_forcing.py"),
    )
    for current in (candidate, *candidate.parents):
        if all((current / marker).exists() for marker in markers):
            return current
    raise FileNotFoundError(
        "Could not find the short-response repository root. Pass --repo-root explicitly."
    )


def resolve_recorded_path(value: str | Path, repo_root: Path) -> Path:
    """Resolve catalog paths even if the repository was moved."""

    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(repo_root / raw)
    parts_lower = [part.lower() for part in raw.parts]
    for anchor in ("artifacts", "data", "reports"):
        if anchor in parts_lower:
            index = parts_lower.index(anchor)
            candidates.append(repo_root.joinpath(*raw.parts[index:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the most portable interpretation for an actionable error message.
    return (repo_root / raw).resolve() if not raw.is_absolute() else raw


def read_catalog_endpoint(catalog_path: Path, repo_root: Path, stage: int | None = None) -> tuple[list[int], Path] | None:
    if not catalog_path.exists():
        return None
    payload = json_read(catalog_path)
    frontier = payload.get("frontier")
    if not isinstance(frontier, list) or not frontier:
        return None
    row: dict[str, Any]
    if stage is None:
        row = frontier[-1]
    else:
        matches = [item for item in frontier if int(item.get("stage", -1)) == stage]
        if not matches:
            return None
        row = matches[0]
    drops = parse_int_list(row.get("dropped_original_layer_ids", []))
    model_value = row.get("inference_model") or row.get("training_model_id")
    if not model_value:
        return None
    return drops, resolve_recorded_path(str(model_value), repo_root)


def auto_reference_model(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        resolved = resolve_recorded_path(explicit, repo_root)
        if not resolved.exists():
            raise FileNotFoundError(f"Reference model does not exist: {resolved}")
        return resolved
    candidates = (
        repo_root / "artifacts/phase_2_progressive_reduction_thorough_v2/reference/full_model",
        repo_root / "artifacts/phase_2_progressive_reduction/reference/full_model",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No full Phase 2B reference checkpoint was found. Pass --reference-model, for example "
        "artifacts/phase_2_progressive_reduction_thorough_v2/reference/full_model."
    )


def core_variant_sets(quick: set[int], thorough: set[int]) -> dict[str, tuple[str, set[int]]]:
    common = quick & thorough
    quick_only = quick - thorough
    thorough_only = thorough - quick
    nonoverlap = quick ^ thorough
    union = quick | thorough
    return {
        "common_only": (
            "Drop only the layers selected by both runs; tests the stable overlap.",
            common,
        ),
        "quick_unique_only": (
            "Drop only layers unique to the quick path.",
            quick_only,
        ),
        "thorough_unique_only": (
            "Drop only layers unique to the thorough path.",
            thorough_only,
        ),
        "nonoverlap_all": (
            "Drop every disagreement layer while retaining the shared drops.",
            nonoverlap,
        ),
        "quick_set_raw": (
            "Rebuild the quick run's complete drop set from the same full reference without recovery.",
            quick,
        ),
        "thorough_set_raw": (
            "Rebuild the thorough run's complete drop set from the same full reference without recovery.",
            thorough,
        ),
        "union_all": (
            "Drop every original layer selected by either run; the main union experiment.",
            union,
        ),
    }


def layer26_variant_sets(quick: set[int], thorough: set[int]) -> dict[str, tuple[str, set[int]]]:
    union = quick | thorough
    return {
        "common_only": (
            "Drop only the layers selected by both runs.",
            quick & thorough,
        ),
        "quick_set_raw": (
            "Quick set rebuilt from the full reference without recovery.",
            quick,
        ),
        "thorough_set_raw": (
            "Thorough set rebuilt from the full reference without recovery.",
            thorough,
        ),
        "thorough_plus_drop_6": (
            "Start with the thorough drop set and additionally remove original layer 6.",
            thorough | {6},
        ),
        "thorough_plus_drop_26": (
            "Start with the thorough drop set and additionally remove original layer 26.",
            thorough | {26},
        ),
        "union_all": (
            "Thorough drop set plus both quick-only layers 6 and 26.",
            union,
        ),
    }


def build_raw_specs(
    suite: str,
    reference_model: Path,
    quick: set[int],
    thorough: set[int],
) -> list[VariantSpec]:
    if suite == "core":
        mapping = core_variant_sets(quick, thorough)
    elif suite == "layer26":
        mapping = layer26_variant_sets(quick, thorough)
    elif suite == "all":
        mapping = {**core_variant_sets(quick, thorough), **layer26_variant_sets(quick, thorough)}
    else:
        raise ValueError(f"Unsupported suite: {suite}")
    return [
        VariantSpec(
            name=name,
            description=description,
            source_model=reference_model,
            additional_drops=tuple(sorted(drops)),
            total_drops=tuple(sorted(drops)),
        )
        for name, (description, drops) in mapping.items()
    ]


def add_leave_one_in_specs(
    specs: list[VariantSpec],
    reference_model: Path,
    quick: set[int],
    thorough: set[int],
) -> None:
    union = quick | thorough
    for retained in sorted(union):
        drops = union - {retained}
        specs.append(
            VariantSpec(
                name=f"union_leave_{retained}_in",
                description=(
                    f"Drop the full union except original layer {retained}; useful after union_all regresses."
                ),
                source_model=reference_model,
                additional_drops=tuple(sorted(drops)),
                total_drops=tuple(sorted(drops)),
            )
        )


def add_existing_endpoint_specs(
    specs: list[VariantSpec],
    repo_root: Path,
    quick_catalog: Path,
    thorough_catalog: Path,
) -> None:
    for name, description, catalog, stage in (
        (
            "quick_actual_endpoint",
            "Evaluate the actual quick endpoint, including any recovery updates.",
            quick_catalog,
            None,
        ),
        (
            "thorough_actual_stage06",
            "Evaluate the thorough run at six drops for a same-depth path comparison.",
            thorough_catalog,
            6,
        ),
        (
            "thorough_actual_endpoint",
            "Evaluate the actual thorough endpoint, including recovery updates.",
            thorough_catalog,
            None,
        ),
    ):
        endpoint = read_catalog_endpoint(catalog, repo_root, stage=stage)
        if endpoint is None:
            continue
        drops, model = endpoint
        if not model.exists():
            print(f"[skip] Catalog endpoint is missing: {model}")
            continue
        specs.append(
            VariantSpec(
                name=name,
                description=description,
                source_model=model,
                additional_drops=(),
                total_drops=tuple(sorted(drops)),
                existing_model=True,
            )
        )


def add_recovered_extensions(
    specs: list[VariantSpec],
    repo_root: Path,
    thorough_catalog: Path,
) -> None:
    endpoint = read_catalog_endpoint(thorough_catalog, repo_root, stage=None)
    if endpoint is None:
        raise FileNotFoundError(
            "Recovered extensions require the thorough model catalog and endpoint artifacts."
        )
    thorough_drops, thorough_model = endpoint
    if not thorough_model.exists():
        raise FileNotFoundError(f"Thorough endpoint does not exist: {thorough_model}")
    base = set(thorough_drops)
    for name, extra, description in (
        (
            "thorough_actual_plus_drop_6",
            {6},
            "Physically remove layer 6 from the recovered thorough endpoint.",
        ),
        (
            "thorough_actual_plus_drop_26",
            {26},
            "Physically remove layer 26 from the recovered thorough endpoint.",
        ),
        (
            "thorough_actual_plus_drop_6_26",
            {6, 26},
            "Physically remove both quick-only layers from the recovered thorough endpoint.",
        ),
    ):
        specs.append(
            VariantSpec(
                name=name,
                description=description,
                source_model=thorough_model,
                additional_drops=tuple(sorted(extra)),
                total_drops=tuple(sorted(base | extra)),
            )
        )


def deduplicate_specs(specs: Iterable[VariantSpec]) -> list[VariantSpec]:
    seen: set[str] = set()
    result: list[VariantSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


def run_command(command: list[str], repo_root: Path, sentinel: Path | None, resume: bool) -> None:
    if resume and sentinel is not None and sentinel.exists():
        print(f"[reuse] {sentinel}")
        return
    printable = subprocess.list2cmdline(command)
    print(f"[run] {printable}")
    subprocess.run(command, cwd=repo_root, check=True)


def install_repo_imports(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.common.depth_reduction import reduce_model_depth
    from scripts.common.experiment_utils import sha256_tree
    from scripts.common.model_introspection import find_decoder_stack
    from scripts.phase_2_layer_selection.create_depth_candidate import lineage_for_model

    return reduce_model_depth, sha256_tree, find_decoder_stack, lineage_for_model


def create_candidate(
    spec: VariantSpec,
    output_dir: Path,
    repo_root: Path,
    resume: bool,
    skip_reload_check: bool,
) -> Path:
    if spec.existing_model:
        return spec.source_model
    manifest_path = output_dir / "playground_depth_manifest.json"
    if resume and manifest_path.exists():
        print(f"[reuse] {output_dir}")
        return output_dir

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reduce_model_depth, sha256_tree, find_decoder_stack, lineage_for_model = install_repo_imports(repo_root)

    source = spec.source_model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    stack = find_decoder_stack(model)
    current_count = len(stack.layers)
    current_original_ids = lineage_for_model(source, current_count)
    requested = set(spec.additional_drops)
    missing = sorted(requested - set(current_original_ids))
    if missing:
        raise ValueError(
            f"{spec.name}: original layers {missing} are not present in source model {source}. "
            f"Current original IDs are {current_original_ids}."
        )
    keep_current = [
        current_index
        for current_index, original_id in enumerate(current_original_ids)
        if original_id not in requested
    ]
    if len(keep_current) < 2:
        raise ValueError(f"{spec.name}: refusing to create a model with fewer than two layers.")
    if len(keep_current) == current_count:
        raise ValueError(f"{spec.name}: no additional layers would be removed from its source model.")

    reduction = reduce_model_depth(
        model,
        retained_current_layer_ids=keep_current,
        current_original_layer_ids=current_original_ids,
    )

    smoke = tokenizer("Reply briefly: hello", return_tensors="pt")
    with torch.inference_mode():
        logits = model(**smoke).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{spec.name}: reduced model produced non-finite logits.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    if not skip_reload_check:
        reloaded = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        reloaded_count = len(find_decoder_stack(reloaded).layers)
        if reloaded_count != len(keep_current):
            raise RuntimeError(
                f"{spec.name}: reload check found {reloaded_count} layers, expected {len(keep_current)}."
            )
        del reloaded

    source_key = str(source)
    if source_key not in SOURCE_FINGERPRINT_CACHE:
        SOURCE_FINGERPRINT_CACHE[source_key] = sha256_tree(source)

    json_write(
        manifest_path,
        {
            "experiment": "phase2b_layer_combination_playground",
            "variant": spec.name,
            "description": spec.description,
            "source_model": str(source),
            "source_fingerprint": SOURCE_FINGERPRINT_CACHE[source_key],
            "source_current_original_layer_ids": current_original_ids,
            "additional_dropped_original_layer_ids": sorted(requested),
            "total_dropped_original_layer_ids": list(spec.total_drops),
            "retained_original_layer_ids": reduction.retained_original_layer_ids,
            "source_layer_count": current_count,
            "new_layer_count": len(keep_current),
            "parameters_before": reduction.parameters_before,
            "parameters_after": reduction.parameters_after,
            "reload_check_performed": not skip_reload_check,
            "evidence_boundary": (
                "Independent raw structural experiment. This checkpoint is not a Phase 2B frontier, "
                "does not modify any gate, and requires separate quality review."
            ),
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def evaluate_one(
    name: str,
    model_path: Path,
    evidence_dir: Path,
    repo_root: Path,
    probe_file: Path,
    case_limit: int | None,
    max_new_tokens: int,
    require_cuda: bool,
    baseline_results: Path | None,
    resume: bool,
    skip_nll: bool,
) -> dict[str, Any]:
    python = sys.executable
    evaluator = repo_root / "scripts/phase_1_dense_sft/evaluate_model.py"
    teacher = repo_root / "scripts/phase_2_layer_selection/evaluate_teacher_forcing.py"
    measure = repo_root / "scripts/common/measure_model_footprint.py"
    compare = repo_root / "scripts/phase_2_layer_selection/compare_outputs.py"

    generation_dir = evidence_dir / "generation"
    eval_cmd = [
        python,
        str(evaluator),
        "--model-id",
        str(model_path),
        "--benchmark",
        str(probe_file),
        "--output-dir",
        str(generation_dir),
        "--max-new-tokens",
        str(max_new_tokens),
        "--response-control",
        "eos_only",
    ]
    if case_limit is not None:
        eval_cmd += ["--limit", str(case_limit)]
    if require_cuda:
        eval_cmd.append("--require-cuda")
    generation_summary_path = generation_dir / "summary.json"
    if resume and generation_summary_path.exists():
        try:
            require_quality_v4(json_read(generation_summary_path), f"cached playground evaluation {name}")
        except ValueError:
            print(f"[stale evaluator] rebuilding {generation_summary_path}")
            shutil.rmtree(generation_dir, ignore_errors=True)
    run_command(eval_cmd, repo_root, generation_summary_path, resume)

    nll_dir = evidence_dir / "teacher_forcing"
    if not skip_nll:
        nll_cmd = [
            python,
            str(teacher),
            "--model-id",
            str(model_path),
            "--input",
            str(probe_file),
            "--output-dir",
            str(nll_dir),
        ]
        if case_limit is not None:
            nll_cmd += ["--limit", str(case_limit)]
        if require_cuda:
            nll_cmd.append("--require-cuda")
        run_command(nll_cmd, repo_root, nll_dir / "summary.json", resume)

    footprint_path = evidence_dir / "footprint.json"
    run_command(
        [python, str(measure), "--model", str(model_path), "--out", str(footprint_path)],
        repo_root,
        footprint_path,
        resume,
    )

    deviation_path: Path | None = None
    if baseline_results is not None:
        deviation_path = evidence_dir / "output_deviation.json"
        run_command(
            [
                python,
                str(compare),
                "--reference",
                str(baseline_results),
                "--candidate",
                str(generation_dir / "results.jsonl"),
                "--out-json",
                str(deviation_path),
                "--out-report",
                str(evidence_dir / "OUTPUT_DEVIATION.md"),
            ],
            repo_root,
            deviation_path,
            resume,
        )

    payload: dict[str, Any] = {
        "name": name,
        "model_path": str(model_path),
        "generation_summary": json_read(generation_dir / "summary.json"),
        "footprint": json_read(footprint_path),
        "nll_summary": None if skip_nll else json_read(nll_dir / "summary.json"),
        "output_deviation_summary": None,
        "evidence_dir": str(evidence_dir),
    }
    if deviation_path is not None:
        payload["output_deviation_summary"] = json_read(deviation_path)["summary"]
    json_write(evidence_dir / "combined_metrics.json", payload)
    return payload


def metric_row(
    spec: VariantSpec | None,
    payload: dict[str, Any],
    baseline: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    generation = payload["generation_summary"]
    require_quality_v4(generation, f"playground evaluation {payload['name']}")
    footprint = payload["footprint"]
    nll = payload.get("nll_summary")
    base_generation = baseline["generation_summary"]
    base_footprint = baseline["footprint"]
    base_nll = baseline.get("nll_summary")
    deviation = payload.get("output_deviation_summary") or {}

    base_bytes = float(base_footprint["parameter_tensor_bytes_loaded"])
    candidate_bytes = float(footprint["parameter_tensor_bytes_loaded"])
    weight_reduction = resource_reduction(base_bytes, candidate_bytes) or 0.0
    base_latency = float(base_generation.get("mean_generation_seconds", 0.0))
    candidate_latency = float(generation.get("mean_generation_seconds", 0.0))
    latency_reduction = None if base_latency <= 0 else 1.0 - candidate_latency / base_latency
    base_vram = float(base_generation.get("peak_vram_gb", 0.0))
    candidate_vram = float(generation.get("peak_vram_gb", 0.0))
    vram_reduction = None if base_vram <= 0 else 1.0 - candidate_vram / base_vram
    nll_value = None if nll is None else float(nll["mean_target_nll"])
    base_nll_value = None if base_nll is None else float(base_nll["mean_target_nll"])

    if spec is None:
        comparison = {
            "eligible": True,
            "decision": "REFERENCE",
            "decision_explanation": "full reference",
            "technical_valid": True,
            "technical_invalidity_reasons": [],
            "resource_gain_fraction": 0.0,
            "quality_opportunity_loss_fraction": 0.0,
            "opportunity_margin": 0.0,
            "quality_dimension_losses": {},
            "category_quality_relative_losses": {},
            "maximum_category_relative_loss": 0.0,
            "quality_profile": policy["quality_profile"],
        }
        status = "REFERENCE"
    else:
        comparison = evaluate_opportunity(
            base_generation,
            generation,
            resource_gain=weight_reduction,
            resource_name="loaded_weight_tensor_memory_reduction",
            profile=policy["quality_profile"],
            reference_target_nll=base_nll_value,
            candidate_target_nll=nll_value,
        )
        status = "WORTHWHILE" if comparison["eligible"] else "NOT_WORTHWHILE"

    return {
        "variant": payload["name"],
        "description": "Full reference" if spec is None else spec.description,
        "model_path": payload["model_path"],
        "additional_drops": [] if spec is None else list(spec.additional_drops),
        "total_drops": [] if spec is None else list(spec.total_drops),
        "existing_model": False if spec is None else spec.existing_model,
        "status": status,
        "eligible": bool(comparison["eligible"]),
        "decision_explanation": comparison["decision_explanation"],
        "technical_invalidity_reasons": list(comparison.get("technical_invalidity_reasons", [])),
        "remaining_layers": int(footprint["num_hidden_layers"]),
        "loaded_weight_bytes": int(footprint["parameter_tensor_bytes_loaded"]),
        "weight_reduction_fraction": weight_reduction,
        "resource_gain_fraction": float(comparison["resource_gain_fraction"]),
        "quality_opportunity_loss_fraction": float(comparison["quality_opportunity_loss_fraction"]),
        "opportunity_margin": float(comparison["opportunity_margin"]),
        "quality_dimension_losses": comparison.get("quality_dimension_losses", {}),
        "category_quality_relative_losses": comparison.get("category_quality_relative_losses", {}),
        "quality_profile": comparison.get("quality_profile", policy["quality_profile"]),
        "mean_quality_score": summary_quality_score(generation),
        "quality_index": quality_index(generation),
        "worst_decile_mean_quality_score": float(generation.get("worst_decile_mean_quality_score", 0.0)),
        "quality_pass_rate": float(generation.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(generation.get("intelligibility_pass_rate", 0.0)),
        "hard_failure_case_count": int(generation.get("hard_failure_case_count", 0)),
        "observable_quality_pass_rate": float(generation.get("observable_quality_pass_rate", generation.get("semantic_pass_rate", 0.0))),
        "complete_utterance_rate": float(generation.get("complete_utterance_rate", 0.0)),
        "critical_failure_count": int(generation.get("critical_failure_count", 0)),
        "major_case_count": int(generation.get("major_case_count", 0)),
        "major_issue_count": int(generation.get("major_issue_count", 0)),
        "mean_target_nll": nll_value,
        "exact_reply_match_rate": deviation.get("exact_reply_match_rate"),
        "mean_fuzzy_reply_similarity": deviation.get("mean_fuzzy_reply_similarity"),
        "new_critical_failure_count": deviation.get("new_critical_failure_count"),
        "new_major_issue_count": deviation.get("new_major_issue_count"),
        "mean_generation_seconds": candidate_latency,
        "latency_reduction_fraction": latency_reduction,
        "peak_vram_gb": candidate_vram,
        "vram_reduction_fraction": vram_reduction,
        "opportunity_assessment": comparison,
    }


def render_report(
    rows: list[dict[str, Any]], quick: set[int], thorough: set[int],
    probe_file: Path, case_limit: int | None, suite: str,
    policy: dict[str, Any],
) -> str:
    lines: list[str] = []
    candidate_rows = [row for row in rows if row["status"] != "REFERENCE"]
    frontier_ids = {row["variant"] for row in pareto_frontier(candidate_rows)}
    for row in rows:
        nll = "n/a" if row["mean_target_nll"] is None else f"{row['mean_target_nll']:.4f}"
        explanation = row["decision_explanation"].replace("|", "/")
        lines.append(
            f"| {row['variant']} | {row['status']} | {row['remaining_layers']} | `{row['total_drops']}` | "
            f"{percent(row['resource_gain_fraction'])} | {percent(row['quality_opportunity_loss_fraction'])} | "
            f"{percent(row['opportunity_margin'], signed=True)} | {row['mean_quality_score']:.3f} | "
            f"{row['worst_decile_mean_quality_score']:.3f} | {nll} | "
            f"{'YES' if row['variant'] in frontier_ids or row['status'] == 'REFERENCE' else 'NO'} | {explanation} |"
        )
    quick_only=sorted(quick-thorough); thorough_only=sorted(thorough-quick); common=sorted(quick&thorough); union=sorted(quick|thorough)
    limit_text="all available cases" if case_limit is None else str(case_limit)
    worthwhile=[row for row in candidate_rows if row["eligible"]]
    worthwhile.sort(key=lambda row:(row["opportunity_margin"], row["resource_gain_fraction"]), reverse=True)
    recommendations = "\n".join(
        f"- `{row['variant']}`: {percent(row['resource_gain_fraction'])} weight-memory gain, "
        f"{percent(row['quality_opportunity_loss_fraction'])} quality loss, "
        f"{percent(row['opportunity_margin'], signed=True)} margin"
        for row in worthwhile[:5]
    ) or "- No variant produced a positive proportional opportunity margin."
    return f"""# PHASE 2B LAYER-COMBINATION PLAYGROUND REPORT

## Experiment boundary

This is an independent exploratory experiment. It does **not** modify or approve
Phase 2B. Playground version `{PLAYGROUND_VERSION}` uses the same Quality
Evaluator v3 and central relative-opportunity engine as the official phases.

- Suite: `{suite}`
- Probe: `{probe_file}`
- Cases requested: {limit_text}
- Response control: `eos_only`
- Quality evaluator: `3.0`
- Quality profile: `{policy['quality_profile']}`
- Decision: loaded weight-memory gain must be strictly greater than combined proportional quality loss
- Quick drops: `{sorted(quick)}`
- Thorough drops: `{sorted(thorough)}`
- Shared drops: `{common}`
- Quick-only drops: `{quick_only}`
- Thorough-only drops: `{thorough_only}`
- Full union: `{union}`

## Results

| Variant | Decision | Remaining | Drops | Resource gain | Quality opportunity loss | Margin | Mean quality | Worst decile | Target NLL | Pareto | Explanation |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(lines)}

## Strongest proportionally worthwhile candidates

{recommendations}

Issue counts, informal grammar, follow-up questions, category deviations, tail
quality, and NLL remain visible in the JSON evidence and contribute continuously
where relevant. None acts as a hidden absolute veto. Technical invalidity still
blocks comparison. Human review of `OUTPUT_DEVIATION.md` and raw generations is
required before any official conclusion.
"""


def parse_args() -> argparse.Namespace:
    script_parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_parent.parent,
        help="Repository root. Auto-detected by walking upward from this path.",
    )
    parser.add_argument("--reference-model", type=Path, default=None)
    parser.add_argument(
        "--probe-file",
        type=Path,
        default=Path("data/phase_2_structural_reduction/probe_150_thorough_v4_1.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("playground_artifacts/phase2b_layer_combinations"),
    )
    parser.add_argument("--quick-drops", default=",".join(map(str, DEFAULT_QUICK_DROPS)))
    parser.add_argument("--thorough-drops", default=",".join(map(str, DEFAULT_THOROUGH_DROPS)))
    parser.add_argument(
        "--quick-catalog",
        type=Path,
        default=Path("artifacts/phase_2_progressive_reduction/model_catalog.json"),
    )
    parser.add_argument(
        "--thorough-catalog",
        type=Path,
        default=Path("artifacts/phase_2_progressive_reduction_thorough_v2/model_catalog.json"),
    )
    parser.add_argument("--suite", choices=("core", "layer26", "all"), default="core")
    parser.add_argument("--case-limit", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--include-existing-endpoints", action="store_true")
    parser.add_argument("--include-recovered-extensions", action="store_true")
    parser.add_argument("--leave-one-in-ablation", action="store_true")
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--quality-profile",
        choices=("balanced", "logic_oriented", "conversational_permissive"),
        default="balanced",
    )
    for deprecated in (
        "--max-mean-quality-loss", "--max-quality-index-loss",
        "--max-worst-decile-loss", "--max-category-quality-loss",
        "--max-semantic-pass-loss", "--max-reference-alignment-loss",
        "--min-quality-pass-rate", "--min-semantic-pass-rate",
        "--min-complete-utterance-rate", "--max-nll-increase",
    ):
        parser.add_argument(deprecated, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-major-issue-cases", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-new-major-issues", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = infer_repo_root(args.repo_root)
    probe_file = resolve_recorded_path(args.probe_file, repo_root)
    output_root = resolve_recorded_path(args.output_root, repo_root)
    quick_catalog = resolve_recorded_path(args.quick_catalog, repo_root)
    thorough_catalog = resolve_recorded_path(args.thorough_catalog, repo_root)
    reference_model = auto_reference_model(repo_root, args.reference_model)
    policy = {
        "quality_profile": args.quality_profile,
        "decision_rule": "loaded weight-memory gain > combined proportional quality loss",
        "absolute_quality_floors": "prohibited",
    }


    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file does not exist: {probe_file}")
    if args.case_limit is not None and args.case_limit < 1:
        raise ValueError("--case-limit must be positive.")

    quick = set(parse_int_list(args.quick_drops))
    thorough = set(parse_int_list(args.thorough_drops))
    # Prefer catalogs when present so later reruns automatically follow the
    # user's real accepted frontier lists rather than the packaged defaults.
    quick_endpoint = read_catalog_endpoint(quick_catalog, repo_root)
    thorough_endpoint = read_catalog_endpoint(thorough_catalog, repo_root)
    if quick_endpoint is not None:
        quick = set(quick_endpoint[0])
    if thorough_endpoint is not None:
        thorough = set(thorough_endpoint[0])

    specs = build_raw_specs(args.suite, reference_model, quick, thorough)
    if args.leave_one_in_ablation:
        add_leave_one_in_specs(specs, reference_model, quick, thorough)
    if args.include_existing_endpoints:
        add_existing_endpoint_specs(specs, repo_root, quick_catalog, thorough_catalog)
    if args.include_recovered_extensions:
        add_recovered_extensions(specs, repo_root, thorough_catalog)
    specs = deduplicate_specs(specs)

    plan = {
        "repo_root": str(repo_root),
        "reference_model": str(reference_model),
        "probe_file": str(probe_file),
        "output_root": str(output_root),
        "suite": args.suite,
        "playground_version": PLAYGROUND_VERSION,
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "relative_opportunity_policy": policy,
        "quick_drops": sorted(quick),
        "thorough_drops": sorted(thorough),
        "shared_drops": sorted(quick & thorough),
        "nonoverlap_drops": sorted(quick ^ thorough),
        "union_drops": sorted(quick | thorough),
        "variants": [
            {
                **asdict(spec),
                "source_model": str(spec.source_model),
            }
            for spec in specs
        ],
    }
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    json_write(output_root / "experiment_plan.json", plan)

    baseline_evidence = output_root / "reference/evidence"
    baseline = evaluate_one(
        name="reference",
        model_path=reference_model,
        evidence_dir=baseline_evidence,
        repo_root=repo_root,
        probe_file=probe_file,
        case_limit=args.case_limit,
        max_new_tokens=args.max_new_tokens,
        require_cuda=args.require_cuda,
        baseline_results=None,
        resume=args.resume,
        skip_nll=args.skip_nll,
    )
    baseline_results = baseline_evidence / "generation/results.jsonl"
    assert_valid_reference(baseline["generation_summary"], label="playground full reference")

    rows = [metric_row(None, baseline, baseline, policy)]
    raw_payloads: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        print(f"\n=== [{index}/{len(specs)}] {spec.name} ===")
        candidate_dir = output_root / "models" / spec.name
        model_path = create_candidate(
            spec,
            candidate_dir,
            repo_root,
            resume=args.resume,
            skip_reload_check=args.skip_reload_check,
        )
        evidence_dir = output_root / "evidence" / spec.name
        payload = evaluate_one(
            name=spec.name,
            model_path=model_path,
            evidence_dir=evidence_dir,
            repo_root=repo_root,
            probe_file=probe_file,
            case_limit=args.case_limit,
            max_new_tokens=args.max_new_tokens,
            require_cuda=args.require_cuda,
            baseline_results=baseline_results,
            resume=args.resume,
            skip_nll=args.skip_nll,
        )
        raw_payloads.append(payload)
        rows.append(metric_row(spec, payload, baseline, policy))

    json_write(output_root / "summary.json", {
        "schema_version": 3, "playground_version": PLAYGROUND_VERSION,
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION, "plan": plan, "results": rows,
        "worthwhile_variants": [row["variant"] for row in rows if row["status"] == "WORTHWHILE"],
        "not_worthwhile_variants": [row["variant"] for row in rows if row["status"] == "NOT_WORTHWHILE"],
    })
    report = render_report(rows, quick, thorough, probe_file, args.case_limit, args.suite, policy)
    (output_root / "PLAYGROUND_REPORT.md").write_text(report, encoding="utf-8")
    print(f"\nSummary: {output_root / 'summary.json'}")
    print(f"Report:  {output_root / 'PLAYGROUND_REPORT.md'}")


if __name__ == "__main__":
    main()
