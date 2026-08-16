from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError
from .utils import canonical_json, deep_merge, sha256_file


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "project": {
        "name": "rineya_anchor_greedy_layerwise_soup",
        "seed": 20260815,
        "run_root": "runs",
        "run_name": None,
        "research_mode": True,
    },
    "models": {
        "anchor_path": None,
        "greedy_path": None,
        "tokenizer_path": None,
        "base_model_path": None,
        "trust_remote_code": False,
        "expected_layers": 12,
        "layer_key_regex": r"(?:^|\.)layers\.(\d+)\.",
        "merge_accumulation_dtype": "float32",
        "output_dtype": "preserve_anchor",
        "max_shard_size_gb": 1.5,
        "copy_non_floating_from": "anchor",
        "require_non_floating_equal": True,
        "allow_quantized_sources": False,
        "lineage_acknowledged": False,
    },
    "datasets": {
        "dev": None,
        "test": None,
        "allow_id_overlap": False,
        "allow_prompt_overlap": False,
    },
    "evaluation": {
        "backend": "bundled_v45",
        "official_repo_root": None,
        "python_executable": None,
        "quality_config": "official_v45/quality_evaluation.json",
        "protocols": {
            "official_no_think": {
                "enable_thinking": False,
                "extract_final_after_thinking": False,
                "response_control": "eos_only",
                "max_new_tokens": 32,
                "min_new_tokens_before_boundary": 1,
                "do_sample": False,
                "temperature": 0.6,
                "top_p": 0.9,
                "capture_token_probabilities": False,
                "require_cuda": False,
                "system_prompt": (
                    "Reply like a casual text message. Use only a few words or one short sentence. "
                    "Be accurate, use the conversation context, ask one brief clarification when "
                    "information is missing, and keep safety boundaries concise."
                ),
            },
            "think_then_short": {
                "enabled": False,
                "enable_thinking": True,
                "extract_final_after_thinking": True,
                "response_control": "eos_only",
                "max_new_tokens": 256,
                "min_new_tokens_before_boundary": 1,
                "do_sample": False,
                "temperature": 0.6,
                "top_p": 0.9,
                "capture_token_probabilities": True,
                "require_cuda": False,
                "system_prompt": (
                    "Think internally if useful, then give only the shortest complete final answer that is "
                    "accurate, natural, context-consistent, and safe."
                ),
            },
        },
        "selection_protocol": "official_no_think",
        "run_secondary_protocols_on_finalists": True,
        "external_command": None,
        "external_summary_path": "{output_dir}/summary.json",
        "external_results_path": "{output_dir}/results.jsonl",
        "mock_allowed": False,
    },
    "objective": {
        "metrics": [
            {"path": "mean_quality_score", "weight": 0.30, "transform": "maximize_01"},
            {"path": "worst_decile_mean_quality_score", "weight": 0.20, "transform": "maximize_01"},
            {"path": "complete_utterance_rate", "weight": 0.10, "transform": "maximize_01"},
            {"path": "observable_quality_pass_rate", "weight": 0.10, "transform": "maximize_01"},
            {"path": "mean_quality_dimensions.task_adequacy", "weight": 0.10, "transform": "maximize_01"},
            {"path": "mean_quality_dimensions.factual_correctness", "weight": 0.08, "transform": "maximize_01"},
            {"path": "mean_quality_dimensions.context_consistency", "weight": 0.05, "transform": "maximize_01"},
            {
                "path": "median_reply_tokens",
                "weight": 0.04,
                "transform": "target_range",
                "minimum": 1.0,
                "maximum": 12.0,
                "soft_upper": 24.0,
            },
            {"path": "duplicate_reply_rate", "weight": 0.03, "transform": "minimize_01"},
        ],
        "case_utility": {
            "quality_weight": 0.72,
            "completeness_weight": 0.10,
            "observable_pass_weight": 0.08,
            "brevity_weight": 0.10,
            "brevity_target_min": 1,
            "brevity_target_max": 12,
            "brevity_soft_upper": 24,
        },
        "normalization": "predeclared",
        "tail_fraction": 0.10,
        "category_max_loss_weight": 0.0,
    },
    "statistics": {
        "bootstrap_samples": 2000,
        "confidence_level": 0.95,
        "seed": 20260815,
        "minimum_accept_probability": 0.60,
        "minimum_objective_delta": 0.0001,
    },
    "experiments": {
        "run_endpoints": True,
        "uniform": {"enabled": True, "alpha": 0.5},
        "global_sweep": {
            "enabled": True,
            "alphas": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "refine_around_best": True,
            "refine_step": 0.025,
            "refine_radius": 0.10,
        },
        "probability_profile": {
            "enabled": False,
            "max_cases": 150,
            "top_k": 128,
            "include_reference_trajectory": True,
        },
        "layer_influence": {
            "enabled": True,
            "layers": "all",
            "bidirectional_swaps": True,
            "centered_alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "smoothing_penalty": 0.05,
            "global_shrinkage_penalty": 0.10,
            "minimum_reliability": 0.0,
        },
        "layer_search": {
            "enabled": True,
            "coordinate_descent": True,
            "max_passes": 2,
            "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "top_layers_only": None,
            "max_candidate_evaluations": 120,
            "pairwise_top_k": 0,
        },
        "finalists": {
            "include": ["anchor", "greedy", "uniform", "global_best", "layerwise_seed", "layerwise_best"],
            "maximum": 6,
            "selection_strategy": "robust_sequential",
            "sequence": ["best_endpoint", "uniform", "global_best", "layerwise_seed", "layerwise_best"],
        },
    },
    "storage": {
        "retain_candidate_weights": ["uniform", "global_best", "layerwise_seed", "layerwise_best"],
        "delete_ephemeral_after_evaluation": True,
        "candidate_cache_dir": "candidate_cache",
        "max_disk_gb": 80.0,
    },
    "sealed_test": {
        "enabled": True,
        "allow_repeat": False,
        "require_selection_freeze": True,
    },
    "reporting": {
        "make_plots": True,
        "write_markdown": True,
        "top_case_examples": 20,
    },
}


PATH_FIELDS = {
    "project.run_root",
    "models.anchor_path",
    "models.greedy_path",
    "models.tokenizer_path",
    "models.base_model_path",
    "datasets.dev",
    "datasets.test",
    "evaluation.official_repo_root",
    "evaluation.python_executable",
    "evaluation.quality_config",
}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    root: Path
    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    @property
    def digest(self) -> str:
        return sha256_file(self.path)

    def as_json(self) -> str:
        return canonical_json(self.raw)


def _resolve_path(value: Any, root: Path) -> Any:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Configuration root must be a mapping.")
    merged = deep_merge(copy.deepcopy(DEFAULT_CONFIG), payload)
    root = path.parent
    for dotted in PATH_FIELDS:
        parts = dotted.split(".")
        current: Any = merged
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            resolved = _resolve_path(current, root)
            if dotted == "evaluation.quality_config" and resolved is not None and not resolved.exists():
                repository_root = Path(__file__).resolve().parents[2]
                raw_path = Path(str(current)).expanduser()
                fallbacks = [
                    repository_root / "configs" / raw_path,
                    repository_root / raw_path,
                    Path(
                        str(
                            files("rineya_soup_lab").joinpath(
                                "resources", "official_v45", raw_path.name
                            )
                        )
                    ),
                ]
                resolved = next((candidate.resolve() for candidate in fallbacks if candidate.exists()), resolved)
            _set_dotted(merged, dotted, resolved)
    validate_config(merged)
    return ExperimentConfig(path=path, root=root, raw=merged)


def validate_config(config: Mapping[str, Any]) -> None:
    if str(config.get("schema_version")) != "1.0":
        raise ConfigurationError("Only configuration schema_version '1.0' is supported.")
    models = config.get("models") or {}
    for key in ("anchor_path", "greedy_path"):
        if not models.get(key):
            raise ConfigurationError(f"models.{key} is required.")
    try:
        re.compile(str(models.get("layer_key_regex")))
    except re.error as exc:
        raise ConfigurationError(f"Invalid models.layer_key_regex: {exc}") from exc
    expected_layers = models.get("expected_layers")
    if expected_layers is not None and int(expected_layers) <= 0:
        raise ConfigurationError("models.expected_layers must be positive or null.")
    if str(models.get("merge_accumulation_dtype")) != "float32":
        raise ConfigurationError("models.merge_accumulation_dtype must be float32 in v1.0.")
    if str(models.get("output_dtype")) not in {"preserve_anchor", "float32", "float16", "bfloat16"}:
        raise ConfigurationError("models.output_dtype has an unsupported value.")
    if str(models.get("copy_non_floating_from")) not in {"anchor", "greedy"}:
        raise ConfigurationError("models.copy_non_floating_from must be anchor or greedy.")
    if float(models.get("max_shard_size_gb", 0.0)) <= 0:
        raise ConfigurationError("models.max_shard_size_gb must be positive.")
    if not bool(models.get("lineage_acknowledged", False)):
        raise ConfigurationError(
            "models.lineage_acknowledged must be true after manually confirming that the "
            "anchor and greedy checkpoints share the same initialization and lineage."
        )
    datasets = config.get("datasets") or {}
    if not datasets.get("dev"):
        raise ConfigurationError("datasets.dev is required.")
    if (config.get("sealed_test") or {}).get("enabled") and not datasets.get("test"):
        raise ConfigurationError("datasets.test is required when sealed_test.enabled is true.")
    evaluation = config.get("evaluation") or {}
    backend = str(evaluation.get("backend"))
    allowed = {"bundled_v45", "official_phase_v45", "external", "mock"}
    if backend not in allowed:
        raise ConfigurationError(f"evaluation.backend must be one of {sorted(allowed)}")
    if backend == "official_phase_v45" and not evaluation.get("official_repo_root"):
        raise ConfigurationError("evaluation.official_repo_root is required for official_phase_v45.")
    if backend == "external" and not evaluation.get("external_command"):
        raise ConfigurationError("evaluation.external_command is required for external backend.")
    if backend == "mock" and not evaluation.get("mock_allowed"):
        raise ConfigurationError("Mock evaluation is disabled. Set evaluation.mock_allowed: true only for tests.")
    protocols = evaluation.get("protocols") or {}
    selected = str(evaluation.get("selection_protocol"))
    if selected not in protocols:
        raise ConfigurationError(f"evaluation.selection_protocol {selected!r} is not defined.")
    for name, protocol in protocols.items():
        if protocol.get("enabled", True) is False:
            continue
        if protocol.get("response_control") not in {"eos_only", "one_sentence"}:
            raise ConfigurationError(f"Protocol {name!r} has invalid response_control.")
        if int(protocol.get("max_new_tokens", 0)) <= 0:
            raise ConfigurationError(f"Protocol {name!r} must have max_new_tokens > 0.")
        minimum_boundary = int(protocol.get("min_new_tokens_before_boundary", 0))
        if minimum_boundary < 0 or minimum_boundary > int(protocol.get("max_new_tokens", 0)):
            raise ConfigurationError(
                f"Protocol {name!r} has an invalid min_new_tokens_before_boundary."
            )
        if protocol.get("do_sample"):
            if float(protocol.get("temperature", 0.0)) <= 0.0:
                raise ConfigurationError(f"Protocol {name!r} requires temperature > 0.")
            if not 0.0 < float(protocol.get("top_p", 0.0)) <= 1.0:
                raise ConfigurationError(f"Protocol {name!r} requires top_p in (0, 1].")
    uniform_alpha = float((config.get("experiments") or {}).get("uniform", {}).get("alpha", 0.5))
    if not 0.0 <= uniform_alpha <= 1.0:
        raise ConfigurationError("experiments.uniform.alpha must lie in [0, 1].")
    for alpha in (config.get("experiments") or {}).get("global_sweep", {}).get("alphas", []):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ConfigurationError(f"Global alpha out of range: {alpha}")
    for alpha in (config.get("experiments") or {}).get("layer_influence", {}).get("centered_alpha_grid", []):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ConfigurationError(f"Layer alpha out of range: {alpha}")
    experiments = config.get("experiments") or {}
    if not experiments.get("run_endpoints", True):
        raise ConfigurationError("experiments.run_endpoints is mandatory for controlled comparisons.")
    global_settings = experiments.get("global_sweep") or {}
    if global_settings.get("enabled", True) and not global_settings.get("alphas"):
        raise ConfigurationError("experiments.global_sweep.alphas must not be empty.")
    if float(global_settings.get("refine_step", 0.0)) <= 0.0:
        raise ConfigurationError("experiments.global_sweep.refine_step must be positive.")
    if float(global_settings.get("refine_radius", 0.0)) < 0.0:
        raise ConfigurationError("experiments.global_sweep.refine_radius cannot be negative.")
    influence_settings = experiments.get("layer_influence") or {}
    search_settings = experiments.get("layer_search") or {}
    if influence_settings.get("enabled", True) and not global_settings.get("enabled", True):
        raise ConfigurationError("Layer influence requires experiments.global_sweep.enabled: true.")
    if search_settings.get("enabled", True) and not influence_settings.get("enabled", True):
        raise ConfigurationError("Layer search requires experiments.layer_influence.enabled: true.")
    if search_settings.get("enabled", True) and not search_settings.get("alpha_grid"):
        raise ConfigurationError("experiments.layer_search.alpha_grid must not be empty.")
    if int(search_settings.get("max_candidate_evaluations", 0)) < 0:
        raise ConfigurationError("max_candidate_evaluations cannot be negative.")
    if int(search_settings.get("max_passes", 0)) <= 0:
        raise ConfigurationError("experiments.layer_search.max_passes must be positive.")
    top_layers_only = search_settings.get("top_layers_only")
    if top_layers_only is not None and int(top_layers_only) < 0:
        raise ConfigurationError("experiments.layer_search.top_layers_only cannot be negative.")
    if int(search_settings.get("pairwise_top_k", 0)) < 0:
        raise ConfigurationError("experiments.layer_search.pairwise_top_k cannot be negative.")
    for key in ("smoothing_penalty", "global_shrinkage_penalty", "minimum_reliability"):
        value = float(influence_settings.get(key, 0.0))
        if value < 0.0:
            raise ConfigurationError(f"experiments.layer_influence.{key} cannot be negative.")
    if float(influence_settings.get("minimum_reliability", 0.0)) > 1.0:
        raise ConfigurationError("experiments.layer_influence.minimum_reliability cannot exceed 1.")
    finalists = experiments.get("finalists") or {}
    if int(finalists.get("maximum", 0)) <= 0:
        raise ConfigurationError("experiments.finalists.maximum must be positive.")
    if str(finalists.get("selection_strategy", "robust_sequential")) not in {
        "robust_sequential",
        "objective_max",
    }:
        raise ConfigurationError("Unsupported finalists.selection_strategy.")
    probability = experiments.get("probability_profile") or {}
    if int(probability.get("top_k", 128)) <= 0:
        raise ConfigurationError("experiments.probability_profile.top_k must be positive.")
    if int(probability.get("max_cases", 150)) <= 0:
        raise ConfigurationError("experiments.probability_profile.max_cases must be positive.")
    storage = config.get("storage") or {}
    if float(storage.get("max_disk_gb", 0.0)) <= 0:
        raise ConfigurationError("storage.max_disk_gb must be positive.")
    cache_dir = Path(str(storage.get("candidate_cache_dir", "candidate_cache")))
    if cache_dir.is_absolute() or ".." in cache_dir.parts or str(cache_dir) in {"", "."}:
        raise ConfigurationError(
            "storage.candidate_cache_dir must be a non-empty relative path inside the run directory."
        )
    for alpha in search_settings.get("alpha_grid", []):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ConfigurationError(f"Layer-search alpha out of range: {alpha}")
    objective_metrics = (config.get("objective") or {}).get("metrics") or []
    if not objective_metrics:
        raise ConfigurationError("objective.metrics must not be empty.")
    if sum(max(0.0, float(item.get("weight", 0.0))) for item in objective_metrics) <= 0:
        raise ConfigurationError("objective metric weights must have a positive total.")
    allowed_transforms = {
        "maximize_01",
        "minimize_01",
        "identity",
        "negative",
        "target_range",
        "log_minimize",
    }
    for item in objective_metrics:
        if not item.get("path"):
            raise ConfigurationError("Every objective metric requires a path.")
        if float(item.get("weight", 0.0)) < 0.0:
            raise ConfigurationError("Objective metric weights cannot be negative.")
        transform = str(item.get("transform", "maximize_01"))
        if transform not in allowed_transforms:
            raise ConfigurationError(f"Unsupported objective transform: {transform}")
        if transform == "target_range":
            minimum = float(item.get("minimum", 0.0))
            maximum = float(item.get("maximum", minimum))
            if maximum < minimum:
                raise ConfigurationError("target_range maximum must be >= minimum.")
    tail_fraction = float((config.get("objective") or {}).get("tail_fraction", 0.10))
    if not 0.0 < tail_fraction <= 1.0:
        raise ConfigurationError("objective.tail_fraction must lie in (0, 1].")
    statistics_config = config.get("statistics") or {}
    if int(statistics_config.get("bootstrap_samples", 0)) <= 0:
        raise ConfigurationError("statistics.bootstrap_samples must be positive.")
    confidence = float(statistics_config.get("confidence_level", 0.0))
    if not 0.0 < confidence < 1.0:
        raise ConfigurationError("statistics.confidence_level must lie in (0, 1).")
    accept_probability = float(statistics_config.get("minimum_accept_probability", 0.0))
    if not 0.0 <= accept_probability <= 1.0:
        raise ConfigurationError("statistics.minimum_accept_probability must lie in [0, 1].")
    if float(statistics_config.get("minimum_objective_delta", 0.0)) < 0.0:
        raise ConfigurationError("statistics.minimum_objective_delta cannot be negative.")
    case_utility_config = (config.get("objective") or {}).get("case_utility") or {}
    for key in (
        "quality_weight",
        "completeness_weight",
        "observable_pass_weight",
        "brevity_weight",
    ):
        if float(case_utility_config.get(key, 0.0)) < 0.0:
            raise ConfigurationError(f"objective.case_utility.{key} cannot be negative.")
    brevity_min = float(case_utility_config.get("brevity_target_min", 1.0))
    brevity_max = float(case_utility_config.get("brevity_target_max", 12.0))
    brevity_soft_upper = float(case_utility_config.get("brevity_soft_upper", 24.0))
    if brevity_min < 0.0 or brevity_max < brevity_min or brevity_soft_upper < brevity_max:
        raise ConfigurationError(
            "objective.case_utility brevity bounds must satisfy 0 <= min <= max <= soft_upper."
        )
    if float((config.get("objective") or {}).get("category_max_loss_weight", 0.0)) < 0.0:
        raise ConfigurationError("objective.category_max_loss_weight cannot be negative.")
