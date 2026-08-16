from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .analysis.probability_profile import run_probability_profile
from .candidates import CandidateManager
from .experiments.endpoints import run_endpoints
from .experiments.global_soup import run_global_sweep, run_uniform
from .experiments.layerwise import run_layer_influence, run_layer_search
from .experiments.preflight import run_preflight
from .experiments.protocols import run_secondary_protocols
from .model.sources import ResolvedSources, resolve_model_sources
from .reporting import write_research_report
from .run_context import RunContext
from .selection import freeze_selection, run_sealed_test
from .state import RunLock


STAGES = (
    "preflight",
    "endpoints",
    "uniform",
    "global",
    "probability",
    "layer_influence",
    "layer_search",
    "freeze",
    "secondary_protocols",
    "sealed",
    "report",
)


class ExperimentPipeline:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.sources: ResolvedSources | None = None
        self.manager: CandidateManager | None = None

    def _ensure_sources(self) -> ResolvedSources:
        if self.sources is None:
            self.sources = resolve_model_sources(self.context)
        return self.sources

    def _ensure_manager(self) -> CandidateManager:
        if self.manager is None:
            self.manager = CandidateManager(self.context, self._ensure_sources())
        return self.manager

    def _execute(self, name: str, action: Callable[[], Any], *, force: bool = False) -> Any:
        if self.context.state.stage_status(name) == "complete" and not force:
            return None
        self.context.state.begin_stage(name)
        try:
            result = action()
            outputs: dict[str, Any] = {}
            if result is not None:
                if hasattr(result, "candidate_id"):
                    outputs["candidate_id"] = result.candidate_id
                elif isinstance(result, dict):
                    outputs["keys"] = sorted(result)
                elif isinstance(result, Path):
                    outputs["path"] = str(result)
            self.context.state.complete_stage(name, **outputs)
            return result
        except BaseException as exc:
            self.context.state.fail_stage(name, exc)
            raise


    def _require_dependencies(self, name: str) -> None:
        dependencies: dict[str, tuple[str, ...]] = {
            "endpoints": ("preflight",),
            "uniform": ("endpoints",),
            "global": ("endpoints",),
            "probability": ("endpoints",),
            "layer_influence": ("global",),
            "layer_search": ("layer_influence",),
            "secondary_protocols": ("freeze",),
            "sealed": ("freeze",),
        }
        if name == "freeze":
            required = ["endpoints"]
            if self.context.config.get("experiments.uniform.enabled", True):
                required.append("uniform")
            if self.context.config.get("experiments.global_sweep.enabled", True):
                required.append("global")
            if self.context.config.get("experiments.layer_influence.enabled", True):
                required.append("layer_influence")
            if self.context.config.get("experiments.layer_search.enabled", True):
                required.append("layer_search")
            dependencies["freeze"] = tuple(required)
        missing = [
            stage
            for stage in dependencies.get(name, ())
            if self.context.state.stage_status(stage) != "complete"
        ]
        if missing:
            raise RuntimeError(
                f"Stage {name!r} requires completed stages {missing}. Run the pipeline in order."
            )


    def _prepare_force(self, name: str) -> None:
        if name == "report":
            return
        sealed_ledger = self.context.path("sealed", "access_ledger.json")
        if sealed_ledger.exists():
            from .utils import read_json

            ledger = read_json(sealed_ledger)
            if ledger.get("accesses"):
                raise RuntimeError(
                    "The sealed split has already been accessed in this run. Upstream stages cannot "
                    "be forced; create a new run to avoid test-guided tuning."
                )
        start = STAGES.index(name)
        for stage in STAGES[start:]:
            if stage == "report":
                continue
            existing = (self.context.state.payload.get("stages") or {}).get(stage)
            if existing:
                existing["status"] = "invalidated"
                existing["invalidated_by"] = name
        alias_by_stage = {
            "uniform": ("uniform",),
            "global": ("global_best",),
            "layer_influence": ("layerwise_seed",),
            "layer_search": ("layerwise_best",),
            "freeze": ("selected",),
            "secondary_protocols": (),
            "sealed": ("selected_tested",),
        }
        aliases = self.context.state.payload.get("aliases") or {}
        for stage in STAGES[start:]:
            for alias in alias_by_stage.get(stage, ()):
                aliases.pop(alias, None)
        if start <= STAGES.index("freeze"):
            self.context.state.payload["selection"] = None
            self.context.path("selection", "selection_freeze.json").unlink(missing_ok=True)
        if start <= STAGES.index("sealed"):
            self.context.path("sealed", "sealed_test_result.json").unlink(missing_ok=True)
        self.context.state.save()

    def run_stage(self, name: str, *, force: bool = False) -> Any:
        if name not in STAGES:
            raise ValueError(f"Unknown stage {name!r}; choose from {STAGES}")
        if force:
            self._prepare_force(name)
        if name != "preflight":
            self._require_dependencies(name)
        if name == "preflight":
            return self._execute(name, lambda: self._set_sources(run_preflight(self.context)), force=force)
        if name == "endpoints":
            return self._execute(name, lambda: run_endpoints(self.context, self._ensure_manager()), force=force)
        if name == "uniform":
            return self._execute(name, lambda: run_uniform(self.context, self._ensure_manager()), force=force)
        if name == "global":
            return self._execute(name, lambda: run_global_sweep(self.context, self._ensure_manager()), force=force)
        if name == "probability":
            return self._execute(name, self._run_probability, force=force)
        if name == "layer_influence":
            return self._execute(
                name, lambda: run_layer_influence(self.context, self._ensure_manager()), force=force
            )
        if name == "layer_search":
            return self._execute(
                name, lambda: run_layer_search(self.context, self._ensure_manager()), force=force
            )
        if name == "freeze":
            return self._execute(
                name, lambda: freeze_selection(self.context, self._ensure_manager()), force=force
            )
        if name == "secondary_protocols":
            return self._execute(
                name,
                lambda: run_secondary_protocols(self.context, self._ensure_manager()),
                force=force,
            )
        if name == "sealed":
            return self._execute(
                name, lambda: run_sealed_test(self.context, self._ensure_manager()), force=force
            )
        if name == "report":
            return self._execute(name, lambda: write_research_report(self.context), force=True)
        raise AssertionError(name)

    def _set_sources(self, sources: ResolvedSources) -> ResolvedSources:
        self.sources = sources
        self.manager = CandidateManager(self.context, sources)
        return sources

    def _run_probability(self) -> dict[str, Any] | None:
        settings = self.context.config.get("experiments.probability_profile")
        if not settings.get("enabled", False):
            return None
        manager = self._ensure_manager()
        anchor = manager.load_result("anchor")
        sources = self._ensure_sources()
        return run_probability_profile(
            self.context,
            anchor_path=sources.anchor,
            greedy_path=sources.greedy,
            tokenizer_path=sources.tokenizer,
            anchor_rows=anchor.evidence.rows,
        )

    def run(
        self,
        *,
        through: str | None = None,
        skip_sealed: bool = False,
        force_stage: str | None = None,
    ) -> RunContext:
        if through is not None and through not in STAGES:
            raise ValueError(f"Unknown --through stage: {through}")
        with RunLock(self.context.lock_path):
            for stage in STAGES:
                if skip_sealed and stage == "sealed":
                    continue
                self.run_stage(stage, force=(stage == force_stage))
                if stage == through:
                    break
        return self.context
