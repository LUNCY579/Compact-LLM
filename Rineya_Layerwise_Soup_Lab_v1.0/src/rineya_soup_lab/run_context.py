from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig
from .state import RunState
from .utils import (
    canonical_json,
    environment_manifest,
    require_non_overlapping_path,
    sha256_file,
    stable_id,
    utc_timestamp,
    write_json,
)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "run"


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class RunContext:
    config: ExperimentConfig
    root: Path
    run_id: str
    state: RunState

    @classmethod
    def create(cls, config: ExperimentConfig, *, existing_run: Path | None = None) -> "RunContext":
        resolved = _plain(config.raw)
        resolved_digest = stable_id(resolved, prefix="cfg_", length=64)
        if existing_run is not None:
            root = existing_run.expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Run directory does not exist: {root}")
            run_id = root.name
        else:
            run_root = Path(config.get("project.run_root")).expanduser().resolve()
            configured_name = config.get("project.run_name")
            if configured_name:
                run_id = _safe_name(str(configured_name))
            else:
                # Include microseconds so independent invocations in the same second cannot
                # silently resolve to the same run directory.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                run_id = f"{_safe_name(str(config.get('project.name')))}-{stamp}"
            root = run_root / run_id
            if root.exists() and not root.is_dir():
                raise FileExistsError(f"Run path exists but is not a directory: {root}")
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(
                    f"Refusing to reuse non-empty run directory: {root}. "
                    "Pass --run-dir to resume it, or choose a new project.run_name."
                )
            root.mkdir(parents=True, exist_ok=True)
        protected_inputs = [
            Path(value)
            for value in (
                config.get("models.anchor_path"),
                config.get("models.greedy_path"),
                config.get("models.tokenizer_path"),
                config.get("models.base_model_path"),
                config.get("datasets.dev"),
                config.get("datasets.test"),
            )
            if value
        ]
        require_non_overlapping_path(root, protected_inputs, label="run directory")
        for relative in (
            "state",
            "manifests",
            "sources",
            "candidate_cache",
            "evaluations/dev",
            "evaluations/test",
            "analysis/global",
            "analysis/layers",
            "analysis/probability",
            "analysis/protocols",
            "selection",
            "sealed",
            "reports/plots",
            "logs",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        resolved_path = root / "config.resolved.yaml"
        if resolved_path.exists():
            existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
            if canonical_json(existing) != canonical_json(resolved):
                raise RuntimeError(
                    "The supplied configuration does not match config.resolved.yaml in the run. "
                    "Use a new run directory for changed settings."
                )
        else:
            resolved_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            shutil.copy2(config.path, root / "config.input.yaml")
            write_json(root / "manifests" / "environment.json", environment_manifest())
            write_json(
                root / "manifests" / "run_manifest.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "created_at": utc_timestamp(),
                    "config_sha256": sha256_file(config.path),
                    "resolved_config_digest": resolved_digest,
                    "models": {
                        "anchor_path": str(config.get("models.anchor_path")),
                        "greedy_path": str(config.get("models.greedy_path")),
                        "tokenizer_path": str(config.get("models.tokenizer_path")),
                    },
                    "datasets": {
                        "dev": str(config.get("datasets.dev")),
                        "test": str(config.get("datasets.test")),
                    },
                    "evaluation_backend": config.get("evaluation.backend"),
                    "selection_protocol": config.get("evaluation.selection_protocol"),
                },
            )
        state = RunState.open(
            root / "state" / "run_state.json",
            run_id=run_id,
            config_digest=resolved_digest,
        )
        return cls(config=config, root=root, run_id=run_id, state=state)

    @property
    def lock_path(self) -> Path:
        return self.root / "state" / "RUN.lock"

    def path(self, *parts: str) -> Path:
        """Return a run-owned path and reject configuration-driven path traversal.

        All mutable experiment artifacts must remain underneath the run directory.
        User-owned model and dataset paths are handled separately through the resolved
        configuration and are never passed through this helper.
        """

        root = self.root.resolve()
        candidate = root.joinpath(*(str(part) for part in parts)).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(
                f"Run artifact path escapes the run directory: {candidate} (root: {root})"
            )
        return candidate
