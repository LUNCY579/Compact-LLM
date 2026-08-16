from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation.objective import ObjectiveResult, score_summary
from .evaluation.runner import EvaluationEvidence, run_evaluation, validate_evidence
from .model.checkpoint import checkpoint_fingerprint, checkpoint_fingerprint_digest
from .model.layer_map import MergeRecipe
from .model.merge import merge_checkpoints
from .model.sources import ResolvedSources
from .model.compatibility import TOKENIZER_FILES
from .run_context import RunContext
from .utils import read_json, sha256_file, write_json


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    label: str
    recipe: MergeRecipe
    model_path: Path
    evidence: EvaluationEvidence
    objective: ObjectiveResult

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "recipe": self.recipe.to_dict(),
            "model_path": str(self.model_path),
            "evaluation_dir": str(self.evidence.output_dir),
            "backend": self.evidence.backend,
            "protocol": self.evidence.protocol_name,
            "objective": self.objective.to_dict(),
            "summary": self.evidence.summary,
        }


class CandidateManager:
    def __init__(self, context: RunContext, sources: ResolvedSources) -> None:
        self.context = context
        self.sources = sources
        self.config = context.config
        self._source_fingerprints = self._load_source_fingerprints()

    def _load_source_fingerprints(self) -> dict[str, Any]:
        path = self.context.path("manifests", "compatibility.json")
        if path.exists():
            report = read_json(path)
            return {
                "anchor": report.get("anchor"),
                "greedy": report.get("greedy"),
                "tokenizer": report.get("configured_tokenizer"),
            }
        return {}

    def _verify_sources_unchanged(self) -> None:
        """Fail closed if user-owned checkpoint/tokenizer files changed after preflight."""

        for name in ("anchor", "greedy"):
            fingerprint = self._source_fingerprints.get(name) or {}
            if not fingerprint:
                continue
            root = Path(str(fingerprint.get("path", "")))
            if not root.is_dir():
                raise RuntimeError(f"{name} source disappeared after preflight: {root}")
            for entry in [
                *(fingerprint.get("weight_files") or []),
                *(fingerprint.get("index_files") or []),
            ]:
                file_path = root / str(entry["name"])
                if not file_path.exists():
                    raise RuntimeError(f"{name} source checkpoint file disappeared: {file_path}")
                stat = file_path.stat()
                if int(stat.st_size) != int(entry.get("bytes", -1)):
                    raise RuntimeError(f"{name} source checkpoint file size changed: {file_path}")
                recorded_mtime = entry.get("mtime_ns")
                if recorded_mtime is not None and int(stat.st_mtime_ns) != int(recorded_mtime):
                    if sha256_file(file_path) != str(entry.get("sha256")):
                        raise RuntimeError(f"{name} source checkpoint file changed: {file_path}")
            config_path = root / "config.json"
            expected_config = fingerprint.get("config_sha256")
            if expected_config and (
                not config_path.exists() or sha256_file(config_path) != expected_config
            ):
                raise RuntimeError(f"{name} source config changed after preflight: {config_path}")
        tokenizer = self._source_fingerprints.get("tokenizer") or {}
        tokenizer_root = Path(str(tokenizer.get("path", ""))) if tokenizer else None
        if tokenizer_root is not None:
            for entry in tokenizer.get("files") or []:
                file_path = tokenizer_root / str(entry["name"])
                if not file_path.exists() or sha256_file(file_path) != str(entry.get("sha256")):
                    raise RuntimeError(f"Tokenizer source changed after preflight: {file_path}")

    def _ensure_tokenizer_files(self, destination: Path) -> None:
        """Copy configured tokenizer artifacts without overwriting checkpoint-local files."""

        for name in TOKENIZER_FILES:
            source_file = self.sources.tokenizer / name
            destination_file = destination / name
            if source_file.exists() and not destination_file.exists():
                shutil.copy2(source_file, destination_file)

    def model_path_for(self, recipe: MergeRecipe, *, retain: bool = False) -> Path:
        self._verify_sources_unchanged()
        if recipe.kind == "endpoint":
            return self.sources.anchor if recipe.endpoint == "anchor" else self.sources.greedy
        cache_name = str(self.config.get("storage.candidate_cache_dir", "candidate_cache"))
        destination = self.context.path(cache_name, recipe.id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        recipe_path = destination / "merge_recipe.json"
        if destination.exists() and recipe_path.exists():
            existing = MergeRecipe.from_dict(read_json(recipe_path))
            if existing.identity_dict() != recipe.identity_dict():
                raise RuntimeError(f"Candidate cache recipe mismatch at {destination}")
            manifest_path = destination / "merge_manifest.json"
            source_match = False
            output_match = False
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                source_match = (
                    manifest.get("source_fingerprints") == self._source_fingerprints
                )
                expected_output_digest = manifest.get("output_checkpoint_digest")
                if expected_output_digest and self._checkpoint_files_present(destination):
                    current_output_digest = checkpoint_fingerprint_digest(
                        checkpoint_fingerprint(destination)
                    )
                    output_match = current_output_digest == expected_output_digest
            if self._checkpoint_files_present(destination) and source_match and output_match:
                self._ensure_tokenizer_files(destination)
                return destination
            # The cache is run-owned. A recipe without complete weights indicates an
            # interrupted or manually damaged materialization and is safe to rebuild.
            shutil.rmtree(destination)
        elif destination.exists():
            shutil.rmtree(destination)
        self._check_disk_budget()
        merge_checkpoints(
            self.sources.anchor,
            self.sources.greedy,
            destination,
            recipe,
            layer_key_regex=str(self.config.get("models.layer_key_regex")),
            accumulation_dtype=str(self.config.get("models.merge_accumulation_dtype")),
            output_dtype=str(self.config.get("models.output_dtype")),
            max_shard_size_gb=float(self.config.get("models.max_shard_size_gb")),
            require_non_floating_equal=bool(
                self.config.get("models.require_non_floating_equal", True)
            ),
            copy_non_floating_from=str(
                self.config.get("models.copy_non_floating_from", "anchor")
            ),
            source_fingerprints=self._source_fingerprints,
        )
        # A separate tokenizer_path is supported; copy missing tokenizer artifacts so a
        # retained candidate remains a self-contained Hugging Face directory.
        self._ensure_tokenizer_files(destination)
        return destination

    def evaluate(
        self,
        recipe: MergeRecipe,
        *,
        split: str = "dev",
        protocol_name: str | None = None,
        alias: str | None = None,
        retain: bool = False,
        resume: bool = True,
    ) -> CandidateResult:
        if split not in {"dev", "test"}:
            raise ValueError(f"Unknown evaluation split: {split}")
        protocol = protocol_name or str(self.config.get("evaluation.selection_protocol"))
        dataset_path = Path(self.config.get(f"datasets.{split}"))
        candidate_id = recipe.id
        model_path = self.model_path_for(recipe, retain=retain)
        output_dir = self.context.path("evaluations", split, protocol, candidate_id)
        self.context.state.register_candidate(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "label": recipe.label,
                "recipe": recipe.to_dict(),
                "model_path": str(model_path),
                "materialized": recipe.kind != "endpoint",
                "retained": bool(retain),
                "evaluations": self.context.state.candidate(candidate_id).get("evaluations", {})
                if candidate_id in (self.context.state.payload.get("candidates") or {})
                else {},
            },
        )
        evidence = run_evaluation(
            self.config,
            model_path=model_path,
            tokenizer_path=self.sources.tokenizer,
            dataset_path=dataset_path,
            output_dir=output_dir,
            protocol_name=protocol,
            resume=resume,
            recipe=recipe,
            provenance={
                "candidate_id": candidate_id,
                "source_fingerprints": self._source_fingerprints,
            },
        )
        objective = score_summary(evidence.summary, self.config.get("objective"))
        record = self.context.state.candidate(candidate_id)
        evaluations = dict(record.get("evaluations") or {})
        evaluations[f"{split}:{protocol}"] = {
            "output_dir": str(output_dir),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "objective": objective.to_dict(),
        }
        self.context.state.update_candidate(
            candidate_id,
            evaluations=evaluations,
            retained=bool(record.get("retained") or retain),
        )
        if alias:
            self.context.state.set_alias(alias, candidate_id)
        result = CandidateResult(
            candidate_id=candidate_id,
            label=recipe.label,
            recipe=recipe,
            model_path=model_path,
            evidence=evidence,
            objective=objective,
        )
        write_json(output_dir / "candidate_result.json", result.to_record())
        if recipe.kind != "endpoint" and self._should_delete(recipe, retain=retain):
            shutil.rmtree(model_path, ignore_errors=True)
            self.context.state.update_candidate(candidate_id, model_path=None, materialized=False)
        return result

    def load_result(
        self,
        alias_or_id: str,
        *,
        split: str = "dev",
        protocol_name: str | None = None,
    ) -> CandidateResult:
        candidate_id = self.context.state.resolve_alias(alias_or_id)
        record = self.context.state.candidate(candidate_id)
        recipe = MergeRecipe.from_dict(record["recipe"])
        protocol = protocol_name or str(self.config.get("evaluation.selection_protocol"))
        output_dir = self.context.path("evaluations", split, protocol, candidate_id)
        from .evaluation.external import load_external_evidence

        manifest_path = output_dir / "lab_evaluation_manifest.json"
        integrity_path = output_dir / "lab_evaluation_result.json"
        if not manifest_path.exists() or not integrity_path.exists():
            raise RuntimeError(f"Evaluation evidence is incomplete: {output_dir}")
        integrity = read_json(integrity_path)
        checks = {
            "manifest_sha256": manifest_path,
            "summary_sha256": output_dir / "summary.json",
            "results_sha256": output_dir / "results.jsonl",
        }
        for field, evidence_path in checks.items():
            expected = integrity.get(field)
            if not expected or not evidence_path.exists() or sha256_file(evidence_path) != str(expected):
                raise RuntimeError(
                    f"Evaluation evidence integrity check failed for {evidence_path}"
                )
        summary, rows = load_external_evidence(
            output_dir / "summary.json", output_dir / "results.jsonl"
        )
        dataset_path = Path(self.config.get(f"datasets.{split}"))
        validate_evidence(summary, rows, dataset_path)
        if int(integrity.get("case_count", -1)) != len(rows):
            raise RuntimeError(f"Evaluation integrity case count mismatch: {output_dir}")
        evidence = EvaluationEvidence(
            summary=summary,
            rows=rows,
            output_dir=output_dir,
            backend=str(self.config.get("evaluation.backend")),
            protocol_name=protocol,
        )
        objective = score_summary(summary, self.config.get("objective"))
        model_path_value = record.get("model_path")
        if model_path_value:
            model_path = Path(model_path_value)
        elif recipe.kind == "endpoint":
            model_path = self.sources.anchor if recipe.endpoint == "anchor" else self.sources.greedy
        else:
            model_path = self.context.path(
                str(self.config.get("storage.candidate_cache_dir", "candidate_cache")), candidate_id
            )
        return CandidateResult(
            candidate_id=candidate_id,
            label=str(record.get("label", recipe.label)),
            recipe=recipe,
            model_path=model_path,
            evidence=evidence,
            objective=objective,
        )

    def ensure_materialized(self, alias_or_id: str) -> Path:
        candidate_id = self.context.state.resolve_alias(alias_or_id)
        record = self.context.state.candidate(candidate_id)
        recipe = MergeRecipe.from_dict(record["recipe"])
        model_path = self.model_path_for(recipe, retain=True)
        self.context.state.update_candidate(
            candidate_id, model_path=str(model_path), materialized=recipe.kind != "endpoint", retained=True
        )
        return model_path

    def _should_delete(self, recipe: MergeRecipe, *, retain: bool) -> bool:
        if retain:
            return False
        record = self.context.state.candidate(recipe.id)
        if bool(record.get("retained")):
            return False
        if not bool(self.config.get("storage.delete_ephemeral_after_evaluation", True)):
            return False
        retained_labels = set(self.config.get("storage.retain_candidate_weights", []))
        known_labels = set(record.get("labels") or [record.get("label"), recipe.label])
        return retained_labels.isdisjoint(str(label) for label in known_labels if label)

    @staticmethod
    def _checkpoint_files_present(path: Path) -> bool:
        return any(
            (path / name).exists()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )

    def _check_disk_budget(self) -> None:
        max_gb = float(self.config.get("storage.max_disk_gb", 0.0))
        if max_gb <= 0:
            return
        total = 0
        for file_path in self.context.root.rglob("*"):
            if file_path.is_file():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    pass
        estimated_candidate = 0
        compatibility = self.context.path("manifests", "compatibility.json")
        if compatibility.exists():
            estimated_candidate = int(read_json(compatibility).get("anchor_weight_bytes_unsharded", 0))
        limit = max_gb * 1024**3
        if total + estimated_candidate > limit:
            raise RuntimeError(
                f"Materializing another candidate would use approximately "
                f"{(total + estimated_candidate) / 1024**3:.2f} GiB, exceeding "
                f"storage.max_disk_gb={max_gb}."
            )
