from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..run_context import RunContext
from ..utils import canonical_json, read_json, sha256_directory, write_json
from .checkpoint import checkpoint_fingerprint, checkpoint_fingerprint_digest
from .materialize import is_peft_adapter, materialize_peft_adapter


@dataclass(frozen=True)
class ResolvedSources:
    anchor: Path
    greedy: Path
    tokenizer: Path
    report: dict[str, Any]


def resolve_model_sources(context: RunContext) -> ResolvedSources:
    """Resolve full checkpoints while keeping user-owned sources read-only."""

    config = context.config
    base = config.get("models.base_model_path")
    anchor_original = Path(config.get("models.anchor_path")).expanduser().resolve()
    greedy_original = Path(config.get("models.greedy_path")).expanduser().resolve()
    tokenizer_original = Path(
        config.get("models.tokenizer_path") or base or anchor_original
    ).expanduser().resolve()
    if not anchor_original.exists():
        raise FileNotFoundError(f"Anchor source does not exist: {anchor_original}")
    if not greedy_original.exists():
        raise FileNotFoundError(f"Greedy source does not exist: {greedy_original}")
    if not tokenizer_original.exists():
        raise FileNotFoundError(f"Tokenizer source does not exist: {tokenizer_original}")

    materialized: dict[str, str] = {}
    anchor = anchor_original
    greedy = greedy_original
    base_fingerprint: dict[str, Any] | None = None
    for name, source in (("anchor", anchor_original), ("greedy", greedy_original)):
        if not is_peft_adapter(source):
            continue
        if not base:
            raise ValueError(
                f"{name} is a PEFT adapter, but models.base_model_path is not configured."
            )
        base_path = Path(base).expanduser().resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Configured PEFT base model does not exist: {base_path}")
        if base_fingerprint is None:
            base_fingerprint = checkpoint_fingerprint(base_path)
        source_identity = {
            "schema_version": "1.0",
            "adapter_path": str(source),
            "adapter_directory_sha256": sha256_directory(source),
            "base_model_path": str(base_path),
            "base_checkpoint_digest": checkpoint_fingerprint_digest(base_fingerprint),
            "trust_remote_code": bool(config.get("models.trust_remote_code", False)),
        }
        destination = context.path("sources", f"{name}_materialized")
        provenance_path = destination / "source_provenance.json"
        reusable = False
        if destination.exists() and provenance_path.exists():
            existing = read_json(provenance_path)
            existing_identity = {key: existing.get(key) for key in source_identity}
            reusable = canonical_json(existing_identity) == canonical_json(source_identity)
            reusable = reusable and (
                (destination / "model.safetensors").exists()
                or (destination / "model.safetensors.index.json").exists()
                or (destination / "pytorch_model.bin").exists()
                or (destination / "pytorch_model.bin.index.json").exists()
            )
            expected_output_digest = existing.get("materialized_checkpoint_digest")
            if reusable and expected_output_digest:
                current_output_digest = checkpoint_fingerprint_digest(
                    checkpoint_fingerprint(destination)
                )
                reusable = current_output_digest == expected_output_digest
            else:
                reusable = False
        if destination.exists() and not reusable:
            import shutil

            shutil.rmtree(destination)
        if not reusable:
            materialize_peft_adapter(
                source,
                base_path,
                destination,
                trust_remote_code=bool(config.get("models.trust_remote_code", False)),
            )
            materialized_fingerprint = checkpoint_fingerprint(destination)
            write_json(
                provenance_path,
                {
                    **source_identity,
                    "materialized_checkpoint_fingerprint": materialized_fingerprint,
                    "materialized_checkpoint_digest": checkpoint_fingerprint_digest(
                        materialized_fingerprint
                    ),
                },
            )
        materialized[name] = str(destination)
        if name == "anchor":
            anchor = destination
        else:
            greedy = destination

    report = {
        "schema_version": "1.0",
        "source_checkpoints_are_read_only": True,
        "anchor_original": str(anchor_original),
        "greedy_original": str(greedy_original),
        "tokenizer_original": str(tokenizer_original),
        "anchor_resolved": str(anchor),
        "greedy_resolved": str(greedy),
        "tokenizer_resolved": str(tokenizer_original),
        "materialized_adapters": materialized,
    }
    write_json(context.path("manifests", "resolved_sources.json"), report)
    return ResolvedSources(anchor=anchor, greedy=greedy, tokenizer=tokenizer_original, report=report)
