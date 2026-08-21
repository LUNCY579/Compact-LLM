from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from importlib.resources import as_file, files
from pathlib import Path

from .config import load_config
from .model.compatibility import (
    TOKENIZER_FILES,
    compare_checkpoints,
    require_compatible,
)
from .model.checkpoint import checkpoint_fingerprint, checkpoint_fingerprint_digest
from .model.layer_map import MergeRecipe
from .model.materialize import is_peft_adapter, materialize_peft_adapter
from .model.merge import merge_checkpoints
from .pipeline import STAGES, ExperimentPipeline
from .run_context import RunContext
from .utils import read_json, require_non_overlapping_path, sha256_directory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rineya-soup",
        description="Standalone Rineya anchor↔greedy global and layer-wise model-soup laboratory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="Copy the fully annotated example configuration.")
    init.add_argument("--output", type=Path, default=Path("experiment.yaml"))
    init.add_argument("--overwrite", action="store_true")

    run = sub.add_parser("run", help="Run the complete resumable pipeline.")
    _add_run_arguments(run)
    run.add_argument("--through", choices=STAGES, default=None)
    run.add_argument("--skip-sealed", action="store_true")
    run.add_argument("--force-stage", choices=STAGES, default=None)

    stage = sub.add_parser("stage", help="Run one named stage.")
    _add_run_arguments(stage)
    stage.add_argument("name", choices=STAGES)
    stage.add_argument("--force", action="store_true")

    report = sub.add_parser("report", help="Regenerate plots and the research report.")
    _add_run_arguments(report)

    status = sub.add_parser("status", help="Print durable run state.")
    status.add_argument("--run-dir", type=Path, required=True)

    merge = sub.add_parser("merge", help="Materialize one explicit coefficient recipe.")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--recipe", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Resume an existing run. Required for separate stage invocations when run_name is null.",
    )


def _context(args: argparse.Namespace) -> RunContext:
    config = load_config(args.config)
    return RunContext.create(config, existing_run=args.run_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-config":
            source = REPOSITORY_ROOT / "configs" / "experiment.example.yaml"
            if args.output.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copy2(source, args.output)
            else:  # Installed wheel rather than editable/source-tree execution.
                resource = files("rineya_soup_lab").joinpath(
                    "resources", "experiment.example.yaml"
                )
                with as_file(resource) as packaged_source:
                    shutil.copy2(packaged_source, args.output)
            print(args.output.resolve())
            return 0
        if args.command == "status":
            path = args.run_dir / "state" / "run_state.json"
            print(json.dumps(read_json(path), indent=2, ensure_ascii=False))
            return 0
        if args.command == "merge":
            config = load_config(args.config)
            recipe = MergeRecipe.from_dict(read_json(args.recipe))
            anchor_original = Path(config.get("models.anchor_path")).expanduser().resolve()
            greedy_original = Path(config.get("models.greedy_path")).expanduser().resolve()
            base_value = config.get("models.base_model_path")
            base_path = Path(base_value).expanduser().resolve() if base_value else None
            tokenizer_path = Path(
                config.get("models.tokenizer_path")
                or base_path
                or anchor_original
            ).expanduser().resolve()
            protected_inputs = [anchor_original, greedy_original, tokenizer_path]
            if base_path is not None:
                protected_inputs.append(base_path)
            output_path = require_non_overlapping_path(
                args.output, protected_inputs, label="manual merge destination"
            )
            require_non_overlapping_path(
                output_path.with_name(f".{output_path.name}.building"),
                protected_inputs,
                label="manual merge temporary directory",
            )

            with tempfile.TemporaryDirectory(prefix="rineya-manual-merge-") as temp_name:
                temporary_root = Path(temp_name)

                def resolve_endpoint(name: str, original: Path) -> tuple[Path, dict[str, object]]:
                    if not is_peft_adapter(original):
                        return original, {
                            "kind": "full_checkpoint",
                            "original_path": str(original),
                        }
                    if base_path is None:
                        raise ValueError(
                            f"{name} is a PEFT adapter, but models.base_model_path is not configured."
                        )
                    destination = temporary_root / f"{name}_materialized"
                    materialize_peft_adapter(
                        original,
                        base_path,
                        destination,
                        trust_remote_code=bool(
                            config.get("models.trust_remote_code", False)
                        ),
                    )
                    base_fingerprint = checkpoint_fingerprint(base_path)
                    return destination, {
                        "kind": "peft_adapter",
                        "original_path": str(original),
                        "adapter_directory_sha256": sha256_directory(original),
                        "base_model_path": str(base_path),
                        "base_checkpoint_digest": checkpoint_fingerprint_digest(
                            base_fingerprint
                        ),
                    }

                anchor_path, anchor_provenance = resolve_endpoint(
                    "anchor", anchor_original
                )
                greedy_path, greedy_provenance = resolve_endpoint(
                    "greedy", greedy_original
                )
                compatibility = compare_checkpoints(
                    anchor_path,
                    greedy_path,
                    layer_key_regex=str(config.get("models.layer_key_regex")),
                    expected_layers=config.get("models.expected_layers"),
                    allow_quantized_sources=bool(
                        config.get("models.allow_quantized_sources", False)
                    ),
                    require_non_floating_equal=bool(
                        config.get("models.require_non_floating_equal", True)
                    ),
                    lineage_acknowledged=bool(
                        config.get("models.lineage_acknowledged", False)
                    ),
                    tokenizer_path=tokenizer_path,
                    allow_hf_serialization_drift=bool(
                        config.get("models.allow_hf_serialization_drift", False)
                    ),
                )
                require_compatible(compatibility)
                manifest = merge_checkpoints(
                    anchor_path,
                    greedy_path,
                    output_path,
                    recipe,
                    layer_key_regex=str(config.get("models.layer_key_regex")),
                    accumulation_dtype=str(
                        config.get("models.merge_accumulation_dtype")
                    ),
                    output_dtype=str(config.get("models.output_dtype")),
                    max_shard_size_gb=float(config.get("models.max_shard_size_gb")),
                    require_non_floating_equal=bool(
                        config.get("models.require_non_floating_equal", True)
                    ),
                    copy_non_floating_from=str(
                        config.get("models.copy_non_floating_from", "anchor")
                    ),
                    overwrite=args.overwrite,
                    allow_tied_lm_head_alias=bool(
                        config.get("models.allow_hf_serialization_drift", False)
                    ),
                    source_fingerprints={
                        "anchor": {
                            **anchor_provenance,
                            "resolved_checkpoint": compatibility.get("anchor"),
                        },
                        "greedy": {
                            **greedy_provenance,
                            "resolved_checkpoint": compatibility.get("greedy"),
                        },
                        "tokenizer": compatibility.get("configured_tokenizer"),
                    },
                )
            for name in TOKENIZER_FILES:
                source_file = tokenizer_path / name
                destination_file = output_path / name
                if source_file.exists() and not destination_file.exists():
                    shutil.copy2(source_file, destination_file)
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return 0

        context = _context(args)
        pipeline = ExperimentPipeline(context)
        if args.command == "run":
            pipeline.run(
                through=args.through,
                skip_sealed=args.skip_sealed,
                force_stage=args.force_stage,
            )
        elif args.command == "stage":
            from .state import RunLock

            with RunLock(context.lock_path):
                pipeline.run_stage(args.name, force=args.force)
        elif args.command == "report":
            from .state import RunLock

            with RunLock(context.lock_path):
                pipeline.run_stage("report", force=True)
        print(f"Run directory: {context.root}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
