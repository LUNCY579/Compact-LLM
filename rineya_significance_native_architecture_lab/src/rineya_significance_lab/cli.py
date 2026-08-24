"""Command-line entry point for the standalone significance lab."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType

from . import __version__
from .config import ConfigError, load_config

LAB_ROOT = Path(__file__).resolve().parents[2]


def _overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ConfigError(f"override must be KEY=VALUE, got {value!r}")
        result[key] = raw
    return result


def _load_runner(filename: str) -> ModuleType:
    path = LAB_ROOT / "scripts" / filename
    if not path.is_file():
        raise RuntimeError(f"runner is not implemented: {path}")
    spec = importlib.util.spec_from_file_location(f"_rineya_lab_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_config(args: argparse.Namespace) -> int:
    overrides = _overrides(args.set)
    if args.model_path is not None:
        overrides["paths.model"] = str(args.model_path.expanduser().resolve())
        overrides["paths.tokenizer"] = str(args.model_path.expanduser().resolve())
    if args.output is not None:
        overrides["paths.output_root"] = str(args.output.expanduser().resolve())
    config = load_config(args.config, overrides=overrides)
    tier = str(config.get("experiment", {}).get("compute_tier", "")).upper()
    mode = str(config.get("protocol", {}).get("execution_mode", ""))
    experiment_name = str(config.get("experiment", {}).get("name", ""))
    if (
        experiment_name == "tier_a_state_and_policy_validation"
        and tier == "A"
        and mode == "open_loop_trace_simulation"
    ):
        runner_name = "run_tier_a.py"
    elif (
        experiment_name.endswith("_cornerstone1_cpu_smoke")
        and tier == "B"
        and mode == "closed_loop_model_execution"
    ):
        runner_name = "run_cpu_smoke.py"
    else:
        raise RuntimeError(
            f"no runner is implemented for experiment {experiment_name!r}; "
            "its config remains NOT_RUN by design"
        )
    runner = _load_runner(runner_name)
    run = getattr(runner, "run", None)
    if not callable(run):
        raise RuntimeError(f"{runner_name} does not expose run(config_path, output_root)")

    # Pass the exact resolved configuration to the runner. JSON is also valid
    # YAML, so runners may preserve it under their conventional resolved YAML
    # filename without losing information.
    with tempfile.TemporaryDirectory(prefix="rineya-significance-config-") as temporary:
        resolved_path = Path(temporary) / "resolved_config.json"
        resolved_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outcome = run(resolved_path, args.output)
    # The runner received an ephemeral resolved file, but provenance must point
    # back to the durable declaring configuration. Keep the resolved artifact
    # hash/path and replace only the temporary source path.
    manifest = outcome.get("manifest") if isinstance(outcome, dict) else None
    output_value = outcome.get("output") if isinstance(outcome, dict) else None
    if isinstance(manifest, dict) and isinstance(manifest.get("config"), dict) and output_value:
        manifest["config"]["source_path"] = str(Path(args.config).expanduser().resolve())
        manifest_path = Path(str(output_value)) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(outcome, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.config, overrides=_overrides(args.set))
    print(
        json.dumps(
            {
                "status": "PASS",
                "config": str(Path(args.config).resolve()),
                "config_sha256": config["_meta"]["config_sha256"],
                "execution_mode": config["protocol"]["execution_mode"],
            },
            indent=2,
        )
    )
    return 0


def _validate_result(args: argparse.Namespace) -> int:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("jsonschema is required to validate result artifacts") from exc
    value_path = Path(args.path).expanduser().resolve()
    schema_path = LAB_ROOT / "schemas" / args.schema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    if value_path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in value_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not values:
            raise ValueError(f"JSONL artifact contains no records: {value_path}")
    else:
        values = [json.loads(value_path.read_text(encoding="utf-8"))]
    for value in values:
        validator.validate(value)
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(value_path),
                "schema": str(schema_path),
                "records": len(values),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rineya-significance-lab",
        description=(
            "CPU-first Cornerstone-1 architecture experiments for published "
            "Rineya Qwen3-0.6B cores, with Halo primary."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Resolve and validate one experiment config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    validate.set_defaults(handler=_validate_config)

    run = subparsers.add_parser("run", help="Run a Tier-A or CPU-smoke configuration")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--model-path", type=Path)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.set_defaults(handler=_run_config)

    result = subparsers.add_parser("validate-result", help="Validate JSON against a lab schema")
    result.add_argument("path", type=Path)
    result.add_argument(
        "--schema",
        choices=("manifest.schema.json", "case.schema.json", "trace.schema.json"),
        default="manifest.schema.json",
    )
    result.set_defaults(handler=_validate_result)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
