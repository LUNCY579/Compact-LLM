from __future__ import annotations

import argparse
import compileall
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from export_schemas import export

REQUIRED = {
    "README.md",
    "ARCHITECTURE_SPECIFICATION.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "RELEASING.md",
    "LICENSE",
    "NOTICE",
    "pyproject.toml",
    "configs/default.yaml",
    "migrations/0001_initial.sql",
    "migrations/0002_idempotency_request_hash.sql",
    "src/rineya_harness/migrations/0001_initial.sql",
    "src/rineya_harness/migrations/0002_idempotency_request_hash.sql",
    "src/rineya_harness/context/assembler.py",
    "src/rineya_harness/memory/service.py",
    "src/rineya_harness/orchestration/workflow.py",
    "src/rineya_harness/security/network.py",
    "src/rineya_harness/state/invariants.py",
    "src/rineya_harness/tools/memory.py",
    "tests/integration/test_workflow.py",
    "docs/evaluation/RSF_EVALUATION_PROTOCOL.md",
    "docs/operations/MIGRATIONS.md",
    "docs/operations/DATA_RETENTION_AND_PRIVACY.md",
    "docs/operations/OBSERVABILITY.md",
    "docs/security/THREAT_MODEL.md",
}
EXPECTED_TOOLS = {
    "get_state",
    "search_web",
    "inspect_evidence",
    "update_memory",
    "delete_memory",
    "finish",
}


def _yaml_files(root: Path) -> list[Path]:
    candidates = [root / "configs", root / "docker", root / ".github"]
    return sorted(
        path
        for directory in candidates
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )


def _parse_test_summary(stdout: str) -> dict[str, Any]:
    """Return stable, release-manifest-safe test facts.

    Wall-clock duration and raw stdout are deliberately omitted because they make two
    otherwise identical release packages differ byte-for-byte. Full diagnostics are
    still surfaced when the test command fails.
    """

    passed = None
    coverage = None
    passed_match = re.search(r"(\d+) passed", stdout)
    if passed_match:
        passed = int(passed_match.group(1))

    precise_coverage = re.search(r"Total coverage:\s*([0-9]+(?:\.[0-9]+)?)%", stdout)
    if precise_coverage:
        coverage = float(precise_coverage.group(1))
    else:
        for line in stdout.splitlines():
            if line.strip().startswith("TOTAL"):
                percentage_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", line)
                if percentage_match:
                    coverage = float(percentage_match.group(1))

    return {
        "passed": passed,
        "coverage_percent": coverage,
    }


def run(root: Path, *, with_tests: bool) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    missing = sorted(path for path in REQUIRED if not (root / path).is_file())
    if missing:
        errors.append(f"missing required files: {missing}")
    checks["required_files"] = {"expected": len(REQUIRED), "missing": missing}

    project_version = "unknown"
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project_version = str(project["project"]["version"])
        version_namespace: dict[str, object] = {}
        exec(
            (root / "src" / "rineya_harness" / "_version.py").read_text(encoding="utf-8"),
            version_namespace,
        )
        runtime_version = str(version_namespace.get("__version__", ""))
        if project_version != runtime_version:
            errors.append(
                f"version mismatch: pyproject={project_version!r}, runtime={runtime_version!r}"
            )
        if f"v{project_version}" not in (root / "README.md").read_text(encoding="utf-8"):
            errors.append("README title does not contain the project version")
        checks["version_consistency"] = {
            "project": project_version,
            "runtime": runtime_version,
        }
    except Exception as exc:
        errors.append(f"invalid project/version metadata: {exc}")

    root_migrations = sorted((root / "migrations").glob("*.sql"))
    packaged_directory = root / "src" / "rineya_harness" / "migrations"
    migration_errors: list[str] = []
    for root_migration in root_migrations:
        packaged_migration = packaged_directory / root_migration.name
        if not packaged_migration.is_file():
            migration_errors.append(f"packaged migration missing: {root_migration.name}")
        elif root_migration.read_bytes() != packaged_migration.read_bytes():
            migration_errors.append(f"packaged migration differs: {root_migration.name}")
    unexpected_packaged = {path.name for path in packaged_directory.glob("*.sql")} - {
        path.name for path in root_migrations
    }
    if unexpected_packaged:
        migration_errors.append(f"unexpected packaged migrations: {sorted(unexpected_packaged)}")
    errors.extend(migration_errors)
    checks["migrations"] = {
        "repository_count": len(root_migrations),
        "errors": migration_errors,
    }

    yaml_errors: list[str] = []
    for path in _yaml_files(root):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if payload is not None and not isinstance(payload, (dict, list)):
                yaml_errors.append(f"YAML root is not a mapping/list: {path.relative_to(root)}")
        except Exception as exc:
            yaml_errors.append(f"invalid YAML {path.relative_to(root)}: {exc}")
    errors.extend(yaml_errors)
    checks["yaml"] = {"files": len(_yaml_files(root)), "errors": yaml_errors}

    json_errors: list[str] = []
    json_count = 0
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        if relative.startswith("artifacts/release/"):
            # Release reports are outputs of this validator/packager and must not make a
            # subsequent identical package depend on whether a prior package was built.
            continue
        json_count += 1
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            json_errors.append(f"invalid JSON {path.relative_to(root)}: {exc}")
    errors.extend(json_errors)
    checks["json"] = {"files": json_count, "errors": json_errors}

    jsonl_errors: list[str] = []
    jsonl_count = 0
    for path in sorted(root.rglob("*.jsonl")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("artifacts/runtime/") or relative.startswith("artifacts/release/"):
            continue
        jsonl_count += 1
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:
                jsonl_errors.append(
                    f"invalid JSONL {path.relative_to(root)}:{line_number}: {exc}"
                )
    errors.extend(jsonl_errors)
    checks["jsonl"] = {"files": jsonl_count, "errors": jsonl_errors}

    compile_ok = compileall.compile_dir(root / "src", quiet=1)
    if not compile_ok:
        errors.append("compileall failed")
    checks["compileall"] = "pass" if compile_ok else "fail"

    schemas_current = export(root / "schemas" / "generated", check=True)
    if not schemas_current:
        errors.append("generated schemas are stale")
    checks["schema_drift"] = "pass" if schemas_current else "fail"

    with tempfile.TemporaryDirectory(prefix="rineya-validation-") as temporary:
        from rineya_harness.config import DatabaseConfig, HarnessConfig, TraceConfig
        from rineya_harness.factory import create_container

        temporary_path = Path(temporary)
        container = create_container(
            HarnessConfig(
                database=DatabaseConfig(path=temporary_path / "validation.sqlite3"),
                trace=TraceConfig(directory=temporary_path / "traces"),
            )
        )
        tool_names = {definition["name"] for definition in container.tools.definitions()}
        if tool_names != EXPECTED_TOOLS:
            errors.append(
                f"tool registry mismatch: expected {sorted(EXPECTED_TOOLS)}, got {sorted(tool_names)}"
            )
        checks["tool_registry"] = sorted(tool_names)

        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        demo_results: dict[str, str] = {}
        for demo in ("restaurant", "hanoi", "research"):
            completed = subprocess.run(
                [sys.executable, "-m", "rineya_harness", "demo", demo, "--workdir", temporary],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if completed.returncode != 0:
                errors.append(f"demo {demo} failed: {completed.stderr[-1000:]}")
                demo_results[demo] = "fail"
            else:
                demo_results[demo] = "pass"
        checks["offline_demos"] = demo_results

    tests: dict[str, Any] = {"requested": with_tests, "status": "not_run"}
    if with_tests:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=rineya_harness",
                "--cov-report=term-missing",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        tests = {
            "requested": True,
            "status": "pass" if completed.returncode == 0 else "fail",
            **_parse_test_summary(completed.stdout),
        }
        if completed.returncode != 0:
            errors.append(
                "test suite failed:\n"
                f"{completed.stdout[-5000:]}\n{completed.stderr[-2000:]}"
            )
    checks["tests"] = tests

    # These analyzers are deliberately CI-owned optional development dependencies. The validator
    # reports their policy rather than pretending they ran in an offline release environment.
    checks["static_analysis"] = {
        "ruff": "configured_in_ci_not_executed_by_this_script",
        "mypy": "configured_in_ci_not_executed_by_this_script",
    }

    if not with_tests:
        warnings.append("test suite was not requested")

    return {
        "schema_version": "1.0",
        "project": "rineya-context-harness",
        "version": project_version,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "checks": checks,
        "limitations": [
            "No live external search provider or public Rineya checkpoint is exercised by offline validation.",
            "No browser-grade JavaScript rendering or runtime-specific KV-cache mutation is claimed.",
            "Ruff, mypy, container build, and wheel build remain CI/release-environment checks.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/release/validation_report.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report = run(root, with_tests=args.with_tests)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
