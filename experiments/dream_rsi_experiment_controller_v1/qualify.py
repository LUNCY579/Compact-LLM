"""Reproducible local CPU qualification, with exclusive receipts and raw test output.

Run from repository root:
  python -m experiments.dream_rsi_experiment_controller_v1.qualify --out <NEW_DIR>
No repository-wide or native model test suite is launched.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .fixtures import create_bundle
from .replay import Plan
from .schema import bytes_hash, require, write_new
from .study import engine_hash, evaluate_frozen, fit


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and not any(p in ("__pycache__", ".pytest_cache", "evidence", "runs") for p in relative.parts)
                and path.name != "MANIFEST.json"):
            result[str(relative).replace("\\", "/")] = bytes_hash(path.read_bytes())
    return result


def qualify(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent
    repo_root = package.parents[1]
    before = source_hashes()
    command = [sys.executable, "-m", "pytest", "-q", str(package / "tests"),
               "--junitxml=" + str(output / "junit.xml")]
    test = subprocess.run(command, cwd=repo_root, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False, timeout=180)
    with (output / "pytest.txt").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(test.stdout)
    write_new(output / "test_command.json", {"argv": command, "returncode": test.returncode,
                                             "scope": "isolated package only"})
    require(test.returncode == 0, "qualification tests failed; raw output preserved")
    suites = ET.parse(output / "junit.xml").getroot().findall(".//testsuite")
    counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0, "invalid JUnit qualification")
    compiled = subprocess.run([sys.executable, "-m", "compileall", "-q", str(package)],
                              cwd=repo_root, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False, timeout=60)
    write_new(output / "compile.json", {"returncode": compiled.returncode, "output": compiled.stdout})
    require(compiled.returncode == 0, "compile qualification failed")
    catalog = create_bundle(output / "synthetic_worlds")
    fit(catalog, Plan(), output / "synthetic_run")
    report = evaluate_frozen(catalog, output / "synthetic_run")
    require(before == source_hashes(), "source changed during qualification")
    artifacts = {str(p.relative_to(output)).replace("\\", "/"): bytes_hash(p.read_bytes())
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {
        "schema": "rineya-dream-qualification-v1", "status": "PASS_SYNTHETIC_MECHANICS_ONLY",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "tests": counts,
        "engine_hash": engine_hash(), "source_sha256": before, "artifact_sha256": artifacts,
        "synthetic_episodes": report["episodes"], "candidate_count": report["candidate_count"],
        "admitted_historical_worlds": 0, "historical_efficacy": "NOT_RUN",
        "full_repository_suite": "NOT_RUN", "remote_ci": "NOT_RUN",
        "native_model_loads": 0, "native_training_steps": 0, "llm_calls": 0,
        "live_deployments": 0,
    }
    write_new(output / "QUALIFICATION_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(args.out)
    print(receipt["status"], receipt["tests"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
