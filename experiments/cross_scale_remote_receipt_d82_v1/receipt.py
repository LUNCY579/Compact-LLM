"""XSC-D82-MECH-001: exact-head CPU mechanics, never an official P1 receipt."""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

TARGET = "d82e27c48b2a75b3a9459148990127cd9b14a2e9"
SCOPE = "experiments/cross_scale_ced_v1/tests"
NATIVE = {f"{SCOPE}/test_native_optional.py::test_native_random_qwen[{r}]" for r in ("encoder", "decoder")}
COUNT = 134
RUNTIME = {"torch": "2.10.0+cpu", "transformers": "4.57.1", "safetensors": "0.7.0", "psutil": "7.2.2", "pytest": "9.0.2"}
PERMISSIONS = {k: False for k in ("official_pretrained_execution", "P1", "P2", "training", "recovery", "DEV", "TEST", "efficacy", "historical_regression", "GPU_fit", "quantization", "deployment")}


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], cwd: Path, log: Path, timeout: int = 180) -> int:
    """Bound a child and its process group; never leave a timed-out test running."""
    with log.open("xb") as f:
        p = subprocess.Popen(args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise TimeoutError(f"command exceeded {timeout}s: {args[0:3]}") from None


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()


def checkout(root: Path) -> dict:
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise ValueError("subject must be its own checkout root")
    head = git(root, "rev-parse", "HEAD")
    if head != TARGET:
        raise ValueError(f"wrong target HEAD: {head}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("subject checkout is not clean")
    return {"sha": head, "tree": git(root, "rev-parse", "HEAD^{tree}")}


def source_hashes(root: Path) -> dict:
    """Bind package bytes and the three unchanged inherited Python dependencies."""
    paths = git(root, "ls-files", "experiments/cross_scale_ced_v1").splitlines()
    paths += [f"experiments/architecture_benchmark_001/{p}" for p in ("__init__.py", "model.py", "budget.py")]
    if len(paths) != 28 or len(set(paths)) != 28:
        raise ValueError("expected 25 XSC files and three inherited files")
    result = {}
    for name in paths:
        p = root/name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"missing or nonregular source: {name}")
        expected = git(root, "rev-parse", f"{TARGET}:{name}")
        actual = git(root, "hash-object", "--no-filters", "--", name)
        if expected != actual:
            raise ValueError(f"source differs from target Git blob: {name}")
        result[name] = {"sha256": sha(p), "git_blob": actual}
    return result


def validate_collection(text: str) -> list[str]:
    nodes = [line.strip() for line in text.splitlines() if "::" in line and line.startswith("experiments/")]
    if len(nodes) != COUNT or len(set(nodes)) != COUNT:
        raise ValueError(f"expected {COUNT} unique collected node IDs, found {len(nodes)}")
    permitted = {f"{SCOPE}/test_xsc.py", f"{SCOPE}/test_native_optional.py"}
    if any(n.split("::")[0] not in permitted for n in nodes) or not NATIVE.issubset(nodes):
        raise ValueError("missing native cases or widened test scope")
    return nodes


def junit_identity(node: str) -> tuple[str, str]:
    file, name = node.split("::", 1)
    return (file.removesuffix(".py").replace("/", "."), name)


def assess_junit(path: Path, nodes: list[str], exit_code: int) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    identities = [(c.get("classname", ""), c.get("name", "")) for c in cases]
    inventory_ok = collections.Counter(identities) == collections.Counter(junit_identity(n) for n in nodes)
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "passed": 0}
    failed_cases = []
    for case in cases:
        bad = [tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None]
        for tag in bad:
            counts[{"failure": "failures", "error": "errors", "skipped": "skipped"}[tag]] += 1
        if bad:
            failed_cases.append({"class": case.get("classname"), "name": case.get("name"), "kinds": bad})
        else:
            counts["passed"] += 1
    expected_native = {junit_identity(n) for n in NATIVE}
    native = [c for c in cases if (c.get("classname"), c.get("name")) in expected_native]
    native_ok = len(native) == 2 and all(not list(c) for c in native)
    summary_ok = True
    suites = [s for s in root.iter("testsuite") if not s.findall("testsuite")]
    for key in ("tests", "failures", "errors", "skipped"):
        summary_ok = summary_ok and sum(int(s.get(key, "-1")) for s in suites) == counts[key]
    passed = (exit_code == 0 and inventory_ok and summary_ok and native_ok
              and counts == {"tests": COUNT, "passed": COUNT, "failures": 0, "errors": 0, "skipped": 0})
    return dict(passed=passed, counts=counts, inventory_exact=inventory_ok,
                summary_consistent=summary_ok, both_native_cases_passed=native_ok,
                pytest_exit_code=exit_code, failed_cases=failed_cases)


def run(subject: Path, output: Path) -> int:
    subject = subject.resolve()
    output = output.absolute()
    if output.is_relative_to(subject):
        raise ValueError("receipt output must be outside the tested checkout")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(schema=1, experiment="XSC-D82-MECH-001", target_sha=TARGET,
                  status="BLOCKED", licenses="nothing", permissions=PERMISSIONS.copy(),
                  scope="134 unchanged XSC tests; random CPU/native fixtures and official-shape meta counts only",
                  synthetic_optimizer_tests="inherited tiny fixtures only; not official P2 or training evidence",
                  official_models_loaded=False, model_downloads_requested=False,
                  full_historical_suite_executed=False,
                  runner_script_sha256=sha(Path(__file__)),
                  workflow={k: os.getenv(k) for k in ("GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW_REF")})
    phase = "checkout"
    try:
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          PYTHONPATH=str(subject), HF_HOME=str(output/"empty-hf-cache"))
        report["checkout"] = checkout(subject)
        before = source_hashes(subject)
        write_new(output/"source_manifest.json", before)
        phase = "runtime"
        versions = {name: importlib.metadata.version(name) for name in RUNTIME}
        report["runtime"] = dict(python=platform.python_version(), platform=platform.platform(), libraries=versions)
        if versions != RUNTIME or platform.python_version() != "3.13.5":
            raise ValueError("runtime differs from the frozen CPU/Python pins")
        report["installed_distributions"] = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        native_probe = "import torch,transformers; from transformers import Qwen3ForCausalLM; assert torch.version.cuda is None; print(torch.__version__, transformers.__version__, Qwen3ForCausalLM.__name__)"
        if command([sys.executable, "-c", native_probe], subject, output/"native_import.log", 60):
            raise ValueError("native Qwen import preflight failed; see native_import.log")
        args = [sys.executable, "-m", "pytest", "-c", os.devnull, f"--rootdir={subject}",
                f"--confcutdir={subject/SCOPE}", "-p", "no:cacheprovider", "-o", "junit_family=xunit2", SCOPE, "-q"]
        phase = "collection"
        if command(args+["--collect-only"], subject, output/"collection.log", 120):
            raise ValueError("collection failed")
        nodes = validate_collection((output/"collection.log").read_text())
        write_new(output/"collected_nodes.json", nodes)
        phase = "tests"
        code = command(args+[f"--junitxml={output/'junit.xml'}"], subject, output/"pytest.log", 600)
        report["test_result"] = assess_junit(output/"junit.xml", nodes, code)
        phase = "post_execution_integrity"
        if checkout(subject) != report["checkout"] or source_hashes(subject) != before:
            raise ValueError("source changed during execution")
        report["post_execution_integrity"] = True
        report["status"] = "MECHANICAL_PASS_ONLY" if report["test_result"]["passed"] else "MECHANICAL_FAIL"
    except Exception as exc:
        report["failure"] = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        with (output/"supervisor_error.log").open("x") as f:
            traceback.print_exc(file=f)
    report["wall_seconds"] = round(time.monotonic()-start, 3)
    report["artifact_sha256"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_new(output/"MECHANICAL.json", report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "MECHANICAL_PASS_ONLY" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    raise SystemExit(run(a.subject, a.output))
