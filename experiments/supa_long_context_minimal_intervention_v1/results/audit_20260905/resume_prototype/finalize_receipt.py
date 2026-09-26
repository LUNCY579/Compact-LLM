"""Derive the bounded CPU result receipt from saved JUnit reports."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import torch

HERE = Path(__file__).resolve().parent


def summary(filename):
    root = ET.parse(HERE / filename).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {"report": filename,
            **{key: sum(int(s.get(key, "0")) for s in suites) for key in ["tests", "failures", "errors", "skipped"]},
            "seconds": sum(float(s.get("time", "0")) for s in suites)}


source = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
source_matches = hashlib.sha256(Path(source["live_source_path"]).read_bytes()).hexdigest() == source["live_source_sha256"]
artifacts = ["src/slcmi/training.py", "src/slcmi/checkpoint_resume.py", "training.patch",
             "baseline_training.py", "tests/test_training_safety.py", "tests/test_resume_equivalence.py",
             "cpu_process_worker.py", "README.md", "REVIEW.md"]
receipt = {
    "status": "UNADOPTED",
    "scope": "CPU-only synthetic adapter checkpoint proof; no canonical model or GPU loaded",
    "python_executable": sys.executable,
    "torch_version": str(torch.__version__),
    "full_suite_before_final_numerical_fix": summary("test_results.xml"),
    "targeted_scaler_regression_after_final_numerical_fix": summary("scaler_review_results.xml"),
    "full_suite_repeated_after_final_numerical_fix": False,
    "unique_test_cases_exercised": 57,
    "true_process_death_cases": ["os._exit(37) after committed attempt 2 with validation pending",
                                 "os._exit(37) inside accumulation after prior attempt 1 commit"],
    "equivalence": "torch.equal for tensors plus exact recursive state/receipt equality, excluding wall timing and throughput",
    "canonical_source_hash_matches_build_input": source_matches,
    "canonical_sources_modified_by_prototype": False,
    "review_findings_resolved": ["dose-accounting corruption", "AdamW/scheduler state mismatch",
                                "RNG validation before checkpoint tensor copy", "float32 scaler underflow/growth saturation"],
    "artifact_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in artifacts},
}
assert source_matches
assert all(receipt[k]["failures"] == 0 and receipt[k]["errors"] == 0 for k in
           ["full_suite_before_final_numerical_fix", "targeted_scaler_regression_after_final_numerical_fix"])
(HERE / "test_summary.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
