"""Independent CPU-only review of frozen V2 auditor/driver. No native probes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, subprocess, sys, tempfile, time, unittest

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "long_v2_audit_adaptation_work"
PINNED = {
    "long_native_auditor_v2.py": "1dcea32718c1599acaf0b6265a4d6e89285e55a49452be946cf35d91b4a0b2e7",
    "long_terminal_driver_v2.py": "a8448f856aa07e8263d314190e28e5c9c397557d767637b801224765999d3c8f",
    "test_long_v2_audit_adaptation.py": "70d87a266c5ef632a31319cae6fe333236d24598bd01e7df8426792d0c605b24",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def main():
    receipt_path = HERE / "FROZEN_CPU_AND_MUTATION_REVIEW_001.json"
    if receipt_path.exists():
        raise ValueError("write-once review receipt already exists")
    before = {name: sha(STAGE / name) for name in PINNED}
    if before != PINNED:
        raise ValueError("frozen source changed before independent review")
    started = time.monotonic()
    argv = [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", str(STAGE), "-p", "test_long_v2_audit_adaptation.py", "-v"]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    spec = importlib.util.spec_from_file_location("independent_frozen_fixture_builder", STAGE / "test_long_v2_audit_adaptation.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    d = fixture_module.d
    controls = fixture_module.FilesystemControls()
    results = []
    for name in ("source_snapshot_changed_after_semantics", "lease_terminal_changed_after_semantics", "registration_changed_before_fixed_baseline", "ignored_nonfinite_lease_field"):
        with tempfile.TemporaryDirectory() as root:
            kw = controls.fixture(root)
            changed = {}
            def change(path, suffix=b" "):
                changed.update(path=str(path), before_sha256=sha(path))
                path.write_bytes(path.read_bytes() + suffix)
                changed["after_sha256"] = sha(path)
            if name == "source_snapshot_changed_after_semantics":
                path = next((kw["preserved"] / "source_snapshot").iterdir())
                def checker(scope):
                    change(path, b"changed after immutable snapshot verification")
                    return True
                kw["asset_checker"] = checker
            elif name == "lease_terminal_changed_after_semantics":
                sup = d.selected_json(kw["attempt"] / "SUPERVISOR_RECEIPT.json")
                path = Path(sup["lease_path"]) / "terminal.json"
                def checker(scope):
                    change(path)
                    return True
                kw["asset_checker"] = checker
            elif name == "registration_changed_before_fixed_baseline":
                path = Path(d.selected_json(kw["config"])["encoder_registration"]["path"])
                calls = [0]
                def commit_checker(require_clean):
                    calls[0] += 1
                    if calls[0] == 1:
                        change(path)
                    return kw["anchors"]["source_commit"]
                kw["commit_checker"] = commit_checker
            else:
                sup = d.selected_json(kw["attempt"] / "SUPERVISOR_RECEIPT.json")
                path = Path(sup["lease_path"]) / "resume.json"
                changed.update(path=str(path), before_sha256=sha(path))
                raw = path.read_bytes()
                path.write_bytes(raw[:-2] + b',"ignored":1e999}\n')
                changed["after_sha256"] = sha(path)
            try:
                report = d.check_filesystem_fixture(**kw)
                results.append({"control": name, "outcome": "ACCEPTED_CPU_REPORT_DESPITE_MUTATION", "kind": report["kind"], "status": report["status"], "native_PASS_issued": report["native_AUDIT_KIND_PASS_issued"], **changed})
            except Exception as error:
                results.append({"control": name, "outcome": "REJECTED", "error_type": type(error).__name__, "error": str(error), **changed})
    after = {name: sha(STAGE / name) for name in PINNED}
    forbidden = sorted(name for name in sys.modules if name.split(".", 1)[0] in {"torch", "transformers", "tensorflow", "jax", "flax", "tokenizers", "jinja2", "markupsafe"})
    receipt = {"schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(), "status": "INDEPENDENT_FROZEN25_PASS_WITH_IMMUTABLE_CLOSURE_GAPS" if completed.returncode == 0 and before == after and not forbidden else "REVIEW_EXECUTION_FAILURE", "argv": argv, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "elapsed_seconds": time.monotonic() - started, "source_before": before, "source_after": after, "source_unchanged": before == after, "mutation_controls": results, "forbidden_modules_loaded": forbidden, "actual_native_identity_assets_model_binary_encoder_GPU_probes": False, "native_AUDIT_KIND_PASS_issued": False, "three_arm_goal_complete": False}
    receipt["sha256"] = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    with receipt_path.open("xb") as stream:
        stream.write((canonical(receipt) + "\n").encode())
    print(canonical({key: receipt[key] for key in ("status", "returncode", "elapsed_seconds", "source_unchanged", "mutation_controls", "sha256")}))

if __name__ == "__main__":
    main()
