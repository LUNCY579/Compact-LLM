"""Single mixed-candidate BIND DEV worker using the unchanged baseline generator.

Gold, categories and decoder/scoring plans are never passed to the worker. Only
this new grant authorizes execution; the consumed baseline grant is not opened.
Training-data verification runs in the controller, outside the gold-free worker.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_mixture_v1 import binding_common as c
from research.binding_signal_eval_v1.probe import generate


def setup(gate, base):
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    head = subprocess.check_output(["git", "-C", str(c.run_probe.PILOT), "rev-parse", "HEAD"],
                                   text=True, stderr=subprocess.PIPE, timeout=30, **kwargs).strip()
    c.require(head == gate["pilot_checkout"] == base["pilot_head"], "mixed BIND pilot checkout differs")
    sys.path.insert(0, str(c.run_probe.PILOT))


def worker(args):
    grant, gate, base = c.context(args)
    setup(gate, base)
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    from experiments.model_quality_gpu_v1.resources import guard, lease
    _, out = c.slot(args); claim_worker(out)
    before = after = None; started = time.monotonic()
    terminal = {"schema": 1, "status": "FAILED", "complete": False, "failure": "unfinished",
                "grant_sha256": c.sha(args.grant), "build_gate_sha256": c.sha(args.build_gate),
                "baseline_plan_sha256": c.sha(args.baseline_plan)}
    try:
        c.require(Path(sys.executable).resolve() == c.absolute(base["python_path"]), "mixed BIND interpreter differs")
        with lease(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"):
            c.time_guard(grant, 900); c.baseline.disk_guard(out)
            import psutil
            c.require(psutil.virtual_memory().available >= 5 * 2**30, "unchanged 5 GiB probe host guard refused")
            resources_before = guard(before_load=True, phase="qualify")
            before = c.model_snapshot(args, grant, gate, base)
            rows = c.baseline.validate_inputs(c.baseline.read_jsonl(base["input_path"]))
            import torch
            from experiments.architecture_benchmark_001 import run
            torch.set_num_threads(1); torch.manual_seed(17); torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            c.write_new(out / "PRECHECK.json", {"status": "PRECHECK_PASSED", "before": before,
                        "numerics": c.baseline.NUMERICS, "resources_before": resources_before,
                        "gold_file_read": False, "model_input_fields": ["prompt_ids"]})
            model, tokenizer, unused = run._load(Path(args.arm), "cuda:0")
            c.require(unused is None and all(p.dtype == torch.float32 for p in model.parameters()), "single FP32 RAB model required")
            eos = run._eos_ids(Path(base["source_dir"]), tokenizer)
            predictions = generate(model, tokenizer, rows, eos, torch)
            del model, tokenizer; torch.cuda.empty_cache()
            c.time_guard(grant); c.baseline.disk_guard(out); guard(phase="qualify")
            after = c.model_snapshot(args, grant, gate, base)
            c.require(before == after and c.sha(args.grant) == terminal["grant_sha256"] and
                      c.sha(args.build_gate) == terminal["build_gate_sha256"], "mixed BIND execution custody changed")
            result = {"schema": 1, "record": "BINDING_MIXTURE_DEV_GENERATIONS", "model_role": "mixed",
                      "baseline_plan_sha256": terminal["baseline_plan_sha256"], "grant_sha256": terminal["grant_sha256"],
                      "build_gate_sha256": terminal["build_gate_sha256"], "rows": 384, "worlds": 32,
                      "predictions": predictions, "eos_ids": sorted(eos), "numerics": dict(c.baseline.NUMERICS),
                      "before": before, "after": after, "gold_file_read": False,
                      "actual_budget_charges": {"rows": 384,
                           "generated_tokens_including_eos": sum(p["generated_tokens"] for p in predictions),
                           "forward_calls": sum(p["forward_calls"] for p in predictions),
                           "processed_input_tokens": sum(p["processed_input_tokens"] for p in predictions)},
                      "extra_warmup_or_qualification_calls": 0, "quality_claim": False, "training_permission": False}
            c.write_new(out / "GENERATIONS.json", result)
            terminal.update(status="COMPLETE", complete=True, failure=None, generations_sha256=c.sha(out / "GENERATIONS.json"))
    except BaseException as exc:
        terminal.update(status="FAILED", complete=False, failure={"type": type(exc).__name__, "message": str(exc)})
        c.write_new(out / "FAILURE.json", terminal)
        raise
    finally:
        terminal.update(before=before, after=after, wall_seconds=time.monotonic() - started)
        c.write_new(out / "worker_terminal.json", terminal)
    return 0


def controller(args):
    grant, gate, base = c.context(args); setup(gate, base)
    root, out = c.slot(args)
    c.require(Path(sys.executable).resolve() == c.absolute(base["python_path"]), "pinned BIND interpreter required")
    c.require(not out.exists() and not (Path(tempfile.gettempdir()) / "rineya-gpu-0.lock").exists(), "attempt or shared lease already exists")
    c.time_guard(grant, 900); c.baseline.disk_guard(out)
    before = c.model_snapshot(args, grant, gate, base)
    training_before = c.run_probe.arm_snapshot(Path(args.arm), gate)
    if out.name.endswith("-retry1"):
        c.require_retry_failure(root, before)
    else:
        c.require(not (root / "eval/mixed-binding-retry1").exists(), "retry cannot precede first attempt")
    command = [base["python_path"], str(Path(__file__).resolve())]
    arguments = {name: str(getattr(args, name)) for name in ("arm", "out", "build_gate", "baseline_plan", "baseline_plan_sha256", "grant")}
    for name, value in arguments.items(): command += ["--" + name.replace("_", "-"), value]
    command += ["--confirm-model-execution", "--_worker"]
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    code = bounded_child(command, out, 900)
    supervisor = c.read(out / "supervisor.json")
    success = False; failure = None; training_after = None; launch = {}
    try:
        c.require(code == 0 and not (out / "FAILURE.json").exists(), "mixed BIND worker failed")
        terminal = c.read(out / "worker_terminal.json"); c.baseline.verify_terminal(supervisor, terminal)
        c.require(terminal["generations_sha256"] == c.sha(out / "GENERATIONS.json") and terminal["before"] == before,
                  "mixed BIND terminal artifact/custody differs")
        launch = c.execution_identity(out, args, base, supervisor)
        training_after = c.run_probe.arm_snapshot(Path(args.arm), gate)
        c.require(training_before == training_after and before == c.model_snapshot(args, grant, gate, base), "mixed training/model changed during BIND probe")
        success = True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = str(exc)
    c.write_new(out / "BINDING.json", {"schema": 1, "status": "COMPLETE" if success else "FAILED", "complete": success,
                "controller_returncode": code, "arguments": arguments, "grant_sha256": c.sha(args.grant),
                "before": before, "training_before": training_before, "training_after": training_after,
                "supervisor_sha256": c.sha(out / "supervisor.json"),
                "worker_terminal_sha256": c.sha(out / "worker_terminal.json") if (out / "worker_terminal.json").exists() else None,
                "generations_sha256": c.sha(out / "GENERATIONS.json") if (out / "GENERATIONS.json").exists() else None,
                "failure": failure, "quality_claim": False, "training_permission": False, **launch})
    print({"status": "COMPLETE" if success else "FAILED", "out": str(out)})
    return 0 if success else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arm", "out", "build-gate", "baseline-plan", "grant"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-plan-sha256", required=True)
    parser.add_argument("--confirm-model-execution", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    c.require(args.confirm_model_execution, "mixed BIND model execution requires explicit parent confirmation")
    return worker(args) if args._worker else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
