"""Local-only CPU supervisor: preflight -> resource guard -> bounded worker.

No remote downloads, CUDA, automatic retries, historical output rewrites or
implicit training. Use an existing exact-host manifest; see HOST_HANDOFF.md.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .binding import package_digest, preflight, qualification_identity


def write_json(path: Path, value: dict) -> None:
    # Output locations are new per session; never replace historical receipts.
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def worker(m: dict, out: Path) -> int:
    import psutil
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from .data import load_rows
    from .diagnostics import historical_diagnostics, qualify
    from .study import TrainSpec, run_study
    from .tangent import CallBudget, FrozenAnchor, tensor_digest

    preflight(m)
    # Conservative planning allowance, NOT an observed peak. Applies after imports.
    free_gib = psutil.virtual_memory().available / 2**30
    if free_gib < m["limits"]["min_available_ram_gib"]:
        raise RuntimeError(f"RAM_GUARD: {free_gib:.3f} GiB available")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    cfg = json.loads((Path(m["base"]["directory"]) / "config.json").read_text())
    forbidden = set(json.loads(Path(m["forbidden_parents"]["path"]).read_text()))
    bank = m["qualification_probe"] if m["action"] == "qualify" else m["data"]
    rows = load_rows(bank["path"], vocab_size=cfg["vocab_size"],
                     max_length=m["limits"]["max_length"], forbidden_parents=forbidden)
    if len(rows) > m["limits"]["max_rows"]:
        raise ValueError("row ceiling exceeded")
    eos = bank["eos_token_id"]
    configured_eos = cfg.get("eos_token_id")
    allowed_eos = configured_eos if isinstance(configured_eos, list) else [configured_eos]
    if eos not in allowed_eos or any(r.labels[-1] != eos for r in rows):
        raise ValueError("assistant_response_plus_eos serialization not satisfied")
    if m["action"] == "study":
        spec = TrainSpec(**m["training"])
        spec.validate()
        if 4 * spec.steps > m["limits"]["max_optimizer_steps"]:
            raise ValueError("four independent training arms exceed optimizer budget")
    else:
        if any(r.split != "diagnostic" for r in rows):
            raise ValueError("qualification/diagnostic requires a diagnostic-only manifest")
    write_json(out / "LOAD_ATTEMPT.json", {"model_load_attempts": 1, "model_loads_completed": 0})
    model = AutoModelForCausalLM.from_pretrained(
        m["base"]["directory"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True)
    model.config.use_cache = False
    model.eval()
    named = dict(model.named_parameters())
    for p in m["scope_prefixes"]:
        if not any(k.startswith(p) for k in named):
            raise ValueError(f"scope prefix matched nothing: {p}")
    scope = sorted(k for k in named if any(k.startswith(p) for p in m["scope_prefixes"]))
    write_json(out / "LOAD_COMPLETE.json", {"model_loads_completed": 1, "scope": scope,
               "canonical_parameter_count": sum(p.numel() for p in named.values())})
    base_bytes = sum(p.numel()*p.element_size() for p in named.values())
    scope_bytes = sum(named[k].numel()*named[k].element_size() for k in scope)
    # Planning allowance, not measured peak: snapshot, deltas/moments/gradients,
    # transient merges, and output/activation headroom. Fail before allocation.
    copies = 10 if m["action"] == "study" else 5
    allowance = base_bytes + copies*scope_bytes + 2*2**30 + 8*m["limits"]["max_length"]*cfg["vocab_size"]*4
    write_json(out / "RESOURCE_PLAN.json", {"additional_bytes_allowance": allowance,
               "basis": "conservative analytical; not observed peak", "scope_bytes": scope_bytes})
    if psutil.virtual_memory().available < allowance:
        raise RuntimeError("RAM_GUARD: insufficient additional tangent allocation allowance")
    anchor = FrozenAnchor(model, scope, CallBudget(m["limits"]["max_logical_calls"]))
    if m["action"] == "qualify":
        result = qualify(anchor, rows[0])
    elif m["action"] == "diagnostic":
        ds = []
        for item in m["historical_endpoints"]:
            trained = load_file(item["path"], device="cpu")
            if set(trained) != set(scope) or any(t.dtype != torch.float32 for t in trained.values()):
                raise ValueError("FP32 partial trained endpoint must exactly cover scope")
            d = {k: trained[k] - anchor.params[k] for k in scope}
            anchor.validate(d); ds.append(d)
        result = historical_diagnostics(anchor, ds[0], rows, ds[1] if len(ds) == 2 else None)
    else:
        from safetensors.torch import save_file
        def save_delta(mode, task, delta):
            # Numeric names avoid interpreting a task label as a filesystem path.
            index = list(m["tasks"]).index(task)
            destination = out / f"delta-{mode}-task{index}.safetensors"
            if destination.exists():
                raise FileExistsError(destination)
            save_file({k: t.contiguous() for k, t in delta.items()}, str(destination))
        result = run_study(anchor, rows, tuple(m["tasks"]), spec, on_delta=save_delta)
        result["status"] = "STUDY_COMPLETE_DEVELOPMENTAL"
    if package_digest() != m["package_sha256"]:
        raise RuntimeError("source changed during run")
    anchor.assert_unchanged()
    result.update({"qualification_identity": qualification_identity(m),
                   "source_commit": m["source_commit"], "package_sha256": package_digest(),
                   "runtime": m["runtime"], "model_loads_completed": 1,
                   "anchor_value_sha256": tensor_digest(anchor.params),
                   "logical_calls": {"nonlinear": anchor.budget.nonlinear, "jvp": anchor.budget.jvp},
                   "quality_claim": False, "release_permitted": False})
    write_json(out / "RESULT.json", result)
    return 0 if result["status"] != "QUALIFICATION_FAIL" else 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.preflight_only:
        print(json.dumps(preflight(m), indent=2)); return 0
    if args.out is None:
        p.error("--out is required for execution")
    out = args.out.resolve()
    if args.worker:
        if os.environ.get("TTA_SUPERVISOR_OUT") != str(out) or not (out / "MANIFEST.json").is_file():
            raise RuntimeError("worker requires a supervisor-created reservation")
        return worker(m, out)
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo):
        raise ValueError("session outputs must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=False)  # one-use reservation, never reuse
    write_json(out / "MANIFEST.json", m)
    started = time.time()
    try:
        preflight(m)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="",
                   TTA_SUPERVISOR_OUT=str(out))
        with (out / "worker.log").open("x") as log:
            proc = subprocess.run([sys.executable, "-m", "experiments.titan_tangent_arithmetic_v1.host",
                                   "--manifest", str(out / "MANIFEST.json"), "--out", str(out), "--worker"],
                                  cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=m["limits"]["max_wall_seconds"], check=False)
        status = "WORKER_SUCCEEDED" if proc.returncode == 0 and (out / "RESULT.json").exists() else "WORKER_FAILED"
        receipt = {"status": status, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        receipt = {"status": "TIMED_OUT", "note": "worker terminated; partial output is not success"}
    except Exception as exc:
        receipt = {"status": "BLOCKED_PREWORKER", "error": f"{type(exc).__name__}: {exc}"}
    receipt.update({"elapsed_seconds": time.time() - started, "quality_claim": False,
                    "model_load_attempt_observed": (out / "LOAD_ATTEMPT.json").exists(),
                    "model_load_complete_observed": (out / "LOAD_COMPLETE.json").exists(),
                    "automatic_retry": False})
    write_json(out / "SUPERVISOR.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "WORKER_SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
