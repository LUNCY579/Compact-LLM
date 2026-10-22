"""Explicitly permitted, local-only native Stage 0/1 screen; no native training.

Reuses the pinned ARMR scorer and bounded greedy generation without modifying it.
Fresh first-pass and restoration identity are tested in addition to the inherited
receipt. Every injection interface is ZERO-START here: this is apparatus
qualification, not a trained architectural comparison.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .protocol import BASE_SHA, CHECKPOINT_SHA, preflight, sha256, verify_bundle, new_run, write_json


def run_native(permit: dict, *, repo: Path, checkpoint: Path, cases_path: Path, out: Path) -> dict:
    if out.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("output must not be inside the canonical checkpoint tree")
    out = new_run(out)
    initial = preflight(permit)
    write_json(out / "preflight.json", initial)
    if initial["status"] == "BLOCKED":
        write_json(out / "status.json", initial)
        return initial
    status = {"status": "BLOCKED", "native_loads": 0, "native_optimizer_steps": 0,
              "synthetic_evidence": False, "trained_interfaces": False}
    start, prior_path = time.perf_counter(), list(sys.path)
    resource_hook = None
    try:
        import psutil
        import torch
        from .core import InjectionInterface
        from .host import FixedPointHost
        def guard():
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError("runtime free-RAM guard: 6 GiB")
            process = psutil.Process()
            rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
            if rss > 8 * 1024**3:
                raise RuntimeError("process-tree RSS guard: 8 GiB")
            if time.perf_counter() - start > 3600:
                raise RuntimeError("one-hour cooperative screen budget")
        if psutil.virtual_memory().available < 7.4 * 1024**3:
            raise RuntimeError("pre-load free-RAM guard: 7.4 GiB")
        repo = repo.resolve()
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=repo, check=True, capture_output=True)
        verify_bundle(checkpoint, permit["checkpoint_files"])
        if permit["checkpoint_files"].get("model.safetensors") != CHECKPOINT_SHA:
            raise ValueError("wrong canonical checkpoint")
        # Pin all Python files in both imported source trees, not just entry points.
        roots = [repo / "experiments/anchor_routed_middle_recurrence_v1/src",
                 repo / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"]
        required = {p.relative_to(repo).as_posix() for root in roots for p in root.rglob("*.py")}
        if not required or not required <= set(permit["source_files"]):
            raise ValueError("all inherited Python sources must be pinned")
        verify_bundle(repo, permit["source_files"], exact=False)
        receipt_path = Path(permit["host_qualification"]["path"])
        if sha256(receipt_path) != permit["host_qualification"]["sha256"]:
            raise ValueError("qualification receipt digest mismatch")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "PASS" or not receipt.get("checks") or not all(v is True for v in receipt["checks"].values()):
            raise ValueError("passing inherited qualification receipt required")
        dataset = permit["dataset"]
        if dataset.get("split") != "DEV" or sha256(cases_path) != dataset["sha256"]:
            raise ValueError("explicit unsealed DEV file hash required")
        cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
        ids = [r["case_id"] for r in cases]
        if not 1 <= len(cases) <= 16 or len(set(ids)) != len(ids) or ids != dataset["case_ids"]:
            raise ValueError("exact ordered allowlist, at most 16 DEV cases")
        # Only now import code capable of loading a model; loading is local_files_only.
        transformers = importlib.import_module("transformers")
        actual = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}
        if actual != permit["runtime_versions"]:
            raise ValueError("runtime differs from pinned qualification")
        manifest_path = Path(permit["host_qualification"]["manifest_path"])
        if sha256(manifest_path) != permit["host_qualification"]["manifest_sha256"]:
            raise ValueError("qualification run-manifest digest mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "PASS" or manifest.get("checkpoint_files_before", {}).get("model.safetensors") != CHECKPOINT_SHA:
            raise ValueError("qualification manifest does not bind canonical weights")
        if manifest.get("torch") != actual["torch"] or manifest.get("transformers") != actual["transformers"]:
            raise ValueError("qualification runtime mismatch")
        guard()
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=True,
            dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True).eval().requires_grad_(False)
        status["native_loads"] = 1
        if not 500_000_000 <= sum(p.numel() for p in model.parameters()) <= 700_000_000:
            raise ValueError("this lane is the bound ~0.6B Supa host only")
        resource_hook = model.register_forward_pre_hook(lambda _m, _a, _k: guard(), with_kwargs=True)
        versions = [(p, p._version) for p in model.parameters()]
        sys.path[:0] = [str(root) for root in roots]
        evaluation = importlib.import_module("armr.evaluation")
        if Path(evaluation.__file__).resolve() != roots[0] / "armr/evaluation.py":
            raise ValueError("foreign cached armr module")
        inherited_hooks = importlib.import_module("armr.hooks")
        for name, module in list(sys.modules.items()):
            if name == "armr" or name.startswith("armr.") or name == "rineya_cross_scale" or name.startswith("rineya_cross_scale."):
                source = getattr(module, "__file__", None)
                if source is not None and Path(source).resolve().relative_to(repo).as_posix() not in permit["source_files"]:
                    raise ValueError("unbound inherited module")
        config = {"max_new_tokens": 24, "max_prompt_seconds": 90, "max_rss_gib": 8, "top_k": 10}
        bases, base_rows, all_rows = {}, {}, []
        # Match arm parameters exactly. These zero-start arms must match raw recurrence.
        additive = InjectionInterface(model.config.hidden_size, mode="additive")
        orthogonal = InjectionInterface(model.config.hidden_size, mode="orthogonal")
        orthogonal.load_state_dict(additive.state_dict())
        arms = [("unwrapped", 1, None), ("intact", 1, None), ("raw", 2, None), ("additive_zero", 2, additive), ("orthogonal_zero", 2, orthogonal), ("restored", 1, None)]
        raw_logits, raw_ids = {}, {}
        for name, applications, interface in arms:
            for case in cases:
                guard()
                manager = inherited_hooks.Reapplication(model, enabled=False) if name == "unwrapped" else FixedPointHost(model, applications=applications, alpha=permit["alpha"], interface=interface)
                with manager as wrapper, torch.inference_mode():
                    row, first = evaluation.evaluate_case(model, tokenizer, case, config, wrapper, bases.get(case["case_id"]))
                key = case["case_id"]
                if row["safety_stopped"]:
                    raise RuntimeError(f"safety stop for {name}/{key}: {row['termination']}")
                if name == "unwrapped":
                    bases[key], base_rows[key] = first, row
                if name == "raw":
                    raw_logits[key], raw_ids[key] = first, row["token_ids"]
                if name in {"additive_zero", "orthogonal_zero"} and (not torch.equal(first, raw_logits[key]) or row["token_ids"] != raw_ids[key]):
                    raise RuntimeError("zero-start interface does not reproduce raw recurrence")
                if name in {"intact", "restored"} and (not torch.equal(first, bases[key]) or row["token_ids"] != base_rows[key]["token_ids"]):
                    raise RuntimeError("baseline restoration identity failed")
                row.update(arm=name, split="DEV", seed=17, trained_interface=False)
                all_rows.append(row)
                write_json(out / f"row_{len(all_rows):04d}.json", row)
        if any(p._version != version or p.grad is not None for p, version in versions):
            raise RuntimeError("base tensor mutation or unexpected gradients")
        verify_bundle(checkpoint, permit["checkpoint_files"])
        status.update(status="NATIVE_APPARATUS_PASS_NOT_EFFICACY", cases=len(cases), arms=len(arms),
                      zero_start_identity=True, restoration_identity=True, runtime=actual,
                      dataset_sha256=sha256(cases_path), checkpoint_files=permit["checkpoint_files"])
    except Exception as error:
        status.update(status="BLOCKED" if status["native_loads"] == 0 else "FAIL",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if resource_hook is not None:
            resource_hook.remove()
        sys.path[:] = prior_path
        status["wall_seconds"] = time.perf_counter() - start
        write_json(out / "status.json", status)
    return status
