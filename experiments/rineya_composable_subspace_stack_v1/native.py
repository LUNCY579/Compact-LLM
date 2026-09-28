"""One-arm, batch-one, local-only native collector. Disabled in shipped plan.

This collector emits raw predictions, never labels its own answers correct.
Reviewed native qualification and ancestry receipts must be supplied explicitly.
"""
from __future__ import annotations
import importlib.metadata
import platform
from pathlib import Path
import time
from .core import require
from .protocol import (admit, execution_lease, digest, file_hash, hash_arg, read_json,
                       write_new, source_digest)
from . import EXPERIMENT


def runtime_identity(device):
    return {"python": platform.python_version(), "platform": platform.system(), "device": device,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
            "dtype": "float32", "attention": "eager", "threads": 1}


def collect(plan, grant, model_dir, suite_path, output, frozen, selection=None):
    arm = grant.get("arm")
    admit(plan, grant, "collect", arm=arm)
    require(grant.get("split") in ("CAL", "CHECK"), "explicit split required")
    split = grant["split"]
    require(grant.get("freeze_sha256") == digest(frozen) and frozen["plan_sha256"] == digest(plan), "freeze not bound")
    require(frozen["code_sha256"] == source_digest(), "frozen source changed")
    if split == "CHECK":
        require(selection is not None and selection.get("selected") == arm and selection.get("freeze_sha256") == digest(frozen)
                and selection.get("selection_split") == "CAL", "locked CAL selection required before CHECK")
        require(grant.get("selection_sha256") == digest(selection), "selection not grant-bound")
    device = grant.get("device")
    require(device in ("cpu", "cuda:0"), "explicit qualified device required")
    runtime = runtime_identity(device)
    require(digest(runtime) == grant.get("runtime_sha256"), "runtime changed/unqualified")
    for field in ("qualification", "ancestry"):
        item = grant.get(field, {})
        hash_arg(item.get("sha256"), field)
        if field == "ancestry":
            require(item["sha256"] == frozen["audit_sha256"], "ancestry receipt differs from frozen audit")
        require(file_hash(item["path"]) == item["sha256"], f"{field} receipt mismatch")
        receipt = read_json(item["path"])
        require(receipt.get("experiment") == EXPERIMENT and receipt.get("reviewed") is True,
                f"{field} receipt must be operator-reviewed for CSS-001")
        require(receipt.get("plan_sha256") == digest(plan) and receipt.get("code_sha256") == source_digest(), f"{field} receipt not bound")
        if field == "qualification":
            require(receipt.get("runtime_sha256") == digest(runtime) and receipt.get("passed") is True,
                    "native runtime qualification required")
            require(receipt.get("model_files_sha256") == digest(grant.get("model_files", {})), "qualification not bound to model files")
    output, model_dir = Path(output).absolute(), Path(model_dir).absolute()
    require(output.parent.resolve() == Path(grant["output_parent"]).resolve() and not output.exists(), "fresh granted output required")
    require(not model_dir.is_symlink() and not (model_dir / "INCOMPLETE.json").exists()
            and not (model_dir / "FAILED.json").exists(), "model artifact incomplete/symlink")
    with execution_lease(grant) as guard:
        # All imports/loaders that could instantiate the native model occur below admission.
        snapshot = grant.get("model_files", {})
        require(set(snapshot) >= {"model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"}, "complete model sidecar binding required")
        actual_files = {p.name for p in model_dir.iterdir() if p.is_file() and p.name != "BUILD.json"}
        require(actual_files == set(snapshot), "unbound model sidecars or files")
        for name, expected in snapshot.items():
            require(Path(name).name == name and not (model_dir/name).is_symlink(), "unsafe model file name")
            require(file_hash(model_dir/name) == hash_arg(expected, name), "model/sidecar hash mismatch")
        require(snapshot["model.safetensors"] == frozen["artifacts"].get(arm), "artifact not frozen")
        build = read_json(model_dir/"BUILD.json")
        require(build.get("experiment") == EXPERIMENT and build.get("plan_sha256") == digest(plan)
                and build.get("arm") == arm and build.get("weight_sha256") == snapshot["model.safetensors"], "build provenance mismatch")
        suite = read_json(suite_path)
        expected = frozen["cal_sha256"] if split == "CAL" else frozen["check_sha256"]
        require(file_hash(suite_path) == expected and grant.get("dataset_sha256") == expected, "suite not frozen/granted")
        rows = suite.get("rows", [])
        require(0 < len(rows) <= 128, "1-128 native rows required")
        seen, parents = set(), set()
        for row in rows:
            require(set(row) == {"id", "parent_id", "domain", "role", "prompt"}, "prompt-only suite fields required")
            require(all(isinstance(row[k], str) and row[k].strip() for k in row), "invalid suite row")
            require(row["role"] in ("capability", "retention"), "suite role invalid")
            require(row["id"] not in seen and row["parent_id"] not in parents, "one canonical view per native parent required")
            seen.add(row["id"]); parents.add(row["parent_id"])
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        if device.startswith("cuda"):
            require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 7.0*1024**3, "native GPU requires >=7 GiB free")
        config = read_json(model_dir/"config.json")
        require(config.get("model_type") == "qwen3" and config.get("architectures") == ["Qwen3ForCausalLM"]
                and not any(k in config for k in ("auto_map", "quantization_config")), "unqualified model architecture")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
                trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32,
                attn_implementation="eager").to(device).eval()
        predictions = []
        started = time.monotonic()
        with torch.inference_mode():
            for row in rows:
                guard()
                inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
                length = inputs["input_ids"].shape[1]
                require(0 < length <= 4032, "prompt plus 64 new tokens exceeds 4096")
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
                predictions.append({"id": row["id"], "parent_id": row["parent_id"],
                                    "domain": row["domain"], "role": row["role"],
                                    "prediction": tokenizer.decode(tokens[0, length:], skip_special_tokens=True)})
        for name, expected in snapshot.items():
            require(file_hash(model_dir/name) == expected, "source changed during native evaluation")
        require(file_hash(suite_path) == frozen["cal_sha256" if split == "CAL" else "check_sha256"], "suite changed")
        result = {"schema": 1, "status": "RAW_PREDICTIONS_NOT_SCORED", "experiment": EXPERIMENT,
                  "arm": arm, "split": split, "rows": predictions, "runtime": runtime,
                  "elapsed_seconds": time.monotonic()-started, "generation_calls": len(rows), "model_forward_count": None,
                  "optimizer_steps": 0, "release_permitted": False,
                  "grant_sha256": digest(grant), "freeze_sha256": digest(frozen)}
        write_new(output, result)
        return result
