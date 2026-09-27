"""Bounded native loader/state/forward identity probe; no generation or training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

LEASE_MODULE = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hplat-rehearsal-20260921/gpu_lease.py")
LEASE_SHA = "049742734eea9bddd7aaf149d2da824c3d9eec1e2157fab6c6c586d033fda00a"
LEASE_LABEL = "comp-rs-q1-native-loader"
RAM_FLOOR_GIB = 5.5
DISK_FLOOR_GIB = 6.0
VRAM_FLOOR_MIB = 4096
MAX_WALL_SECONDS = 1200
EXPECTED_PY = Path(r"C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe")
MESSAGES = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": 'Reply with the JSON object {"answer": [1]}.'}]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def vram_free_mib() -> int:
    run = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True)
    values = [int(line.strip()) for line in run.stdout.splitlines() if line.strip()]
    require(len(values) == 1, "expected one visible GPU")
    return values[0]


def state_fingerprint(model, source: Path, torch) -> dict:
    from safetensors import safe_open

    state = model.state_dict()
    with safe_open(str(source), framework="pt", device="cpu") as f:
        source_keys = set(f.keys())
        extra = set(state) - source_keys
        require(extra == {"lm_head.weight"} and not (source_keys - set(state)),
                f"loaded state key mismatch: extra={sorted(extra)} missing={sorted(source_keys-set(state))}")
        inp = model.get_input_embeddings().weight
        out = model.get_output_embeddings().weight
        require(inp.data_ptr() == out.data_ptr(), "lm_head is not tied to input embedding")
        digest = hashlib.sha256()
        checked = 0
        for name in sorted(source_keys):
            expected = f.get_tensor(name)
            actual = state[name].detach().to("cpu").contiguous()
            require(expected.dtype == actual.dtype and tuple(expected.shape) == tuple(actual.shape)
                    and torch.equal(expected, actual), f"loaded tensor differs: {name}")
            raw = actual.view(torch.uint8).numpy()
            piece = hashlib.sha256(memoryview(raw)).hexdigest()
            digest.update(name.encode() + b"\0" + piece.encode() + b"\n")
            checked += 1
            del expected, actual, raw
    return {"stored_tensors_matched": checked, "tied_head_alias": True,
            "state_fingerprint_sha256": digest.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-receipt", required=True, type=Path)
    ap.add_argument("--tokenizer-receipt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("receipt exists (create-only)")
    start = time.monotonic()
    import psutil
    from shutil import disk_usage

    a = psutil.virtual_memory().available / 2**30
    time.sleep(5)
    b = psutil.virtual_memory().available / 2**30
    disk = disk_usage("C:/").free / 2**30
    print(f"launch gate RAM {a:.2f}/{b:.2f} GiB, disk {disk:.2f} GiB", flush=True)
    if min(a, b) < RAM_FLOOR_GIB or disk < DISK_FLOOR_GIB:
        print("NOT_STARTED_RESOURCE_FLOOR", flush=True)
        return 3
    require(Path(sys.executable).resolve() == EXPECTED_PY.resolve(), "wrong Python runtime")
    require(sha256(LEASE_MODULE) == LEASE_SHA, "GPU lease implementation drift")
    asset = json.loads(args.asset_receipt.read_text(encoding="utf-8"))
    token_rec = json.loads(args.tokenizer_receipt.read_text(encoding="utf-8"))
    require(asset["stage"] == "Q1_ASSET_CANDIDATE" and token_rec["stage"] == "Q1_TOKENIZER_PROMPT_LENGTHS",
            "Q1 prerequisite receipts missing")
    model_dir = Path(asset["model_path"])
    require(sha256(model_dir / "model.safetensors") == asset["assets"]["model.safetensors"]["sha256"],
            "model weight bytes drift")
    require(vram_free_mib() >= VRAM_FLOOR_MIB, "GPU free VRAM below Q1 floor")
    sys.path.insert(0, str(LEASE_MODULE.parent))
    import gpu_lease
    require(gpu_lease.LOCK.name == "rineya-gpu-0.lock", "GPU lease path mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.maxrl_comp002_pretraining_v1.mechanics import parameter_inventory

    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "safetensors")}
    require(versions == asset["runtime_candidate"]["packages"], "runtime version drift")
    token = gpu_lease.acquire(timeout_s=120, label=LEASE_LABEL)
    try:
        require(time.monotonic() - start < MAX_WALL_SECONDS, "wall ceiling before load")
        pre_load_ram = psutil.virtual_memory().available / 2**30
        pre_load_vram = vram_free_mib()
        if pre_load_ram < RAM_FLOOR_GIB or pre_load_vram < VRAM_FLOOR_MIB:
            print(f"NOT_STARTED_PRELOAD_FLOOR RAM={pre_load_ram:.2f} VRAM={pre_load_vram}", flush=True)
            return 3
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                   trust_remote_code=False)
        require(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
                == token_rec["chat_template_sha256"], "tokenizer template drift")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16,
                    local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    low_cpu_mem_usage=False).eval()
        require(type(model).__name__ == "Qwen3ForCausalLM", "unexpected loaded model class")
        model.to("cuda")
        require(next(model.parameters()).device.type == "cuda", "model not on CUDA")
        inventory = parameter_inventory(model)
        before = state_fingerprint(model, model_dir / "model.safetensors", torch)
        rendered = tokenizer.apply_chat_template(MESSAGES, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        require(ids.shape[1] < 128, "synthetic loader prompt unexpectedly long")
        with torch.inference_mode():
            logits1 = model(input_ids=ids, use_cache=False).logits
            logits2 = model(input_ids=ids, use_cache=False).logits
        torch.cuda.synchronize()
        require(torch.isfinite(logits1).all().item() and torch.equal(logits1, logits2),
                "nonfinite/nonrepeatable forward logits")
        last = logits1[0, -1].float().cpu().contiguous()
        last_sha = hashlib.sha256(memoryview(last.numpy())).hexdigest()
        top_ids = torch.topk(last, 5).indices.tolist()
        del logits1, logits2, last
        after = state_fingerprint(model, model_dir / "model.safetensors", torch)
        require(before == after, "model state changed during inference")
        require(time.monotonic() - start < MAX_WALL_SECONDS, "Q1 wall ceiling exceeded")
        receipt = {"experiment": "COMP-RS-001", "stage": "Q1_NATIVE_LOADER_FORWARD",
                   "asset_receipt_sha256": sha256(args.asset_receipt),
                   "tokenizer_receipt_sha256": sha256(args.tokenizer_receipt),
                   "lease_module_sha256": LEASE_SHA, "lease_label": LEASE_LABEL,
                   "model_class": type(model).__name__, "model_path": str(model_dir),
                   "source_weight_sha256": asset["assets"]["model.safetensors"]["sha256"],
                   "state_before": before, "state_after": after,
                   "parameter_inventory": inventory,
                   "synthetic_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                   "synthetic_tokens": int(ids.shape[1]), "forward_calls": 2,
                   "last_logits_sha256": last_sha, "top_token_ids": top_ids,
                   "runtime": {"python": platform.python_version(), "packages": versions,
                               "torch_cuda_build": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0),
                               "gpu_capability": list(torch.cuda.get_device_capability(0)),
                               "device": "cuda", "weights_dtype": "bfloat16", "tf32": False},
                   "resources": {"launch_ram_gib": [a, b], "launch_disk_gib": disk,
                                 "required_ram_gib": RAM_FLOOR_GIB,
                                 "required_free_vram_mib": VRAM_FLOOR_MIB,
                                 "preload_ram_gib": pre_load_ram, "preload_free_vram_mib": pre_load_vram,
                                 "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                 "wall_seconds": time.monotonic() - start},
                   "generated_completions": 0, "backward_calls": 0, "optimizer_steps": 0,
                   "native_loader_qualified": True, "native_training_qualified": False,
                   "training_authorized": False}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"stage": receipt["stage"], "stored_tensors_matched": before["stored_tensors_matched"],
                          "peak_gpu_mib": receipt["resources"]["peak_gpu_allocated_mib"]}), flush=True)
        del model, tokenizer, ids
        torch.cuda.empty_cache()
    finally:
        gpu_lease.release(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
