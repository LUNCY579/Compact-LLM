"""Fail-closed, local-only, single-smoke qualification. Never starts a native study."""
import importlib.metadata
import platform
from pathlib import Path
import time
from . import BRANCH, EXPERIMENT, PARENT_COMMIT
from .data import DATA_MANIFEST, file_hash, load_atomic, read_json

ENTRY_RAM_GIB = 7.4
WORKER_RAM_GIB = 6.0


def runtime():
    result = {"python": platform.python_version(), "platform": platform.system()}
    for package in ("torch", "transformers", "safetensors", "psutil", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def blockers(binding):
    errors = []
    for key in ("model_dir", "data_dir", "eligible_names", "runtime", "smoke_row_id", "git_commit"):
        if not binding.get(key):
            errors.append(f"unbound:{key}")
    if binding.get("parent_commit") != PARENT_COMMIT:
        errors.append("wrong_parent_commit")
    if binding.get("dataset_sha256") != DATA_MANIFEST:
        errors.append("wrong_dataset_pin")
    if binding.get("runtime") and binding["runtime"] != runtime():
        errors.append("runtime_mismatch")
    if platform.system() != "Windows":
        errors.append("exact_Windows_GPU_lease_host_required")
    if binding.get("model_dir") and not Path(binding["model_dir"]).is_dir():
        errors.append("local_model_missing")
    if binding.get("data_dir") and not Path(binding["data_dir"]).is_dir():
        errors.append("local_corpus_missing")
    for pkg, version in runtime().items():
        if version is None:
            errors.append(f"dependency_missing:{pkg}")
    return errors


def preflight(binding):
    return {"experiment": EXPERIMENT, "branch": BRANCH, "parent_commit": PARENT_COMMIT,
            "status": "BLOCKED" if blockers(binding) else "BOUND_NOT_SMOKE_QUALIFIED",
            "blockers": blockers(binding), "runtime": runtime(), "native_model_loads": 0,
            "native_training_steps": 0, "full_training_authorized": False}


def ram_guard(threshold=WORKER_RAM_GIB):
    import psutil
    free = psutil.virtual_memory().available / 2**30
    if free < threshold:
        raise RuntimeError(f"host RAM guard: {free:.3f} GiB < {threshold:.3f} GiB")
    return free


def smoke(binding, config, *, authorized=False):
    report = preflight(binding)
    if report["blockers"] or not authorized:
        report["status"] = "BLOCKED"
        if not authorized:
            report["blockers"].append("explicit_smoke_authorization_required")
        return report
    # Reuse inherited immutable checkpoint/tokenizer checks and the shared-card lease.
    from experiments.titan_model_soup_mini_v1 import soup
    from experiments.fresh_dual_skill_instrument_v1.gpu_lease import lease
    from dataclasses import replace
    import subprocess
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .training import SkillRunner
    from .artifacts import tensor_hash
    cwd = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=cwd, text=True).strip()
    if head != binding["git_commit"] or branch != BRANCH or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, text=True).strip():
        raise ValueError("native checkout is not the bound clean branch/commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    model_dir = Path(binding["model_dir"])
    with lease(label="clare-rineya-001:smoke", job="one rollback-only smoke", timeout_seconds=0):
        report["entry_free_ram_gib"] = [ram_guard(ENTRY_RAM_GIB)]
        time.sleep(1)
        report["entry_free_ram_gib"].append(ram_guard(ENTRY_RAM_GIB))
        soup.checked_file(model_dir, "model.safetensors", soup.BASE_SHA)
        for name, digest in soup.ASSETS.items():
            soup.checked_file(model_dir, name, digest)
        soup.checked_file(model_dir, "config.json", "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd")
        soup.canonical_config(soup.read_json(model_dir / "config.json"))
        from safetensors import safe_open
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
            inventory = soup.inventory(handle)
            soup.check_tied_head(handle)
        if sum(__import__("math").prod(v) for v in inventory.values()) != 596049920:
            raise ValueError("full-model inventory mismatch")
        names = binding["eligible_names"]
        if len(names) != len(set(names)) or any(n not in inventory or not n.startswith("model.layers.") for n in names):
            raise ValueError("eligible pool must explicitly name unique transformer parameters")
        import math
        count = sum(math.prod(inventory[n]) for n in names)
        if count > config.max_coordinates:
            raise ValueError("native coordinate budget exceeded")
        # Direct FP32 weights, gradients/moments/snapshots and conservative activations.
        estimated = 596049920 * 4 + count * 32 + 2 * 2**30
        free_vram, _ = torch.cuda.mem_get_info()
        if free_vram < estimated:
            raise RuntimeError("VRAM estimate exceeds currently free memory")
        rows = load_atomic(binding["data_dir"], config.order[0])
        row = next((r for r in rows if r.row_id == binding["smoke_row_id"]), None)
        if row is None:
            raise ValueError("smoke row not in the registered atomic TRAIN bank")
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        text = tok.apply_chat_template(row.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        if not 2 <= enc["input_ids"].shape[1] <= 512:
            raise ValueError("smoke token length outside [2,512]; no truncation")
        ram_guard(ENTRY_RAM_GIB)
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, attn_implementation="eager")
        report["native_model_loads"] = 1
        tiny = replace(config, arm="learned_disjoint", order=(config.order[0],), discovery_steps=1, train_steps=1)
        runner = SkillRunner(model, names, tiny, data_digest=DATA_MANIFEST)
        initial = runner.coords.vector()
        batch = {k: v.to("cuda:0") for k, v in enc.items()}
        def plumbing_loss(m, _row):
            # Prompt next-token loss exercises autograd only. NOT skill supervision or a study.
            return m(**batch, labels=batch["input_ids"], use_cache=False).loss
        try:
            rec = runner.train_skill(tiny.order[0], [row], plumbing_loss, guard=ram_guard)
            report.update({"status": "SMOKE_ONLY_PASS", "native_training_steps": 2,
                           "eligible_coordinates": count, "coordinate_names": list(names),
                           "estimated_vram_bytes": estimated, "reset_exact": rec["reset_exact"],
                           "mask_sha256": tensor_hash(rec["mask"]), "delta_norm": rec["delta_norm"],
                           "device": torch.cuda.get_device_name(0), "dtype": "float32",
                           "objective": "prompt_next_token_plumbing_only_not_skill_training",
                           "token_count": enc["input_ids"].shape[1]})
        finally:
            runner.coords.restore(initial)
        report["rollback_exact"] = torch.equal(initial, runner.coords.vector())
        del runner, model
        torch.cuda.empty_cache()
    return report
