"""Read-only local HF/LoRA evaluator. All heavy imports are lazy and explicit."""
from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from .common import Blocked, digest, exclusive, require


def preflight(config: dict, needs_peft: bool = False) -> dict:
    packages = ["torch", "transformers", "safetensors", "psutil"] + (["peft"] if needs_peft else [])
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    reasons = [f"missing dependency: {p}" for p in missing]
    versions = {p: importlib.metadata.version(p) for p in packages if p not in missing}
    free_ram = None
    if "psutil" not in missing:
        import psutil
        free_ram = psutil.virtual_memory().available / 2 ** 30
        if free_ram < config["min_free_ram_gib"]:
            reasons.append("host RAM below registered guard")
    if shutil.disk_usage(Path.cwd()).free / 2 ** 30 < config["min_free_disk_gib"]:
        reasons.append("disk below registered guard")
    free_vram = None
    if config["device"] == "cuda" and "torch" not in missing:
        import torch
        if not torch.cuda.is_available():
            reasons.append("CUDA unavailable")
        else:
            free_vram = torch.cuda.mem_get_info(0)[0] / 2 ** 30
            if free_vram < config["min_free_vram_gib"]:
                reasons.append("CUDA free memory below registered guard")
            try:
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                      check=True, capture_output=True, text=True, timeout=10)
                foreign = [line.strip() for line in apps.stdout.splitlines() if line.strip().isdigit() and int(line.strip()) != os.getpid()]
                if foreign:
                    reasons.append("another CUDA compute process is active; do not interrupt it")
            except (OSError, subprocess.SubprocessError):
                reasons.append("cannot verify exclusive GPU availability")
    return {"status": "BLOCKED" if reasons else "PREFLIGHT_ONLY", "reasons": reasons,
            "versions": versions, "python": platform.python_version(),
            "free_ram_gib": free_ram, "free_vram_gib": free_vram,
            "does_not_authorize_training_or_establish_compatibility": True}


def continuation_logp(model, prompt_ids: list[int], candidate_ids: list[int], device: str) -> float:
    """Score every candidate token, shifted once; never include prompt loss."""
    import torch
    require(bool(prompt_ids) and bool(candidate_ids), "Empty prompt/continuation")
    ids = torch.tensor([prompt_ids + candidate_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        # Cast only answer positions to FP32, not all prompt logits.
        logp = logits[0, len(prompt_ids)-1:-1].float().log_softmax(-1)
        targets = ids[0, len(prompt_ids):]
        value = logp.gather(-1, targets.unsqueeze(-1)).sum().item()
    return value


class NativeEvaluator:
    def __init__(self, registry: dict, config: dict):
        self.registry, self.config = registry, config
        self.model = self.tokenizer = None
        self.runtime = None

    def load(self, checkpoint: dict) -> dict:
        require(self.model is None, "Unload the previous checkpoint first")
        flight = preflight(self.config, any(c["kind"] == "peft" for c in self.registry["checkpoints"]))
        require(flight["status"] == "PREFLIGHT_ONLY", "; ".join(flight["reasons"]))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.config["cpu_threads"])
        if self.config["device"] == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.cuda.reset_peak_memory_stats()
        dtype = getattr(torch, self.config["dtype"])
        directory = self.registry["base"]["path"] if checkpoint["kind"] in {"base", "peft"} else checkpoint["asset"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.registry["tokenizer"]["path"], local_files_only=True, trust_remote_code=False)
        model, info = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=dtype,
            attn_implementation="eager", output_loading_info=True)
        require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "Non-strict full-model load")
        base_parameters = sum(p.numel() for p in model.parameters())
        require(base_parameters == self.registry["expected_base_parameters"], "Exact base parameter count mismatch")
        if checkpoint["kind"] == "peft":
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            model = PeftModel.from_pretrained(model, checkpoint["asset"]["path"], is_trainable=False,
                        local_files_only=True, autocast_adapter_dtype=False)
            actual = get_peft_model_state_dict(model)
            expected = load_file(str(Path(checkpoint["asset"]["path"]) / "adapter_model.safetensors"))
            require(set(actual) == set(expected), "Adapter missing/unexpected tensor keys")
            for key in expected:
                require(actual[key].shape == expected[key].shape and torch.equal(actual[key].detach().cpu(), expected[key].to(actual[key].dtype)), f"Adapter tensor not loaded exactly: {key}")
        model.requires_grad_(False)
        self.model = model.to(self.config["device"]).eval()
        require(self.tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
        self.runtime = {"versions": flight["versions"], "python": flight["python"],
                        "device": self.config["device"], "dtype": self.config["dtype"],
                        "base_parameters": base_parameters, "parameters_with_adapter": sum(p.numel() for p in model.parameters()),
                        "torch_threads": torch.get_num_threads(), "attention": "eager",
                        "prompt_format": self.config["prompt_format"], "deterministic_algorithms": True,
                        "hardware": torch.cuda.get_device_name(0) if self.config["device"] == "cuda" else platform.machine()}
        # Adapter presence legitimately changes PEFT dependency and parameter count.
        return self.runtime

    def predict(self, row: dict) -> dict:
        import torch
        from transformers import GenerationConfig
        require(self.model is not None, "No model loaded")
        cfg, tok = self.config, self.tokenizer
        if cfg["prompt_format"] == "plain":
            prompt = tok.encode(row["prompt"], add_special_tokens=False)
        else:
            require(bool(tok.chat_template), "Chat mode requires a registered template")
            prompt = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False)
        require(0 < len(prompt) <= cfg["max_input_tokens"], "Prompt over budget; truncation forbidden")
        require(len(prompt) + cfg["max_new_tokens"] <= self.model.config.max_position_embeddings, "Context limit exceeded")
        candidates = [tok.encode(" " + row[k], add_special_tokens=False) for k in ("answer", "distractor")]
        require(all(candidates) and max(map(len, candidates)) <= cfg["max_new_tokens"], "Candidate over budget")
        ids = torch.tensor([prompt], device=cfg["device"], dtype=torch.long)
        start = time.monotonic()
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"],
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                         repetition_penalty=1.0, use_cache=True)
        with torch.inference_mode():
            generated = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)[0, len(prompt):].tolist()
        stopped = bool(generated) and generated[-1] == tok.eos_token_id
        text = tok.decode(generated[:-1] if stopped else generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        scores = [continuation_logp(self.model, prompt, candidate, cfg["device"]) for candidate in candidates]
        return {"id": row["id"], "text": text, "generated_token_ids": generated,
                "prompt_token_sha256": digest(prompt), "prompt_tokens": len(prompt),
                "gold_logp": scores[0], "distractor_logp": scores[1],
                "gold_tokens": len(candidates[0]), "distractor_tokens": len(candidates[1]),
                "stop_reason": "eos" if stopped else "length", "wall_seconds": time.monotonic() - start}

    def resources(self) -> dict:
        import psutil
        resource = {"process_rss_bytes": psutil.Process().memory_info().rss}
        if self.config["device"] == "cuda":
            import torch
            resource.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return resource

    def close(self) -> None:
        self.model = self.tokenizer = None
        gc.collect()
        if self.config["device"] == "cuda" and importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def device_lock(config: dict):
    # Coordinates GD-001 jobs only. The operator must also use the host scheduler.
    return exclusive(Path(tempfile.gettempdir()) / f"rineya-gd001-{config['device']}.lock")
