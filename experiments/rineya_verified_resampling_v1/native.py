"""Exact Titan loader and inference-only qualification. No optimizer exists here."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time

from .contracts import (PACKAGE, digest, directory_bytes, encoded, file_sha, read_json, require,
                        source_identity, write_jsonl, write_new)
from .torch_sampler import TorchSampler

WEIGHT_SHA = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
ASSETS = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
VERSIONS = {"torch": "2.10.0+cu126", "transformers": "4.57.1", "safetensors": "0.7.0", "tokenizers": "0.22.2"}
TEMPLATE_SHA = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
EOS = (151645, 151643)


class NativeBackend:
    def __init__(self, model_dir: Path, output: Path, config: dict, *, allow_model_load: bool = False):
        require(allow_model_load, "explicit model-load permission required")
        require(os.name == "nt", "native Titan path is registered for Windows only")
        self.config, self.output, self.started = config, output, time.monotonic()
        self.events = output / "native_events.jsonl"
        self.source = source_identity(native=True)
        self.event("runtime_import_attempt")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
        actual = {key: importlib.metadata.version(key) for key in VERSIONS}
        require(actual == VERSIONS, "unregistered runtime; use the pinned Titan interpreter")
        import psutil
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        self.torch, self.psutil = torch, psutil
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA unavailable")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        self.guard(admission=True)
        model_dir = model_dir.resolve()
        source_files = {"model.safetensors": WEIGHT_SHA, **ASSETS}
        for name, expected in source_files.items():
            path = model_dir / name
            require(path.is_file() and not path.is_symlink(), "missing or linked source asset: " + name)
            self.event("asset_hash_attempt", name=name)
            require(file_sha(path) == expected, "source asset drift: " + name)
            self.event("asset_hash_returned", name=name)
        # Bind extra tokenizer metadata too; never execute remote/model-supplied Python.
        extras = {p.name: file_sha(p) for p in sorted(model_dir.iterdir())
                  if p.is_file() and p.suffix in {".json", ".txt", ".model"} and p.name not in source_files}
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer.eos_token_id == 151645 and tokenizer.eos_token == "<|im_end|>", "EOS drift")
        require(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == TEMPLATE_SHA, "template drift")
        self.tokenizer = tokenizer
        cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
        cfg._attn_implementation = "eager"
        expected_geometry = {"num_hidden_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
                             "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                             "vocab_size": 151936, "rope_theta": 1000000, "tie_word_embeddings": True}
        require(all(getattr(cfg, k) == v for k, v in expected_geometry.items()), "Titan geometry mismatch")
        self.event("model_construct_attempt")
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = Qwen3ForCausalLM(cfg)
        finally:
            torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        self.event("model_construct_returned")
        self.guard()
        states = model.state_dict()
        self.event("checkpoint_load_attempt")
        with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as weights:
            keys = set(weights.keys())
            require(len(keys) == 310 and set(states)-keys == {"lm_head.weight"} and not keys-set(states),
                    "strict checkpoint key mismatch")
            with torch.no_grad():
                for name in sorted(keys):
                    source = weights.get_tensor(name)
                    require(source.shape == states[name].shape and source.dtype == states[name].dtype, "tensor shape/dtype drift")
                    self.event("tensor_copy_attempt", name=name)
                    states[name].copy_(source)
                    torch.cuda.synchronize()
                    require(torch.equal(states[name].cpu(), source), "loaded tensor differs")
                    self.event("tensor_copy_returned", name=name)
                    del source
            del states
        require(model.lm_head.weight is model.model.embed_tokens.weight, "embedding tie lost")
        require(sum(p.numel() for p in model.parameters()) == 596049920, "parameter count mismatch")
        require(not any(p.requires_grad for p in model.parameters()), "unexpected trainable parameter")
        self.event("checkpoint_load_returned")
        self.model = model
        self.sampler = TorchSampler(model, torch, "cuda:0", self.forward_guard)
        gpu = torch.cuda.get_device_properties(0)
        self.identity = {"name": "Titan_596M_BF16_eager", "evidence": "NATIVE_MODEL_EXECUTION",
                         "assets": source_files, "extra_assets": extras, "template_sha256": TEMPLATE_SHA,
                         "runtime": {**actual, "python": sys.version, "psutil": importlib.metadata.version("psutil"),
                                     "cuda": torch.version.cuda, "platform": platform.platform()},
                         "host": digest([platform.node(), str(getattr(gpu, "uuid", "unavailable"))]),
                         "gpu": gpu.name, "total_vram": gpu.total_memory,
                         "sampler": "manual_temperature_softmax", "cache": config["cache"]}
        self.guard()
        write_new(output / "model_identity.json", self.identity)

    def event(self, event: str, **values) -> None:
        require(not self.events.exists() or self.events.stat().st_size < 16*1024**2, "native log limit")
        with self.events.open("ab") as f:
            f.write(encoded({"event": event, "seconds": time.monotonic()-self.started, **values}) + b"\n")
            f.flush()
            os.fsync(f.fileno())

    def guard(self, admission: bool = False) -> None:
        # Segment output is run/segments/NNNN; include all prior journals and logs.
        require(directory_bytes(self.output.parents[1]) < self.config["max_output_bytes"],
                "aggregate output storage budget")
        t, ps = self.torch, self.psutil
        t.cuda.synchronize()
        free, _ = t.cuda.mem_get_info()
        snapshot = {"host_free_gib": ps.virtual_memory().available / 2**30,
                    "gpu_free_gib": free / 2**30, "disk_free_gib": ps.disk_usage(str(self.output)).free / 2**30,
                    "peak_gpu_allocated": t.cuda.max_memory_allocated(),
                    "gpu_reserved": t.cuda.memory_reserved(), "admission": admission}
        self.event("resource", **snapshot)
        require(snapshot["host_free_gib"] >= (7.4 if admission else 6.0), "HOST_RAM_GUARD")
        require(snapshot["gpu_free_gib"] >= (4.0 if admission else 0.5), "GPU_MEMORY_GUARD")
        require(snapshot["disk_free_gib"] >= 4.0, "DISK_GUARD")
        require(time.monotonic()-self.started < self.config["segment_seconds"], "WALL_CEILING")

    def forward_guard(self) -> None:
        self.guard()
        self.event("forward_boundary")  # Boundary check, not proof the next CUDA operation returned.

    def prompt_ids(self, task: dict) -> list[int]:
        kw = dict(add_generation_prompt=True, enable_thinking=False)
        text = self.tokenizer.apply_chat_template(task["messages"], tokenize=False, **kw)
        ids = self.tokenizer.apply_chat_template(task["messages"], tokenize=True, **kw)
        require(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "non-thinking template mismatch")
        require(ids == self.tokenizer.encode(text, add_special_tokens=False), "prompt tokenization mismatch")
        require(0 < len(ids) <= self.config["max_prompt_tokens"], "prompt exceeds frozen cap; no truncation")
        return ids

    def bind_prompts(self, public: list[dict]) -> tuple[list[dict], str]:
        rows = [{"id": row["id"], "ids": self.prompt_ids(row)} for row in public]
        return rows, digest(rows)

    def generate(self, task: dict, seed: int, greedy: bool, config: dict, emit_token) -> dict:
        ids = self.prompt_ids(task)
        result = self.sampler.sample(ids, seed=seed, temperature=config["temperature"],
                                     max_tokens=config["max_new_tokens"], greedy=greedy,
                                     cache=config["cache"], eos=EOS, emit=emit_token)
        tokens = result["token_ids"][:-1] if result["stop"] == "eos" else result["token_ids"]
        text = self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return {"text": text, "stop": result["stop"], "prompt_tokens": len(ids),
                "seconds": result["seconds"], "forward_positions": result["forward_positions"]}

    def qualify(self, study: dict, public: list[dict], out: Path) -> dict:
        rows, token_binding = self.bind_prompts(public)
        write_jsonl(out / "tokenized_public.jsonl", rows)
        task = {"messages": [{"role": "user", "content": 'Mechanical qualification only. Return {"answer":12}.'}]}
        ids = self.prompt_ids(task)
        checks = self.sampler.qualify(ids, self.config, EOS)
        # Probe the maximum configured context without generating any DEV answer.
        filler = self.tokenizer.encode(" neutral", add_special_tokens=False)
        require(filler and not any(tok in EOS for tok in filler), "invalid neutral fixture tokens")
        count = self.config["max_context_tokens"]-1
        maximum = (filler * (count//len(filler)+1))[:count]
        self.sampler.logits(maximum)
        # Exercise cache allocation at the maximum envelope as well, including a one-token continuation.
        self.sampler.sample(maximum, seed=self.config["seed"], temperature=self.config["temperature"],
                            max_tokens=1, greedy=True, cache=self.config["cache"], eos=EOS, emit=lambda *_: None)
        self.guard()
        checks["maximum_prefix_forward_tokens"] = count
        checks["configured_total_sequence_envelope"] = self.config["max_context_tokens"]
        checks["probe_note"] = "prefix n-1 plus one emitted token; final emitted token is not fed back"
        receipt = {"status": "PASS_NATIVE_INFERENCE_ONLY", "study_id": study["study_id"],
                   "code_digest": self.source["code_digest"], "backend": self.identity,
                   "public_token_ids_digest": token_binding, "checks": checks,
                   "optimizer_steps": 0, "teacher_calls": 0, "efficacy": "UNMEASURED",
                   "seconds_including_load": time.monotonic()-self.started}
        write_new(out / "qualification.json", receipt)
        return receipt
