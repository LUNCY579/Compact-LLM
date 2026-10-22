"""Exact registered dense Qwen3, FP32/eager/greedy; local-only inference."""
from __future__ import annotations
import os
import time
from .budget import PINS, runtime_versions
from .core import digest, model_assets, read_json


class Native:
    def __init__(self, model_spec, meter):
        import torch
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        self.torch, self.meter, self.spec = torch, meter, model_spec
        libs = runtime_versions()
        if any(libs[k].split("+")[0] != v for k, v in PINS.items()):
            raise ValueError("unqualified runtime version")
        model_assets(model_spec)
        cfg = read_json(__import__('pathlib').Path(model_spec["path"]) / "config.json")
        if cfg.get("model_type") != "qwen3" or cfg.get("auto_map") or cfg.get("quantization_config"):
            raise ValueError("unsupported native model")
        if any("abstain" in k or "gate" in k for k in cfg):
            raise ValueError("custom policy cannot be normalized away")
        if model_spec["policy"] != "plain_dense_greedy_no_bias" or model_spec["runtime_dtype"] != "float32":
            raise ValueError("unregistered model policy")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec["path"], local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        self.model, info = Qwen3ForCausalLM.from_pretrained(
            model_spec["path"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
            output_loading_info=True)
        if info["unexpected_keys"] or info["mismatched_keys"] or info["error_msgs"] or set(info["missing_keys"]) - {"lm_head.weight"}:
            raise ValueError(f"strict native loading failed: {info}")
        self.model.eval().requires_grad_(False)
        if sum(p.numel() for p in self.model.parameters()) != 596049920:
            raise ValueError("wrong dense parameter inventory")
        if self.model.lm_head.weight.data_ptr() != self.model.model.embed_tokens.weight.data_ptr():
            raise ValueError("lost tied storage")
        for p in self.model.parameters():
            if p.dtype != torch.float32 or not bool(torch.isfinite(p).all()):
                raise ValueError("invalid materialized tensor")
        self.model = self.model.to("cuda:0")
        if self.tokenizer.eos_token_id != 151645:
            raise ValueError("wrong EOS")
        self.meter.guard(force=True)
        self.loading_info = info
        self.state_before = self.state_digest()

    def state_digest(self):
        import hashlib
        h = hashlib.sha256()
        for name, p in sorted(self.model.named_parameters()):
            h.update(name.encode())
            flat = p.detach().reshape(-1)
            for start in range(0, flat.numel(), 262144):
                h.update(flat[start:start+262144].cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def tokens(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
        if not ids or len(ids) > 2048:
            raise ValueError("prompt exceeds frozen 2048-token input cap; no truncation")
        return ids

    def generate(self, messages, max_new_tokens):
        torch = self.torch
        self.meter.reserve(max_new_tokens)
        ids = self.tokens(messages)
        if len(ids) + max_new_tokens > 4096:
            raise ValueError("context allowance exceeded")
        emitted, past, terminated = [], None, False
        current = torch.tensor([ids], device="cuda:0")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                self.meter.guard()
                out = self.model(input_ids=current, past_key_values=past, use_cache=True,
                                 return_dict=True, logits_to_keep=1)
                past = out.past_key_values
                if not bool(torch.isfinite(out.logits).all()):
                    raise FloatingPointError("nonfinite native logits")
                token = int(out.logits[0, -1].argmax())
                emitted.append(token)
                if token == 151645:
                    terminated = True
                    break
                current = torch.tensor([[token]], device="cuda:0")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        del out, past, current
        torch.cuda.empty_cache()
        self.meter.guard(force=True)
        return {"raw_text": self.tokenizer.decode(emitted[:-1] if terminated else emitted, skip_special_tokens=False),
                "generated_ids": emitted, "terminated": terminated, "prompt_tokens": len(ids),
                "output_tokens": len(emitted), "total_tokens": len(ids) + len(emitted),
                "generation_calls": 1, "forward_calls": len(emitted), "wall_seconds": elapsed}

    def qualify(self, public_rows):
        from .prompts import messages
        torch = self.torch
        probes = [(len(self.tokens(messages(item, "R0"))), item) for item in public_rows]
        _, longest = max(probes, key=lambda p: p[0])
        ids = torch.tensor([self.tokens(messages(longest, "R0"))], device="cuda:0")
        self.meter.guard(force=True)
        with torch.inference_mode():
            full = self.model(input_ids=ids, use_cache=False, return_dict=True).logits
            part = self.model(input_ids=ids[:, :-1], use_cache=True, return_dict=True)
            final = self.model(input_ids=ids[:, -1:], past_key_values=part.past_key_values,
                               use_cache=True, return_dict=True).logits
            torch.testing.assert_close(full[:, -1:], final, atol=2e-4, rtol=2e-4)
            error = float((full[:, -1:] - final).abs().max())
            changed = ids.clone()
            changed[0, -1] = (int(changed[0, -1]) + 1) % 151936
            perturbed = self.model(input_ids=changed, use_cache=False, return_dict=True).logits
            torch.testing.assert_close(full[:, :-1], perturbed[:, :-1], atol=0, rtol=0)
        del full, part, final, changed, perturbed, ids
        torch.cuda.empty_cache()
        probe = [{"role": "user", "content": "Reply with the digit 7."}]
        first = self.generate(probe, 16)
        second = self.generate(probe, 16)
        if first["generated_ids"] != second["generated_ids"]:
            raise ValueError("deterministic repeat failed")
        if self.state_digest() != self.state_before:
            raise ValueError("inference modified model state")
        return {"status": "PASS", "cache_max_abs": error, "causality": True,
                "repeat_ids_identical": True, "longest_direct_prompt_tokens": max(n for n, _ in probes),
                "qualification_forward_calls": 4 + first["forward_calls"] + second["forward_calls"],
                "qualification_generations": [first, second], "strict_loading": self.loading_info,
                "state_digest": self.state_before, "training_steps": 0, "efficacy": False}
