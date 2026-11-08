"""Interchangeable CPU fixture and real single-device causal-LM runtimes.

The HF runtime is deliberately single-turn, exact-verifier and local-files-only.
External multi-turn MiMo runners must implement the same controller boundary,
with their own audited assistant-token masks; they are not silently emulated.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .objectives import advantages, policy_loss
from .schema import Episode, Plan, Prompt, Task, digest, require


class Backend(Protocol):
    capabilities: frozenset[str]
    identity: dict
    version: int

    def collect(self, prompt: Prompt, count: int, cap: int, seed: int) -> list[Episode]: ...
    def update(self, groups: list[tuple[list[Episode], list[float]]], replay: list[tuple[Task, str]]) -> dict: ...
    def sft(self, demonstrations: list[tuple[Task, str]]) -> dict: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...


class DemoBackend:
    """Synthetic Bernoulli policy for controller tests, NEVER model efficacy evidence."""
    capabilities = frozenset({"text"})
    identity = {"kind": "synthetic", "implementation": "bernoulli-arithmetic-v1"}

    def __init__(self, plan: Plan):
        self.plan, self.version, self.logit = plan, 0, -.8

    def collect(self, prompt: Prompt, count: int, cap: int, seed: int) -> list[Episode]:
        numbers = re.search(r"(\d+)\s*\+\s*(\d+)", prompt.text)
        require(numbers is not None, "demo only accepts arithmetic fixture prompts")
        answer = int(numbers[1]) + int(numbers[2])
        rng = random.Random(seed)
        p = 1 / (1 + math.exp(-self.logit))
        outputs = []
        for _ in range(count):
            success = rng.random() < p
            text = str(answer if success else answer + 1)
            response = tuple(text.encode()) + (0,)
            require(len(response) <= cap, "demo token cap too small")
            outputs.append(Episode(prompt.key, text, tuple(prompt.text.encode()), response,
                                   (math.log(p if success else 1-p) / len(response),) * len(response), self.version))
        return outputs

    def update(self, groups, replay) -> dict:
        gradient = 0.0
        for _, rewards in groups:
            a = advantages(rewards)
            gradient += sum(x * r for x, r in zip(a, rewards)) / len(a)
        self.logit = min(3.0, self.logit + .3 * gradient / max(1, len(groups)))
        self.version += 1
        return {"synthetic_update": True, "groups": len(groups), "replay_rows": len(replay)}

    def sft(self, demonstrations) -> dict:
        require(bool(demonstrations), "empty SFT batch")
        self.logit = min(3.0, self.logit + .15)
        self.version += 1
        return {"synthetic_sft": True, "rows": len(demonstrations)}

    def save(self, directory: Path) -> None:
        (directory / "backend.json").write_text(json.dumps({"version": self.version, "logit": self.logit}))

    def load(self, directory: Path) -> None:
        value = json.loads((directory / "backend.json").read_text())
        self.version, self.logit = value["version"], value["logit"]


class TorchBackend:
    """Actual GRPO + verifier-approved CE. Frozen reference is kept on CPU.

    Microbatch size one; fixed temperature 1, top_k=0, top_p=1; dropout disabled.
    The sampled policy and teacher-forced old-logprob distribution therefore match.
    Only declared parameter names are trained. Capacity does not change mid-run.
    """
    capabilities = frozenset({"text"})

    def __init__(self, model, tokenizer, plan: Plan, *, device: str, identity: dict):
        import torch
        self.torch, self.model, self.tokenizer, self.plan = torch, model, tokenizer, plan
        self.device, self.identity, self.version = device, identity, 0
        require(tokenizer.eos_token_id is not None, "EOS must be explicit")
        require(tokenizer.pad_token_id is not None, "PAD must be explicit")
        self.names = []
        for name, parameter in model.named_parameters():
            enabled = any(re.fullmatch(pattern, name) for pattern in plan.trainable_patterns)
            parameter.requires_grad_(enabled)
            if enabled:
                self.names.append(name)
        require(bool(self.names), "trainable patterns matched no parameters")
        total = sum(p.numel() for p in model.parameters())
        trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
        require(trained / total <= .20, "more than 20% trainable; use a separately qualified full-weight runtime")
        self.reference = copy.deepcopy(model).cpu().eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        self.model.to(device).eval()
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=plan.learning_rate, weight_decay=0.0)
        self.identity = {**identity, "trainable_names": self.names, "trainable_count": trained,
                         "torch": torch.__version__, "device": device,
                         "reference_device": "cpu", "objective": "sequence_mean_grpo_v1"}

    def _prompt_ids(self, prompt: Prompt) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt.text}],
                                                      tokenize=True, add_generation_prompt=True)
        else:
            ids = self.tokenizer.encode(prompt.text, add_special_tokens=True)
        require(0 < len(ids) <= self.plan.limits.max_prompt_tokens, "prompt exceeds limit; no silent truncation")
        return list(ids)

    def _logps(self, model, prompt: tuple[int, ...], response: tuple[int, ...], device: str):
        torch = self.torch
        ids = torch.tensor([(*prompt, *response)], device=device, dtype=torch.long)
        require(ids.shape[1] <= self.plan.limits.max_context_tokens, "context overflow")
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        scores = logits[0, len(prompt)-1:-1].float().log_softmax(-1)
        target = torch.tensor(response, device=device).unsqueeze(1)
        return scores.gather(-1, target).squeeze(-1)

    def collect(self, prompt: Prompt, count: int, cap: int, seed: int) -> list[Episode]:
        torch = self.torch
        ids = self._prompt_ids(prompt)
        require(len(ids) + cap <= self.plan.limits.max_context_tokens, "context cap exceeded")
        outputs = []
        devices = [torch.cuda.current_device()] if self.device.startswith("cuda") else []
        self.model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            for _ in range(count):
                inp = torch.tensor([ids], device=self.device)
                generated = self.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                                do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                                max_new_tokens=cap, pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                use_cache=True, return_dict_in_generate=True, output_scores=True,
                                                repetition_penalty=1.0, no_repeat_ngram_size=0,
                                                min_length=0, min_new_tokens=0, forced_eos_token_id=None,
                                                forced_bos_token_id=None)
                response = tuple(generated.sequences[0, len(ids):].tolist())
                require(bool(response) and len(response) <= cap, "invalid generation length")
                eos = self.tokenizer.eos_token_id
                if eos in response:
                    response = response[:response.index(eos) + 1]
                require(len(generated.scores) >= len(response), "sampler did not return behavior scores")
                logps = tuple(float(score[0].float().log_softmax(-1)[token])
                              for score, token in zip(generated.scores, response))
                recomputed = self._logps(self.model, tuple(ids), response, self.device)
                mismatch = (recomputed - torch.tensor(logps, device=self.device)).abs().max()
                require(float(mismatch) <= 1e-3, "sampler/trainer distribution mismatch during qualification")
                text = self.tokenizer.decode(response, skip_special_tokens=True)
                outputs.append(Episode(prompt.key, text, tuple(ids), response, logps, self.version,
                                       terminated=response[-1] == eos))
                del generated, recomputed
        return outputs

    def _ce(self, task: Task, answer: str):
        ids = tuple(self._prompt_ids(task.public()))
        response = tuple(self.tokenizer.encode(answer, add_special_tokens=False))
        eos = self.tokenizer.eos_token_id
        if not response or response[-1] != eos:
            response += (eos,)
        require(len(ids) + len(response) <= self.plan.limits.max_context_tokens, "SFT context overflow")
        return -self._logps(self.model, ids, response, self.device).mean()

    def _step(self):
        torch = self.torch
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), "nonfinite gradient; step rejected")
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0,
                                      error_if_nonfinite=True)
        self.optimizer.step()
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                require(bool(torch.isfinite(parameter).all()), "nonfinite parameter after step; recover prior checkpoint")
        self.version += 1

    def update(self, groups, replay) -> dict:
        torch = self.torch
        require(bool(groups), "no policy groups")
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        count = sum(len(episodes) for episodes, _ in groups)
        total_loss = 0.0
        for episodes, rewards in groups:
            require(len(episodes) == len(rewards), "reward/group mismatch")
            for episode, advantage in zip(episodes, advantages(rewards)):
                require(episode.policy_version == self.version, "stale/off-policy episode")
                require(episode.terminated and not episode.infrastructure_error, "incomplete episode")
                current = self._logps(self.model, episode.prompt_ids, episode.response_ids, self.device)
                with torch.no_grad():
                    reference = self._logps(self.reference, episode.prompt_ids, episode.response_ids, "cpu").to(self.device)
                old = torch.tensor(episode.old_logps, device=self.device, dtype=current.dtype)
                require(float((current.detach() - old).abs().max()) <= 1e-3,
                        "sampler/trainer logprob mismatch before on-policy update")
                loss = policy_loss(current[None], old[None], reference[None],
                                   torch.tensor([advantage], device=self.device, dtype=current.dtype),
                                   torch.ones_like(current[None], dtype=torch.bool),
                                   epsilon=self.plan.clip_epsilon, beta=self.plan.kl_beta) / count
                require(bool(torch.isfinite(loss)), "nonfinite objective")
                loss.backward()
                total_loss += float(loss.detach())
        for task, answer in replay:
            loss = self.plan.replay_ce_weight * self._ce(task, answer) / len(replay)
            loss.backward()
            total_loss += float(loss.detach())
        self._step()
        return {"loss": total_loss, "episodes": count, "replay_rows": len(replay)}

    def sft(self, demonstrations) -> dict:
        require(bool(demonstrations), "empty SFT batch")
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for task, answer in demonstrations:
            loss = self._ce(task, answer) / len(demonstrations)
            require(bool(self.torch.isfinite(loss)), "nonfinite SFT loss")
            loss.backward()
            total += float(loss.detach())
        self._step()
        return {"sft_loss": total, "rows": len(demonstrations)}

    def save(self, directory: Path) -> None:
        torch = self.torch
        parameters = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if n in self.names}
        torch.save({"parameters": parameters, "optimizer": self.optimizer.state_dict(),
                    "version": self.version, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if self.device.startswith("cuda") else []},
                   directory / "backend.pt")

    def load(self, directory: Path) -> None:
        torch = self.torch
        value = torch.load(directory / "backend.pt", map_location="cpu", weights_only=True)
        require(set(value["parameters"]) == set(self.names), "checkpoint capacity mismatch")
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in value["parameters"]:
                    parameter.copy_(value["parameters"][name].to(parameter.device))
        self.optimizer.load_state_dict(value["optimizer"])
        torch.set_rng_state(value["rng"])
        if value["cuda_rng"]:
            torch.cuda.set_rng_state_all(value["cuda_rng"])
        self.version = value["version"]


def local_hf_backend(model_dir: Path, plan: Plan, *, device: str, allow_model_load: bool) -> TorchBackend:
    require(allow_model_load, "explicit --allow-model-load required")
    from .evidence import file_hash
    require(model_dir.is_dir(), "local model directory required")
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and ".git" not in p.parts)
    require(any(p.suffix == ".safetensors" for p in files), "local safetensors weights required")
    fingerprints = {str(p.relative_to(model_dir)): file_hash(p) for p in files}
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(plan.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True,
                                                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    require(plan.limits.max_context_tokens <= getattr(model.config, "max_position_embeddings", 0),
            "configured context exceeds model capacity")
    require(fingerprints == {str(p.relative_to(model_dir)): file_hash(p) for p in files},
            "model/tokenizer files changed during loading")
    # Fixed low-rank capacity prevents a confounded data+capacity curriculum.
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    identity = {"kind": "native_hf", "model_files": fingerprints,
                "model_fingerprint": digest(fingerprints), "transformers": transformers.__version__,
                "peft": peft.__version__,
                "plan": digest(asdict(plan))}
    return TorchBackend(model, tokenizer, plan, device=device, identity=identity)
