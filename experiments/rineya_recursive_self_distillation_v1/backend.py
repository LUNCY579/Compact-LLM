"""One frozen-base model; role snapshots contain only trainable adapter tensors."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

import torch

from .contracts import Plan, Problem, Rollout, Task, require
from .objectives import guidance_kl, token_ce
from .prompts import Views


def tensor_hash(values: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TorchBackend:
    """The model exposes .model(...).last_hidden_state and a bias-free frozen lm_head.

    A trusted local loader must construct the model. This class never downloads,
    loads checkpoints, dispatches tools, or executes generated text.
    """
    def __init__(self, model, tokenizer, plan: Plan, *, eos_ids=(151645, 151643), meter=None):
        self.model, self.plan = model, plan
        self.model.eval()
        self.views = Views(tokenizer, tuple(eos_ids), plan.max_context_tokens)
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        require(bool(self.params) and all(n.endswith(("lora_A", "lora_B")) for n in self.params),
                "only fixed-scope LoRA factors may be trainable")
        require(all(p.dtype == torch.float32 for p in self.params.values()), "FP32 adapter contract")
        require(not model.lm_head.weight.requires_grad and getattr(model.lm_head, "bias", None) is None,
                "frozen, bias-free output projection required")
        self.device = next(model.parameters()).device
        require(all(p.device == self.device for p in model.parameters()), "single-device model required")
        self.initial = self.snapshot()
        self.initial_digest = tensor_hash(self.initial)
        self.version = 0
        self.meter = meter

    def charge(self, kind: str, tokens: int, *, grad: bool = False, generated: int = 0):
        if self.meter:
            self.meter(kind, tokens * (4 if grad else 1), generated)

    def snapshot(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.params.items()}

    def restore(self, state: dict) -> None:
        require(set(state) == set(self.params), "adapter key drift")
        require(all(v.shape == self.params[n].shape and v.dtype == self.params[n].dtype
                    and bool(torch.isfinite(v).all()) for n, v in state.items()), "adapter shape/dtype/nonfinite")
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(state[n])

    @contextmanager
    def teacher_role(self, frozen: bool):
        # Must precede creation/backward of the student graph: copy_ bumps versions.
        saved = self.snapshot() if frozen else None
        try:
            if frozen:
                self.restore(self.initial)
            with torch.no_grad():
                yield
        finally:
            if frozen:
                self.restore(saved)

    def hidden(self, prefix, response, *, grad: bool):
        self.views.fit(prefix, len(response))
        full = prefix + response[:-1]
        self.charge("hidden", len(full), grad=grad)
        with torch.set_grad_enabled(grad):
            ids = torch.tensor([full], dtype=torch.long, device=self.device)
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
            selected = hidden[0, len(prefix) - 1:len(prefix) - 1 + len(response)]
            require(selected.shape[0] == len(response), "causal-position alignment")
            return selected

    def sample(self, problem: Problem, *, seed: int, prefix=None, max_new=None, greedy=False) -> Rollout:
        require(type(problem) is Problem, "sample requires public DTO")
        cap = self.plan.max_new_tokens if max_new is None else max_new
        require(type(cap) is int and 0 < cap <= self.plan.max_new_tokens, "generation cap")
        prefix = self.views.student(problem) if prefix is None else tuple(prefix)
        self.views.fit(prefix, cap)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        response, logps = [], []
        for _ in range(cap):
            self.charge("generated", 0, generated=1)
            ids = tuple(response) + (0,)  # The dummy next token is excluded by hidden().
            with torch.no_grad():
                h = self.hidden(prefix, ids, grad=False)[-1:]
                logits = self.model.lm_head(h).float()[0]
                require(bool(torch.isfinite(logits).all()), "nonfinite generation logits")
                lp = logits.log_softmax(-1)
                token = int(lp.argmax()) if greedy else int(torch.multinomial(lp.exp(), 1, generator=gen))
            response.append(token)
            logps.append(float(lp[token]))
            if token in self.views.eos_ids:
                break
        terminated = response[-1] in self.views.eos_ids
        visible = response[:-1] if terminated else response
        text = self.views.tokenizer.decode(visible, skip_special_tokens=False)
        return Rollout(problem.key, prefix, tuple(response), text, tuple(logps), self.version, terminated)

    def audit_rollout(self, task: Task, rollout: Rollout, *, student=True) -> None:
        require(rollout.key == task.key and rollout.policy_version == self.version, "stale/mismatched rollout")
        self.views.response(rollout.response_ids, rollout.terminated)
        visible = rollout.response_ids[:-1] if rollout.terminated else rollout.response_ids
        require(self.views.tokenizer.decode(visible, skip_special_tokens=False) == rollout.text, "text/token mismatch")
        if student:
            require(rollout.prompt_ids == self.views.student(task.public()), "student prompt drift or private context")

    def sampled_logprob_error(self, rollout: Rollout) -> tuple[float, float]:
        with torch.no_grad():
            h = self.hidden(rollout.prompt_ids, rollout.response_ids, grad=False)
            chunks = []
            for start in range(0, len(h), self.plan.chunk_tokens):
                end = min(len(h), start + self.plan.chunk_tokens)
                lp = self.model.lm_head(h[start:end]).float().log_softmax(-1)
                ids = torch.tensor(rollout.response_ids[start:end], device=self.device)
                chunks.append(lp.gather(1, ids[:, None]).squeeze(1).cpu())
            error = (torch.cat(chunks) - torch.tensor(rollout.logps)).abs()
            return float(error.max()), float(error.mean())

    def guidance(self, task: Task, rollout: Rollout, *, frozen: bool):
        require(task.split == "train" and rollout.terminated, "guidance requires completed TRAIN rollout")
        self.audit_rollout(task, rollout)
        with self.teacher_role(frozen):
            q = self.hidden(self.views.privileged(task), rollout.response_ids, grad=False).detach()
        p = self.hidden(rollout.prompt_ids, rollout.response_ids, grad=True)
        return guidance_kl(p, q, self.model.lm_head.weight, self.plan.chunk_tokens)

    def supervised(self, task: Task, response_ids: tuple[int, ...]):
        require(task.split in ("train", "retention_train"), "held-out gradients forbidden")
        self.views.response(response_ids, True)
        require(len(response_ids) <= self.plan.max_new_tokens, "supervised response exceeds frozen budget")
        h = self.hidden(self.views.student(task.public()), response_ids, grad=True)
        return token_ce(h, self.model.lm_head.weight,
                        torch.tensor(response_ids, dtype=torch.long, device=self.device), self.plan.chunk_tokens)

    def finite_gradients(self):
        require(all(p.grad is None for n, p in self.model.named_parameters() if n not in self.params),
                "gradient leaked to frozen base")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.params.values()),
                "missing/nonfinite adapter gradients")
        return float(torch.nn.utils.clip_grad_norm_(list(self.params.values()), self.plan.max_grad_norm,
                                                   error_if_nonfinite=True))
