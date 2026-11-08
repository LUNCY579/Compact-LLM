"""Concrete PyTorch training backend; receives already constructed local models.

No model download or generated-code execution. Teacher scoring is gold-free.
One model occupies the compute device at a time; gradients are consumed before
switching roles. Full-prefix generation is deliberately simple and costed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .contracts import Plan, digest, file_hash, require, sealed, write_new
from .data import Problem, Task, normalized
from .manifests import FORMAT, TOKENIZER_NAMES, public_row, validate_manifest
from .objectives import projected_ce, projected_kl, logprobs


def tensor_hash(state: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features, generator=generator,
                                              dtype=torch.float32) / base.in_features**0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x):
        d = nn.functional.linear(nn.functional.linear(x.float(), self.lora_A), self.lora_B)
        return self.base(x) + (self.scale * d).to(x.dtype)


def install_lora(model, plan: Plan, seed: int):
    model.requires_grad_(False)
    require(not any(isinstance(m, LoRALinear) for m in model.modules()), "adapter already installed")
    targets = [(n, m) for n, m in model.named_modules() if n.endswith((".q_proj", ".v_proj"))]
    require(bool(targets) and all(type(m) is nn.Linear for _, m in targets), "expected plain q/v linear modules")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, module in targets:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(module, plan.rank, plan.alpha, gen))
    model.eval()  # Deterministic no-dropout path; eval() does NOT disable autograd.


@dataclass(frozen=True)
class Rollout:
    key: str
    source: str
    policy_step: int
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    text: str
    terminated: bool
    token_logps: tuple[float, ...]
    seed: int


class Meter:
    def __init__(self):
        self.data = {}
        self.phase = "setup"

    def add(self, event: str, amount: int = 1):
        require(type(amount) is int and amount >= 0, "negative resource count")
        key = self.phase + "/" + event
        self.data[key] = self.data.get(key, 0) + amount

    def snapshot(self):
        return dict(self.data)


class TorchBackend:
    def __init__(self, student, teacher, tokenizer, plan: Plan, *, eos_ids: tuple[int, ...],
                 device="cpu", guard=None, evidence_kind="mechanical_fixture"):
        self.student, self.teacher, self.tokenizer, self.plan = student, teacher, tokenizer, plan
        self.device, self.guard = torch.device(device), guard or (lambda: None)
        self.eos_ids = tuple(eos_ids)
        require(bool(self.eos_ids) and all(type(x) is int for x in self.eos_ids), "explicit EOS ids required")
        self.params = {n: p for n, p in student.named_parameters() if p.requires_grad}
        require(bool(self.params) and all(n.endswith((".lora_A", ".lora_B")) for n in self.params),
                "only q/v LoRA factors may train")
        require(all(p.dtype == torch.float32 for p in self.params.values()), "FP32 adapter factors")
        require(not any(p.requires_grad for p in teacher.parameters()), "teacher must be frozen before enrollment")
        student.eval()
        teacher.eval()
        for model in (student, teacher):
            require(not model.lm_head.weight.requires_grad and getattr(model.lm_head, "bias", None) is None,
                    "frozen bias-free output projection required")
        self.vocab = student.lm_head.weight.shape[0]
        require(teacher.lm_head.weight.shape[0] == self.vocab, "vocabulary size mismatch")
        require(all(0 <= x < self.vocab for x in self.eos_ids), "EOS outside vocabulary")
        require(not ({p.data_ptr() for p in student.parameters()} & {p.data_ptr() for p in teacher.parameters()}),
                "teacher/student storage must be independent")
        self.initial = self.snapshot()
        self.version = 0
        self.meter = Meter()
        self.evidence_kind = evidence_kind
        self.prompt_catalog = {}
        self.prompt_identity = None
        self.prompt_events = []
        self.prompt_history = []
        self.prompt_renderings = {}
        self.prompt_rendered_token_ids = {}
        self.tokenizer_content_sha256 = None
        self.export_base = None

    def bind_prompt_catalog(self, tasks, identity=None, *, tokenizer_content_sha256=None):
        rows = {t.key: public_row(t) for t in tasks}
        require(len(rows) == len(tasks), "consumed prompt catalog duplicate key")
        for key in self.prompt_catalog.keys() & rows.keys():
            require(self.prompt_catalog[key] == rows[key], "consumed prompt catalog drift")
        if identity is not None:
            require(self.prompt_identity in {None, identity}, "consumed prompt registration drift")
            self.prompt_identity = identity
        self.prompt_catalog.update(rows)
        if tokenizer_content_sha256 is not None:
            require(self.tokenizer_content_sha256 in {None, tokenizer_content_sha256}, "consumed tokenizer drift")
            self.tokenizer_content_sha256 = tokenizer_content_sha256

    def bind_native_export(self, registration):
        require(self.evidence_kind == "qualified_native_hf", "native export binding requires native backend")
        require(self.prompt_identity in {None, digest(registration)}, "native export registration drift")
        self.prompt_identity = digest(registration)
        self.export_base = registration["assets"]["student"]
        self.tokenizer_content_sha256 = digest({n: h for n, h in self.export_base["files"].items()
                                              if n in TOKENIZER_NAMES})

    def record_prompt_consumption(self, problem, *, role, purpose, kind,
                                  input_tokens=0, rendered_prompt_ids=None, forward_input_ids=None):
        require(type(problem) is Problem and role in {"student", "teacher"}, "public consumption DTO/model")
        require(type(purpose) is str and purpose, "consumption purpose required")
        require(kind in {"generation", "forward", "external_forward", "teacher_bank_reuse", "rollout_consumption"},
                "known consumption kind required")
        row = self.prompt_catalog.get(problem.key)
        complete = row is not None and row["prompt"] == problem.prompt and rendered_prompt_ids is not None
        public = dict(row) if complete else {"key": problem.key, "prompt": problem.prompt,
                                             "prompt_hash": digest(normalized(problem.prompt))}
        event = {**public, "ordinal": len(self.prompt_events), "complete": complete,
                 "model": role, "purpose": purpose, "kind": kind, "phase": self.meter.phase,
                 "step": self.version, "input_tokens": input_tokens,
                 "rendered_prompt_sha256": digest(tuple(rendered_prompt_ids)) if rendered_prompt_ids is not None else None}
        if complete:
            ids = list(rendered_prompt_ids)
            cache_key = (problem.key, digest(problem.prompt))
            if cache_key not in self.prompt_renderings:
                rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": problem.prompt}],
                    tokenize=False, add_generation_prompt=True)
                if type(rendered) is not str:
                    require(self.evidence_kind == "mechanical_fixture", "native template must render actual text")
                    rendered = self.tokenizer.decode(ids, skip_special_tokens=False)
                self.prompt_renderings[cache_key] = rendered
            if self.evidence_kind == "qualified_native_hf":
                if cache_key not in self.prompt_rendered_token_ids:
                    self.prompt_rendered_token_ids[cache_key] = list(self.tokenizer.encode(
                        self.prompt_renderings[cache_key], add_special_tokens=False))
                require(self.prompt_rendered_token_ids[cache_key] == ids,
                        "actual rendered text versus consumed prompt token mismatch")
            event.update({"input_ids": ids, "raw_rendered_prompt": self.prompt_renderings[cache_key],
                          "consumed_prompt": self.prompt_renderings[cache_key],
                          "prompt_token_sha256": digest(ids), "tokenizer_content_sha256": self.tokenizer_content_sha256,
                          "rendering_history": [{"operation": "apply_chat_template", "add_generation_prompt": True,
                                                 "public_user_only": True}],
                          "truncation": {"policy": "none", "original_input_ids": ids, "max_tokens": None, "history": []}})
        if forward_input_ids is not None:
            require(kind in {"forward", "external_forward"}
                    and type(forward_input_ids) in {tuple, list}
                    and len(forward_input_ids) == input_tokens
                    and all(type(t) is int and t >= 0 for t in forward_input_ids),
                    "actual causal input IDs/count required")
            event.update({"forward_input_ids": list(forward_input_ids),
                          "causal_input_token_sha256": digest(list(forward_input_ids))})
        if kind == "external_forward":
            self.meter.add(role + "_external_forward_input_tokens", input_tokens)
        self.prompt_events.append(event)
        return event

    def _prompt_session(self, relationship="current_run"):
        events = [dict(e) for e in self.prompt_events]
        return {"identity": self.prompt_identity, "relationship": relationship,
                "event_count": len(events), "events_sha256": digest(events), "events": events,
                "meter": self.meter.snapshot()}

    def prompt_manifest(self, *, current_only=False):
        sessions = [] if current_only else list(self.prompt_history)
        sessions.append(self._prompt_session())
        events = [e for s in sessions for e in s["events"]]
        payload = {"schema": 1, "format": FORMAT, "identity": self.prompt_identity,
                   "evidence_kind": self.evidence_kind, "sessions": sessions,
                   "event_count": len(events), "complete": bool(events) and all(e["complete"] for e in events),
                   "public_prompt_hashes": sorted({e["prompt_hash"] for e in events}),
                   "contains_private_solution_text": False,
                   "contains_response_conditioning_tokens": any("forward_input_ids" in e for e in events),
                   "public_prompt_fields_contain_labels": False, "semantic_decontamination": False}
        return validate_manifest(payload, complete=False)

    def inherit_prompt_manifest(self, manifest, *, relationship):
        validate_manifest(manifest)
        require(manifest["identity"] == self.prompt_identity, "inherited manifest registration drift")
        for session in manifest["sessions"]:
            self.prompt_history.append({**session, "relationship": relationship})

    def write_prompt_manifest(self, path, *, complete=False):
        payload = self.prompt_manifest()
        validate_manifest(payload, complete=complete)
        write_new(path, sealed(payload))
        return {"path": str(Path(path).resolve()), "sha256": file_hash(path),
                "payload_sha256": digest(payload), "complete": payload["complete"],
                "event_count": payload["event_count"]}

    def activate(self, role: str):
        require(role in {"student", "teacher"}, "model role")
        self.guard()
        inactive = self.teacher if role == "student" else self.student
        model = self.student if role == "student" else self.teacher
        inactive.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(self.device)
        model.eval()
        return model

    def prompt(self, problem: Problem) -> tuple[int, ...]:
        require(type(problem) is Problem, "model input must be the public Problem DTO")
        ids = tuple(self.tokenizer.apply_chat_template([{"role": "user", "content": problem.prompt}],
                                                      tokenize=True, add_generation_prompt=True))
        require(bool(ids) and all(type(x) is int and 0 <= x < self.vocab for x in ids), "prompt token contract")
        require(len(ids) + self.plan.max_new_tokens <= self.plan.max_context_tokens,
                "context overflow; no truncation allowed")
        return ids

    def response(self, ids):
        require(0 < len(ids) <= self.plan.max_new_tokens
                and all(type(x) is int and 0 <= x < self.vocab for x in ids), "response token contract")
        require(not any(x in self.eos_ids for x in ids[:-1]), "EOS before final response token")

    def hidden(self, model, prefix, response, grad: bool, role: str, *, problem=None, purpose=None):
        require(bool(prefix) and bool(response), "empty causal sequence")
        full = tuple(prefix) + tuple(response[:-1])
        require(len(full) < self.plan.max_context_tokens, "context overflow")
        self.guard()
        self.meter.add(role + "_forward_input_tokens", len(full))
        event = self.record_prompt_consumption(problem or Problem("unbound:raw-prefix", "unbound raw token prefix"),
            role=role, purpose=purpose or ("objective_forward" if grad else "model_forward"), kind="forward",
            input_tokens=len(full), rendered_prompt_ids=prefix, forward_input_ids=full)
        event.update({"gradient": grad, "forward_input_ids": list(full), "causal_input_token_sha256": digest(full),
                      "response_tokens": len(response), "response_token_sha256": digest(tuple(response))})
        if grad:
            self.meter.add("student_gradient_forward_tokens", len(full))
        device = next(model.parameters()).device
        with torch.set_grad_enabled(grad):
            h = model.model(input_ids=torch.tensor([full], dtype=torch.long, device=device),
                            use_cache=False).last_hidden_state
            chosen = h[0, len(prefix)-1:len(prefix)-1+len(response)]
            require(chosen.shape[0] == len(response), "causal shift/response alignment")
            return chosen

    def sample(self, problem: Problem, role: str, seed: int, *, greedy=False, purpose=None) -> Rollout:
        model = self.activate(role)
        prefix = self.prompt(problem)
        self.meter.add(role + "_generation_attempts")
        purpose = purpose or self.meter.phase + "_generation"
        event = self.record_prompt_consumption(problem, role=role, purpose=purpose, kind="generation",
                                              rendered_prompt_ids=prefix)
        event.update({"seed": seed, "greedy": greedy})
        gen = torch.Generator(device=self.device).manual_seed(seed)
        response, lps = [], []
        for _ in range(self.plan.max_new_tokens):
            self.guard()
            with torch.no_grad():
                h = self.hidden(model, prefix, tuple(response) + (0,), False, role,
                                problem=problem, purpose=purpose + "_prefix")[-1:]
                lp = (model.lm_head(h).float()[0] / self.plan.sampling_temperature).log_softmax(-1)
                require(bool(torch.isfinite(lp).all()), "nonfinite generation distribution")
                token = int(lp.argmax()) if greedy else int(torch.multinomial(lp.exp(), 1, generator=gen))
            response.append(token)
            lps.append(float(lp[token]))
            self.meter.add(role + "_generated_tokens")
            if token in self.eos_ids:
                break
        ended = response[-1] in self.eos_ids
        visible = response[:-1] if ended else response
        return Rollout(problem.key, role, self.version if role == "student" else 0, prefix,
                       tuple(response), self.tokenizer.decode(visible, skip_special_tokens=False),
                       ended, tuple(lps), seed)

    def check_rollout(self, task: Task, r: Rollout, source: str):
        require(type(r) is Rollout and r.key == task.key and r.source == source, "rollout identity/source")
        require(r.policy_step == (self.version if source == "student" else 0), "stale rollout policy")
        require(r.prompt_ids == self.prompt(task.public()), "prompt drift/private teacher conditioning")
        self.response(r.response_ids)
        require(r.terminated == (r.response_ids[-1] in self.eos_ids), "termination flag drift")
        require(r.terminated or len(r.response_ids) == self.plan.max_new_tokens, "unaccounted short rollout")
        require(len(r.token_logps) == len(r.response_ids), "missing rollout likelihoods")
        visible = r.response_ids[:-1] if r.terminated else r.response_ids
        require(self.tokenizer.decode(visible, skip_special_tokens=False) == r.text, "text/token mismatch")

    def distill(self, task: Task, r: Rollout, direction: str, source: str, *, purpose=None):
        require(task.role == "train", "no held-out gradients")
        self.check_rollout(task, r, source)
        purpose = purpose or self.meter.phase + "_" + source + "_" + direction + "_kl"
        self.record_prompt_consumption(task.public(), role=source, purpose=purpose + "_rollout_consumed",
                                      kind="rollout_consumption", rendered_prompt_ids=r.prompt_ids)
        teacher = self.activate("teacher")
        with torch.no_grad():
            th = self.hidden(teacher, r.prompt_ids, r.response_ids, False, "teacher",
                             problem=task.public(), purpose=purpose + "_teacher_feedback").detach().cpu()
        student = self.activate("student")
        sh = self.hidden(student, r.prompt_ids, r.response_ids, True, "student",
                         problem=task.public(), purpose=purpose + "_student_objective")
        self.meter.add("supervised_response_tokens", len(r.response_ids))
        self.meter.add("censored_rollouts", int(not r.terminated))
        return projected_kl(sh, student.lm_head.weight, th, teacher.lm_head.weight,
                            direction, self.plan.chunk_tokens)

    def supervised(self, task: Task):
        require(task.role in {"train", "retention_train"}, "no held-out gradients")
        prefix = self.prompt(task.public())
        ids = tuple(self.tokenizer.encode(task.solution, add_special_tokens=False))
        require(not any(x in self.eos_ids for x in ids), "solution already contains EOS")
        ids += (self.eos_ids[0],)
        self.response(ids)
        model = self.activate("student")
        h = self.hidden(model, prefix, ids, True, "student", problem=task.public(),
                        purpose="independent_retention_replay" if task.role == "retention_train" else "verified_solution_sft")
        self.meter.add("replay_tokens" if task.role == "retention_train" else "supervised_response_tokens", len(ids))
        return projected_ce(h, model.lm_head.weight, torch.tensor(ids, dtype=torch.long, device=h.device),
                            self.plan.chunk_tokens)

    def snapshot(self):
        return {n: p.detach().cpu().clone() for n, p in self.params.items()}

    def restore(self, state):
        require(set(state) == set(self.params), "adapter key mismatch")
        with torch.no_grad():
            for name, p in self.params.items():
                v = state[name]
                require(v.shape == p.shape and v.dtype == p.dtype and bool(torch.isfinite(v).all()),
                        "invalid adapter tensor")
                p.copy_(v.to(p.device))
        for p in self.params.values():
            p.grad = None

    def reset(self):
        if self.prompt_events:
            self.prompt_history.append(self._prompt_session("prior_qualification_or_setup"))
        self.prompt_events = []
        self.restore(self.initial)
        self.version = 0
        self.meter = Meter()

    def frozen_hashes(self):
        return {"student_base": tensor_hash({n: p for n, p in self.student.named_parameters() if n not in self.params}),
                "teacher": tensor_hash(dict(self.teacher.named_parameters()))}

    def gradient_step(self, optimizer):
        require(all(p.grad is None for n, p in self.student.named_parameters() if n not in self.params),
                "frozen-base gradient leak")
        require(all(p.grad is None for p in self.teacher.parameters()), "teacher gradient leak")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.params.values()),
                "missing/nonfinite adapter gradient")
        norm = torch.nn.utils.clip_grad_norm_(list(self.params.values()), self.plan.max_grad_norm,
                                             error_if_nonfinite=True)
        self.guard()
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in self.params.values()), "nonfinite updated adapter")
        self.version += 1
        return float(norm)

    def save(self, directory: str | Path) -> dict:
        from safetensors.torch import save_file
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        state = self.snapshot()
        save_file(state, str(path / "adapter.safetensors"))
        receipt = {"schema": 1, "step": self.version, "format": "rdd-qv-lora-v1",
                   "adapter_sha256": file_hash(path / "adapter.safetensors"),
                   "tensor_sha256": tensor_hash(state), "rank": self.plan.rank, "alpha": self.plan.alpha,
                   "evidence_kind": self.evidence_kind}
        write_new(path / "receipt.json", sealed(receipt))
        if self.evidence_kind == "qualified_native_hf":
            require(self.export_base is not None, "exact base binding required for original PEFT save")
            from .integration import write_standard_peft_state
            write_standard_peft_state(state, self.plan, path / "peft", self.export_base,
                identity=self.prompt_identity, source_receipt=receipt,
                artifact_role="original_training_snapshot" if self.meter.phase == "training" else "qualification_disposable")
        return receipt

    def load(self, directory: str | Path):
        from safetensors.torch import load_file
        from .contracts import read_json, unseal
        path = Path(directory)
        r = unseal(read_json(path / "receipt.json"))
        require(r["format"] == "rdd-qv-lora-v1" and r["rank"] == self.plan.rank and r["alpha"] == self.plan.alpha,
                "checkpoint configuration mismatch")
        require(r["adapter_sha256"] == file_hash(path / "adapter.safetensors"), "checkpoint file drift")
        state = load_file(str(path / "adapter.safetensors"), device="cpu")
        require(r["tensor_sha256"] == tensor_hash(state), "checkpoint tensor drift")
        self.restore(state)
        self.version = r["step"]

    def update_geometry(self):
        total, tiny, squared, count = 0.0, {str(t): 0 for t in (1e-8, 1e-6, 1e-4)}, 0.0, 0
        for name, module in self.student.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            a0, b0 = self.initial[name + ".lora_A"], self.initial[name + ".lora_B"]
            a, b = module.lora_A.detach().cpu(), module.lora_B.detach().cpu()
            delta = module.scale * (b @ a - b0 @ a0)
            squared += float(delta.double().square().sum())
            total += float(delta.abs().sum())
            count += delta.numel()
            for threshold in tiny:
                tiny[threshold] += int((delta.abs() < float(threshold)).sum())
        return {"scope": "effective updates to adapted q/v matrices, NOT all frozen parameters",
                "numel": count, "l2": squared**0.5, "mean_abs": total/count,
                "fraction_below_threshold": {k: v/count for k, v in tiny.items()}}
