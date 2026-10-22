"""Batch-one, no-cache, full-vocabulary self-distillation mechanics.

The same sampled response is scored at different teacher/student prompt offsets.
A single model shares its frozen backbone with an FP32 CPU teacher parameter bank.
"""
from __future__ import annotations
from contextlib import nullcontext, contextmanager
import hashlib
import math
import time
import torch
from .data import messages, score
from .objectives import reference_nll, response_logits, token_kl
from .teacher import TeacherBank, tensor_digest
from .analysis import last_three_endpoint
from .plan import digest_json


class Engine:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = torch.device(config["device"])
        self.bank = TeacherBank(model)
        self.counts = {"forward_calls": 0, "processed_input_tokens": 0, "sampled_tokens": 0}
        self.eos = tokenizer.eos_token_id
        if type(self.eos) is not int or not 0 <= self.eos < model.config.vocab_size:
            raise ValueError("One valid, explicit tokenizer EOS id required")
        model.eval()  # eval disables dropout; gradients are still enabled for training

    def ids(self, row, teacher=False):
        ids = self.tokenizer.apply_chat_template(messages(row, teacher), tokenize=True,
                                                 add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
            raise ValueError("Chat template must return a nonempty list of token IDs")
        return ids

    def reference_ids(self, row):
        ids = self.tokenizer.encode(row["reference"], add_special_tokens=False)
        if not ids or ids[-1] != self.eos:
            ids = ids + [self.eos]
        if len(ids) > self.config["max_new_tokens"]:
            raise ValueError("Reference exceeds response budget; no silent truncation")
        return ids

    def check_rows(self, rows):
        for row in rows:
            if row["split"] == "train" and row["task"] != self.config["task"]:
                raise ValueError("Adapt exactly one task per run; do not mix contradictory targets")
            for privileged in ((False, True) if row["split"] == "train" else (False,)):
                if len(self.ids(row, privileged)) + self.config["max_new_tokens"] > self.config["max_sequence_tokens"]:
                    raise ValueError("Complete prompt plus response budget exceeds context limit")
            if row["split"] == "train":
                self.reference_ids(row)

    def forward(self, tokens):
        if not tokens or len(tokens) > self.config["max_sequence_tokens"]:
            raise ValueError("Sequence budget exceeded")
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in tokens):
            raise ValueError("Token outside pinned model vocabulary")
        self.counts["forward_calls"] += 1
        self.counts["processed_input_tokens"] += len(tokens)
        inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            out = self.model(input_ids=inputs, use_cache=False, return_dict=True)
        return out.logits

    @torch.no_grad()
    def generate(self, prefix, seed, greedy=False):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        response = []
        for _ in range(self.config["max_new_tokens"]):
            logits = self.forward(prefix + response)[0, -1].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite rollout logits")
            token = int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            response.append(token)
            self.counts["sampled_tokens"] += 1
            if token == self.eos:
                break
        return response, "eos" if response[-1] == self.eos else "length_limit"

    @contextmanager
    def objective(self, row, arm, step):
        prefix = self.ids(row)
        seed = int(digest_json([self.config["seed"], step, row["id"], "rollout"])[:8], 16)
        teacher_logits = None
        if arm.method == "sft":
            response, stop = self.reference_ids(row), "reference_eos"
        else:
            teacher_prefix = self.ids(row, True)
            source_prefix = prefix if arm.rollout_source == "student" else teacher_prefix
            context = self.bank.use() if arm.rollout_source == "teacher" else nullcontext()
            with context:
                response, stop = self.generate(source_prefix, seed)
            # This teacher forward occurs before creating a student autograd graph.
            with self.bank.use():
                teacher_logits = response_logits(self.forward(teacher_prefix + response), len(teacher_prefix), len(response)).detach().clone()
        with self.bank.student_graph():
            student_logits = response_logits(self.forward(prefix + response), len(prefix), len(response))
            targets = torch.tensor([response], dtype=torch.long, device=self.device)
            loss = (reference_nll(student_logits, targets) if arm.method == "sft"
                    else token_kl(student_logits, teacher_logits, arm.kl_direction).mean())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite objective")
            details = {"row_id": row["id"], "response_ids": response, "response_tokens": len(response),
                       "stop_reason": stop, "student_prefix_sha256": digest_json(prefix),
                       "student_prefix_tokens": len(prefix)}
            if teacher_logits is not None:
                with torch.no_grad():
                    qlog = teacher_logits.float().log_softmax(-1)
                    details["teacher_entropy_nats"] = float(-(qlog.exp() * qlog).sum(-1).mean())
                    details["teacher_prefix_tokens"] = len(teacher_prefix)
            yield loss, details

    @torch.no_grad()
    def evaluate(self, rows):
        records = []
        for row in rows:
            if row["split"] != "dev":
                continue
            response, stop = self.generate(self.ids(row), self.config["seed"], greedy=True)
            text = self.tokenizer.decode(response, skip_special_tokens=True)
            records.append({"id": row["id"], "parent_id": row["parent_id"], "task": row["task"],
                            "purpose": row["purpose"], "correct": score(text, row["answer"], row["scorer"]),
                            "prediction": text, "response_tokens": len(response), "stop_reason": stop})
        return records

    def qualify(self, rows, arm_list):
        """No optimizer updates. Test both rollouts/KLs and SFT on one TRAIN row."""
        self.check_rows(rows)
        before = tensor_digest(self.model.named_parameters())
        frozen_before = self.bank.frozen_digest()
        row = next(r for r in rows if r["split"] == "train")
        diagnostics = []
        for arm in arm_list:
            self.model.zero_grad(set_to_none=True)
            with self.objective(row, arm, 0) as (loss, detail):
                loss.backward()
            grads = [p.grad for p in self.bank.selected.values()]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise RuntimeError("Missing/nonfinite trainable gradient")
            diagnostics.append(dict(detail, arm_id=arm.name, loss=float(loss.detach()),
                                    gradient_l2=math.sqrt(sum(float(g.double().square().sum()) for g in grads))))
        self.model.zero_grad(set_to_none=True)
        if before != tensor_digest(self.model.named_parameters()) or frozen_before != self.bank.frozen_digest():
            raise RuntimeError("Qualification mutated model parameters or buffers")
        return {"diagnostics": diagnostics, "parameters_unchanged": True, "frozen_digest": frozen_before,
                "teacher_initial_identity": tensor_digest(self.bank.initial.items()) == tensor_digest(self.bank.teacher.items())}

    def train(self, rows, arm, on_step=None, memory_guard=None):
        self.check_rows(rows)
        frozen_before = self.bank.frozen_digest()
        baseline = self.evaluate(rows)
        train_rows = [r for r in rows if r["split"] == "train"]
        import random
        random.Random(self.config["seed"]).shuffle(train_rows)
        parameters = list(self.bank.selected.values())
        optimizer = torch.optim.AdamW(parameters, lr=self.config["learning_rate"],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        steps, checkpoints = [], []
        training_start = time.perf_counter()
        for step in range(self.config["steps"]):
            if memory_guard:
                memory_guard()
            self.model.zero_grad(set_to_none=True)
            row = train_rows[step % len(train_rows)]
            before = [p.detach().clone() for p in parameters]
            counts = dict(self.counts)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            # A fixed cosine schedule is shared by every arm and reaches zero after the final step.
            lr = self.config["learning_rate"] * .5 * (1 + math.cos(math.pi * step / self.config["steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            with self.objective(row, arm, step) as (loss, details):
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError("Invalid gradient before optimizer step")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise RuntimeError("Nonfinite updated parameters")
            if arm.method == "sd":
                self.bank.update(arm.ema_rate)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            record = dict(details, step=step + 1, learning_rate=lr, loss=float(loss.detach()),
                          preclip_gradient_norm=float(norm),
                          update_l2=math.sqrt(sum(float((p.detach() - old).double().square().sum()) for p, old in zip(parameters, before))),
                          step_seconds=time.perf_counter() - started,
                          cost={k: self.counts[k] - counts[k] for k in counts})
            steps.append(record)
            if on_step:
                on_step(record)
            if step + 1 >= self.config["steps"] - 2:
                checkpoints.append({"step": step + 1, "rows": self.evaluate(rows)})
        if frozen_before != self.bank.frozen_digest():
            raise RuntimeError("Frozen backbone/buffers changed")
        return {"evidence": "LOCAL_DEV_BEHAVIOR_ONLY", "arm_id": arm.name, "baseline": baseline,
                "checkpoints": checkpoints, "steps": steps, "endpoints": last_three_endpoint(baseline, checkpoints),
                "frozen_parameters_unchanged": True, "all_forward_counts_including_eval": self.counts,
                "total_seconds_including_eval": time.perf_counter() - training_start,
                "promotion_allowed": False}
