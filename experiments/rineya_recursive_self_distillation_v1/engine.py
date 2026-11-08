"""Fresh-rollout updates, matched parent schedules and reviewer-free evaluation."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict

import torch

from .contracts import Plan, Task, arm_spec, digest, purpose_seed, require
from .data import accept_rewrite, require_training_roles, validate_catalog, verify
from .evidence import RunStore, write_new
from .metrics import retention_stop, summary


def scheduled(tasks, split, count, seed, step):
    """Fixed cyclic parent schedule: variants do not increase parent exposure."""
    groups = {}
    for t in sorted(tasks, key=lambda x: x.key):
        if t.split == split:
            groups.setdefault(t.parent, []).append(t)
    require(len(groups) >= count, "insufficient independent parent groups for " + split)
    names = sorted(groups)
    rng = random.Random(purpose_seed(seed, 0, split, "schedule"))
    rng.shuffle(names)
    return [groups[names[(step * count + i) % len(names)]][
            (step * count + i) // len(names) % len(groups[names[(step * count + i) % len(names)]])]
            for i in range(count)]


def update(backend, optimizer, tasks: list[Task], replay: list[Task], arm: str, plan: Plan):
    spec, version = arm_spec(arm), backend.version
    require(all(t.split == "train" for t in tasks) and all(t.split == "retention_train" for t in replay),
            "gradient role violation")
    require(bool(tasks) and bool(replay), "acquisition and replay required")
    require(len({t.key for t in tasks + replay}) == len(tasks) + len(replay), "duplicate update task")
    optimizer.zero_grad(set_to_none=True)
    complete, rewrites, rejected, rollouts = [], [], Counter(), []
    if not spec["sft"]:
        # All rollouts and rewrites are generated BEFORE the single optimizer step.
        for task in tasks:
            rollout = backend.sample(task.public(), seed=purpose_seed(plan.seed, version, task.key, "rollout"))
            backend.audit_rollout(task, rollout)
            rollouts.append(rollout)
            if not rollout.terminated:
                rejected["original_truncated"] += 1
                continue
            complete.append((task, rollout))
            if spec["srcl"]:
                prefix = backend.views.rewriter(task.public(), rollout.text)
                # Overflow is an explicit rejection, never silent target truncation.
                if len(prefix) + plan.max_new_tokens > plan.max_context_tokens:
                    rejected["rewrite_context_overflow"] += 1
                    continue
                rewritten = backend.sample(task.public(), prefix=prefix,
                         seed=purpose_seed(plan.seed, version, task.key, "rewrite"))
                backend.audit_rollout(task, rewritten, student=False)
                accepted, reason = accept_rewrite(task, rollout, rewritten)
                if accepted:
                    rewrites.append((task, rewritten))
                else:
                    rejected[reason] += 1
        if not complete or (spec["teacher"] == "none" and not rewrites):
            return {"status": "NO_SIGNAL", "optimizer_step": False, "rejections": dict(rejected),
                    "attempts": len(tasks), "complete": len(complete), "accepted_rewrites": len(rewrites)}
        if len(complete) / len(tasks) < 1 - plan.max_truncation:
            return {"status": "STOPPED_TRUNCATION", "optimizer_step": False, "rejections": dict(rejected),
                    "attempts": len(tasks), "complete": len(complete), "accepted_rewrites": len(rewrites)}
    components = {"guidance": 0.0, "concise": 0.0, "sft": 0.0, "replay": 0.0}
    supervised_tokens = {"concise": 0, "sft": 0, "replay": 0, "guidance": 0}
    def backward(name, loss, coefficient):
        require(bool(torch.isfinite(loss)), "nonfinite " + name + " loss")
        components[name] += float(loss.detach()) * coefficient
        (loss * coefficient).backward()  # Release this graph BEFORE the next role swap.
    if spec["sft"]:
        for task in tasks:
            ids = backend.views.target(task.solution)
            backward("sft", backend.supervised(task, ids), 1 / len(tasks))
            supervised_tokens["sft"] += len(ids)
    elif spec["teacher"] != "none":
        for task, rollout in complete:
            maximum, mean = backend.sampled_logprob_error(rollout)
            require(maximum <= .03125 and mean <= .01, "behavior/teacher-forced logprob disagreement")
            backward("guidance", backend.guidance(task, rollout, frozen=spec["teacher"] == "frozen"),
                     plan.guidance_weight / len(complete))
            supervised_tokens["guidance"] += len(rollout.response_ids)
    total_rewrite_tokens = sum(len(r.response_ids) for _, r in rewrites)
    for task, rewrite in rewrites:
        ids = rewrite.response_ids
        backward("concise", backend.supervised(task, ids),
                 plan.concise_weight * len(ids) / total_rewrite_tokens)
        supervised_tokens["concise"] += len(ids)
    for task in replay:
        ids = backend.views.target(task.solution)
        backward("replay", backend.supervised(task, ids), plan.replay_weight / len(replay))
        supervised_tokens["replay"] += len(ids)
    norm = backend.finite_gradients()
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in backend.params.values()), "nonfinite updated adapter")
    backend.version += 1
    optimizer.zero_grad(set_to_none=True)
    return {"status": "UPDATED", "optimizer_step": True, "version_before": version,
            "version_after": backend.version, "teacher_version": 0 if spec["teacher"] == "frozen" else version,
            "attempts": len(tasks), "complete": len(complete), "accepted_rewrites": len(rewrites),
            "rejections": dict(rejected), "weighted_losses": components, "gradient_norm": norm,
            "supervised_tokens": supervised_tokens,
            "rollout_token_digest": digest([(r.key, r.policy_version, list(r.response_ids)) for r in rollouts])}


def evaluate(backend, tasks, plan):
    rows = []
    for task in sorted(tasks, key=lambda x: x.key):
        if task.split not in ("calibration", "retention_eval"):
            continue
        # Same key-specific seed and greedy decoder for every arm and checkpoint.
        response = backend.sample(task.public(), seed=purpose_seed(plan.seed, 0, task.key, "evaluation"), greedy=True)
        backend.audit_rollout(task, response)
        ok, reason = verify(task, response.text, response.terminated)
        rows.append({"key": task.key, "parent": task.parent, "source": task.source, "split": task.split,
                     "budget": plan.max_new_tokens, "correct": ok, "reason": reason,
                     "terminated": response.terminated, "tokens": len(response.response_ids)})
    require(bool(rows), "empty evaluation enrollment")
    return rows


def train(backend, tasks, plan, arm, output, binding, *, max_updates=1, resume=False):
    require(type(max_updates) is int and max_updates > 0, "explicit positive update ceiling required")
    arm_spec(arm)
    validate_catalog(tasks)
    require_training_roles(tasks)
    require(binding["arm"] == arm and binding["plan"] == plan.fingerprint
            and binding["catalog"] == digest([asdict(t) for t in tasks]), "training binding drift")
    store = RunStore(output, binding, plan, resume=resume)
    previous_meter = backend.meter
    def meter(kind, tokens, generated):
        store.meter(kind, tokens, generated)
        if previous_meter:
            previous_meter(kind, tokens, generated)
    backend.meter = meter
    optimizer = torch.optim.AdamW(list(backend.params.values()), lr=plan.learning_rate,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    try:
        if resume:
            store.load(backend, optimizer)
        else:
            store.begin("baseline_evaluation")
            baseline = evaluate(backend, tasks, plan)
            store.state["evaluations"].append({"step": 0, "rows": baseline})
            store.save(backend, optimizer, status="PAUSED")
        target = min(plan.rounds, store.state["step"] + max_updates)
        status = "PAUSED"
        while store.state["step"] < target:
            step = store.state["step"]
            acquisition = scheduled(tasks, "train", plan.batch_tasks, plan.seed, step)
            replay = scheduled(tasks, "retention_train", plan.replay_tasks, plan.seed, step)
            store.begin("fresh_rollout_update")
            result = update(backend, optimizer, acquisition, replay, arm, plan)
            store.state["history"].append(result)
            if not result["optimizer_step"]:
                status = result["status"]
                store.save(backend, optimizer, status=status)
                break
            store.state["step"] += 1
            store.state["optimizer_steps"] += 1
            require(store.state["step"] == backend.version, "policy/controller version mismatch")
            if backend.version % plan.eval_every == 0 or backend.version == plan.rounds:
                rows = evaluate(backend, tasks, plan)
                stopped, details = retention_stop(store.state["evaluations"][0]["rows"], rows,
                                                  plan.max_retention_drop, plan.max_truncation)
                store.state["evaluations"].append({"step": backend.version, "rows": rows, "retention": details})
                if stopped:
                    status = "STOPPED_RETENTION"
                elif summary(rows)["truncation_rate"] > plan.max_truncation:
                    status = "STOPPED_TRUNCATION"
            if status not in ("STOPPED_RETENTION", "STOPPED_TRUNCATION"):
                status = "COMPLETE" if backend.version == plan.rounds else "PAUSED"
            store.save(backend, optimizer, status=status)
            if status in ("STOPPED_RETENTION", "STOPPED_TRUNCATION", "COMPLETE"):
                break
        return {"status": status, "steps": store.state["step"], "native_efficacy_claim": False,
                "last_evaluated_step": store.state["evaluations"][-1]["step"],
                "last_summary": summary(store.state["evaluations"][-1]["rows"]),
                "costs": {k: store.state[k] for k in ("generated_tokens", "processed_tokens", "model_calls",
                                                     "optimizer_steps", "wall_seconds")}}
    except BaseException as exc:
        store.fail(exc)
        raise
    finally:
        backend.meter = previous_meter
        store.close()
