"""Replay-verified, per-action SFT; private review and observations are not targets."""
from __future__ import annotations

from dataclasses import asdict
from .contracts import (EMPTY, HARNESS_HASH, MODES, PROCEDURAL, TERMINALS, Action,
                        Limits, Reference, Review, ReviewRequest, Task, digest,
                        render, require, source_hash, validate_catalog)
from .environment import Environment


def replay(record: dict, task: Task, references: tuple[Reference, ...] = (EMPTY, PROCEDURAL)):
    """Recompute environment observations and grade; never trust imported success."""
    obj = dict(record)
    trace_hash = obj.pop("trace_hash")
    require(trace_hash == digest(obj), "trace integrity mismatch")
    require(record["source_hash"] == source_hash(), "collector/compiler source drift")
    require(record["schema"] == 1 and record["harness_hash"] == HARNESS_HASH, "harness/schema drift")
    require(record["task_key"] == task.key and record["task_hash"] == task.fingerprint, "task drift")
    require(record["mode"] in MODES and record["status"] in TERMINALS, "unknown mode/status")
    require(type(record["success"]) is bool and type(record["evaluation"]) is bool, "invalid flags")
    limits = Limits(**record["limits"])
    require(type(record["attempt"]) is int and record["attempt"] >= 0
            and type(record["seed"]) is int and record["seed"] >= 0, "invalid attempt/seed")
    expected_session = digest({"task": task.fingerprint, "attempt": record["attempt"],
                               "seed": record["seed"], "mode": record["mode"],
                               "student": record["student"], "h": HARNESS_HASH,
                               "reviewer": record["reviewer"], "reference": record["reference_hash"],
                               "source": record["source_hash"], "limits": record["limits"]})
    require(record["session"] == expected_session, "session identity drift")
    refs = {r.fingerprint: r for r in references}
    require(record["reference_hash"] in refs, "unregistered reference")
    ref = refs[record["reference_hash"]]
    reviewing = record["mode"].startswith("review_")
    require((record["mode"] == "review_k") == bool(ref.rules), "reference/arm mismatch")
    require(bool(record["reviewer"]) == reviewing, "reviewer identity mismatch")
    if record["evaluation"]:
        require(task.split in {"calibration", "retention_eval"}
                and record["mode"] == "student_only" and not reviewing, "invalid deployment trace")
    else:
        require(task.split == "train", "training trace has non-training task")
    require(type(record["steps"]) is list and len(record["steps"]) <= limits.max_steps,
            "too many/invalid steps")
    env, rows, pairs = Environment(task), [], []
    for index, step in enumerate(record["steps"]):
        require(env.final is None, "post-terminal step")
        view = env.view()
        require(step["view_hash"] == view.fingerprint, "trajectory context drift")
        require(len(render(view)) <= limits.max_context_chars, "context exceeded")
        require(type(step["proposal"]) is str
                and len(step["proposal"]) <= limits.max_response_chars, "proposal cap/type")
        request = ReviewRequest(view, step["proposal"], ref, record["session"], index)
        if reviewing:
            require(type(step["review"]) is dict, "missing review")
            review = Review(**step["review"])
            accepted = review.accept(request)
        else:
            require(step["review"] is None, "unexpected review")
            review = None
            accepted = step["proposal"]
            Action.parse(accepted)
        require(accepted == step["accepted"] and len(accepted) <= limits.max_response_chars,
                "accepted action drift/cap")
        row = {"schema": 1, "id": digest([record["session"], index]), "split": task.split,
               "source": task.source, "revision": task.revision, "parent": task.parent,
               "task_key": task.key, "skill": task.skill, "attempt": record["attempt"],
               "step": index, "mode": record["mode"], "prompt": render(view),
               "completion": accepted, "trajectory_hash": trace_hash,
               "harness_hash": HARNESS_HASH}
        rows.append(row)
        if review and review.decision == "REPLACE":
            pairs.append({"schema": 1, "row_id": row["id"], "prompt": row["prompt"],
                          "proposed": step["proposal"], "accepted": accepted,
                          "reason": review.reason, "trajectory_hash": trace_hash,
                          "preference_eligible": False,
                          "exclusion": "independent_counterfactual_verification_required"})
        require(env.execute(accepted) == step["observation"], "environment replay mismatch")
    success = env.success()
    require(success == record["success"], "untrusted success flag disagrees with verifier")
    if env.final is not None:
        require(record["status"] == ("SUCCESS" if success else "FAILURE"), "terminal status mismatch")
    else:
        require(record["status"] not in {"SUCCESS", "FAILURE"}, "missing terminal action")
    for name in ("student_calls", "reviewer_calls", "input_chars", "output_chars"):
        require(type(record["cost"][name]) is int and record["cost"][name] >= 0, "invalid costs")
    require(record["cost"]["student_calls"] >= len(rows), "undercounted student calls")
    require(record["cost"]["reviewer_calls"] >= (len(rows) if reviewing else 0),
            "undercounted review calls")
    return rows, pairs, success


def compile_dataset(records: list[dict], tasks: list[Task]) -> tuple[list, list, dict]:
    catalog = validate_catalog(tasks)
    require(bool(records), "no trajectories")
    require(len({r["mode"] for r in records}) == 1, "do not mix ablation arms in one SFT dataset")
    sft, candidates, seen = [], [], set()
    counts = {"attempted": 0, "retained": 0, "sft_rows": 0, "statuses": {}, "by_mode": {},
              "passed_reviews": 0, "replacements": 0, "review_calls": 0}
    for record in records:
        require(record["task_key"] in catalog, "unknown task")
        task = catalog[record["task_key"]]
        require(task.split == "train" and record["evaluation"] is False,
                "calibration/retention/sealed data cannot enter SFT")
        require(record["session"] not in seen, "duplicate trajectory session")
        seen.add(record["session"])
        rows, pairs, success = replay(record, task)
        counts["attempted"] += 1
        counts["review_calls"] += record["cost"]["reviewer_calls"]
        for step in record["steps"]:
            if step["review"]:
                field = "passed_reviews" if step["review"]["decision"] == "PASS" else "replacements"
                counts[field] += 1
        status = record["status"]
        counts["statuses"][status] = counts["statuses"].get(status, 0) + 1
        arm = counts["by_mode"].setdefault(record["mode"], {"attempted": 0, "retained": 0})
        arm["attempted"] += 1
        if success:
            counts["retained"] += 1
            arm["retained"] += 1
            sft.extend(rows)
            candidates.extend(pairs)
    sft.sort(key=lambda row: (row["task_key"], row["mode"], row["attempt"], row["step"]))
    counts["sft_rows"] = len(sft)
    reviewed = counts["passed_reviews"] + counts["replacements"]
    counts["intervention_rate_executed_reviews"] = counts["replacements"] / reviewed if reviewed else None
    counts["preference_eligible"] = 0
    counts["catalog_hash"] = digest([asdict(t) for t in sorted(tasks, key=lambda t: t.key)])
    counts["sft_hash"] = digest(sft)
    return sft, candidates, counts


def tokenize_sft(row: dict, tokenizer, *, max_prompt_tokens: int, max_context_tokens: int) -> dict:
    """Token contract matching PRLC's prompt wrapper and completion+EOS CE.

    No truncation, padding, packing, or native template qualification is assumed.
    Repeated earlier accepted actions and tool observations are prompt-only here.
    """
    require(row["split"] == "train" and row["harness_hash"] == HARNESS_HASH, "SFT role/harness")
    Action.parse(row["completion"])
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos >= 0, "explicit integer EOS required")
    if getattr(tokenizer, "chat_template", None):
        prompt = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True))
    else:
        prompt = list(tokenizer.encode(row["prompt"], add_special_tokens=True))
    target = list(tokenizer.encode(row["completion"], add_special_tokens=False))
    require(bool(prompt) and len(prompt) <= max_prompt_tokens, "prompt overflow/empty")
    require(bool(target), "empty target")
    require(eos not in target, "embedded EOS in action")
    target.append(eos)
    require(len(prompt) + len(target) <= max_context_tokens, "context overflow; no truncation")
    require(all(type(t) is int and t >= 0 for t in prompt + target), "invalid token IDs")
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target,
            "prompt_tokens": len(prompt), "supervised_tokens": len(target)}
