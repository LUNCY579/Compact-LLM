"""Fresh RHD environments, public-evidence checks, collection and replay."""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict

from .core import (Action, Backend, BudgetExceeded, Environment, Ledger, Runbook,
                   Task, View, VERSION, STEPS, canonical, digest, loads, render, require,
                   seed_for, source_hash, student_prompt)
from experiments.rineya_harness_distillation_v1.contracts import (
    Reference, Review, ReviewRequest,
)


def observations(view: View) -> list[dict]:
    return [loads(m.content) for m in view.messages if m.role == "tool"]


def supported(action: Action, view: View) -> bool:
    """Conservative sufficiency predicate for the restricted RHD environment only.

    This is not a semantic-leakage detector for arbitrary text or terminal tasks.
    A correct guess without observed evidence is deliberately ineligible for SFT.
    """
    obs = observations(view)
    prompt = view.messages[0].content
    if action.kind == "retrieve":
        return re.search(r"(?<![\w-])" + re.escape(action.value) + r"(?![\w-])", prompt) is not None
    if action.kind == "check":
        normalized = re.sub(r"\s+", "", action.value)
        sources = [m[0] for m in re.finditer(r"(?<![\w-])-?\d{1,6}\s*[+*\-]\s*-?\d{1,6}(?!\d)", prompt)]
        sources += [o["result"] for o in obs if o.get("kind") == "retrieve" and o.get("found")]
        return bool(normalized) and any(normalized == re.sub(r"\s+", "", s) for s in sources)
    if action.kind == "answer":
        return any(str(o.get("result")).strip() == action.value.strip()
                   for o in obs if o.get("ok") is True or o.get("found") is True)
    return any(o.get("kind") == "retrieve" and o.get("found") is False
               and re.search(r"(?<![\w-])" + re.escape(o["key"]) + r"(?![\w-])", prompt)
               for o in obs)


def harness_hint(harness: str, view: View) -> str:
    require(harness in {"H0", "H1", "H2"}, "unknown harness")
    if harness == "H0":
        return ""
    obs = observations(view)
    if harness == "H1":
        return canonical({"public_state": {"retrievals": sum(o["kind"] == "retrieve" for o in obs),
                          "checks": sum(o["kind"] == "check" for o in obs)},
                          "obligations": ["inspect requested evidence", "check observed arithmetic",
                                          "answer from observations or abstain if missing"]})
    if obs and (obs[-1].get("ok") is False or obs[-1].get("found") is False):
        return "Use the last public tool failure to repair, or abstain when evidence is absent."
    return "Check public tool feedback before completing. No hidden verifier feedback is available."


def execute(task: Task, backend: Backend, ledger: Ledger, *, harness="H0", arm="discovery",
            attempt=0, seed=37, run_id="run", runbook: Runbook | None = None,
            source_id: str | None = None, reviewer: Backend | None = None) -> dict:
    require(task.split in {"train", "calibration", "retention_eval"}, "task role cannot execute")
    require(type(attempt) is int and attempt >= 0, "attempt")
    require(harness in {"H0", "H1", "H2"}, "harness")
    if task.split != "train":
        require(harness == "H0" and runbook is None and reviewer is None, "evaluation privilege")
        require(arm == "evaluation", "evaluation purpose")
    require(runbook is None or (harness == "H0" and source_id is not None), "rewrite binding")
    require(reviewer is None or (task.split == "train" and arm == "corrected" and runbook is not None), "correction arm")
    reference = Reference("rsr-procedures", tuple(STEPS[s] for s in runbook.steps)) if reviewer else None
    env = Environment(task)  # Never reuse the discovery environment or its final state.
    session = uuid.uuid4().hex
    record = dict(schema=VERSION, task=task.key, task_hash=task.fingerprint, parent=task.parent,
                  split=task.split, skill=task.skill, source=task.source, code_hash=source_hash(),
                  harness=harness, arm=arm, attempt=attempt, seed=seed, run_id=run_id,
                  backend=backend.identity, origin=backend.origin, sandbox=session,
                  source_id=source_id, runbook=runbook.fingerprint if runbook else None,
                  reviewer=reviewer.identity if reviewer else None,
                  review_reference=asdict(reference) if reference else None, steps=[], status="TRUNCATED")
    record["id"] = digest([run_id, task.key, arm, harness, attempt, seed, source_id, session])
    start = dict(ledger.used)
    for step in range(ledger.limits.steps):
        view = env.view()
        assistance = runbook.guidance() if runbook else harness_hint(harness, view)
        prompt = student_prompt(view, assistance)
        try:
            raw = ledger.call(backend, "executor", prompt,
                              seed_for(task.key, attempt, seed, step, "executor"))
            proposal, review = raw, None
            if reviewer:
                request = ReviewRequest(view, proposal, reference, session, step)
                payload = canonical({"review": asdict(request), "request_hash": request.fingerprint,
                                     "instruction": "Return RHD Review JSON: PASS unchanged or REPLACE; no gold access."})
                review = Review(**loads(ledger.call(reviewer, "reviewer", payload,
                                                   seed_for(task.key, attempt, seed, step, "reviewer"))))
                raw = review.accept(request)
            action = Action.parse(raw)
            grounded = supported(action, view)
            if action.kind in {"check", "retrieve"}:
                ledger.tool()
            observation = env.execute(raw)
            record["steps"].append(dict(before=view.fingerprint, action=raw,
                                        observation=observation, supported=bool(grounded),
                                        student_prompt=prompt, proposal=proposal,
                                        review=asdict(review) if review else None))
            if env.final:
                record["status"] = "SUCCESS" if env.success() else "FAILURE"
                break
        except BudgetExceeded as exc:
            record.update(status="BUDGET", error=str(exc))
            break
        except (ValueError, TypeError, KeyError) as exc:
            record.update(status="INVALID", error=type(exc).__name__)
            break
        except Exception as exc:
            record.update(status="ERROR", error=type(exc).__name__)
            break
    record["success"] = env.success()
    record["grounded"] = bool(record["steps"]) and all(s["supported"] for s in record["steps"])
    record["usage"] = {k: ledger.used[k] - start[k] for k in start}
    if not ledger.token_accounting:
        record["usage"]["tokens"] = None
    record["record_hash"] = digest(record)
    return record


def audit(record: dict, task: Task) -> dict:
    require(record["record_hash"] == digest({k: v for k, v in record.items() if k != "record_hash"}),
            "record integrity")
    require(record["schema"] == VERSION and record["code_hash"] == source_hash(), "code drift")
    for key, expected in (("task", task.key), ("task_hash", task.fingerprint),
                          ("split", task.split), ("parent", task.parent),
                          ("skill", task.skill), ("source", task.source)):
        require(record[key] == expected, "task/provenance mismatch")
    require(record["harness"] in {"H0", "H1", "H2"}, "harness")
    require(record["status"] in {"SUCCESS", "FAILURE", "TRUNCATED", "INVALID", "ERROR", "BUDGET"}, "status")
    require(type(record["success"]) is bool and type(record["grounded"]) is bool, "boolean outcomes")
    require(set(record["usage"]) == {"calls", "input_chars", "output_chars", "tools", "tokens"}, "usage fields")
    for name, value in record["usage"].items():
        require((name == "tokens" and value is None) or (type(value) is int and value >= 0), "invalid usage")
    require(record["usage"]["calls"] >= len(record["steps"]), "unaccounted executor calls")
    require(record["usage"]["tools"] == sum(Action.parse(s["action"]).kind in {"retrieve", "check"}
                                            for s in record["steps"]), "tool accounting mismatch")
    if task.split != "train":
        require(task.split in {"calibration", "retention_eval"}, "forbidden role")
        require(record["harness"] == "H0" and record["runbook"] is None
                and record["reviewer"] is None and record["arm"] == "evaluation", "evaluation privilege")
    require((record["reviewer"] is None) == (record["review_reference"] is None), "review reference")
    if record["runbook"] is not None:
        require(record["source_id"] is not None and record["harness"] == "H0"
                and record["arm"] in {"one_pass", "recursive", "corrected"}, "rewrite binding")
    env = Environment(task)
    grounding = []
    for index, step in enumerate(record["steps"]):
        view = env.view()
        require(view.fingerprint == step["before"], "history mismatch")
        if record["reviewer"]:
            require(record["arm"] == "corrected" and step["review"] is not None, "review arm")
            reference = Reference(record["review_reference"]["name"], tuple(record["review_reference"]["rules"]))
            request = ReviewRequest(view, step["proposal"], reference, record["sandbox"], index)
            require(Review(**step["review"]).accept(request) == step["action"], "review replay mismatch")
        else:
            require(step["review"] is None and step["proposal"] == step["action"], "unrecorded correction")
        if record["runbook"] is None:
            require(step["student_prompt"] == student_prompt(view, harness_hint(record["harness"], view)),
                    "collection prompt mismatch")
        else:
            require(step["student_prompt"].startswith(student_prompt(view) + "\nPRIVATE_TRAINING_GUIDANCE\n"),
                    "rewrite public prefix mismatch")
        grounded = bool(supported(Action.parse(step["action"]), view))
        require(step["supported"] is grounded, "evidence audit mismatch")
        grounding.append(grounded)
        require(env.execute(step["action"]) == step["observation"], "observation replay mismatch")
    require(env.success() is record["success"], "untrusted success flag")
    require((bool(grounding) and all(grounding)) is record["grounded"], "grounding mismatch")
    if env.final:
        require(record["status"] == ("SUCCESS" if env.success() else "FAILURE"), "terminal status")
    else:
        require(record["status"] not in {"SUCCESS", "FAILURE"}, "missing terminal")
    return dict(success=env.success(), grounded=record["grounded"],
                eligible=task.split == "train" and env.success() and record["grounded"])
