"""Hand-written fixtures, never represented as model competence or distillation."""
from __future__ import annotations

import re

from .core import Action, Backend, Reply, Runbook, Task, canonical, loads
from experiments.rineya_harness_distillation_v1.environment import arithmetic


def fixture_backend() -> Backend:
    def respond(prompt, seed, max_new_tokens):
        value = loads(prompt.split("\nPRIVATE_TRAINING_GUIDANCE\n")[0])
        if value.get("role") == "planner":
            text = Runbook(("inspect_requested", "check_observed", "answer_observed", "abstain_missing")).encode()
        elif value.get("role") == "critic":
            text = canonical(dict(request_hash=value["request_hash"], decision="ACCEPT", reason="accepted"))
        else:
            messages = value["messages"]
            observations = [loads(m["content"]) for m in messages if m["role"] == "tool"]
            if observations:
                last = observations[-1]
                if last.get("found") is False:
                    action = Action("abstain", "Requested evidence is absent.")
                elif last["kind"] == "retrieve":
                    try:
                        arithmetic(last["result"])
                        action = Action("check", last["result"])
                    except ValueError:
                        action = Action("answer", last["result"])
                else:
                    action = Action("answer", last["result"])
            else:
                public = messages[0]["content"]
                key = re.search(r"record_\d+", public)
                expression = re.search(r"-?\d+\s*[+*\-]\s*-?\d+", public)
                action = Action("retrieve", key[0]) if key else Action("check", expression[0])
            text = action.encode()
        return Reply(text)
    return Backend("handwritten-fixture-v1", "fixture", respond)


def catalog(count=12, *, split="train", offset=0):
    tasks = []
    for i in range(offset, offset + count):
        key, expression = f"record_{i}", f"{i + 3} * {i + 2}"
        if i % 3 == 0:
            task = Task(f"task-{i}", "rsr-fixture", "v1", f"parent-{i}", split, "verify",
                        f"Check {expression} and answer the observed result.", arithmetic(expression))
        else:
            absent = i % 3 == 2
            task = Task(f"task-{i}", "rsr-fixture", "v1", f"parent-{i}", split,
                        "abstain" if absent else "retrieve",
                        f"Retrieve {key}; check its expression and answer the observed result. Abstain if absent.",
                        None if absent else arithmetic(expression), () if absent else ((key, expression),))
        tasks.append(task)
    return tasks, {t.key: [t.parent] for t in tasks}


class ByteTokenizer:
    """Only a loss-mask fixture; does not approximate a Rineya tokenizer."""
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [b + 2 for b in text.encode("utf-8")]
