"""Scripted protocol fixtures, never a learned or distilled model."""
from __future__ import annotations

import re
from .contracts import (Action, PROCEDURAL, Review, ReviewRequest, Task, View, loads, require)


def make_tasks(count: int = 24, *, split: str = "train", offset: int = 0) -> list[Task]:
    require(type(count) is int and count > 0, "positive fixture count required")
    tasks = []
    for i in range(offset, offset + count):
        skill = ("verify", "retrieve", "abstain")[i % 3]
        key = f"{split}-{i}"
        if skill == "verify":
            prompt, expected, records = f"Compute {i + 1} + 7.", str(i + 8), ()
        else:
            record_key = f"record_{split}_{i}"
            prompt = f"Report the value for record key {record_key}."
            expected = f"value_{i}" if skill == "retrieve" else None
            records = ((record_key, expected),) if expected is not None else ()
        tasks.append(Task(key, "owned_fixture", "rhd-fixture-v1", f"fixture-parent-{i}",
                          split, skill, prompt, expected, records))
    return tasks


def _observed(view: View):
    observations = [loads(m.content) for m in view.messages if m.role == "tool"]
    return observations[-1] if observations else None


class FixtureStudent:
    identity = "scripted-fixture-student-not-rineya"

    def propose(self, view: View, seed: int) -> str:
        observed = _observed(view)
        if observed and observed.get("result") is not None:
            return Action("answer", observed["result"]).encode()
        if observed and observed.get("found") is False:
            return Action("abstain", "The requested record was not found.").encode()
        return Action("answer", "unverified").encode()


class RuleReviewer:
    identity = "scripted-rule-reviewer-not-llm"

    def review(self, request: ReviewRequest) -> Review:
        if not request.reference.rules:
            return Review(request.fingerprint, "PASS")
        observed = _observed(request.view)
        if observed:
            if observed.get("result") is not None:
                wanted = Action("answer", observed["result"]).encode()
                reason = "repair"
            elif observed.get("found") is False:
                wanted = Action("abstain", "The requested record was not found.").encode()
                reason = "unsupported"
            else:
                raise ValueError("fixture cannot recover this tool error")
        else:
            prompt = request.view.messages[0].content
            expression = re.fullmatch(r"Compute (\d+ \+ 7)\.", prompt)
            if expression:
                wanted, reason = Action("check", expression[1]).encode(), "verify_first"
            else:
                key = re.fullmatch(r"Report the value for record key (\w+)\.", prompt)
                require(key is not None, "unknown fixture prompt")
                wanted, reason = Action("retrieve", key[1]).encode(), "retrieve_first"
        # Semantic identity can PASS with the student's original whitespace unchanged.
        try:
            sound = Action.parse(request.proposal) == Action.parse(wanted)
        except (ValueError, TypeError):
            sound = False
        if sound:
            return Review(request.fingerprint, "PASS")
        return Review(request.fingerprint, "REPLACE", wanted, reason,
                      private_note="PRIVATE_REVIEW_SENTINEL: protocol fixture only")


class ScriptedExpert:
    identity = "scripted-expert-fixture-not-distilled"

    def propose(self, view: View, seed: int) -> str:
        proposal = FixtureStudent().propose(view, seed)
        request = ReviewRequest(view, proposal, PROCEDURAL, "fixture-expert", 0)
        return RuleReviewer().review(request).accept(request)
