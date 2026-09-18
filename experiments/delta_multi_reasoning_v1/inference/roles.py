"""Frozen role and paraphrase definitions for the O / T / REP / MR inference conditions.

DESIGN CONSTRAINTS, all from the protocol and all enforced here rather than remembered
-------------------------------------------------------------------------------------
* The **final-answer contract is identical in every condition**. Every role's prompt ends with
  the same contract line, so the parser is unchanged and a score difference cannot be a
  formatting artifact. `CONTRACT` is defined once and appended by the builder, not written into
  each template.
* **R0-R3 are independent perspectives.** None sees another's output. R2 and R3 therefore audit
  the PROBLEM or their own independently reconstructed solution -- never an unseen earlier
  candidate. Sequential critique would be a different, separately registered protocol.
* **REP is four benign paraphrases, not temperature sampling.** Decoding is greedy and frozen;
  introducing sampling to get variation would be an unregistered change to the decoding policy.
  The paraphrases are frozen on CAL and are deliberately semantics-preserving.
* **No role or selector ever receives the gold answer**, the scorer's output, private task
  metadata, or an independent solver's answer. `build()` structurally cannot see a reference:
  it accepts the item's prompt messages only, and `assert_label_isolated()` re-checks the
  constructed text against the reference at call time so a regression fails loudly.
* **Roles must actually do different things.** Different names are not evidence. The companion
  check is `analysis/paired.py::roles_perform_different_operations`, which compares raw decoded
  text across roles and returns ROLES_COLLAPSED when the prompts induced no difference.

A NOTE ON THIS BACKEND, recorded because it bounds what MR can show here
------------------------------------------------------------------------
The 596M Rineya family is drilled to emit exactly one JSON field. Asking it to "audit premises"
is asking a heavily format-constrained model to do something its training never rewarded, and
the honest expectation is that R2/R3 either collapse onto R0 or emit malformed output. That is
a real possible result, not a reason to skip the measurement -- but it is why the malformed and
collapsed paths are frozen in advance instead of handled when they appear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# The registered abstention sentinel. Defined by the long-context suite and reused unchanged so
# that abstention means the same thing in every condition.
ABSTAIN = "UNKNOWN"

# The single final-answer contract. Appended verbatim to every role prompt in every condition.
CONTRACT = (
    "Reply with the final answer only, as a single JSON value, and nothing else. "
    f'If the question cannot be answered from what is given, reply exactly {ABSTAIN}.'
)


@dataclass(frozen=True)
class Role:
    """One registered operation. `operation` names what it does differently from R0."""

    rid: str
    operation: str
    instruction: str
    sees_other_role_outputs: bool = False
    max_new_tokens: int = 96


# --------------------------------------------------------------------------------------- #
# MR roles. Frozen; the mapping from role to task is fixed on CAL and never re-tuned.
# --------------------------------------------------------------------------------------- #

R0 = Role(
    rid="R0",
    operation="direct",
    instruction="",  # the ordinary prompt, unmodified: MR's R0 IS condition O's call
)

R1 = Role(
    rid="R1",
    operation="constraint_first_then_minimal_edit",
    instruction=(
        "Work in a different order from the obvious one. First decide whether the described "
        "action applies at all under the stated condition. If it does not apply, the answer is "
        "the starting state unchanged. If it does apply, change only the entries the action "
        "touches and copy every other entry across unmodified."
    ),
)

R2 = Role(
    rid="R2",
    operation="premise_and_evidence_audit",
    instruction=(
        "Before answering, check independently whether everything needed is actually present: "
        "whether the stated condition is met, whether any required value is missing, and "
        "whether any two statements conflict. If a needed premise is missing or the statements "
        "contradict each other, abstain. Otherwise give the answer those premises support."
    ),
)

R3 = Role(
    rid="R3",
    operation="state_reconstruction_and_check",
    instruction=(
        "Reconstruct the starting state and the rule that applies to it from the text, without "
        "relying on a first impression of the answer. Derive the result from that "
        "reconstruction, then check the result is consistent with every constraint given."
    ),
)

MR_SOLVER_ROLES: tuple[Role, ...] = (R0, R1, R2, R3)

R4 = Role(
    rid="R4",
    operation="selector",
    instruction="",           # built by selector_prompt(); listed here so the roster is complete
    sees_other_role_outputs=True,
    max_new_tokens=8,
)

# --------------------------------------------------------------------------------------- #
# REP paraphrases. Frozen on CAL. Semantics-preserving restatements of the SAME operation,
# so REP varies presentation while MR varies operation. That is the contrast that lets a gain
# be attributed to differentiated reasoning rather than to four tries.
# --------------------------------------------------------------------------------------- #

REP_PARAPHRASES: tuple[str, ...] = (
    "",
    "Answer the question above.",
    "Give the answer the information above determines.",
    "Using only the information above, state the answer.",
)


# --------------------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------------------- #

def _messages(prompt: list[dict]) -> list[dict]:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty list of chat messages")
    for m in prompt:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"malformed chat message: {m!r}")
    return [dict(m) for m in prompt]


def build(prompt: list[dict], role: Role | None = None, paraphrase: str = "") -> list[dict]:
    """Build the message list for one call.

    Takes ONLY the item's chat messages. There is no parameter through which a reference,
    label, family name, suite name or answerability flag could enter, which is the structural
    half of label isolation; `assert_label_isolated` is the runtime half.
    """
    msgs = _messages(prompt)
    extra = [t for t in ((role.instruction if role else ""), paraphrase) if t]
    tail = " ".join(extra + [CONTRACT])
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            msgs[i]["content"] = f"{msgs[i]['content']}\n\n{tail}"
            return msgs
    msgs.append({"role": "user", "content": tail})
    return msgs


def selector_prompt(prompt: list[dict], candidates: list[dict],
                    include_checker_findings: bool) -> list[dict]:
    """Build the selector call. `candidates` is [{'id','answer','finding'?}, ...].

    `include_checker_findings` distinguishes the two registered selectors on a common
    interface: S2 shows the audit roles' findings, S1 withholds them. Candidate CONTENT and
    checker FINDINGS are kept in separate labelled fields so the selector cannot confuse a
    finding for an answer.

    The selector chooses an ID. It never writes an answer of its own: a synthesising selector
    would be a separate variant, not this one.
    """
    msgs = _messages(prompt)
    lines = ["Several independent attempts at the answer are listed below."]
    for c in candidates:
        lines.append(f"[{c['id']}] answer: {c['answer']}")
        if include_checker_findings and c.get("finding"):
            lines.append(f"[{c['id']}] note: {c['finding']}")
    ids = ", ".join(c["id"] for c in candidates)
    lines.append(
        f"Reply with the identifier of the single best attempt and nothing else, "
        f"one of: {ids}.")
    msgs.append({"role": "user", "content": "\n".join(lines)})
    return msgs


# --------------------------------------------------------------------------------------- #
# Label isolation
# --------------------------------------------------------------------------------------- #

FORBIDDEN_FIELDS = ("reference", "reference_text", "answer_key", "accepted_answers",
                    "gold", "label", "is_correct", "family", "view_role", "source_split",
                    "gold_verified_by", "difficulty", "variant", "needle_profile")


def assert_label_isolated(built: list[dict], item: dict,
                          allow: Iterable[str] = ()) -> None:
    """Fail loudly if prompt CONSTRUCTION introduced a forbidden field's value.

    `allow` lists strings that are legitimately present because a MODEL produced them -- the
    candidate answers on the selector path. A selector must see its candidates, and a candidate
    that happens to be correct is byte-identical to the reference; refusing that would refuse
    every selector call on an item any role got right. What the check still guarantees is the
    property that matters: gold enters a prompt ONLY as something the model itself emitted,
    never from the bank's label field.

    The invariant is about this module, not about the bank. A retrieval task legitimately
    contains its own answer: a needle-in-haystack item's reference IS a sentence inside the
    document, and that is the task rather than a leak. What must never happen is for a role
    instruction, a paraphrase, the contract line or a selector payload to interpolate a
    reference that was not already in the item's own prompt.

    So the comparison is against the item's ORIGINAL prompt text, and only values the
    construction ADDED are flagged. The first version compared against the whole constructed
    prompt and refused every answerable long-context item -- correctly reporting that the seal
    number was present, and wrongly calling that a leak.

    Checked at call time rather than trusted, because this failure mode produces a *better*
    score and so cannot be caught by looking at results.

    `family` stays on the forbidden list deliberately: family labels are hidden suite metadata,
    and a role that knew the family would be reading the evaluation suite's structure rather
    than the user's request.
    """
    built_text = "\n".join(m["content"] for m in built)
    original = item.get("prompt") or []
    original_text = "\n".join(m.get("content", "") for m in original) if original else ""
    allowed = [str(x) for x in allow if x is not None]

    for field in FORBIDDEN_FIELDS:
        if field not in item:
            continue
        val = item[field]
        for v in (val if isinstance(val, (list, tuple)) else [val]):
            s = str(v)
            # Short values collide with ordinary prose; only non-trivial strings are decisive.
            if len(s) < 4:
                continue
            if s not in built_text:
                continue
            if s in original_text:
                continue            # the item's own evidence -- a retrieval task's needle
            if any(s in cand for cand in allowed):
                continue            # a model-produced candidate the selector must be shown
            raise AssertionError(
                f"label leak: item field {field!r} value {s!r} was INTRODUCED by prompt "
                f"construction for item {item.get('id')!r} -- it is not in the item's own "
                f"prompt and not a model-produced candidate. Refusing to generate.")


def role_roster() -> dict:
    """The frozen roster, for the run receipt."""
    return {
        "contract": CONTRACT,
        "abstention_sentinel": ABSTAIN,
        "mr_solver_roles": [
            {"id": r.rid, "operation": r.operation, "max_new_tokens": r.max_new_tokens,
             "sees_other_role_outputs": r.sees_other_role_outputs,
             "instruction": r.instruction}
            for r in MR_SOLVER_ROLES],
        "selector": {"id": R4.rid, "operation": R4.operation,
                     "max_new_tokens": R4.max_new_tokens,
                     "sees_other_role_outputs": R4.sees_other_role_outputs},
        "rep_paraphrases": list(REP_PARAPHRASES),
        "mr_independence": ("R0-R3 see the original problem and never each other's outputs; "
                            "only R4 sees candidates"),
        "decoding": "greedy, frozen; REP varies presentation, never temperature",
        "forbidden_fields_checked_at_call_time": list(FORBIDDEN_FIELDS),
    }
