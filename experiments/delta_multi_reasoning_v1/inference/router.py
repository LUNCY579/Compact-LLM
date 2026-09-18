"""The frozen input-only router, and the two ways a routed score may be computed.

WHAT THIS IS
------------
A dispatch decision taken BEFORE any generation, from the user's request alone. The chosen
backend then supplies EVERY role of whatever inference condition is running (O/T/REP/MR). The
4B is never used merely as a judge of the small model's candidates -- that would be an
output-dependent cascade, a different protocol with different costs, and it is not this.

THE INFORMATION BOUNDARY, enforced structurally
-----------------------------------------------
`dispatch()` takes the request's chat messages and nothing else. It is given no item dict, so
it cannot read an evaluation-suite name, a hidden family label, gold answerability,
correctness, or per-item best-component knowledge even by accident. `assert_input_only()`
re-checks at call time against the very fields a leak would come from.

The SAME rule is frozen across R_D0, R_L0, R_D1 and R_L1, so those four systems differ only in
which component is substituted. That is what makes the component comparison interpretable.

WHY THE MEASURED RESULT WILL BE CLOSE TO AN UPPER BOUND, stated before it is run
-------------------------------------------------------------------------------
The two instruments in this programme are near-trivially separable by request shape: the
acquisition views are short JSON state-transition prompts, the long-context rows are long
documents with a question. A rule that separates them will look excellent, and that appearance
is mostly instrument separability rather than routing skill. Three things are therefore
required of any scout this runs on, and are registered here:

  1. AMBIGUOUS cases -- requests that genuinely could go either way -- must be present, or the
     dispatch rate is measured only where the answer is obvious.
  2. SHORTCUT CHECKS -- cases whose surface form points at the wrong component -- must be
     present, so misrouting is observable rather than assumed absent.
  3. The dispatch rate, the conditional accuracy WITHIN each dispatch branch, and the
     misrouting loss must all be reported, not just the routed total.

CACHED FUNCTIONAL vs END-TO-END NATIVE
--------------------------------------
When the dispatch is input-only and deterministic, the routed system's answer on an item is
*identical* to the chosen component's already-recorded answer, so the routed functional score
can be computed from immutable stored outputs without regenerating anything. That is legitimate
and it is labelled `cached_functional`.

It is NOT a measurement of the routed runtime. A cache lookup is not model latency, and
nothing here measures loading, switching or serving cost. Those need a separate bounded
`end_to_end_native` check, and until one runs the routed runtime is NOT qualified. The two
labels are kept distinct in every receipt this module writes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

# The two components. SMALL is the 596M specialist (B/D/L/N16); LARGE is the 4B (G0/G1).
SMALL = "small_specialist"
LARGE = "large_general"

# Frozen thresholds. Chosen on CAL before any CHECK outcome and not re-tuned afterwards.
LONG_REQUEST_CHARS = 1200          # above this the request carries a document to read
STRUCTURED_TARGET_MIN_KEYS = 2     # a JSON object of this many keys is a state to transform

_JSON_OBJ = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class Dispatch:
    component: str
    reason: str
    features: dict


def _request_text(messages: Sequence[Mapping]) -> str:
    """The user-visible request. System scaffolding is included because a deployment sees it."""
    return "\n".join(str(m.get("content", "")) for m in messages)


def features(messages: Sequence[Mapping]) -> dict:
    """Features drawn from the request only. Every one is something a deployment can see."""
    text = _request_text(messages)
    objs = _JSON_OBJ.findall(text)
    widest = 0
    for o in objs:
        try:
            parsed = json.loads(o)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            widest = max(widest, len(parsed))
    return {
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "widest_json_object_keys": widest,
        "has_json_object": widest > 0,
        "is_long": len(text) > LONG_REQUEST_CHARS,
        "asks_for_structured_state": widest >= STRUCTURED_TARGET_MIN_KEYS,
    }


def dispatch(messages: Sequence[Mapping]) -> Dispatch:
    """The frozen rule. Input-only, deterministic, identical across all routed systems.

    The rule is deliberately simple and stated positively: a request that hands over a
    structured state to transform goes to the specialist that was trained to transform one;
    everything else goes to the general model. Length is the tie-breaker, not the primary
    signal, so a long request that still carries a state to transform stays with the
    specialist -- which is the shortcut case worth getting right.
    """
    f = features(messages)
    if f["asks_for_structured_state"]:
        return Dispatch(SMALL, "request supplies a multi-key structured state to transform", f)
    if f["is_long"]:
        return Dispatch(LARGE, "long request with no structured state to transform", f)
    return Dispatch(LARGE, "short request with no structured state to transform", f)


# --------------------------------------------------------------------------------------- #
# Information boundary
# --------------------------------------------------------------------------------------- #

ROUTER_FORBIDDEN = ("reference", "reference_text", "accepted_answers", "answer", "gold",
                    "label", "is_correct", "family", "view_role", "source_split",
                    "gold_verified_by", "variant", "difficulty", "needle_profile",
                    "instrument", "suite", "split", "task_type")


def assert_input_only(messages: Sequence[Mapping], item: Mapping) -> None:
    """Fail if anything the router must not see has reached the request it routes on."""
    text = _request_text(messages)
    for field in ROUTER_FORBIDDEN:
        if field not in item:
            continue
        val = item[field]
        for v in (val if isinstance(val, (list, tuple)) else [val]):
            s = str(v)
            if len(s) >= 4 and s in text:
                raise AssertionError(
                    f"router information leak: field {field!r} value {s!r} is present in the "
                    f"request for item {item.get('id')!r}. The router must decide from the "
                    f"user's request alone. Refusing to dispatch.")


# --------------------------------------------------------------------------------------- #
# Scoring an actual per-item dispatch
# --------------------------------------------------------------------------------------- #

def route_cached(items: Sequence[Mapping],
                 component_correct: Mapping[str, Mapping[str, bool]],
                 small_system: str, large_system: str) -> dict:
    """Score the routed system from immutable recorded component outputs.

    `component_correct[system][item_id]` is that system's recorded per-item outcome.
    Every item is dispatched, then credited to whichever component the dispatch chose. No
    generation happens, and the result is labelled `cached_functional` for that reason.

    Reported per branch, because a routed total alone hides where the routing helped:
      dispatch rate, conditional accuracy inside each branch, and the misrouting loss --
      what was lost relative to always using the OTHER component on the items sent here.
    """
    small, large = component_correct[small_system], component_correct[large_system]
    routed: dict[str, bool] = {}
    chosen: dict[str, str] = {}
    branch: dict[str, list[str]] = {SMALL: [], LARGE: []}

    for it in items:
        iid = it["id"]
        msgs = it["prompt"]
        assert_input_only(msgs, it)
        d = dispatch(msgs)
        chosen[iid] = d.component
        branch[d.component].append(iid)
        src = small if d.component == SMALL else large
        if iid not in src:
            raise KeyError(f"no recorded outcome for {iid} in the dispatched component")
        routed[iid] = bool(src[iid])

    n = len(routed)
    out = {
        "evaluation_label": "cached_functional",
        "what_this_is": ("routed functional score computed from immutable recorded component "
                         "outputs under an exact input-only dispatch; NOT a runtime measurement"),
        "components": {SMALL: small_system, LARGE: large_system},
        "n_items": n,
        "routed_correct": sum(routed.values()),
        "always_small_correct": sum(1 for i in routed if small[i]),
        "always_large_correct": sum(1 for i in routed if large[i]),
        "dispatch_counts": {k: len(v) for k, v in branch.items()},
        "dispatch_rates": {k: (len(v) / n if n else None) for k, v in branch.items()},
        "per_item_dispatch": chosen,
        "routed_per_item_correct": routed,
        "branches": {},
        "latency_note": ("A cache lookup is not routed-model latency. Loading, switching and "
                         "serving cost are unmeasured here and require an end_to_end_native "
                         "check before the routed runtime may be called qualified."),
        "runtime_qualified": False,
    }
    for comp, ids in branch.items():
        if not ids:
            out["branches"][comp] = {"n": 0}
            continue
        used = small if comp == SMALL else large
        other = large if comp == SMALL else small
        got = sum(1 for i in ids if used[i])
        alt = sum(1 for i in ids if other[i])
        # Misrouting is a PAIRED count, not a net. A branch that sends one item to the wrong
        # component and another to the right one has misrouted once, even though the totals
        # cancel -- the same error `analysis/paired.py` exists to prevent on model comparisons.
        misrouted = sorted(i for i in ids if other[i] and not used[i])
        rescued = sorted(i for i in ids if used[i] and not other[i])
        out["branches"][comp] = {
            "n": len(ids),
            "conditional_accuracy": got / len(ids),
            "alternative_component_would_have_scored": alt,
            "misrouted_count": len(misrouted),
            "rescued_count": len(rescued),
            "net_branch_advantage": len(rescued) - len(misrouted),
            "misrouted_ids": misrouted[:50],
            "rescued_ids": rescued[:50],
        }
    out["net_vs_always_small"] = out["routed_correct"] - out["always_small_correct"]
    out["net_vs_always_large"] = out["routed_correct"] - out["always_large_correct"]
    out["total_misrouted_count"] = sum(b.get("misrouted_count", 0)
                                       for b in out["branches"].values())
    out["total_rescued_count"] = sum(b.get("rescued_count", 0)
                                     for b in out["branches"].values())
    out["zero_observed_misrouting"] = out["total_misrouted_count"] == 0
    out["misrouting_note"] = (
        "total_misrouted_count is the number of items the OTHER component would have answered "
        "correctly and the dispatched one did not. It is reported alongside total_rescued_count "
        "rather than netted, because a net of zero can hide misroutes that another item "
        "happened to offset.")
    return out


def rule() -> dict:
    """The frozen rule, for the run receipt."""
    return {
        "kind": "input_only_frozen_dispatch",
        "decided": "before any generation; the chosen backend supplies every role of the condition",
        "not": ("an output-dependent cascade; the large model is never used only as a judge of "
                "the small model's candidates"),
        "components": {SMALL: "596M specialist (B/D/L/N16)", LARGE: "4B general (G0/G1)"},
        "thresholds": {"LONG_REQUEST_CHARS": LONG_REQUEST_CHARS,
                       "STRUCTURED_TARGET_MIN_KEYS": STRUCTURED_TARGET_MIN_KEYS},
        "frozen_identically_across": ["R_D0", "R_L0", "R_D1", "R_L1"],
        "forbidden_information": list(ROUTER_FORBIDDEN),
        "required_of_any_scout": ["ambiguous cases", "shortcut checks",
                                  "per-branch conditional accuracy and misrouting loss"],
        "evaluation_labels": {
            "cached_functional": "scored from immutable recorded outputs under exact dispatch",
            "end_to_end_native": "measures real loading/switching/serving cost; NOT YET RUN",
        },
        "oracle_note": ("a per-item oracle over component answers is diagnostic only and gets "
                        "no deployment credit"),
    }
