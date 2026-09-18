"""One answer extractor and one correctness rule, shared by the selector and the scorer.

Split out for the reason `scoring.py` in the qualified harness was split out: the extractor was
about to exist in two places -- once to build selector candidates, once to score them -- and two
copies drift. When they drift here the failure is silent and favourable, because a selector
that parses answers differently from the scorer can be credited with picking a candidate the
scorer then marks wrong.

EXTRACTION IS LABEL-BLIND. `extract` never sees the reference. It takes the row only for its
`stratum`, which selects the parsing rule, and the stratum is not a label. `is_correct` is the
only function that touches gold, and it is never called on the inference path -- only in
analysis, after every generation is recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PILOT_SCORER = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917/"
                    r"experiments/rab001_exploration")

ABSTAIN = "UNKNOWN"


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _answer_matches():
    """The qualified scorer, imported rather than copied."""
    if str(PILOT_SCORER) not in sys.path:
        sys.path.insert(0, str(PILOT_SCORER))
    from scoring import answer_matches, self_test
    self_test()
    return answer_matches


def extract(row: dict, text: str) -> str | None:
    """The candidate answer, or None if the output does not parse under that stratum's rule.

    None means malformed, which the selector policy handles explicitly -- it is not silently
    turned into a wrong answer, because "the model emitted nothing usable" and "the model was
    wrong" are different failures and the malformed count is reported separately.
    """
    t = (text or "").strip()
    if not t:
        return None
    s = row["stratum"]
    if s in ("transition", "multi_step"):
        try:
            return canon(json.loads(t))
        except (ValueError, TypeError):
            return None
    if s == "arith_competence":
        return t.splitlines()[0].strip() or None
    if s == "longctx_mixed":
        return (t.splitlines()[0].strip() or None)
    raise ValueError(f"unregistered stratum {row['stratum']!r}")


def norm_lc(v: str) -> str:
    """The long-context suite's normalisation, verbatim."""
    return " ".join(str(v).strip().lower().replace(".", " ").split())


def is_correct(row: dict, answer: str | None) -> bool:
    """ANALYSIS ONLY. Never called while building a prompt."""
    if answer is None:
        return False
    s, ref = row["stratum"], row["reference"]
    if s in ("transition", "multi_step"):
        return answer == ref
    if s == "arith_competence":
        return bool(_answer_matches()(answer, ref))
    if s == "longctx_mixed":
        return norm_lc(answer) == norm_lc(ref)
    raise ValueError(f"unregistered stratum {s!r}")


def finding_for(rid: str, answer: str | None, direct_answer: str | None) -> str | None:
    """The checker finding an audit role contributes, derived from answers by CODE.

    Why this shape. The protocol asks the selector to distinguish candidate CONTENT from
    optional checker FINDINGS, and S2 is the checker-informed variant. But this model family
    emits exactly one field -- adding a second would break the identical final-answer contract
    that makes O/T/REP/MR comparable at all. So the finding is DERIVED, at zero extra model
    calls, from what the audit role actually did:

      R2 (premise/evidence audit) abstaining IS its finding: it judged the evidence
         insufficient, which is a real signal a selector can weigh.
      R3 (state/constraint audit) disagreeing with the direct solve IS its finding: an
         independent reconstruction reached a different answer.

    Both are label-blind -- neither consults the reference. Returns None when the role has
    nothing to report, and `roles.selector_prompt` then omits the field.
    """
    if rid == "R2" and answer is not None and answer.strip() == ABSTAIN:
        return "the premise and evidence audit judged the evidence insufficient"
    if rid == "R3" and answer is not None and direct_answer is not None and answer != direct_answer:
        return "an independent state reconstruction reached a different result"
    return None
