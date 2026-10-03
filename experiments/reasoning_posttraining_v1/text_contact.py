"""Report and remove training contact with a declared, already-open panel.

IDs and class exclusions alone missed arithmetic generated under another class.
Eight-word shingles alone miss six-word stems such as ``What is 10% of 100?``.
Use all three checks. Only user content participates: shared system instructions
and assistant answers are not evidence of prompt contact. This is an exact-text
guard, not a semantic contamination certificate or permission to read a bank.
Callers must supply an instrument they are allowed to inspect.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

POLICY = "user_text_contact_v1_id_8gram_digit_blinded_stem"


def _user_turns(row: dict) -> list[str]:
    messages = row.get("messages", row.get("prompt"))
    if not isinstance(messages, list):
        raise ValueError("contact guard needs explicit chat messages")
    turns = [m.get("content") for m in messages if m.get("role") == "user"]
    if not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
        raise ValueError("contact guard needs nonempty user content")
    return turns


def _shingles(text: str) -> set[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return {" ".join(words[i:i + 8]) for i in range(max(0, len(words) - 7))}


def _digit_blind(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\d+", "#", " ".join(text.split()))


def audit_and_filter(rows: Iterable[dict], panel_rows: Iterable[dict]) -> tuple[list[dict], dict]:
    """Return original clean rows in original order, plus every exclusion reason.

    The stem definition follows the existing arithmetic panel audit: last user
    turn, text after its final ``': '``. A match in any candidate user turn counts.
    Numeric runs are blinded so changing operands cannot disguise a shared task.
    Nothing is relabeled; the caller must retain this report with its new corpus.
    """
    panel_ids: set[str] = set()
    panel_shingles: set[str] = set()
    stems: dict[str, list[str]] = {}
    panel_count = 0
    for panel in panel_rows:
        panel_id = panel.get("id")
        if not isinstance(panel_id, str) or not panel_id:
            raise ValueError("panel row needs an id")
        if panel_id in panel_ids:
            raise ValueError("duplicate panel id")
        panel_ids.add(panel_id)
        panel_count += 1
        turns = _user_turns(panel)
        for turn in turns:
            panel_shingles.update(_shingles(turn))
        stem = _digit_blind(turns[-1].rsplit(": ", 1)[-1])
        if not stem:
            raise ValueError("empty panel stem")
        stems.setdefault(stem, []).append(panel_id)
    if not panel_count:
        raise ValueError("empty panel cannot establish zero contact")

    clean, dropped = [], []
    reasons_count: Counter = Counter()
    count = 0
    for row in rows:
        count += 1
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("candidate row needs an id")
        turns = _user_turns(row)
        reasons = []
        if row_id in panel_ids:
            reasons.append("panel_id")
        if any(_shingles(turn) & panel_shingles for turn in turns):
            reasons.append("user_8gram")
        blinded = [_digit_blind(turn) for turn in turns]
        stem_ids = sorted({pid for stem, ids in stems.items()
                           if any(stem in turn for turn in blinded) for pid in ids})
        if stem_ids:
            reasons.append("digit_blinded_panel_stem")
        if reasons:
            reasons_count.update(reasons)
            dropped.append({"id": row_id, "reasons": reasons,
                            "matched_panel_stem_ids": stem_ids,
                            "reply_class": (row.get("metadata") or {}).get("reply_class")})
        else:
            clean.append(row)
    return clean, {
        "policy": POLICY,
        "panel_rows": panel_count,
        "input_rows": count,
        "kept_rows": len(clean),
        "dropped_rows": len(dropped),
        "reason_counts": dict(sorted(reasons_count.items())),
        "dropped": dropped,
        "semantic_contamination_certificate": False,
        "system_and_assistant_text_checked": False,
    }
