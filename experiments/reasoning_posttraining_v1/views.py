"""Public-scope filtering for an explicitly easier same-world training view.

No answer, certificate, minimal-evidence oracle or future state is consulted.
This view is not the full-context model scoreboard.
"""
from __future__ import annotations
import re
from experiments.model_quality_gpu_v1.integrity import decode, digest


def scope_only(prompt: list[dict], parse_prompt, canonical_case) -> tuple[list[dict], dict]:
    original = parse_prompt(prompt)
    lines = prompt[1]["content"].splitlines()
    scopes = [decode(line[6:]) for line in lines if line.startswith("SCOPE ")]
    if len(scopes) != 1 or not isinstance(scopes[0], str):
        raise ValueError("unambiguous public scope required")
    kept, removed, records = [], 0, 0
    pattern = re.compile(r'^Record R[0-9a-f]+ \| scope=("(?:[^"\\]|\\.)*") \| ')
    for line in lines:
        if line.startswith("Record "):
            match = pattern.match(line)
            if match is None:
                raise ValueError("unsupported record scope serialization")
            if decode(match[1]) != scopes[0]:
                removed += 1
                continue
            records += 1
        kept.append(line)
    shortened = [dict(prompt[0]), {"role": "user", "content": "\n".join(kept)}]
    parsed = parse_prompt(shortened)
    if canonical_case(parsed) != canonical_case(original) or records != len(original["records"]):
        raise ValueError("scope filtering changed task evidence")
    return shortened, {"variant": "scope_only", "original_prompt_sha256": digest(prompt),
                       "derived_prompt_sha256": digest(shortened), "target_records_preserved": records,
                       "other_scope_records_removed": removed, "canonical_world_unchanged": True,
                       "minimal_causal_evidence_claim": False, "full_context_efficacy_claim": False}
