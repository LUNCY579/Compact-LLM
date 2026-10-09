"""Explicit semantic banks, boundary-safe encoding and exact-dose paired controls."""
from __future__ import annotations
from collections import defaultdict
import copy
import random
import re
from .io import digest, require, seal, validate_bank, verify


def encode_bank(raw: dict, tokenizer, tokenizer_sha256: str, max_length: int = 512) -> dict:
    verify(raw)
    require(raw.get("kind") == "raw_bank" and raw.get("schema_version") == 1, "Wrong raw bank schema")
    require(getattr(tokenizer, "is_fast", False), "Fast tokenizer offsets required")
    items = []
    for row in raw["items"]:
        prompt, target = row["prompt"], row["target"]
        require(isinstance(prompt, str) and prompt and isinstance(target, str) and target, "Empty text")
        spans = row["content_spans"]
        require(all(isinstance(s, list) and len(s) == 2 and all(type(v) is int for v in s)
                    and 0 <= s[0] < s[1] <= len(target) for s in spans), "Invalid content spans")
        enc = tokenizer(prompt + target, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), enc["offset_mapping"]
        require(len(ids) == len(offsets) and len(ids) <= max_length, "Length limit; truncation is forbidden")
        boundary = len(prompt)
        roles = []
        for start, end in offsets:
            require(0 <= start < end <= len(prompt + target), "Unusable tokenizer offsets")
            require(not start < boundary < end, "Token crosses prompt/target boundary; revise registered frame")
            if end <= boundary:
                roles.append("prompt")
                continue
            start, end = start - boundary, end - boundary
            content_chars = {i for lo, hi in spans for i in range(max(start, lo), min(end, hi))}
            roles.append("format" if not content_chars else "content" if len(content_chars) == end - start else "mixed")
        item = {k: copy.deepcopy(row[k]) for k in ("item_id", "parent_id", "lineage_id", "family", "operation",
                "rule_case", "suite", "format_id", "split", "horizon", "target_rule")}
        item.update(payload_sha256=digest(row["semantic_payload"]), input_ids=ids, roles=roles,
                    text_sha256=digest({"prompt": prompt, "target": target, "spans": spans}))
        items.append(item)
    bank = seal({"schema_version": 1, "kind": "encoded_bank", "evidence_class": raw["evidence_class"],
                 "tokenizer_sha256": tokenizer_sha256, "raw_sha256": raw["content_sha256"], "items": items})
    validate_bank(bank)
    return bank


def match_controls(bank: dict, pairs: int, seed: int = 17, min_coverage_gap: float = 0.15) -> dict:
    """Exact per-pair family/frame/horizon/processed/supervised-token matching.

    The control is random conditional on matched buckets, NOT an unconditional
    population sample. Both arms contain unique payloads and no payload is shared.
    """
    validate_bank(bank)
    require(all(x["split"] == "train" for x in bank["items"]), "Control selection may read only training data")
    require(type(pairs) is int and pairs > 0 and 0 <= min_coverage_gap <= 1, "Invalid control design")
    buckets = defaultdict(list)
    for x in bank["items"]:
        key = (x["family"], x["format_id"], x["horizon"], len(x["input_ids"]),
               sum(r != "prompt" for r in x["roles"]))
        buckets[key].append(x)
    rng = random.Random(seed)
    candidates = [(key, x) for key in sorted(buckets) for x in buckets[key] if x["target_rule"]]
    rng.shuffle(candidates)
    target, control, used = [], [], set()
    for key, x in candidates:
        if len(target) == pairs:
            break
        if x["payload_sha256"] in used:
            continue
        choices = [c for c in buckets[key] if c["payload_sha256"] not in used
                   and c["payload_sha256"] != x["payload_sha256"]]
        if not choices:
            continue
        c = rng.choice(choices)
        target.append(x)
        control.append(c)
        used |= {x["payload_sha256"], c["payload_sha256"]}
    require(len(target) == pairs, f"Infeasible exact matching: only {len(target)}/{pairs} pairs; do not pad/fabricate")
    coverage_gap = 1 - sum(x["target_rule"] for x in control) / pairs
    require(coverage_gap >= min_coverage_gap, "Matching removed the intended coverage contrast")
    def subset(rows):
        return seal({"schema_version": 1, "kind": "encoded_bank", "evidence_class": bank["evidence_class"],
                     "tokenizer_sha256": bank["tokenizer_sha256"], "source_sha256": bank["content_sha256"], "items": rows})
    return {"targeted": subset(target), "control": subset(control),
            "receipt": {"seed": seed, "source_sha256": bank["content_sha256"], "pairs": pairs,
                "processed_tokens_each": sum(len(x["input_ids"]) for x in target),
                "supervised_tokens_each": sum(sum(r != "prompt" for r in x["roles"]) for x in target),
                "targeted_rule_fraction": 1.0, "control_rule_fraction": 1 - coverage_gap,
                "coverage_gap": coverage_gap, "order": "paired index; use identical seed and batching in both arms",
                "control_estimand": "random conditional on exact length/family/frame/horizon buckets"}}


def transition(family: str, state, action: dict):
    """Small executable TEST domains; these are not the RCWM/BWM generators."""
    if family == "ledger_mini":
        a, b = state
        if not action["posted"]:
            return [a, b]
        k = action["amount"]
        return {"add": [a + k, b], "transfer": [a - k, b + k], "scale": [a * k, b]}[action["op"]]
    if family == "prerequisite_mini":
        return state | action["add"] if state & action["need"] == action["need"] else state
    if family == "register_mini":
        a, b = state
        k = action["amount"]
        return {"swap": [b, a], "add": [(a+k) % 97, b], "mix": [(a+b*k) % 97, b]}[action["op"]]
    raise ValueError("Unknown miniature domain")


def mini_bank(groups: int = 16, seed: int = 17, split: str = "development") -> dict:
    """Counterfactual pairs share parent/lineage so bootstrap cannot count them twice."""
    require(type(groups) is int and 1 <= groups <= 10000, "Invalid group count")
    rng, items, seen = random.Random(seed), [], set()
    for family in ("ledger_mini", "prerequisite_mini", "register_mini"):
        for group in range(groups):
            state = rng.randrange(16) if family == "prerequisite_mini" else [rng.randrange(10, 80), rng.randrange(10, 80)]
            amount = rng.randrange(1, 10)
            parent = digest([family, seed, group, state, amount])
            for variant in range(2):
                if family == "ledger_mini":
                    action = {"op": ("add", "transfer", "scale")[group % 3], "amount": amount, "posted": bool(variant)}
                    case, targeted = ("posted", False) if variant else ("void", True)
                    rule = "Apply op to the first balance only when posted is true; transfer moves amount to the second."
                elif family == "prerequisite_mini":
                    action = {"op": "bit_add", "need": rng.randrange(16), "add": rng.randrange(16)}
                    applicable = state & action["need"] == action["need"]
                    case, targeted = ("applicable", False) if applicable else ("inapplicable", True)
                    targeted |= applicable and state + action["add"] != state | action["add"]
                    rule = "If (state & need) == need return state | add; otherwise return state. Use bitwise operations."
                else:
                    action = {"op": "mix" if variant else ("swap" if group % 2 else "add"), "amount": amount}
                    case, targeted = ("composite", True) if variant else ("simple", False)
                    rule = "swap: exchange registers; add: r0=(r0+amount)%97; mix: r0=(r0+r1*amount)%97; keep other values."
                payload = {"family": family, "state": state, "action": action}
                if digest(payload) in seen:
                    continue
                seen.add(digest(payload))
                from .io import canonical
                target = canonical(transition(family, state, action)).decode()
                prompt = rule + "\n" + canonical(payload).decode() + "\nNext state JSON:\n"
                items.append({"item_id": digest([parent, variant]), "parent_id": parent, "lineage_id": parent,
                    "family": family, "operation": action["op"], "rule_case": case, "suite": "skill",
                    "format_id": "mini-json-v1", "split": split, "horizon": 1, "target_rule": bool(targeted),
                    "semantic_payload": payload, "prompt": prompt, "target": target,
                    "content_spans": [[m.start(), m.end()] for m in re.finditer(r"-?\d+|true|false", target)]})
    return seal({"schema_version": 1, "kind": "raw_bank", "evidence_class": "synthetic_fixture", "items": items})


class FixtureTokenizer:
    """Deterministic character tokenizer ONLY for synthetic software tests."""
    is_fast = True
    def __call__(self, text: str, **kwargs):
        return {"input_ids": [ord(c) + 2 for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}
