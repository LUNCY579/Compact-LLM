"""RRP2-PRES-DEV: the preservation / abstention DEV bank, and its own scoring contract.

This is the third lane. It exists to answer a question the Forge banks cannot: when a ladder arm
improves multi-step planning, does the model still give ordinary useful short answers, and does
it still abstain when the decisive fact is absent? A reasoning gain paid for with either of those
is not a gain.

Its contract differs from both Forge lanes, which is why it is a separate module rather than a
fourth branch inside `instrument.py`:

  Forge v1  answer is a plain string, abstention is a literal sentinel in the answer position
  Forge v2  answer is strict JSON, abstention is a structured `reference.status`
  here      answer is ordinary prose, abstention is ONE exact sentence, and a wrong answer and a
            wrong abstention are different failures that must never be added together

Three sub-banks are kept separate and separately reported, because their contracts differ:

  (a) residual_capability_v23 direct_reasoning dev   384 rows   prompt-contained reasoning
  (b) residual_capability_v23 procedural_rag   dev   192 rows   retrieval-grounded procedures,
                                                               and the only source on this
                                                               machine that separates
                                                               missing-input from conflict
  (c) v3 preservation slice, answer_or_abstain       192 rows   ordinary language, and 40
                                                               genuine minimal pairs

DEV ONLY. This bank can never be quoted as confirmation, for reasons recorded on the artifact
and enforced by `quarantine.py`: residual_v23 dev was read by the completed v2.3 run's dose
calibration and DEV/OOD selection, and every v3 slice row is `split: train` with
`human_review_status: pending`.

Torch-free by contract.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, seal, write_new

from . import quarantine
from .sources import stable_hash

# There are TWO abstention sentences in this data, not one, and the difference is load-bearing.
# Both are taken verbatim from the golds and from the `response_contract.requirements` the rows
# ship, so the model is told the contract it is scored against.
#
# Measured distribution of required abstention sentence, by source family:
#   direct_reasoning  insufficient_information  405/405 -> NO_EVIDENCE
#   direct_reasoning  conflicting_evidence      405/405 -> NO_EVIDENCE   <-- note this
#   procedural_rag    abstain (missing input)   180/180 -> NO_EVIDENCE
#   procedural_rag    conflict                  180/180 -> SOURCES_CONFLICT
#
# So `conflicting_evidence` in direct_reasoning requires the *missing-information* sentence: its
# cause is a contradiction but its contract is not distinguishable from absence. Only
# procedural_rag separates the two at the response level. See CORRECTIONS[4] (PRES-C5).
ABSTENTION_SENTENCES = {
    "NO_EVIDENCE": "I don't know based on the available information.",
    "SOURCES_CONFLICT": "The available sources conflict, so I can't give one certain answer.",
}
ABSTAIN = ABSTENTION_SENTENCES["NO_EVIDENCE"]

SALT = "RRP-002-pres-dev"
SEED = "20260915"

DECISIONS = ("answer", "abstain", "conflict")

# Registered sizes. Fixed before any prediction is scored, and not to be adjusted afterwards.
SIZES = {
    "direct_reasoning": {"answer": 192, "insufficient_information": 96, "conflicting_evidence": 96},
    "procedural_rag": {"answer": 96, "abstain": 48, "conflict": 48},
    "v3_answer_or_abstain": {"abstain": 96, "answer": 96},
}
TOTAL = 768

# Sub-bank (c) only: at most this many rows may come from one `template_family`, so no single
# surface template dominates. See CORRECTIONS[1] for why it is template_family and not
# base_template_family.
V3_TEMPLATE_CAP = 4

# Corrections to the registered design, forced by measurement. Recorded on the artifact so the
# difference between what was pre-registered and what was built is legible without a diff.
CORRECTIONS = (
    {"id": "PRES-C1",
     "registered": "(a) sampled one per (domain, reasoning_family, decision) cell by lowest "
                   "digest, giving 384 rows = 192 answer + 192 abstain.",
     "measured": "direct_reasoning dev has exactly 324 cells of exactly 15 rows "
                 "(27 domains x 12 reasoning_families), and `decision` is FULLY DETERMINED by "
                 "`reasoning_family`: 10 answerable families and 2 abstention families, so no "
                 "cell contains both decisions. One-per-cell therefore yields 324 rows split "
                 "270 answer / 54 abstain, not 384 split 192 / 192.",
     "resolution": "Keep the registered totals -- 192/192 is a deliberate property, it puts the "
                   "constant-abstain floor at exactly 0.5000 -- and replace the mechanism with a "
                   "deterministic largest-remainder allocation across families, then across "
                   "domains, then lowest digest within the cell. Per-cell counts are recorded in "
                   "full on the artifact.",
     "changes_a_number": "The mechanism, not the totals."},
    {"id": "PRES-C2",
     "registered": "(c) capped at <=4 rows per base_template_family.",
     "measured": "the answer_or_abstain stratum has only 5 distinct base_template_family values "
                 "over its 250 rows, so a <=4 cap admits at most 20 rows -- while the same line "
                 "requires 192. The cap as written is arithmetically impossible. There are 59 "
                 "distinct `template_family` values (max 9 rows in one).",
     "resolution": "Apply the cap to `template_family`, which admits at most 236 >= 192. Read as "
                   "base_template_family the registered design is self-contradictory; this is "
                   "the only reading under which both halves of the line can hold.",
     "changes_a_number": "No reported accuracy; it changes which rows are admitted."},
    {"id": "PRES-C3",
     "registered": "(c) 'the stratum only has 5 base families over 250 rows, so pairing is by "
                   "shared base template, NOT by a decisive-evidence swap -- state that "
                   "limitation on the artifact'.",
     "measured": "REFUTED by the data. Every row carries `counterfactual_group_id`. Among the 250 "
                 "answer_or_abstain rows there are 40 groups of size 2, and all 40 contain "
                 "exactly one exact-abstain row and one answerable row differing by the presence "
                 "of the decisive fact -- e.g. 'A car travels 30 mph for 4 hours. Distance?' -> "
                 "'120 miles.' against 'A car travels for 4 hours, but its speed is not given. "
                 "Distance?' -> the abstain string. That is precisely a decisive-evidence swap.",
     "resolution": "Admit all 40 pairs (80 rows) first and carry `pair_id` on them, then fill to "
                   "96/96 from the 85 unpaired abstain and 85 unpaired answerable rows. The "
                   "registered limitation is withdrawn for those 40 pairs and still stands for "
                   "the other 112 rows of (c), which are unpaired.",
     "changes_a_number": "It adds a stronger analysis unit than the design believed existed: 40 "
                         "within-pair abstention-calibration contrasts."},
    {"id": "PRES-C4",
     "registered": "HARD EXCLUSION: 'every v3 insufficient_information row that is not an EXACT "
                   "abstain-string final turn'.",
     "measured": "in the answer_or_abstain stratum all 250 rows carry "
                 "reply_class=insufficient_information, and exactly 125 have the exact abstain "
                 "string. The other 125 are their ANSWERABLE counterparts. Excluding all 125 "
                 "would leave 0 answerable rows, while the same design requires 96 of them.",
     "resolution": "Read the exclusion as what it was for -- a ban on RELABELLING a non-abstain "
                   "row as an abstention, which is the refusal MQ-001's build_pool.py already "
                   "recorded. So: a row is ABSTAIN if and only if its final assistant turn is "
                   "the exact string, and every other row is ANSWER scored against its own gold. "
                   "Nothing is relabelled and no row is dropped.",
     "changes_a_number": "Under the literal reading sub-bank (c) is unbuildable; this is the only "
                         "reading that is not self-contradictory."},
    {"id": "PRES-C5",
     "registered": "strata (2): 'abstention cause insufficient_information vs "
                   "conflicting_evidence, cross-tabulated', with the note that residual_v23 "
                   "procedural_rag is one of only two banks that make the distinction.",
     "measured": "the distinction exists in the METADATA of both sub-banks but in the RESPONSE "
                 "CONTRACT of only one. direct_reasoning requires "
                 "\"I don't know based on the available information.\" for 405/405 "
                 "insufficient_information rows AND for 405/405 conflicting_evidence rows -- the "
                 "identical sentence. procedural_rag is the only sub-bank with a second "
                 "sentence, \"The available sources conflict, so I can't give one certain "
                 "answer.\", required by 180/180 of its conflict rows and pinned in each row's "
                 "`response_contract.requirements`. My first scorer assumed one abstention "
                 "sentence and its own guard rejected the build, which is how this was found.",
     "resolution": "Score the required sentence PER ROW, derived from that row's gold, over a "
                   "registered two-sentence vocabulary. Report `wrong_abstention_kind` as its "
                   "own channel: emitting the missing-information sentence where the sources "
                   "contradict each other is a distinct failure from answering anyway, and "
                   "neither is a correct abstention. The metadata cross-tabulation still stands "
                   "as a difficulty contrast -- is abstention as reliable when the cause is "
                   "contradiction rather than absence -- but for direct_reasoning it is NOT a "
                   "contract distinction, and no report may imply that it is.",
     "changes_a_number": "It creates a channel that would otherwise have been silently merged "
                         "into 'correct abstention' for 48 rows of sub-bank (b)."},
    {"id": "PRES-C6",
     "registered": "'run MQ-001 data.overlap_audit ... at its zero tolerance on id / family / "
                   "NFKC-casefold-normalized first user turn / digit-blinded prompt'.",
     "measured": "at zero tolerance the bank refuses ITSELF. Against every other bank it is "
                 "perfectly clean -- 0 collisions on all four axes against sealed.jsonl (110), "
                 "sealed_roleplay_v2_DO_NOT_TRAIN.jsonl (400), dev.jsonl (510), "
                 "dev_roleplay_v2.jsonl (400) and the 32 selected TRAIN worlds. But INTERNALLY, "
                 "while all 768 rows pose distinct questions verbatim, digit blinding collapses "
                 "them to 589 clusters: 19 clusters hold 198 rows, the largest holding 18. All "
                 "the large clusters are sub-bank (b), whose 192 rows share just 16 digit-blinded "
                 "question templates -- one per procedure family per contract. And sub-bank (c)'s "
                 "minimal pairs collide by construction, since the pair IS the same sentence with "
                 "the decisive number removed.",
     "resolution": "Split one verdict into two. LEAKAGE against other banks stays at zero "
                   "tolerance and remains the build gate, together with 'no two rows pose the "
                   "identical question'. INTERNAL REDUNDANCY is measured and published rather "
                   "than gated, because deleting 198 legitimate rows would only be flattering a "
                   "statistic. Instead every row carries `cluster_id` = digit-blinded last user "
                   "turn, so the paired statistics layer clusters on it and cannot silently "
                   "choose a looser unit.",
     "changes_a_number": "Yes, and it is the largest correction here: the effective unit count "
                         "for sub-bank (b) is 16, not 192. Any interval computed on 192 "
                         "independent rows there would be far too narrow."},
)


# ----------------------------------------------------------------------------- normalization

def _norm_space(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def strict_key(text: str) -> str:
    """Byte-exact after NFKC and whitespace collapse. Nothing forgiven."""
    return _norm_space(text)


def tolerant_key(text: str) -> str:
    """The declared tolerance, and the whole of it: case, surrounding quotes, and one trailing
    sentence-final mark. It does NOT strip units, reorder words, or match substrings -- every
    one of those would hand credit to a wrong answer."""
    value = _norm_space(text).casefold()
    value = value.strip("\"'“”‘’")
    value = re.sub(r"[.!]+$", "", value).strip()
    return value


def abstention_kind(text: str, *, tolerant: bool = True) -> str | None:
    """Which registered abstention sentence this is, or None. The response contract, not the
    metadata cause -- those differ, and conflating them is PRES-C5."""
    key = tolerant_key if tolerant else strict_key
    wanted = key(text)
    for name, sentence in ABSTENTION_SENTENCES.items():
        if wanted == key(sentence):
            return name
    return None


def is_abstention(text: str, *, tolerant: bool = True) -> bool:
    return abstention_kind(text, tolerant=tolerant) is not None


def _pick(salt: str, value: str) -> str:
    return digest([SALT, SEED, salt, value])


def _order(rows, salt: str, key="id"):
    return sorted(rows, key=lambda r: (_pick(salt, str(r[key])), str(r[key])))


def _largest_remainder(total: int, groups: list[str], available: dict[str, int]) -> dict[str, int]:
    """Spread `total` over `groups` as evenly as the availability allows, deterministically."""
    if total > sum(available[g] for g in groups):
        raise ValueError(f"cannot draw {total} from {sum(available[g] for g in groups)} rows")
    base, remainder = divmod(total, len(groups))
    quota = {g: min(base, available[g]) for g in groups}
    # Hand out the remainder, then any shortfall from capped groups, in digest order.
    short = total - sum(quota.values())
    for group in sorted(groups, key=lambda g: (_pick("quota", g), g)) * (len(groups) + 2):
        if short <= 0:
            break
        if quota[group] < available[group]:
            quota[group] += 1
            short -= 1
    if sum(quota.values()) != total:
        raise ValueError("allocation failed to reach the registered total")
    _ = remainder
    return quota


def _messages(row) -> list[dict]:
    return row["messages"]


def _prompt(row) -> list[dict]:
    return [m for m in _messages(row) if m["role"] != "assistant"]


def _gold(row) -> str:
    return [m for m in _messages(row) if m["role"] == "assistant"][-1]["content"]


def _user_turns(row) -> list[str]:
    return [m["content"] for m in _messages(row) if m["role"] == "user"]


# ----------------------------------------------------------------------------- bank assembly

def _row(*, sub_bank, source_id, decision, cause, gold, prompt, strata, pair_id=None,
         aliases=()):
    """One admitted row. `answerable` and the required response are derived from the GOLD TEXT,
    never from a family name -- that is the whole of PRES-C4."""
    required = abstention_kind(gold, tolerant=False)
    if decision == "answer" and required is not None:
        raise ValueError(f"{source_id}: labelled answer but its gold IS the {required} sentence")
    if decision != "answer" and required is None:
        raise ValueError(f"{source_id}: labelled {decision} but its gold is no registered "
                         f"abstention sentence; relabelling is forbidden (PRES-C4)")
    return {
        "required_response_kind": required,
        "id": f"{sub_bank}:{source_id}",
        "source_id": source_id,
        "sub_bank": sub_bank,
        "decision": decision,
        "abstention_cause": cause,
        "answerable": decision == "answer",
        "reference_text": gold,
        "reference_aliases": list(aliases),
        "pair_id": pair_id,
        "group_id": pair_id or f"{sub_bank}:{source_id}",
        "prompt": prompt,
        "strata": strata,
    }


def build_direct_reasoning(rows) -> list[dict]:
    """(a) 192 answer + 96 insufficient_information + 96 conflicting_evidence."""
    want = SIZES["direct_reasoning"]
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[row["reasoning_family"]].append(row)

    abstain_families = ("insufficient_information", "conflicting_evidence")
    answer_families = sorted(f for f in by_family if f not in abstain_families)
    if len(answer_families) != 10 or any(f not in by_family for f in abstain_families):
        raise ValueError("direct_reasoning family structure is not the measured 10 + 2")

    def draw(families, total):
        available = {f: len(by_family[f]) for f in families}
        per_family = _largest_remainder(total, sorted(families), available)
        out = []
        for family in sorted(families):
            pool = by_family[family]
            by_domain = collections.defaultdict(list)
            for row in pool:
                by_domain[row["domain"]].append(row)
            domains = sorted(by_domain)
            per_domain = _largest_remainder(per_family[family], domains,
                                            {d: len(by_domain[d]) for d in domains})
            for domain in domains:
                out.extend(_order(by_domain[domain], f"a:{family}:{domain}")[:per_domain[domain]])
        return out

    picked = draw(answer_families, want["answer"])
    for family in abstain_families:
        picked += draw([family], want[family])

    admitted = []
    for row in picked:
        family = row["reasoning_family"]
        abstains = family in abstain_families
        if row["decision"] != ("abstain" if abstains else "answer"):
            raise ValueError(f"{row['id']}: decision/family disagree")
        admitted.append(_row(
            sub_bank="direct_reasoning", source_id=row["id"],
            decision="abstain" if abstains else "answer",
            cause=family if abstains else None,
            gold=_gold(row), prompt=_prompt(row),
            aliases=row.get("answer_aliases") or (),
            strata={"domain": row["domain"], "reasoning_family": family,
                    "difficulty": row["difficulty"], "surface_format": row["surface_format"],
                    "answer_stance": row["answer_stance"]}))
    if len(admitted) != sum(want.values()):
        raise ValueError(f"(a) drew {len(admitted)}, registered {sum(want.values())}")
    return admitted


def build_procedural_rag(rows) -> list[dict]:
    """(b) 96 answer + 48 missing-input abstain + 48 conflict. Conflict stays SEPARATE."""
    want = SIZES["procedural_rag"]
    by_decision = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        by_decision[row["decision"]][row["procedure_family"]].append(row)

    admitted = []
    for decision, total in (("answer", want["answer"]), ("abstain", want["abstain"]),
                            ("conflict", want["conflict"])):
        families = sorted(by_decision[decision])
        if len(families) != 8:
            raise ValueError(f"expected the measured 8 procedure families, got {len(families)}")
        quota = _largest_remainder(total, families,
                                   {f: len(by_decision[decision][f]) for f in families})
        for family in families:
            for row in _order(by_decision[decision][family], f"b:{decision}:{family}")[:quota[family]]:
                gold = _gold(row)
                # In (b) a conflict row's gold is also the abstain sentence, but its CAUSE is a
                # direct contradiction between equally authoritative sources, not a missing
                # input. Collapsing the two would erase the only such distinction on this machine.
                admitted.append(_row(
                    sub_bank="procedural_rag", source_id=row["id"],
                    decision=decision,
                    cause={"abstain": "missing_required_input",
                           "conflict": "direct_conflict_equal_authority"}.get(decision),
                    gold=gold, prompt=_prompt(row),
                    strata={"category": row["category"],
                            "procedure_family": row["procedure_family"],
                            "difficulty": row["difficulty"],
                            "evidence_clarity": row["evidence_clarity"],
                            "answer_stance": row["answer_stance"]}))
    if len(admitted) != sum(want.values()):
        raise ValueError(f"(b) drew {len(admitted)}, registered {sum(want.values())}")
    return admitted


def v3_pairs(rows) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Split the answer_or_abstain stratum into minimal pairs and unpaired remainders."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["metadata"]["counterfactual_group_id"]].append(row)
    pairs, lone_abstain, lone_answer = [], [], []
    for _, members in sorted(groups.items()):
        abstain = [r for r in members if is_abstention(_gold(r), tolerant=False)]
        answer = [r for r in members if not is_abstention(_gold(r), tolerant=False)]
        if len(members) == 2 and len(abstain) == 1 and len(answer) == 1:
            pairs.append((abstain[0], answer[0]))
        else:
            lone_abstain.extend(abstain)
            lone_answer.extend(answer)
    return pairs, lone_abstain, lone_answer


def build_v3(rows) -> list[dict]:
    """(c) 96 abstain + 96 answerable, pairs first, then a template-capped fill."""
    want = SIZES["v3_answer_or_abstain"]
    pairs, lone_abstain, lone_answer = v3_pairs(rows)

    def admit(row, decision, pair_id=None):
        meta = row["metadata"]
        return _row(
            sub_bank="v3_answer_or_abstain", source_id=row["id"], decision=decision,
            cause="insufficient_information" if decision == "abstain" else None,
            gold=_gold(row), prompt=_prompt(row), pair_id=pair_id,
            strata={"reply_class": meta["reply_class"], "domain": meta["domain"],
                    "difficulty": meta["difficulty"],
                    "length_mode_proxy_words_and_punctuation": meta["length_mode"],
                    "base_template_family": meta["base_template_family"],
                    "template_family": meta["template_family"],
                    "multi_turn": meta.get("multi_turn", False),
                    "human_review_status": meta["human_review_status"],
                    "source_split": meta["split"]})

    used = collections.Counter()
    admitted = []
    for index, (abstain, answer) in enumerate(pairs):
        pair_id = f"v3_pair:{index:03d}"
        admitted.append(admit(abstain, "abstain", pair_id))
        admitted.append(admit(answer, "answer", pair_id))
        used[abstain["metadata"]["template_family"]] += 1
        used[answer["metadata"]["template_family"]] += 1

    def fill(pool, decision, target):
        taken = 0
        # The cap applies to the whole sub-bank, pairs included, so it cannot be evaded by
        # spending the allowance on pairs and then ignoring it during the fill.
        for row in _order(pool, f"c:{decision}"):
            if taken >= target:
                break
            family = row["metadata"]["template_family"]
            if used[family] >= V3_TEMPLATE_CAP:
                continue
            used[family] += 1
            admitted.append(admit(row, decision))
            taken += 1
        if taken < target:
            raise ValueError(f"(c) could not fill {decision}: {taken}/{target} under the "
                             f"<={V3_TEMPLATE_CAP}-per-template_family cap")

    fill(lone_abstain, "abstain", want["abstain"] - len(pairs))
    fill(lone_answer, "answer", want["answer"] - len(pairs))
    if len(admitted) != sum(want.values()):
        raise ValueError(f"(c) drew {len(admitted)}, registered {sum(want.values())}")
    return admitted


# ----------------------------------------------------------------------------- admission audit

def text_keys(row) -> dict[str, str]:
    """Four axes. The design named the FIRST user turn; `overlap.keys` uses the LAST. Multi-turn
    rows exist in (c), so both are computed rather than picking one and hoping."""
    turns = [m["content"] for m in row["prompt"] if m["role"] == "user"] \
        if "prompt" in row else _user_turns(row)
    first, last = turns[0], turns[-1]
    out = {}
    for name, text in (("first_user_turn", first), ("last_user_turn", last)):
        normal = _norm_space(text).casefold()
        out[name] = digest(normal)
        out[name + "_digit_blinded"] = digest(re.sub(r"\d", "#", normal))
    return out


def admission_audit(bank: list[dict], others: dict[str, list[dict]]) -> dict:
    """Two different questions, deliberately not merged into one verdict.

    LEAKAGE, against every other named bank, at ZERO tolerance on all four axes. Any hit is a
    build failure: it would mean this DEV bank overlaps training or sealed material.

    INTERNAL REDUNDANCY, within the bank. Measured and reported, NOT gated at zero, because
    digit-blinded collision here is a DESIGNED property rather than a defect: sub-bank (c)'s
    minimal pairs differ by exactly the decisive number, and sub-bank (b)'s questions are one
    fixed template per procedure family with only the numbers varying. Gating that at zero would
    mean deleting legitimate rows to flatter a statistic. What it really bounds is effective
    sample size, so it is emitted as clusters for the statistics layer to consume instead.
    See CORRECTIONS[5] (PRES-C6).
    """
    axes = list(text_keys(bank[0]))
    bank_keys = {axis: collections.defaultdict(list) for axis in axes}
    ids = collections.Counter()
    for row in bank:
        ids[row["source_id"]] += 1
        for axis, value in text_keys(row).items():
            bank_keys[axis][value].append(row["id"])

    def cluster_sizes(axis):
        return dict(sorted(collections.Counter(
            len(v) for v in bank_keys[axis].values()).items()))

    per_sub_bank_clusters = {}
    for sub in sorted({r["sub_bank"] for r in bank}):
        rows = [r for r in bank if r["sub_bank"] == sub]
        groups = {text_keys(r)["last_user_turn_digit_blinded"] for r in rows}
        per_sub_bank_clusters[sub] = {"rows": len(rows), "digit_blinded_clusters": len(groups)}

    internal = {
        "duplicate_source_ids": sorted(k for k, v in ids.items() if v > 1),
        "distinct_questions_exact": len(bank_keys["last_user_turn"]),
        "duplicate_prompt_counts": {
            axis: sum(1 for v in bank_keys[axis].values() if len(v) > 1) for axis in axes},
        "cluster_size_distribution": {axis: cluster_sizes(axis) for axis in axes},
        "per_sub_bank_effective_clusters": per_sub_bank_clusters,
        "reading": ("Every one of the rows poses a distinct question verbatim "
                    "(distinct_questions_exact == rows). Under digit blinding they collapse, and "
                    "that collapse is the honest denominator for any template-driven claim: "
                    "sub-bank (b)'s rows share one question template per procedure family, so "
                    "its effective unit count is its cluster count, not its row count. Any "
                    "paired comparison must cluster on last_user_turn_digit_blinded."),
    }

    against = []
    for name, rows in sorted(others.items()):
        hits = {axis: [] for axis in axes}
        id_hits = []
        for row in rows:
            rid = str(row.get("id", row.get("parent_id", "")))
            if rid and rid in ids:
                id_hits.append(rid)
            try:
                keys = text_keys(row)
            except (KeyError, IndexError, StopIteration):
                continue
            for axis, value in keys.items():
                if value in bank_keys[axis]:
                    hits[axis].append(rid)
        against.append({"bank": name, "rows": len(rows), "id_collisions": sorted(set(id_hits)),
                        "text_collisions": {a: len(set(v)) for a, v in hits.items()},
                        "examples": {a: sorted(set(v))[:3] for a, v in hits.items() if v}})

    no_leakage = all(not b["id_collisions"] and not any(b["text_collisions"].values())
                     for b in against)
    # The gate: no repeated source row, no two rows posing the identical question, and no
    # overlap whatsoever with any other bank.
    gate = (not internal["duplicate_source_ids"]
            and internal["distinct_questions_exact"] == len(bank)
            and no_leakage)
    return {"axes": axes,
            "leakage_tolerance": "ZERO on every axis against every other bank",
            "redundancy_tolerance": ("MEASURED, NOT GATED -- digit-blinded collision inside the "
                                     "bank is designed, and is reported as effective clusters"),
            "internal": internal, "against": against,
            "no_leakage": no_leakage, "clean": gate}


# ----------------------------------------------------------------------------- scoring

def score(prediction: str, row: dict, *, tolerant: bool = True) -> dict:
    """One row. Answer correctness and abstention correctness are DIFFERENT channels."""
    key = tolerant_key if tolerant else strict_key
    predicted_kind = abstention_kind(prediction, tolerant=tolerant)
    required_kind = row["required_response_kind"]
    golds = [row["reference_text"], *row["reference_aliases"]]
    matches_gold = any(key(prediction) == key(g) for g in golds)

    if row["answerable"]:
        correct = matches_gold and predicted_kind is None
        reason = ("exact_match" if correct else
                  "false_abstention" if predicted_kind else "wrong_answer")
    else:
        # Abstaining for the wrong reason is neither correct nor a plain miss. A model that
        # says "I don't know" where the sources contradict each other has not read the
        # contradiction; lumping that in with a flat wrong answer would hide it.
        correct = predicted_kind == required_kind
        reason = ("correct_abstention" if correct else
                  "wrong_abstention_kind" if predicted_kind else "missing_abstention")
    return {"correct": correct, "reason": reason,
            "predicted_abstention": predicted_kind is not None,
            "predicted_abstention_kind": predicted_kind,
            "required_abstention_kind": required_kind,
            "matches_gold_text": matches_gold, "tolerant": tolerant}


def summarize(bank: list[dict], predictions: list[str], *, tolerant: bool = True) -> dict:
    if len(bank) != len(predictions):
        raise ValueError("one prediction per row, or the denominators are a guess")
    scored = [score(p, r, tolerant=tolerant) for p, r in zip(predictions, bank)]

    def bucket(fn):
        out = collections.defaultdict(lambda: {"n": 0, "correct": 0})
        for row, got in zip(bank, scored):
            cell = out[fn(row)]
            cell["n"] += 1
            cell["correct"] += got["correct"]
        return dict(out)

    answerable = [(r, s) for r, s in zip(bank, scored) if r["answerable"]]
    unanswerable = [(r, s) for r, s in zip(bank, scored) if not r["answerable"]]

    # Within-pair abstention calibration: only the 40 (c) pairs can support this, and only when
    # BOTH members are in the bank. It is the strictest available read -- the two prompts differ
    # by the decisive fact alone, so a model that abstains on both, or answers both, is shown to
    # be insensitive to evidence rather than merely inaccurate.
    pairs = collections.defaultdict(list)
    for row, got in zip(bank, scored):
        if row["pair_id"]:
            pairs[row["pair_id"]].append((row, got))
    complete = [v for v in pairs.values() if len(v) == 2]
    calibration = collections.Counter()
    for members in complete:
        abstained = {r["answerable"]: s["predicted_abstention"] for r, s in members}
        if abstained.get(False) and not abstained.get(True):
            calibration["evidence_sensitive"] += 1
        elif abstained.get(False) and abstained.get(True):
            calibration["abstains_regardless"] += 1
        elif not abstained.get(False) and not abstained.get(True):
            calibration["answers_regardless"] += 1
        else:
            calibration["inverted"] += 1

    return {
        "rows": len(bank),
        "correct": sum(s["correct"] for s in scored),
        "accuracy": round(sum(s["correct"] for s in scored) / len(bank), 4),
        "scored_by": ("preservation contract: an answerable row needs its own gold (or a shipped "
                      "alias) under the declared tolerance; an unanswerable row needs the exact "
                      "abstain sentence. Tolerance is case, wrapping quotes and one trailing "
                      "sentence mark -- no unit stripping, no substring credit."),
        "tolerant": tolerant,
        "answer_channel": {
            "n": len(answerable),
            "correct": sum(s["correct"] for _, s in answerable),
            "false_abstention": sum(s["reason"] == "false_abstention" for _, s in answerable),
            "wrong_answer": sum(s["reason"] == "wrong_answer" for _, s in answerable)},
        "abstention_channel": {
            "n": len(unanswerable),
            "correct": sum(s["correct"] for _, s in unanswerable),
            "abstained_at_all": sum(s["predicted_abstention"] for _, s in unanswerable),
            "wrong_abstention_kind": sum(s["reason"] == "wrong_abstention_kind"
                                         for _, s in unanswerable),
            "missing_abstention": sum(s["reason"] == "missing_abstention"
                                      for _, s in unanswerable)},
        "per_required_abstention_kind": bucket(
            lambda r: r["required_response_kind"] or "answerable"),
        "per_sub_bank": bucket(lambda r: r["sub_bank"]),
        "per_decision": bucket(lambda r: r["decision"]),
        "per_abstention_cause": bucket(lambda r: r["abstention_cause"] or "n/a"),
        "per_difficulty": bucket(lambda r: r["strata"].get("difficulty", "n/a")),
        "reasons": dict(collections.Counter(s["reason"] for s in scored)),
        "minimal_pair_calibration": {"complete_pairs": len(complete), **dict(calibration)},
        "quality_claim": False,
        "confirmation_eligible": False,
    }


def positive_control(bank: list[dict]) -> dict:
    """Score each gold against itself. Anything but 1.0000 means the bank is broken, not the
    model. Makes no model call."""
    got = summarize(bank, [r["reference_text"] for r in bank], tolerant=False)
    failures = [r["id"] for r, s in zip(bank, [score(r["reference_text"], r, tolerant=False)
                                               for r in bank]) if not s["correct"]]
    return {"status": "PASS" if not failures else "FAIL", "accuracy": got["accuracy"],
            "failed": len(failures), "failures": failures[:10], "model_calls": 0}


def shortcut_predictions(bank: list[dict], *, train_majority: str = ABSTAIN) -> dict:
    """Prompt-only baselines. None may read a gold; `test_shortcuts_are_answer_agnostic` proves
    it by deleting every reference field and requiring identical output."""
    def last_number(row):
        turns = [m["content"] for m in row["prompt"] if m["role"] == "user"]
        found = re.findall(r"-?\d+(?:\.\d+)?", " ".join(turns))
        return found[-1] if found else "0"

    def last_capitalized(row):
        turns = [m["content"] for m in row["prompt"] if m["role"] == "user"]
        found = re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(turns))
        return (found[-1] + ".") if found else "Unknown."

    return {
        "constant_abstain": [ABSTAIN] * len(bank),
        "constant_conflict_sentence": [ABSTENTION_SENTENCES["SOURCES_CONFLICT"]] * len(bank),
        "constant_train_majority": [train_majority] * len(bank),
        "constant_yes": ["Yes."] * len(bank),
        "copy_last_number": [last_number(r) for r in bank],
        "copy_last_capitalized_entity": [last_capitalized(r) for r in bank],
        "empty": [""] * len(bank),
        "prose_refusal": ["I'm not able to help with that."] * len(bank),
    }


def floor(bank: list[dict], *, tolerant: bool = True) -> dict:
    """The number any preservation claim must clear. Reported per sub-bank too, because the
    sub-banks have different abstain base rates and one number would hide that."""
    results = {}
    for name, predictions in sorted(shortcut_predictions(bank).items()):
        got = summarize(bank, predictions, tolerant=tolerant)
        results[name] = {"accuracy": got["accuracy"], "correct": got["correct"],
                         "per_sub_bank": {k: v["correct"] for k, v in got["per_sub_bank"].items()}}
    best = max(results, key=lambda k: results[k]["accuracy"])
    return {"baselines": results, "shortcut_floor": results[best],
            "shortcut_floor_predictor": best,
            "meaning": ("A preservation result at or below this is not evidence of preserved "
                        "capability. constant_abstain is the one to watch: the bank is built "
                        "half-abstain by construction, so it scores about 0.5 for free.")}


# ----------------------------------------------------------------------------- build entrypoint

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


V23 = ("short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
       "residual_capability_v23")
V3_SLICE = ("Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/"
            "rineya_v1_v2_preservation_slice_14000.jsonl")
SOURCES = {
    "direct_reasoning": V23 + "/direct_reasoning/splits/dev.reviewed.jsonl",
    "procedural_rag": V23 + "/procedural_rag/splits/dev.reviewed.jsonl",
    "v3_answer_or_abstain": V3_SLICE,
}


def build(worktree: Path, registry: dict) -> tuple[list[dict], dict]:
    worktree = Path(worktree)
    for relative in SOURCES.values():
        quarantine.assert_readable(registry, relative)

    raw = {name: read_jsonl(worktree / relative) for name, relative in SOURCES.items()}
    v3_stratum = [r for r in raw["v3_answer_or_abstain"]
                  if r["metadata"].get("answer_type") == "answer_or_abstain"]

    bank = (build_direct_reasoning(raw["direct_reasoning"])
            + build_procedural_rag(raw["procedural_rag"])
            + build_v3(v3_stratum))
    if len(bank) != TOTAL:
        raise ValueError(f"built {len(bank)} rows, registered {TOTAL}")

    # Carry the clustering unit on every row. The statistics layer must not have to re-derive
    # it, and it must not be able to quietly choose a looser one.
    for row in bank:
        row["cluster_id"] = text_keys(row)["last_user_turn_digit_blinded"][:16]

    # DEV is the only use this bank claims. The residual_v23 dev entries bar TRAIN and
    # CONFIRMATION, so a refusal here would mean someone had widened the claim.
    quarantine.refuse([r["source_id"] for r in bank], registry, use="DEV")

    others = {
        "rrp2_train_candidates/microfit-train": _maybe(worktree, "research/"
                                                       "reasoning_posttraining_v1/evidence"),
        **{f"phase012/{p.parent.name}/{p.name}": read_jsonl(p)
           for p in sorted((worktree / "short-response-model-phase-0-1-2/data/sealed").glob("*.jsonl"))
           + sorted((worktree / "short-response-model-phase-0-1-2/data/dev").glob("*.jsonl"))},
    }
    others = {k: v for k, v in others.items() if v}
    audit = admission_audit(bank, others)
    control = positive_control(bank)
    if control["status"] != "PASS":
        raise ValueError(f"positive control failed on {control['failed']} rows: a broken bank "
                         f"is not a model finding")
    if not audit["clean"]:
        raise ValueError("admission audit is not clean; see the report before admitting")

    report = seal({
        "record": "RRP2_PRES_DEV_BUILD",
        "schema": 1,
        "bank": "RRP2-PRES-DEV",
        "what_this_is": ("Preservation / abstention DEV bank: ordinary useful responses plus "
                         "appropriate abstention, from two non-Forge ordinary-language corpora."),
        "dev_only": True,
        "confirmation_eligible": False,
        "why_not_confirmation": (
            "residual_capability_v23 dev was read by the completed v2.3 run's dose calibration "
            "and DEV/OOD selection, so it is selection-exposed; and every admitted v3 row is "
            "`split: train` with `human_review_status: pending`. Both bars are registered in "
            "QUARANTINE.json, which is why refuse(use='DEV') passes and use='CONFIRMATION' "
            "would not."),
        "human_review_status": "MACHINE_VERIFIED_NOT_HUMAN_REVIEWED",
        "salt": SALT, "seed": SEED,
        "registered_sizes": SIZES, "rows": len(bank),
        "sources": {name: {"path": relative, "sha256": stable_hash(worktree / relative),
                           "rows_read": len(raw[name])}
                    for name, relative in SOURCES.items()},
        "v3_stratum_rows": len(v3_stratum),
        "corrections_to_the_registered_design": list(CORRECTIONS),
        "composition": {
            "per_sub_bank": dict(collections.Counter(r["sub_bank"] for r in bank)),
            "per_decision": dict(collections.Counter(r["decision"] for r in bank)),
            "per_abstention_cause": dict(collections.Counter(
                r["abstention_cause"] or "n/a" for r in bank)),
            "answerable": sum(r["answerable"] for r in bank),
            "unanswerable": sum(not r["answerable"] for r in bank),
            "complete_minimal_pairs": len({r["pair_id"] for r in bank if r["pair_id"]}),
        },
        "per_cell_counts": {
            "direct_reasoning_family_domain": dict(collections.Counter(
                f"{r['strata']['reasoning_family']}|{r['strata']['domain']}"
                for r in bank if r["sub_bank"] == "direct_reasoning")),
            "procedural_rag_family_decision": dict(collections.Counter(
                f"{r['strata']['procedure_family']}|{r['decision']}"
                for r in bank if r["sub_bank"] == "procedural_rag")),
            "v3_template_family": dict(collections.Counter(
                r["strata"]["template_family"]
                for r in bank if r["sub_bank"] == "v3_answer_or_abstain")),
        },
        "admission_audit": audit,
        "positive_control": control,
        "floor_tolerant": floor(bank, tolerant=True),
        "floor_strict": floor(bank, tolerant=False),
        "quarantine_sha256": registry["sha256"],
        "model_calls": 0,
        "quality_claim": False,
    })
    return bank, report


def _maybe(worktree: Path, relative: str) -> list[dict]:
    """The 32 selected TRAIN worlds, if the prepared microfit is reachable from here."""
    for candidate in sorted(Path(worktree, relative).rglob("microfit-train.jsonl")):
        return read_jsonl(candidate)
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--out-bank", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)

    registry = quarantine.load(args.quarantine)
    bank, report = build(args.worktree, registry)

    if Path(args.out_bank).exists():
        raise ValueError("bank already exists; this build is create-only")
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in bank)
    Path(args.out_bank).write_bytes(body.encode("utf-8"))
    report = seal({**{k: v for k, v in report.items() if k != "sha256"},
                   "bank_file": Path(args.out_bank).name,
                   "bank_sha256": stable_hash(Path(args.out_bank))})
    write_new(args.out_report, report)

    print(f"rows {report['rows']}  answerable {report['composition']['answerable']}  "
          f"unanswerable {report['composition']['unanswerable']}  "
          f"pairs {report['composition']['complete_minimal_pairs']}")
    print(f"positive control {report['positive_control']['status']}  "
          f"admission audit clean={report['admission_audit']['clean']}")
    print(f"tolerant floor {report['floor_tolerant']['shortcut_floor']['accuracy']} "
          f"({report['floor_tolerant']['shortcut_floor_predictor']})")
    for name, value in sorted(report["floor_tolerant"]["baselines"].items(),
                              key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {name:30} {value['correct']:4}/{report['rows']}  {value['accuracy']:.4f}")


if __name__ == "__main__":
    main()
