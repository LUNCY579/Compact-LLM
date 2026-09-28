"""Population acceptance, label isolation, teacher verification, and matched F/P/T views.

This module never opens a quarantined bank to perform an overlap audit. Supply an
already-authorized fingerprint registry. A missing registry is BLOCKED, not zero overlap.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from .integrity import digest, require, valid_hash

SPLITS = {"TRAIN", "CAL", "CHECK"}
LCR_COUNTS = {"alias_resolution": 1200, "causal_disambiguation": 1000, "rule_exception": 1000,
              "count_with_exclusions": 800, "temporal_correction": 700,
              "object_state_tracking": 500, "latest_verified_record": 400,
              "multi_condition_identity": 400}

@dataclass(frozen=True)
class PublicCase:
    """The ONLY task object accepted by inference and teacher-generation functions."""
    id: str
    world_id: str
    system: str
    user: str

    def __post_init__(self):
        require(all(type(v) is str and v.strip() for v in (self.id, self.world_id, self.system, self.user)),
                "public case fields must be nonempty strings")

    @classmethod
    def from_row(cls, row: dict):
        messages = row["prompt"]
        require(len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"],
                "public prompt must contain exactly system/user turns")
        require(all(type(m["content"]) is str and m["content"].strip() for m in messages),
                "empty or nontext public prompt")
        require(all(type(row[k]) is str and row[k] for k in ("id", "world_id")), "missing IDs")
        return cls(row["id"], row["world_id"], messages[0]["content"], messages[1]["content"])

    def messages(self):
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]

    def identity(self):
        return digest({"id": self.id, "world_id": self.world_id, "prompt": self.messages()})


def normalize(text: str, blind_digits: bool = False) -> str:
    value = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return re.sub(r"\d+", "#", value) if blind_digits else value


def fingerprints(row: dict) -> dict[str, set[str]]:
    case = PublicCase.from_row(row)
    result = {k: {str(row[k])} for k in ("world_id", "structure_id", "template_id", "entity_pool_id")}
    for key, text in (("text", case.user), ("question", row["question"])):
        for mode in ("normalized", "digit_blinded"):
            normalized = normalize(text, mode == "digit_blinded")
            result[f"{key}_{mode}"] = {hashlib.sha256(normalized.encode()).hexdigest()}
        words = normalize(text).split()
        result[f"{key}_8gram"] = {digest(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
    return result


def population_overlap(rows: list[dict], external: dict | None) -> dict:
    by_split = {split: defaultdict(set) for split in SPLITS}
    collisions = []
    for row in rows:
        require(row["split"] in SPLITS, "protected/unknown split cannot enter open-data functions")
        for axis, values in fingerprints(row).items():
            by_split[row["split"]][axis].update(values)
    # Short repeated questions and common prose ngrams are diagnostics, not automatic
    # structural equivalence. Exact complete prompts and declared lineage MUST be disjoint.
    binding_axes = {"world_id", "structure_id", "template_id", "entity_pool_id", "text_normalized",
                    "text_digit_blinded"}
    for a, b in (("TRAIN", "CAL"), ("TRAIN", "CHECK"), ("CAL", "CHECK")):
        for axis in sorted(set(by_split[a]) | set(by_split[b])):
            n = len(by_split[a][axis] & by_split[b][axis])
            if n:
                collisions.append({"a": a, "b": b, "axis": axis, "count": n,
                                   "binding": axis in binding_axes})
    external_hits = {}
    if external is not None:
        require(external.get("schema") == "SCP_FINGERPRINT_REGISTRY_V1", "unsupported fingerprint registry")
        require(external.get("authorized_metadata_only") is True, "unapproved fingerprint registry")
        require(external.get("coverage") and all(valid_hash(x["manifest_sha256"]) for x in external["coverage"]),
                "fingerprint registry needs explicit source coverage")
        require(binding_axes <= set(external["axes"]), "incomplete external fingerprint axes")
        require(all(type(v) is list and all(type(x) is str for x in v) for v in external["axes"].values()),
                "invalid external fingerprint values")
        for axis, forbidden in external["axes"].items():
            present = set().union(*(by_split[s][axis] for s in SPLITS))
            external_hits[axis] = len(present & set(forbidden))
    return {"cross_split": collisions, "external_hits": external_hits,
            "external_registry_sha256": digest(external) if external is not None else None,
            "blocked_external_coverage": external is None,
            "binding_overlap": any(c["binding"] for c in collisions) or any(external_hits.values())}


def render_tokens(tokenizer, prompt: list[dict], target: str, *, limit=2048, reserve=128):
    """Measure BOTH native template calls; no truncation and no token-count proxy."""
    require(type(target) is str and bool(target), "empty target")
    require(type(limit) is int and limit > 0 and type(reserve) is int and 0 < reserve < limit, "invalid context budget")
    kwargs = {"tokenize": True, "enable_thinking": False}
    p = list(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **kwargs))
    s = list(tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": target}],
                                            add_generation_prompt=False, **kwargs))
    require(all(type(i) is int and i >= 0 for i in p + s), "invalid tokenizer IDs")
    require(p and len(s) > len(p) and s[:len(p)] == p, "native prefix/target boundary mismatch")
    require(len(s) <= limit and len(p) + reserve <= limit, "rendered sequence exceeds context contract")
    # Counts only. Native training masks (including EOS suffix) come from RRP compile_row.
    return {"ids": s, "prompt_ids": p,
            "rendered_total_tokens": len(s), "rendered_prompt_tokens": len(p),
            "completion_tokens": len(s) - len(p), "unused_capacity": limit - reserve - len(p)}


def _balanced(actual: list[bool], predicted: list[bool]):
    require(len(actual) == len(predicted), "classifier row mismatch")
    if not actual or not all(any(y == c for y in actual) for c in (False, True)):
        return None
    return sum(sum(p == c for y, p in zip(actual, predicted) if y == c) /
               sum(y == c for y in actual) for c in (False, True)) / 2


def _words(text):
    return re.findall(r"\w+", normalize(text))


def shortcut_audit(train: list[dict], check: list[dict]) -> dict:
    """Train-only NB/threshold choices, evaluated on group-disjoint open data.

    Numeric failures are never fixed by dropping selected rows. This is a necessary
    acceptance instrument, not an exhaustive proof that no shortcut exists.
    """
    require(train and check, "shortcut adversary needs training and open-check rows")
    require(not ({r["world_id"] for r in train} & {r["world_id"] for r in check}), "adversary world leakage")
    y = [r["reference"] == "UNKNOWN" for r in train]
    z = [r["reference"] == "UNKNOWN" for r in check]
    require(set(y) == {False, True}, "adversary training needs both answerability classes")
    counts = {c: Counter(w for r, label in zip(train, y) if label == c for w in _words(r["question"]))
              for c in (False, True)}
    vocabulary = set(counts[False]) | set(counts[True])
    totals = {c: sum(counts[c].values()) + len(vocabulary) + 1 for c in (False, True)}
    def classify(row):
        # Balanced priors avoid making majority-class prevalence look like signal.
        scores = {c: sum(math.log((counts[c][w] + 1) / totals[c]) for w in _words(row["question"]))
                  for c in (False, True)}
        return scores[True] > scores[False]
    surface = {
        "question_chars": lambda r: len(r["question"]),
        "question_words": lambda r: len(_words(r["question"])),
        "prompt_chars": lambda r: len(PublicCase.from_row(r).user),
        "digit_density": lambda r: sum(c.isdigit() for c in PublicCase.from_row(r).user) /
                                    max(1, len(PublicCase.from_row(r).user)),
        "punct_density": lambda r: sum(c in ";=:|/" for c in PublicCase.from_row(r).user) /
                                    max(1, len(PublicCase.from_row(r).user)),
    }
    feature_results = {}
    for name, feature in surface.items():
        values = [feature(r) for r in train]
        # All training cut-points and both orientations, with deterministic tie order.
        candidates = [(float(_balanced(y, [(v >= cut) == direction for v in values])), cut, direction)
                      for cut in sorted(set(values)) + [max(values) + 1]
                      for direction in (False, True)]
        best = max(candidates, key=lambda c: (c[0], -c[1], not c[2]))
        predictions = [(feature(r) >= best[1]) == best[2] for r in check]
        feature_results[name] = {"train_balanced_accuracy": best[0], "cut": best[1],
                                 "direction": best[2], "check_balanced_accuracy": _balanced(z, predictions)}
    question_predictions = [classify(r) for r in check]
    return {"question_only_balanced_accuracy": _balanced(z, question_predictions),
            "surface": feature_results, "train_worlds": len({r["world_id"] for r in train}),
            "check_worlds": len({r["world_id"] for r in check}),
            "subgroups": {family: _balanced([z[i] for i, r in enumerate(check) if r["family"] == family],
                [question_predictions[i] for i, r in enumerate(check) if r["family"] == family])
                for family in sorted({r["family"] for r in check})},
            "classification": "OPEN_DATA_PRE_MODEL_ADVERSARY", "model_calls": 0}


def audit_rows(rows, tokenizer, verifier, *, tokenizer_identity: dict,
               external_fingerprints=None, profile="targeted_pilot", human_review_sha256=None):
    """An independent verifier receives no gold. Returns per-row failures, not filtered data."""
    require(profile in ("targeted_pilot", "lcr_v2"), "unknown data profile")
    require(rows and len({r["id"] for r in rows}) == len(rows), "empty/duplicate corpus")
    require(all(r.get("split") in SPLITS for r in rows), "protected split: refuse before verifier/tokenizer calls")
    require(callable(verifier), "independent verifier is required")
    failures, measured = [], []
    for row in rows:
        try:
            require(row["split"] in SPLITS, "forbidden split")
            case = PublicCase.from_row(row)
            # Deliberately pass only public inputs and world specification, never reference.
            answer, steps = verifier(case, row["task"])
            require(answer == row["reference"], "independent answer disagreement")
            require(type(steps) is list and all(type(s) is str for s in steps), "invalid solver process")
            tokenized = render_tokens(tokenizer, case.messages(), row["reference"])
            text = case.user
            raw_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            require(raw_tokens > 0, "empty raw tokenization")
            record = {"id": row["id"], **{k: v for k, v in tokenized.items() if k not in ("ids", "labels", "prompt_ids")},
                      "chars_per_token": len(text) / raw_tokens,
                      "punct_density": sum(c in ";=:|/" for c in text) / len(text),
                      "digit_density": sum(c.isdigit() for c in text) / len(text)}
            if profile == "lcr_v2":
                require(row["family"] in LCR_COUNTS, "wrong LCR v2 taxonomy")
                require(record["chars_per_token"] >= 4.6 and record["punct_density"] <= .010 and
                        record["digit_density"] <= .020, "LCR prose-style constraint")
                n = record["rendered_prompt_tokens"]
                band = "short" if 600 <= n < 1000 else "medium" if 1000 <= n < 1400 else "long" if 1400 <= n <= 1920 else None
                require(band is not None and row["length_band"] == band, "LCR rendered band mismatch")
            measured.append(record)
        except (KeyError, ValueError, TypeError) as exc:
            failures.append({"id": row.get("id"), "reason": str(exc)})
    overlap = population_overlap(rows, external_fingerprints)
    blockers = []
    if overlap["blocked_external_coverage"]:
        blockers.append("authorized external quarantine fingerprint coverage missing")
    if overlap["binding_overlap"]:
        blockers.append("binding cross-split/external overlap")
    if tokenizer_identity.get("kind") != "native" or not valid_hash(tokenizer_identity.get("asset_manifest_sha256")):
        blockers.append("native tokenizer asset identity missing; fixture counts are not acceptance")
    if not valid_hash(human_review_sha256):
        blockers.append("human review receipt missing")
    train = [r for r in rows if r["split"] == "TRAIN"]
    check = [r for r in rows if r["split"] == "CAL"]
    adversary = None
    try:
        adversary = shortcut_audit(train, check)
        rates = [adversary["question_only_balanced_accuracy"]] + [r["check_balanced_accuracy"] for r in adversary["surface"].values()] + list(adversary["subgroups"].values())
        if any(r is None or r > .55 for r in rates):
            blockers.append("shortcut >0.55 or subgroup answerability not assessable")
    except (KeyError, ValueError, TypeError) as exc:
        blockers.append(f"shortcut audit blocked: {exc}")
    if profile == "lcr_v2":
        if Counter(r["family"] for r in rows) != Counter(LCR_COUNTS):
            blockers.append("LCR v2 exact family counts differ")
        if Counter(r["split"] for r in rows) != {"TRAIN": 4800, "CAL": 600, "CHECK": 600}:
            blockers.append("LCR 4800/600/600 partition differs")
        if Counter(r.get("length_band") for r in rows) != {"short": 2000, "medium": 2000, "long": 2000}:
            blockers.append("LCR length-band counts differ")
        # Rich LCR-specific structure/position/heuristic checks remain separately required.
        blockers.append("LCR eight-family structural, position, and answer-shortcut receipt integration pending")
    return {"schema": "SCP_DATA_AUDIT_V1", "corpus_sha256": digest(rows), "profile": profile,
            "row_bindings": [{"id": r["id"], "world_id": r["world_id"], "split": r["split"],
                              "public_sha256": PublicCase.from_row(r).identity(),
                              "reference_sha256": digest(r["reference"])} for r in rows],
            "rows_total": len(rows), "rows_verified_and_fitting": len(measured),
            "failures": failures, "measurements": measured, "overlap": overlap,
            "shortcut_audit": adversary, "blockers": blockers,
            "mechanical_pass": not failures and not overlap["binding_overlap"],
            "candidate_ready_for_review": not failures and not blockers,
            "training_ready": False, "native_execution_permitted": False,
            "tokenizer_identity": tokenizer_identity, "human_review_sha256": human_review_sha256,
            "quality_claim": False, "model_calls": 0}


def verify_teacher(row, record, teacher_sha256, step_checker):
    case = PublicCase.from_row(row)
    require(valid_hash(teacher_sha256), "teacher checkpoint identity missing")
    require(record["public_sha256"] == case.identity(), "teacher prompt/input mismatch")
    require(record["teacher_checkpoint_sha256"] == teacher_sha256, "teacher checkpoint mismatch")
    require(record["answer"] == row["reference"], "incorrect teacher final answer")
    require(record.get("terminated") is True, "teacher did not terminate")
    require(type(record.get("steps")) is list and record["steps"] and
            all(type(s) is str and s.strip() for s in record["steps"]), "teacher process missing/invalid")
    require(step_checker(case, row["task"], tuple(record["steps"])) is True, "teacher steps not independently valid")
    return {"id": row["id"], "record_sha256": digest(record), "public_sha256": case.identity(),
            "teacher_checkpoint_sha256": teacher_sha256, "steps": list(record["steps"]),
            "answer": record["answer"], "steps_verified": True}


def matched_views(rows, solver, *, arms=("F", "P"), teacher_records=None,
                  teacher_sha256=None, step_checker=None, preservation_rows=()):
    """Same admitted worlds across F/P/T; NO teacher-success filtering or silent repairs.

    FINAL_LINE_V1 is a NEW prospective output contract, not the old native exact
    contract. Retention anchors retain their own original prompt and target bytes.
    """
    require(set(arms) <= {"F", "P", "T"} and len(arms) == len(set(arms)) and arms, "invalid arm set")
    require(rows and len({r["id"] for r in rows}) == len(rows), "empty/duplicate worlds")
    result = {arm: [] for arm in arms}
    for row in rows:
        require(row["split"] == "TRAIN", "only TRAIN may produce training presentations")
        require(row.get("output_contract") == "FINAL_LINE_V1", "new task output contract must be explicit")
        case = PublicCase.from_row(row)
        answer, steps = solver(case, row["task"])
        require(answer == row["reference"], "solver disagrees")
        require(type(steps) is list and steps and all(type(s) is str and s.strip() for s in steps), "solver process missing")
        require(type(answer) is str and answer and "\n" not in answer and "FINAL:" not in answer, "invalid final target")
        targets = {"F": "FINAL: " + answer, "P": "\n".join([*steps, "FINAL: " + answer])}
        if "T" in arms:
            require(teacher_records is not None and row["id"] in teacher_records, "teacher missing; do not filter the population")
            verified = verify_teacher(row, teacher_records[row["id"]], teacher_sha256, step_checker)
            targets["T"] = "\n".join([*verified["steps"], "FINAL: " + answer])
        for arm in arms:
            require(targets[arm].count("FINAL:") == 1, "ambiguous final marker in process")
            result[arm].append({"id": row["id"], "world_id": row["world_id"], "family": row["family"],
                "source_split": "train", "role": "acquisition", "prompt": case.messages(),
                "reference": targets[arm], "canonical_answer": answer, "public_sha256": case.identity()})
    ids = {r["id"] for r in rows}
    for anchor in preservation_rows:
        require(anchor["split"] == "TRAIN" and anchor["id"] not in ids, "invalid or duplicate preservation anchor")
        ids.add(anchor["id"])
        case = PublicCase.from_row(anchor)
        for arm in arms:
            result[arm].append({"id": anchor["id"], "world_id": anchor["world_id"], "family": anchor["family"],
                "source_split": "train", "role": "preservation", "prompt": case.messages(),
                "reference": anchor["reference"], "canonical_answer": anchor["reference"], "public_sha256": case.identity()})
    return result
