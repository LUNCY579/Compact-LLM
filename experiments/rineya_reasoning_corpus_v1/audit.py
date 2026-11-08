"""Acceptance instrumentation: shortcut battery and native token census.

Every attack is FIT ON TRAIN and SCORED ON DEV, so a reported figure is held-out rather
than in-sample. The holdout split is never touched by this module.

The battery includes one probe the scout bank's own audit did not run: a FAMILY-CONDITIONED
information-gap classifier. The scout measured a single pooled `missing_marker` feature and
saw 0.302 against a 0.229 floor, which looks mild -- but conditioned on family that same
cue predicts abstention perfectly in three of its four families. Pooling hid it.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict

ANSWERS = ("0", "1", "2", "3", "UNKNOWN")
NAME_RE = re.compile(r"\bq[a-z]{7}\b")


def blind(text):
    return NAME_RE.sub("<E>", text)


def _majority(counter, default):
    if not counter:
        return default
    return sorted(counter, key=lambda a: (-counter[a], a))[0]


def _fit_score(train, dev, feature, label):
    """Single-feature majority classifier: fit on train, score on dev."""
    votes = defaultdict(Counter)
    for row in train:
        votes[feature(row)][label(row)] += 1
    prior = _majority(Counter(label(r) for r in train), "0")
    correct = 0
    for row in dev:
        correct += _majority(votes[feature(row)], prior) == label(row)
    return {"correct": correct, "n": len(dev),
            "accuracy": correct / len(dev) if dev else None}


def _balanced(train, dev, feature, label):
    """Balanced accuracy for a binary label, so class imbalance cannot inflate it."""
    votes = defaultdict(Counter)
    for row in train:
        votes[feature(row)][label(row)] += 1
    prior = _majority(Counter(label(r) for r in train), True)
    hits, totals = Counter(), Counter()
    for row in dev:
        truth = label(row)
        totals[truth] += 1
        hits[truth] += _majority(votes[feature(row)], prior) == truth
    per_class = {str(k): (hits[k] / totals[k] if totals[k] else None) for k in totals}
    usable = [v for v in per_class.values() if v is not None]
    return {"per_class_recall": per_class,
            "balanced_accuracy": sum(usable) / len(usable) if usable else None,
            "n": len(dev)}


def _digits(text):
    return re.findall(r"\d", text)


FEATURES = {
    "question_only": lambda r: blind(r["problem"].splitlines()[-1]),
    "family": lambda r: r["family"],
    "family_and_gap_marker": lambda r: (r["family"], "not recorded" in r["problem"]),
    "gap_marker_only": lambda r: "not recorded" in r["problem"],
    "length_bin": lambda r: len(r["problem"]) // 100,
    "line_count": lambda r: len(r["problem"].splitlines()),
    "first_digit": lambda r: (_digits(r["problem"]) or [""])[0],
    "last_digit": lambda r: (_digits(r["problem"]) or [""])[-1],
    "modal_digit": lambda r: _majority(Counter(_digits(r["problem"])), ""),
    "digit_sum_mod4": lambda r: sum(map(int, _digits(r["problem"]))) % 4,
    "digit_count": lambda r: len(_digits(r["problem"])),
    "family_and_difficulty": lambda r: (r["family"], r["difficulty"]),
    "tier": lambda r: r["tier"],
}


def shortcuts(rows, gold):
    labels = {g["id"]: g for g in gold}
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    answer = lambda r: labels[r["id"]]["answer"]                      # noqa: E731
    unanswerable = lambda r: not labels[r["id"]]["answerable"]        # noqa: E731

    exact = {name: _fit_score(train, dev, feature, answer)
             for name, feature in FEATURES.items()}
    constant = Counter(answer(r) for r in dev)
    floor = max(constant.values()) / len(dev)
    exact["constant_answer"] = {"correct": max(constant.values()), "n": len(dev),
                                "accuracy": floor}

    answerability = {name: _balanced(train, dev, feature, unanswerable)
                     for name, feature in FEATURES.items()}

    per_family = {}
    for family in sorted({r["family"] for r in rows}):
        ftrain = [r for r in train if r["family"] == family]
        fdev = [r for r in dev if r["family"] == family]
        counts = Counter(answer(r) for r in fdev)
        per_family[family] = {
            "n_dev": len(fdev),
            "constant_answer_floor": max(counts.values()) / len(fdev) if fdev else None,
            "gap_marker_balanced_accuracy":
                _balanced(ftrain, fdev, FEATURES["gap_marker_only"], unanswerable),
            "structural_gap_balanced_accuracy":
                _balanced(ftrain, fdev, lambda r: labels[r["id"]]["information_gap"],
                          unanswerable),
            "question_only_exact": _fit_score(ftrain, fdev, FEATURES["question_only"], answer),
        }

    return {
        "method": ("single-feature majority classifiers; fit on train, scored on dev; "
                   "holdout never read"),
        "exact_answer_attacks": exact,
        "answerability_attacks_balanced": answerability,
        "per_family": per_family,
        "constant_answer_floor_dev": floor,
        "answer_priors": {
            "all": dict(Counter(g["answer"] for g in gold)),
            **{split: dict(Counter(g["answer"] for g in gold if g["split"] == split))
               for split in ("train", "dev", "holdout")}},
        "gap_conditional": {
            family: {
                "P_unknown_given_gap": _conditional(gold, family, True),
                "P_unknown_given_no_gap": _conditional(gold, family, False)}
            for family in sorted({g["family"] for g in gold})},
        "limitation": ("A bounded battery of shallow attacks, not a proof against every "
                       "shortcut. It reports held-out figures for the specific cues that "
                       "the evaluation bank was found to leak."),
    }


def _conditional(gold, family, gap):
    subset = [g for g in gold if g["family"] == family and g["information_gap"] is gap]
    if not subset:
        return None
    return sum(not g["answerable"] for g in subset) / len(subset)


# --------------------------------------------------------------------- token census

def token_census(rows, gold, sft, tokenizer_path, limits=(2048, 1920)):
    """Exact native rendered-chat census. No estimate, no truncation, no dropped row."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    total_cap, prompt_cap = limits
    by_id = {s["id"]: s for s in sft}
    labels = {g["id"]: g for g in gold}
    prompts, totals, completions = [], [], []
    violations = []
    for row in rows:
        entry = by_id[row["id"]]
        prompt_ids = tokenizer.apply_chat_template(
            entry["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        completion_ids = tokenizer.encode(entry["completion"], add_special_tokens=False)
        # The served completion is the answer followed by the EOS the model must emit.
        total = len(prompt_ids) + len(completion_ids) + 1
        prompts.append(len(prompt_ids))
        completions.append(len(completion_ids) + 1)
        totals.append(total)
        if len(prompt_ids) > prompt_cap or total > total_cap:
            violations.append({"id": row["id"], "family": row["family"], "tier": row["tier"],
                               "prompt_tokens": len(prompt_ids), "total_tokens": total})
    def stats(values):
        ordered = sorted(values)
        return {"min": ordered[0], "max": ordered[-1],
                "mean": round(sum(ordered) / len(ordered), 2),
                "p50": ordered[len(ordered) // 2],
                "p95": ordered[int(.95 * len(ordered))], "p99": ordered[int(.99 * len(ordered))]}
    per_family = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [t for r, t in zip(rows, totals) if r["family"] == family]
        per_family[family] = stats(subset)
    supervised = sum(completions)
    return {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_class": type(tokenizer).__name__,
        "caps": {"rendered_total": total_cap, "generation_prefix_prompt": prompt_cap},
        "prompt_tokens": stats(prompts), "total_tokens": stats(totals),
        "completion_tokens": stats(completions),
        "per_family_total_tokens": per_family,
        "violations": violations, "violation_count": len(violations),
        "rows": len(rows), "truncated_rows": 0, "dropped_rows": 0,
        "processed_tokens_one_epoch": sum(totals),
        "supervised_tokens_one_epoch": supervised,
        "supervised_fraction": round(supervised / sum(totals), 6),
        "note": ("Supervision is the bare answer plus EOS, matching the frozen MRS-2 serving "
                 "contract. The supervised fraction is therefore very small by design; the "
                 "prompt is context, not a target."),
        "unlabelled_rows": [r["id"] for r in rows if r["id"] not in labels][:10],
    }


# ------------------------------------------------- semantic equivalence disclosure

def semantic_equivalence(rows, gold, worlds):
    """How much structural isolation actually buys, per family.

    Surfaced by independent review and disclosed rather than assumed away: every
    sequence of ADD and XOR operations on Z4 composes to ONE of just eight maps (the
    order-8 dihedral group), so a register world's answer is fixed by the pair
    (initial value, net map) -- only 40 classes for ADD/XOR-only rows. Sequence-level
    fingerprint isolation between train and holdout therefore does NOT give semantic
    isolation for those two families.

    This is a limitation on what the split tests, not a leak: the net map is never
    written down, so a model must still COMPOSE the listed operations to recover it.
    The holdout measures that composition on unseen sequences, which is the intended
    skill -- much as testing addition on unseen operand pairs is still a real test.
    SET and NOT RECORDED break the group, so extended rows add classes beyond the 40.
    """
    from .families import _register_ops, _register_states, apply_op

    def net_map(world):
        signature = []
        for value in range(4):
            values = {value}
            for op in _register_ops(world):
                values = apply_op(values, op)
            signature.append(tuple(sorted(values)))
        return tuple(signature)

    by_id = {entry["id"]: entry["world"] for entry in worlds}
    labels = {g["id"]: g for g in gold}
    out = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        if family in ("register_machine", "counterfactual_state"):
            classes = {r["id"]: (by_id[r["id"]]["initial"], net_map(by_id[r["id"]]))
                       for r in subset}
        else:
            # No comparable collapse: these answers depend on the whole dependency DAG or
            # route-and-record structure, which the fingerprint already canonicalises.
            classes = {r["id"]: None for r in subset}
        if classes and next(iter(classes.values())) is None:
            out[family] = {"collapses": False,
                           "note": "answer depends on the full structure; fingerprint is the class"}
            continue
        per_split = defaultdict(set)
        for row in subset:
            per_split[row["split"]].add(classes[row["id"]])
        train, holdout = per_split["train"], per_split["holdout"]
        # An answer-determining class must be consistent, or a fingerprint would carry
        # two different gold answers, which would be an outright labelling bug.
        answers = defaultdict(set)
        for row in subset:
            answers[classes[row["id"]]].add(labels[row["id"]]["answer"])
        inconsistent = {str(k): sorted(v) for k, v in answers.items() if len(v) > 1}
        out[family] = {
            "collapses": True,
            "distinct_semantic_classes": len(set(classes.values())),
            "classes_in_train": len(train), "classes_in_holdout": len(holdout),
            "holdout_classes_also_in_train": len(holdout & train),
            "holdout_classes_unseen_in_train": len(holdout - train),
            "semantic_isolation": False,
            "class_answer_inconsistencies": inconsistent,
            "note": ("Sequence-level novelty only. The net map is never rendered, so the "
                     "holdout still tests composition on unseen operation sequences."),
        }
    return out
