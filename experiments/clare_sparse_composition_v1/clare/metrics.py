"""Generation-scored analysis for CLARE-RINEYA-001: accuracy, retention, composition, paired
contrasts and the damage-denominator rule. Pure numpy/stdlib (no torch, no model).

Every function consumes SCORED RECORDS, one per (configuration, row)::

    {"arm": str, "seed": int, "order": str, "config": str, "stage": int, "row_id": str,
     "parent_id": str, "signature": str, "correct": 0|1, "terminated": bool}

``config`` is one of

    "base"        the base model, stage 0
    "prefix:k"    the composed model after the first k skills of ``order`` (1 <= k <= len(order)),
                  stage k
    "single:X"    skill X's delta alone, stage -1
    "pair:X+Y"    the two deltas X and Y together (X != Y), stage -1

Records are validated before anything is counted and the analysis REFUSES (``MetricsError``,
a ``ValueError``) rather than guessing: a missing key, a non-0/1 ``correct`` (the string "0" is
truthy), a stage that disagrees with its config, a prefix stage beyond the order's length, an
unknown config label or signature letter, the same row scored twice under one
(arm, seed, order, config), or one ``row_id`` carrying two different ``parent_id``/``signature``
values anywhere in the input.

What every number here measures: exact-verification accuracy of greedy generation (``correct``
is the dataset verifier's 0/1) over a FIXED row set. None of it says why a row changed, whether
a gain is capability or a surface convention, or whether an effect generalises beyond these rows.
Accuracies are never compared across different row sets: every function that contrasts two
configurations requires the SAME rows on both sides and refuses otherwise (no intersections, no
dropped rows).
"""
from __future__ import annotations

import csv
import io as _stdio
import math
import numbers
import re
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import SKILLS
from .io import write_bytes

RECORD_FIELDS = ("arm", "seed", "order", "config", "stage", "row_id", "parent_id", "signature",
                 "correct", "terminated")
POOLED = "__pooled__"
DEFAULT_SEED = 20260927
DEFAULT_N_BOOT = 10000
BOOT_CHUNK = 1000  # bootstrap draws are generated in blocks of this many; part of the RNG contract

RETENTION_COLUMNS = ("arm", "seed", "order", "skill", "own_stage", "stage", "n", "correct", "acc",
                     "acquisition", "final", "forgetting_final", "forgetting_max", "backward_transfer")
COMPOSITION_COLUMNS = ("arm", "seed", "order", "signature", "stage", "covered", "first_covered_stage",
                       "n", "correct", "acc", "final")

_PREFIX = re.compile(r"prefix:([1-9][0-9]*)")
_SINGLE = re.compile(r"single:([A-Z])")
_PAIR = re.compile(r"pair:([A-Z])\+([A-Z])")
_STR_FIELDS = ("arm", "order", "config", "row_id", "parent_id", "signature")


class MetricsError(ValueError):
    """The records cannot support the requested quantity; nothing is computed."""


# ---------------------------------------------------------------------------------------
# Vendored statistics
# ---------------------------------------------------------------------------------------

# Vendored VERBATIM from experiments/titan_model_soup_mini_v1/score.py :: exact_mcnemar at commit
# 816a1e8f (the function is unchanged since it was introduced in 01bb8b9c). Do not edit; the test
# suite compares this function's AST against the source file. Two-sided exact McNemar test on the
# discordant pairs: p = min(1, 2 * P[Binomial(gains + losses, 1/2) <= min(gains, losses)]).
def exact_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    if not n:
        return 1.0
    # Compute in log space, avoiding overflow at large n. Stable sum of positive terms.
    k = min(gains, losses)
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) - n * math.log(2)
            for j in range(k + 1)]
    largest = max(logs)
    return min(1.0, 2 * math.exp(largest) * math.fsum(math.exp(x - largest) for x in logs))


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------

def _is_int(v: Any) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _check_skills(s: str, what: str) -> None:
    if not isinstance(s, str) or not s:
        raise MetricsError(f"{what} must be a non-empty string, got {s!r}")
    bad = [c for c in s if c not in SKILLS]
    if bad:
        raise MetricsError(f"{what} {s!r}: letters {bad} are not skills {SKILLS}")
    if len(set(s)) != len(s):
        raise MetricsError(f"{what} {s!r} repeats a skill")


def expected_stage(config: str, order: str) -> int:
    """The stage a record with this ``config`` must carry (0 base, k prefix, -1 otherwise)."""
    if config == "base":
        return 0
    m = _PREFIX.fullmatch(config)
    if m:
        k = int(m.group(1))
        _check_skills(order, "order of a prefix record")
        if k > len(order):
            raise MetricsError(f"{config}: stage {k} is beyond order {order!r} (length {len(order)})")
        return k
    m = _SINGLE.fullmatch(config)
    if m:
        _check_skills(m.group(1), config)
        return -1
    m = _PAIR.fullmatch(config)
    if m:
        _check_skills(m.group(1) + m.group(2), config)
        return -1
    raise MetricsError(f"unknown config label {config!r} (base | prefix:k | single:X | pair:X+Y)")


def _check_record(r: Any, i: int) -> dict:
    if not isinstance(r, Mapping):
        raise MetricsError(f"record {i}: not a mapping")
    missing = [k for k in RECORD_FIELDS if k not in r]
    if missing:
        raise MetricsError(f"record {i}: missing {missing}")
    for k in _STR_FIELDS:
        if not isinstance(r[k], str):
            raise MetricsError(f"record {i}: {k} must be a string, got {type(r[k]).__name__}")
        if k != "order" and not r[k]:
            raise MetricsError(f"record {i}: empty {k}")
    for k in ("seed", "stage"):
        if not _is_int(r[k]):
            raise MetricsError(f"record {i}: {k} must be an integer, got {r[k]!r}")
    c = r["correct"]
    if isinstance(c, (bool, np.bool_)):
        correct = int(bool(c))
    elif _is_int(c) and int(c) in (0, 1):
        correct = int(c)
    else:
        raise MetricsError(f"record {i}: correct must be 0 or 1, got {c!r}")
    if not isinstance(r["terminated"], (bool, np.bool_)):
        raise MetricsError(f"record {i}: terminated must be a bool, got {r['terminated']!r}")
    _check_skills(r["signature"], f"record {i}: signature")
    want = expected_stage(r["config"], r["order"])
    if int(r["stage"]) != want:
        raise MetricsError(f"record {i}: config {r['config']!r} requires stage {want}, got {r['stage']}")
    out = dict(r)
    out.update(seed=int(r["seed"]), stage=int(r["stage"]), correct=correct, terminated=bool(r["terminated"]))
    return out


def validate(records: Iterable[Mapping]) -> list[dict]:
    """Validate and normalise records (``correct`` -> int 0/1, ``terminated`` -> bool).

    Refuses on any schema violation, on a duplicate (arm, seed, order, config, row_id), and on a
    row_id that appears with two different (parent_id, signature) values.
    """
    out, seen, ident = [], set(), {}
    for i, r in enumerate(records):
        n = _check_record(r, i)
        key = (n["arm"], n["seed"], n["order"], n["config"], n["row_id"])
        if key in seen:
            raise MetricsError(f"record {i}: row scored twice under {key}")
        seen.add(key)
        prev = ident.setdefault(n["row_id"], (n["parent_id"], n["signature"]))
        if prev != (n["parent_id"], n["signature"]):
            raise MetricsError(f"record {i}: row {n['row_id']!r} has (parent, signature) "
                               f"{(n['parent_id'], n['signature'])} but elsewhere {prev}")
        out.append(n)
    return out


def _match(value: Any, want: Any) -> bool:
    if isinstance(want, (list, tuple, set, frozenset)):
        return value in want
    return value == want


def select(records: Iterable[Mapping], **filters) -> list[dict]:
    """Validated records whose fields equal every filter (a list/tuple/set value = membership).

    Unknown filter keys are refused (a typo must not silently match nothing). An empty result is
    returned as an empty list; callers that count refuse it.
    """
    bad = [k for k in filters if k not in RECORD_FIELDS]
    if bad:
        raise MetricsError(f"unknown filter keys {bad}; filterable fields are {RECORD_FIELDS}")
    return [r for r in validate(records) if all(_match(r[k], v) for k, v in filters.items())]


def _sig_key(sig: str):
    return (len(sig), [SKILLS.index(c) if c in SKILLS else len(SKILLS) for c in sig], sig)


# ---------------------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------------------

def accuracy(records: Iterable[Mapping], **filters) -> dict:
    """Exact-verification accuracy of the records matching ``filters``, per signature and pooled.

    Returns ``{signature: {"n", "correct", "acc", "terminated"}}`` plus ``POOLED`` (key
    ``"__pooled__"``): ``n`` rows, ``correct`` rows with verifier score 1, ``acc = correct / n``,
    ``terminated`` rows whose generation stopped at an EOS token (the rest hit the token cap).
    The pooled entry is ROW-weighted (micro average over every matching row), not the mean of the
    per-signature accuracies. It pools whatever matches: filter to one (arm, seed, order, config)
    unless pooling across configurations is what you intend. Refuses when nothing matches.
    """
    sel = select(records, **filters)
    if not sel:
        raise MetricsError(f"no records match {filters}")
    cells: dict[str, dict] = {}
    for r in sel:
        d = cells.setdefault(r["signature"], {"n": 0, "correct": 0, "terminated": 0})
        d["n"] += 1
        d["correct"] += r["correct"]
        d["terminated"] += int(r["terminated"])
    out = {s: cells[s] for s in sorted(cells, key=_sig_key)}
    out[POOLED] = {k: sum(d[k] for d in cells.values()) for k in ("n", "correct", "terminated")}
    for d in out.values():
        d["acc"] = d["correct"] / d["n"]
    return out


# ---------------------------------------------------------------------------------------
# Retention and composition over a sequential order
# ---------------------------------------------------------------------------------------

def _check_group_args(arm: Any, seed: Any, order: Any) -> None:
    if not isinstance(arm, str) or not arm:
        raise MetricsError(f"arm must be a non-empty string, got {arm!r}")
    if not _is_int(seed):
        raise MetricsError(f"seed must be an integer, got {seed!r}")
    _check_skills(order, "order")


def _prefix_cells(records, arm: str, seed: int, order: str) -> dict:
    """{(signature, stage): {row_id: correct}} for the prefix records of one (arm, seed, order)."""
    cells: dict[tuple[str, int], dict[str, int]] = {}
    for r in validate(records):
        if r["arm"] == arm and r["seed"] == seed and r["order"] == order and r["config"].startswith("prefix:"):
            cells.setdefault((r["signature"], r["stage"]), {})[r["row_id"]] = r["correct"]
    return cells


def _same_rows(cells: dict, sig: str, stages: Sequence[int], who: str) -> dict[int, dict[str, int]]:
    missing = [k for k in stages if not cells.get((sig, k))]
    if missing:
        raise MetricsError(f"{who}: signature {sig!r} has no rows at stage(s) {missing}")
    ref_stage = stages[0]
    ref = set(cells[(sig, ref_stage)])
    for k in stages[1:]:
        got = set(cells[(sig, k)])
        if got != ref:
            raise MetricsError(f"{who}: signature {sig!r} row set at stage {k} differs from stage {ref_stage} "
                               f"({len(got - ref)} rows only at {k}, {len(ref - got)} only at {ref_stage}); "
                               "the same rows must be scored at every stage")
    return {k: cells[(sig, k)] for k in stages}


def retention_matrix(records: Iterable[Mapping], arm: str, seed: int, order: str) -> dict:
    """Continual-learning retention of each skill of ``order`` along its sequential composition.

    Built ONLY from ``prefix:k`` records of (arm, seed, order) on the ATOMIC signatures of the
    skills in ``order`` (composition signatures and other skills' atomic rows are ignored). Every
    skill must be scored at every stage 1..K on the SAME rows; otherwise it refuses.

    With ``a[s][k]`` = accuracy on skill s's atomic rows after the first k skills, and s's own
    stage ``o(s)`` = its 1-based position in ``order``:

    * ``acquisition[s] = a[s][o(s)]``       accuracy right after s was learned
    * ``final[s]       = a[s][K]``          accuracy after the last skill
    * ``forgetting_final[s] = acquisition - final``   (> 0: lost since acquisition)
    * ``forgetting_max[s]   = max_{o(s) <= k <= K} a[s][k] - final``  (the standard CL maximum
      forgetting, Chaudhry et al. 2018, restricted to stages at or after s's own; always >= 0,
      and 0 for the last skill)
    * ``backward_transfer = mean over all skills but the last of (final - acquisition)``
      (Lopez-Paz & Ranzato 2017; negative = forgetting; None when K = 1)

    Differences are computed from integer counts over the shared row set, so they are exact
    ratios rounded once. Stages BEFORE a skill's own stage are reported in ``acc`` (forward
    transfer / floor) but enter none of the summaries.
    """
    _check_group_args(arm, seed, order)
    cells = _prefix_cells(records, arm, seed, order)
    skills, K = list(order), len(order)
    stages = list(range(1, K + 1))
    who = f"retention({arm!r}, seed {seed}, {order!r})"
    n, correct, acc, own = {}, {}, {}, {}
    acquisition, final, f_final, f_max = {}, {}, {}, {}
    bwt_terms = []
    for i, s in enumerate(skills):
        table = _same_rows(cells, s, stages, who)
        n[s] = len(table[1])
        correct[s] = {k: sum(table[k].values()) for k in stages}
        acc[s] = {k: correct[s][k] / n[s] for k in stages}
        own[s] = i + 1
        c_acq, c_fin = correct[s][own[s]], correct[s][K]
        acquisition[s], final[s] = acc[s][own[s]], acc[s][K]
        f_final[s] = (c_acq - c_fin) / n[s]
        f_max[s] = (max(correct[s][k] for k in range(own[s], K + 1)) - c_fin) / n[s]
        if i < K - 1:
            bwt_terms.append(Fraction(c_fin - c_acq, n[s]))
    bwt = float(sum(bwt_terms, Fraction(0)) / len(bwt_terms)) if bwt_terms else None
    return {"arm": arm, "seed": int(seed), "order": order, "skills": skills, "stages": stages,
            "own_stage": own, "n": n, "correct": correct, "acc": acc,
            "acquisition": acquisition, "final": final, "forgetting_final": f_final,
            "forgetting_max": f_max, "backward_transfer": bwt}


def composition_matrix(records: Iterable[Mapping], arm: str, seed: int, order: str,
                       signatures: Sequence[str]) -> dict:
    """Accuracy on (typically held-out combination) signatures along a sequential composition.

    For each signature, reports accuracy at every prefix stage k whose learned skills
    ``set(order[:k])`` cover all of the signature's letters, plus at the final stage K even when
    it does not cover them (``covered`` says which; ``first_covered_stage`` is None when no stage
    covers the signature). Stages that do not cover it and are not final are ignored even if
    scored. Every reported stage must carry the SAME rows; otherwise it refuses.

    This is the accuracy with which the composed model solves a signature once it has all the
    required skills. It is not, by itself, evidence of composition beyond the atomic skills: that
    needs the atomic accuracies (``retention_matrix``) and a floor (``acc_above_floor``).
    """
    _check_group_args(arm, seed, order)
    if isinstance(signatures, str):
        raise MetricsError("signatures must be a sequence of signature strings, not one string")
    sigs = list(signatures)
    if not sigs:
        raise MetricsError("no signatures requested")
    if len(set(sigs)) != len(sigs):
        raise MetricsError(f"duplicate signatures in {sigs}")
    for s in sigs:
        _check_skills(s, "signature")
    cells = _prefix_cells(records, arm, seed, order)
    K = len(order)
    who = f"composition({arm!r}, seed {seed}, {order!r})"
    out = {"arm": arm, "seed": int(seed), "order": order, "skills": list(order), "signatures": sigs,
           "stages": {}, "covered": {}, "first_covered_stage": {}, "n": {}, "correct": {}, "acc": {},
           "final": {}}
    for s in sigs:
        cov = [k for k in range(1, K + 1) if set(s) <= set(order[:k])]
        report = sorted(set(cov) | {K})
        table = _same_rows(cells, s, report, who)
        n = len(table[report[0]])
        out["stages"][s] = report
        out["covered"][s] = {k: k in cov for k in report}
        out["first_covered_stage"][s] = cov[0] if cov else None
        out["n"][s] = n
        out["correct"][s] = {k: sum(table[k].values()) for k in report}
        out["acc"][s] = {k: out["correct"][s][k] / n for k in report}
        out["final"][s] = out["acc"][s][K]
    return out


# ---------------------------------------------------------------------------------------
# Paired contrasts
# ---------------------------------------------------------------------------------------

def _by_row(records: Iterable[Mapping], side: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for r in validate(records):
        if r["row_id"] in rows:
            raise MetricsError(f"{side}: row {r['row_id']!r} appears twice (one configuration per side)")
        rows[r["row_id"]] = r
    if not rows:
        raise MetricsError(f"{side}: no records")
    return rows


def _paired_rows(records_a, records_b, a: str, b: str) -> tuple[list[str], dict, dict]:
    A, B = _by_row(records_a, a), _by_row(records_b, b)
    if set(A) != set(B):
        raise MetricsError(f"row sets differ: {len(set(A) - set(B))} rows only in {a}, "
                           f"{len(set(B) - set(A))} only in {b}; no intersections or dropped rows")
    for i in A:
        if (A[i]["parent_id"], A[i]["signature"]) != (B[i]["parent_id"], B[i]["signature"]):
            raise MetricsError(f"row {i!r}: parent/signature differ between {a} and {b}")
    ra = {r.get("config_sha256") for r in A.values()} - {None}
    rb = {r.get("config_sha256") for r in B.values()} - {None}
    if ra and rb and ra != rb:
        raise MetricsError(f"{a} and {b} were scored under different registrations {sorted(ra)} vs {sorted(rb)}")
    return sorted(A), A, B


def _configs(rows: Mapping[str, dict]) -> list[list]:
    return [list(t) for t in sorted({(r["arm"], r["seed"], r["order"], r["config"]) for r in rows.values()})]


def parent_cluster_bootstrap(parents: Sequence[str], diffs: Sequence[float], n_boot: int = DEFAULT_N_BOOT,
                             seed: int = DEFAULT_SEED) -> np.ndarray:
    """Bootstrap replicates of the mean per-row difference, resampling PARENT clusters.

    Clusters are the distinct ``parents`` in sorted order (so the result does not depend on row
    order). Each replicate draws G = #clusters cluster indices uniformly with replacement and
    returns sum(diffs in drawn clusters) / (rows in drawn clusters) -- the row-weighted mean
    difference of the resampled data set. A fresh ``numpy.random.default_rng(seed)`` is created
    per call; indices are drawn as ``rng.integers(0, G, size=(m, G))`` in blocks of
    ``m = min(BOOT_CHUNK, remaining)`` draws.
    """
    if not _is_int(n_boot) or n_boot < 1:
        raise MetricsError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not _is_int(seed):
        raise MetricsError(f"seed must be an integer, got {seed!r}")
    parents = list(parents)
    d = np.asarray(diffs, dtype=np.float64).reshape(-1)
    if len(parents) != d.size or d.size == 0:
        raise MetricsError("parents and diffs must be the same non-empty length")
    if not np.isfinite(d).all():
        raise MetricsError("non-finite differences")
    labels = sorted(set(parents))
    index = {p: j for j, p in enumerate(labels)}
    inv = np.fromiter((index[p] for p in parents), dtype=np.int64, count=len(parents))
    G = len(labels)
    sums = np.bincount(inv, weights=d, minlength=G)
    sizes = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), BOOT_CHUNK):
        m = min(BOOT_CHUNK, int(n_boot) - start)
        idx = rng.integers(0, G, size=(m, G))
        out[start:start + m] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return out


def paired(records_x: Iterable[Mapping], records_y: Iterable[Mapping], n_boot: int = DEFAULT_N_BOOT,
           seed: int = DEFAULT_SEED) -> dict:
    """Paired contrast of configuration y against x on the IDENTICAL row set.

    Each side must hold each row at most once and both sides must hold exactly the same row_ids
    (with the same parent_id and signature); otherwise it refuses. Returns

    * ``n``, ``x_correct``, ``y_correct``, ``x_acc``, ``y_acc``
    * ``gains``  rows y got right and x got wrong; ``losses`` rows x got right and y got wrong
    * ``net = gains - losses`` and ``diff_acc = net / n`` (= y_acc - x_acc)
    * ``mcnemar_p``: exact two-sided McNemar p on (gains, losses) -- descriptive, not
      multiplicity-adjusted; it tests only whether discordant rows split evenly
    * ``ci95``: 2.5/97.5 percentiles of the parent-cluster bootstrap of ``diff_acc``
      (``parent_cluster_bootstrap``). Rows sharing a parent_id are resampled together because
      they are not independent. This is a descriptive percentile interval; its coverage has not
      been calibrated here.

    ``net`` hides offsetting movement (5 gains + 5 losses = 0): use ``damage`` for the
    reference-denominated loss and gain counts.
    """
    ids, X, Y = _paired_rows(records_x, records_y, "x", "y")
    xc = np.array([X[i]["correct"] for i in ids], dtype=np.int64)
    yc = np.array([Y[i]["correct"] for i in ids], dtype=np.int64)
    gains = int(((yc == 1) & (xc == 0)).sum())
    losses = int(((xc == 1) & (yc == 0)).sum())
    n = len(ids)
    parents = [X[i]["parent_id"] for i in ids]
    boot = parent_cluster_bootstrap(parents, yc - xc, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "x_correct": int(xc.sum()), "y_correct": int(yc.sum()),
            "x_acc": float(xc.sum()) / n, "y_acc": float(yc.sum()) / n,
            "gains": gains, "losses": losses, "net": gains - losses, "diff_acc": (gains - losses) / n,
            "mcnemar_p": exact_mcnemar(gains, losses), "ci95": [float(lo), float(hi)],
            "ci_method": "parent_cluster_percentile_bootstrap", "clusters": len(set(parents)),
            "n_boot": int(n_boot), "seed": int(seed), "x_configs": _configs(X), "y_configs": _configs(Y)}


def _damage_counts(ids: Sequence[str], R: Mapping[str, dict], C: Mapping[str, dict]) -> dict:
    ref_right = [i for i in ids if R[i]["correct"] == 1]
    ref_wrong = [i for i in ids if R[i]["correct"] == 0]
    losses = sum(1 for i in ref_right if C[i]["correct"] == 0)
    gains = sum(1 for i in ref_wrong if C[i]["correct"] == 1)
    retained = len(ref_right) - losses
    return {"n": len(ids), "reference_correct": len(ref_right), "reference_wrong": len(ref_wrong),
            "losses": losses, "retained": retained,
            "retention_fraction": retained / len(ref_right) if ref_right else None,
            "loss_fraction": losses / len(ref_right) if ref_right else None,
            "gains": gains, "gain_fraction": gains / len(ref_wrong) if ref_wrong else None,
            "candidate_correct": retained + gains}


def damage(records_ref: Iterable[Mapping], records_cand: Iterable[Mapping]) -> dict:
    """The damage-denominator rule: damage is counted over the rows the REFERENCE got right.

    * ``losses``  reference right, candidate wrong -- denominator ``reference_correct``
    * ``retained = reference_correct - losses``; ``retention_fraction = retained / reference_correct``
      and ``loss_fraction = losses / reference_correct`` (None when the reference got none right)
    * ``gains``   reference wrong, candidate right -- denominator ``reference_wrong``;
      ``gain_fraction`` (None when the reference got every row right)

    Losses and gains are reported SEPARATELY and never netted (there is deliberately no ``net``
    key): 5 losses and 5 gains are 5 damaged rows, not zero damage. ``by_signature`` repeats the
    counts per signature, each over that signature's own reference-correct rows. Requires the
    identical row set on both sides; refuses otherwise.
    """
    ids, R, C = _paired_rows(records_ref, records_cand, "reference", "candidate")
    out = _damage_counts(ids, R, C)
    by_sig: dict[str, list[str]] = {}
    for i in ids:
        by_sig.setdefault(R[i]["signature"], []).append(i)
    out["by_signature"] = {s: _damage_counts(by_sig[s], R, C) for s in sorted(by_sig, key=_sig_key)}
    return out


def acc_above_floor(acc: float, floor: float) -> float | None:
    """Floor-adjusted accuracy ``(acc - floor) / (1 - floor)``: the fraction of the headroom above
    a content-free floor (e.g. ``data.shortcut_floors``) that the model actually earns.

    1 = perfect, 0 = at the floor, negative = BELOW the floor (reported, not clipped). Returns None
    when ``floor >= 1`` (no headroom: the quantity is undefined). Refuses non-finite or
    non-numeric inputs, ``acc`` outside [0, 1], and a negative floor.
    """
    for name, v in (("acc", acc), ("floor", floor)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise MetricsError(f"{name} must be a finite real number, got {v!r}")
    if floor >= 1:
        return None
    if floor < 0:
        raise MetricsError(f"floor must be >= 0, got {floor!r}")
    if not 0 <= acc <= 1:
        raise MetricsError(f"acc must lie in [0, 1], got {acc!r}")
    return (float(acc) - float(floor)) / (1.0 - float(floor))


# ---------------------------------------------------------------------------------------
# Deterministic CSV
# ---------------------------------------------------------------------------------------

def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if _is_int(v):
        return str(int(v))
    if isinstance(v, numbers.Real):
        f = float(v)
        if not math.isfinite(f):
            raise MetricsError(f"non-finite value {v!r} in CSV")
        return repr(f)
    if isinstance(v, str):
        if "\r" in v or "\n" in v:
            raise MetricsError(f"line break inside CSV field {v!r}")
        return v
    raise MetricsError(f"unsupported CSV value {v!r} ({type(v).__name__})")


def csv_bytes(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    """UTF-8 CSV, LF line endings, minimal quoting; floats as ``repr`` (shortest round-trip)."""
    buf = _stdio.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(list(columns))
    width = len(columns)
    for row in rows:
        row = list(row)
        if len(row) != width:
            raise MetricsError(f"row has {len(row)} fields, header has {width}")
        w.writerow([_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")


def _require(matrix: Mapping, keys: Sequence[str], what: str) -> None:
    missing = [k for k in keys if k not in matrix]
    if missing:
        raise MetricsError(f"not a {what}: missing {missing}")


def write_retention_csv(path, matrix: Mapping) -> str:
    """One row per (skill in order, stage 1..K), columns ``RETENTION_COLUMNS``; the per-skill
    summaries and the matrix-level ``backward_transfer`` are repeated on each of the skill's rows.
    Refuses to overwrite. Returns the file's sha256."""
    _require(matrix, ("arm", "seed", "order", "skills", "stages", "own_stage", "n", "correct", "acc",
                      "acquisition", "final", "forgetting_final", "forgetting_max", "backward_transfer"),
             "retention matrix")
    rows = []
    for s in matrix["skills"]:
        for k in matrix["stages"]:
            rows.append([matrix["arm"], matrix["seed"], matrix["order"], s, matrix["own_stage"][s], k,
                         matrix["n"][s], matrix["correct"][s][k], matrix["acc"][s][k],
                         matrix["acquisition"][s], matrix["final"][s], matrix["forgetting_final"][s],
                         matrix["forgetting_max"][s], matrix["backward_transfer"]])
    return write_bytes(path, csv_bytes(RETENTION_COLUMNS, rows))


def write_composition_csv(path, matrix: Mapping) -> str:
    """One row per (signature in request order, reported stage ascending), columns
    ``COMPOSITION_COLUMNS``. Refuses to overwrite. Returns the file's sha256."""
    _require(matrix, ("arm", "seed", "order", "signatures", "stages", "covered", "first_covered_stage",
                      "n", "correct", "acc", "final"), "composition matrix")
    rows = []
    for s in matrix["signatures"]:
        for k in matrix["stages"][s]:
            rows.append([matrix["arm"], matrix["seed"], matrix["order"], s, k, matrix["covered"][s][k],
                         matrix["first_covered_stage"][s], matrix["n"][s], matrix["correct"][s][k],
                         matrix["acc"][s][k], matrix["final"][s]])
    return write_bytes(path, csv_bytes(COMPOSITION_COLUMNS, rows))


def write_rows_csv(path, rows: Iterable[Mapping], extra_columns: Sequence[str] = ()) -> str:
    """Per-row scored records, columns ``RECORD_FIELDS`` then ``extra_columns`` in the given order.

    Records are validated (``validate``) and written in canonical order, sorted by (arm, seed,
    order, stage, config, signature, row_id), so the same record SET always gives the same bytes.
    A key that is neither a record field nor a declared extra column is refused (nothing is
    dropped silently); a declared extra column missing from a record is written empty. Refuses
    to overwrite. Returns the file's sha256.
    """
    extra = list(extra_columns)
    if len(set(extra)) != len(extra) or set(extra) & set(RECORD_FIELDS):
        raise MetricsError(f"extra_columns {extra} repeat or shadow record fields")
    columns = list(RECORD_FIELDS) + extra
    allowed = set(columns)
    recs = validate(rows)
    if not recs:
        raise MetricsError("no rows to write")
    for r in recs:
        undeclared = sorted(set(r) - allowed)
        if undeclared:
            raise MetricsError(f"row {r['row_id']!r}: undeclared keys {undeclared}; pass extra_columns")
    recs.sort(key=lambda r: (r["arm"], r["seed"], r["order"], r["stage"], r["config"], r["signature"],
                             r["row_id"]))
    return write_bytes(path, csv_bytes(columns, ([r.get(c) for c in columns] for r in recs)))


def write_table_csv(path, rows: Sequence[Mapping], columns: Sequence[str] | None = None) -> str:
    """Generic deterministic CSV for NON-record tables (geometry, allocation, energy): columns are
    the given list or the sorted union of keys; rows keep their given order. Refuses to overwrite."""
    rows = list(rows)
    if not rows:
        raise MetricsError("no rows to write")
    cols = list(columns) if columns is not None else sorted({k for r in rows for k in r})
    return write_bytes(path, csv_bytes(cols, ([r.get(c) for c in cols] for r in rows)))
