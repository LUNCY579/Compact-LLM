"""Phase 6 — activation probes and matched-pair activation patching (brief 16-18).

Three stages, run in this order per arm while the model is resident, because each one
consumes the previous one's output:

``probes``    collect residual-stream / attention-out / MLP-out vectors at BOS, inside the
              evidence span, at the final query tokens and at a seeded random control set,
              then ask what is *decodable* from them. Correlational; it nominates layers.
``patching``  transplant clean activations into the corrupted run of the same scenario and
              measure the change in gold log-probability. Causal; this is the only claim
              about mechanism the phase makes.
``heads``     per-``(layer, q_head)`` positional statistics, with the patch effect joined
              on at layer resolution.

Artifacts (brief section 46 names)::

    results/activation_probe_results.json
    results/activation_patch_results.parquet
    results/head_ablation_results.parquet
    results/phase6_summary.json          (this script's own run record)

    PYTHONPATH=src python scripts/phase6_probes_and_patching.py --dry-run
    PYTHONPATH=src python scripts/phase6_probes_and_patching.py --split calibration

Pair selection is **per model**, not per prior
----------------------------------------------
Phase 3 measured that the worst position differs by model *and* by family:
``exact_retrieval`` is a recency gradient whose worst point is 0.17 for Qwen3 and 0.67 for
Halo, while ``multi_hop`` peaks at 0.50. So the default ``--pair-mode best_worst`` reads
the clean and corrupted positions **off this arm's own Phase-3 gold NLL**: within one
scenario, the variant the model scores best is clean and the variant it scores worst is
corrupted. Nothing here assumes the middle is the bad place. ``--pair-mode edge_middle``
restores the library's ``find_matched_pairs`` window logic for callers who want the
brief's literal edge-versus-middle contrast.

Pairs are formed **within a (scenario, length_bucket, fmt, local_position) block.** One
``scenario_id`` in this bank spans 2K/4K/8K and both rhetorical positions as well as the
seven-point grid, so pairing on ``scenario_id`` alone would happily hand the patcher a 2K
clean run and an 8K corrupted run — two different prompts, not one prompt with the
evidence moved.

Whole-residual patching, and why it is opt-in
---------------------------------------------
``patching.py`` implements an ``all_positions`` alignment, and it is *not* in the default
set. It requires the two runs to tokenize to the same length, and even then position ``i``
holds a different token in the two runs everywhere the evidence moved, so patching every
position at layer ``l`` replaces the whole computation up to ``l``: a near-total recovery
is expected by construction and localises nothing. ``--include-all-positions`` turns it on
as an upper bound; the summary records that it was used and what it does not show. The two
well-defined sites — ``final_query`` (exact: same offset from the end of an identical
generation prompt) and ``span_aligned`` (approximate, with the RoPE-phase caveat recorded
on every row) — are the defaults, together with the ``random_control`` negative control.

DEVIATION: ``patching.find_matched_pairs`` cannot consume rows read straight out of the
Phase-3 parquet. Its private ``_position_of`` evaluates ``record_field(record, name) or ()``
on ``actual_positions``, which is a numpy array in a parquet row; for the multi-evidence
families that array has length 3-4 and ``bool(array)`` raises ``ValueError: truth value of
an array with more than one element is ambiguous``. :func:`load_records` therefore converts
every row to a plain dict with Python lists before the library sees it. The library was not
edited.

DEVIATION: ``head_ablation_results.parquet`` ships with ``ablation_effect`` all-NaN and
``ablation_scope="unset"``. That column is fed by a ``SignificanceMatrix`` from Phases 4-5,
which have produced no artifact yet, and ``significance.matrices`` has a ``save_npz`` but no
loader, so there is nothing to attach and nothing to attach it with. The column is emitted
NaN rather than dropped so the table's schema is stable across the Phase 4-5 rerun, and the
summary says so in ``limitations``.

DEVIATION: the answer-identity probe of brief section 16 is ill-posed on this bank as a
by-scenario k-way classification: every positional variant of a scenario shares one answer
and answers are unique per scenario, so ``ProbeTask.min_class_scenarios=2`` drops every
label and the probe correctly reports ``skipped``. That skip is run and recorded, and two
relabellings are run beside it — ``first_token`` (the first gold answer token) and
``family`` (four classes). Measured on the calibration 4K bank, ``first_token`` gives 24
distinct classes over 85 scenarios of which only two appear in two or more scenarios, one
covering ~70% of them, so it is near-degenerate and must be read against the ``chance``
column; ``family`` is the balanced one and is the informative control. Relabelling happens
on a copy of the dataset through the public ``ActivationDataset`` API; ``collect.py`` was
not edited.

Cost
----
Forward seconds are SPEC 12.1's measured table with the backend pinned. Nothing in this
phase needs gradients, so 4K is the working length (SPEC 11.3's screening scale) rather
than the 2K that backward-bound phases are held to. The capture pass behind ``heads`` runs
the chunked eager kernel and its wall clock is *not* in the measured table; it is estimated
at :data:`CAPTURE_FORWARD_MULTIPLIER` forwards and is labelled as an estimate everywhere it
appears. ``--dry-run`` prints the whole breakdown and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.activation_analysis import collect, head_stats, patching, probes  # noqa: E402
from rineya_psa.backend import efficient_attention, probe_backends  # noqa: E402
from rineya_psa.evaluation.runner import prepare_items  # noqa: E402
from rineya_psa.loading import (  # noqa: E402
    dtype_is_approximation,
    free_model,
    load_model,
    load_tokenizer,
)
from rineya_psa.significance.causal import GPU_HOUR_BUDGET, representative_layers  # noqa: E402
from rineya_psa.statistics.provenance import write_result  # noqa: E402
from rineya_psa.types import BenchmarkItem  # noqa: E402

# Every arm Phase 3 ran. The default is two of them: this phase is the most expensive per
# item in the lab, and Phase 3 answered "which Rineya arm" — Supa-A1.1 has the best
# worst-position robustness and the smallest edge-middle gap on every family. It is taken
# ungated because SPEC section 1 requires the mechanistic phases to run without the
# abstention offset, which shifts one vocabulary row and therefore every gold logprob.
ARMS: tuple[tuple[str, bool], ...] = (
    ("qwen3", False),
    ("supa_a11", True),
    ("halo", False),
    ("nimbus", False),
    ("supa_a11", False),
)
DEFAULT_ARMS: tuple[tuple[str, bool], ...] = (("qwen3", False), ("supa_a11", True))

STAGES: tuple[str, ...] = ("probes", "patching", "heads")

# SPEC 12.1, measured on this card at batch 1 with the backend pinned.
FORWARD_SECONDS: Mapping[str, float] = {
    "2k": 0.12,
    "4k": 0.29,
    "8k": 0.84,
    "16k": 2.27,
    "32k": 7.06,
}

#: ESTIMATE, not a measurement. ``hooks.capture`` runs the chunked eager kernel on the
#: captured layers (SPEC 12.3): at 4K that is 16 query chunks per layer, each materialising
#: a [1, 16, 256, 4096] block, plus a Python-level reduction per chunk. No timing for it
#: exists in SPEC, so the head-stats line of the cost table is marked ``estimated`` and the
#: number is deliberately pessimistic. Replace it with a measurement once one exists.
CAPTURE_FORWARD_MULTIPLIER: float = 8.0

#: Rough seconds per probe fit on this CPU, used only to keep the CPU cost visible next to
#: the GPU cost. Ridge is closed-form and cheap; multinomial logistic on 1024 features is
#: not, which is why the logistic grid defaults to a handful of layers and one site.
PROBE_SECONDS: Mapping[str, float] = {"ridge": 0.08, "logistic": 6.0}

#: Host RAM ceiling for one arm's activation store. Brief section 16: a few hundred 4K
#: items must not blow past a few GB.
DEFAULT_MAX_ACTIVATION_GB: float = 3.0

RIDGE_TASKS: tuple[probes.ProbeTask, ...] = (
    probes.ProbeTask("normalized_position", "ridge", "normalized_position"),
    probes.ProbeTask("evidence_location", "ridge", "evidence_position", roles=("final",)),
)

#: Answer labellings for the logistic probe. See the module DEVIATION note.
ANSWER_LABELS: tuple[str, ...] = ("full", "first_token", "family")


def arm_label(key: str, ungate: bool) -> str:
    return f"{key}_ungated" if ungate else key


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def load_split(split: str) -> list[BenchmarkItem]:
    """The benchmark bank for one split. Mirrors ``phase3_baseline_fingerprint.load_split``."""
    if split == "sealed":  # sealed-ok: refusal
        raise SystemExit("the sealed split is opened only by scripts/phase14_*.py")  # sealed-ok:
    path = paths.DATA / split / "position_examples.jsonl"
    if not path.exists():
        path = paths.DATA_GENERATED / "position_examples.jsonl"
        if not path.exists():
            raise SystemExit(
                f"no benchmark found; run the Phase 2 generator first "
                f"(looked for {paths.DATA / split} and {path})"
            )
        rows = [BenchmarkItem.from_json(json.loads(line)) for line in path.open(encoding="utf-8")]
        return [r for r in rows if r.split == split]
    return [BenchmarkItem.from_json(json.loads(line)) for line in path.open(encoding="utf-8")]


def load_records(path: Path) -> list[dict[str, Any]]:
    """Phase-3 run records as plain dicts with Python lists.

    The list-valued columns arrive from parquet as numpy arrays, and the library's pair
    finder tests them for emptiness with ``or ()`` — which raises on an array of length
    > 1. See the module DEVIATION note. Converting here keeps the library untouched and
    keeps every downstream consumer on ordinary Python types.
    """
    import pandas as pd

    if not path.exists():
        raise SystemExit(
            f"no Phase-3 records at {path}; matched-pair patching needs each arm's gold "
            "NLL per positional variant. Run scripts/phase3_baseline_fingerprint.py first, "
            "or point --records at its parquet."
        )
    frame = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            clean[key] = value.tolist() if isinstance(value, np.ndarray) else value
        rows.append(clean)
    return rows


def filter_items(
    items: Sequence[BenchmarkItem],
    *,
    length_bucket: str | None,
    family: str | None,
    local_position: str | None,
    max_scenarios: int | None,
    limit: int | None,
) -> list[BenchmarkItem]:
    """Narrow the bank. Scenario capping happens before item capping, deliberately.

    Capping items first would take the first N items of the first few scenarios and leave
    the probe with almost no between-scenario variation to split on — and the split is by
    scenario, so that is the axis that has to survive.
    """
    kept = list(items)
    if length_bucket:
        kept = [i for i in kept if i.length_bucket == length_bucket]
    if family:
        kept = [i for i in kept if i.family == family]
    if local_position:
        kept = [i for i in kept if i.local_position == local_position]
    if max_scenarios:
        seen: list[str] = []
        for item in kept:
            if item.scenario_id not in seen:
                seen.append(item.scenario_id)
        keep = set(seen[: int(max_scenarios)])
        kept = [i for i in kept if i.scenario_id in keep]
    if limit:
        kept = kept[: int(limit)]
    return kept


# --------------------------------------------------------------------------------------
# Matched pairs
# --------------------------------------------------------------------------------------

#: A pair is only a positional manipulation if everything except the position is held
#: fixed. ``scenario_id`` alone does not do that in this bank; see the module docstring.
BLOCK_FIELDS: tuple[str, ...] = ("length_bucket", "fmt", "local_position")


@dataclass(frozen=True)
class PairCandidate:
    """One (scenario, block) reduced to its best and worst positional variant."""

    scenario_id: str
    block: tuple[str, ...]
    clean_item_id: str
    corrupted_item_id: str
    clean_gold_nll: float
    corrupted_gold_nll: float
    clean_position: float
    corrupted_position: float
    n_variants: int

    @property
    def gap(self) -> float:
        return float(self.corrupted_gold_nll - self.clean_gold_nll)


def _position_of(row: Mapping[str, Any]) -> float:
    """The decisive evidence sentence's position, matching Phase 3's axis.

    ``primary_target`` is the swept sentence. ``target_positions[0]`` is not: for
    ``state_constraint`` and ``multi_hop`` the non-decisive sentences are pinned by
    construction and reading position off the first one collapses the family onto a
    constant (Phase 3 report, section 5).
    """
    value = row.get("primary_target")
    if value is not None and np.isfinite(float(value)):
        return float(value)
    actual = row.get("actual_positions") or row.get("target_positions") or []
    finite = [float(v) for v in actual if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def select_best_worst(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_nll_gap: float,
    allowed_item_ids: set[str] | None = None,
) -> list[PairCandidate]:
    """Per (scenario, block), the variant this arm scores best against the one it scores worst.

    No position window is applied. Phase 3 showed the worst position is 0.17 for Qwen3 on
    ``exact_retrieval``, 0.67 for Halo on the same family and 0.50 on ``multi_hop``, so any
    fixed window would either miss the real trough or manufacture one.
    """
    blocks: dict[tuple[str, ...], list[tuple[str, float, float]]] = {}
    for row in rows:
        item_id = str(row.get("item_id"))
        if allowed_item_ids is not None and item_id not in allowed_item_ids:
            continue
        nll = row.get("gold_nll")
        if nll is None or not np.isfinite(float(nll)):
            continue
        key = (str(row.get("scenario_id")), *(str(row.get(f)) for f in BLOCK_FIELDS))
        blocks.setdefault(key, []).append((item_id, float(nll), _position_of(row)))

    out: list[PairCandidate] = []
    for key, entries in sorted(blocks.items()):
        if len(entries) < 2:
            continue
        # Ties break on item_id (entries are sorted) so the choice is reproducible.
        entries = sorted(entries)
        best = min(entries, key=lambda e: e[1])
        worst = max(entries, key=lambda e: e[1])
        if best[0] == worst[0] or worst[1] - best[1] < float(min_nll_gap):
            continue
        out.append(
            PairCandidate(
                scenario_id=key[0],
                block=key[1:],
                clean_item_id=best[0],
                corrupted_item_id=worst[0],
                clean_gold_nll=best[1],
                corrupted_gold_nll=worst[1],
                clean_position=best[2],
                corrupted_position=worst[2],
                n_variants=len(entries),
            )
        )
    # Largest positional deficit first: those are the pairs a patch has room to move.
    return sorted(out, key=lambda c: -c.gap)


def build_pairs(
    candidates: Sequence[PairCandidate],
    lookup: Mapping[str, tuple[BenchmarkItem, Any]],
    *,
    max_pairs: int | None,
) -> tuple[list[patching.PatchPair], dict[str, int]]:
    """Turn candidates into :class:`patching.PatchPair`, counting every drop reason."""
    dropped = {"missing_item": 0, "answer_token_mismatch": 0}
    pairs: list[patching.PatchPair] = []
    for cand in candidates:
        if max_pairs is not None and len(pairs) >= int(max_pairs):
            break
        if cand.clean_item_id not in lookup or cand.corrupted_item_id not in lookup:
            dropped["missing_item"] += 1
            continue
        clean_item, clean_tok = lookup[cand.clean_item_id]
        corrupt_item, corrupt_tok = lookup[cand.corrupted_item_id]
        if tuple(clean_tok.answer_ids) != tuple(corrupt_tok.answer_ids):
            # PatchPair raises on this; the two runs would be scored on different token
            # counts and their log-probabilities would not be comparable.
            dropped["answer_token_mismatch"] += 1
            continue
        pairs.append(
            patching.PatchPair(
                scenario_id=cand.scenario_id,
                clean_item=clean_item,
                clean_tok=clean_tok,
                corrupted_item=corrupt_item,
                corrupted_tok=corrupt_tok,
                clean_position=cand.clean_position,
                corrupted_position=cand.corrupted_position,
                clean_gold_nll=cand.clean_gold_nll,
                corrupted_gold_nll=cand.corrupted_gold_nll,
                selection="best_worst_gold_nll",
            )
        )
    return pairs, dropped


def pairs_for_arm(
    rows: Sequence[Mapping[str, Any]],
    lookup: Mapping[str, tuple[BenchmarkItem, Any]],
    *,
    mode: str,
    min_nll_gap: float,
    max_pairs: int | None,
) -> tuple[list[patching.PatchPair], dict[str, Any]]:
    """Matched pairs for one arm, under either selection rule."""
    if mode == "best_worst":
        candidates = select_best_worst(
            rows, min_nll_gap=min_nll_gap, allowed_item_ids=set(lookup)
        )
        pairs, dropped = build_pairs(candidates, lookup, max_pairs=max_pairs)
        info = {
            "mode": mode,
            "n_candidates": len(candidates),
            "n_pairs": len(pairs),
            "dropped": dropped,
            "min_nll_gap": float(min_nll_gap),
            "block_fields": list(BLOCK_FIELDS),
        }
        return pairs, info

    # edge_middle: the brief's literal contrast, via the library's own window logic.
    usable = [r for r in rows if str(r.get("item_id")) in lookup]
    found = patching.find_matched_pairs(usable, lookup, min_nll_gap=min_nll_gap)
    if max_pairs is not None:
        found = found[: int(max_pairs)]
    return found, {
        "mode": mode,
        "n_candidates": len(found),
        "n_pairs": len(found),
        "dropped": {},
        "min_nll_gap": float(min_nll_gap),
        "block_fields": [],
    }


# --------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------


def _first_answer_token(tok: Any, answer: str) -> str:
    ids = list(tok.encode(answer, add_special_tokens=False))
    return f"tok{int(ids[0])}" if ids else "empty"


def relabel_dataset(
    dataset: collect.ActivationDataset,
    labels: Sequence[str],
    *,
    max_classes: int | None,
) -> tuple[collect.ActivationDataset, dict[str, Any]]:
    """A copy of ``dataset`` whose ``answer`` field carries ``labels``.

    Capping the class count bounds the multinomial-logistic cost, which is what makes a
    per-layer answer probe affordable at all. The cap is applied by *dropping rows* of the
    rare classes, so the retained chance level, the permutation baseline and the reported
    accuracy are all computed on the same filtered design — and ``n_classes_capped`` says
    it happened.
    """
    if len(labels) != len(dataset.rows):
        raise ValueError(f"{len(labels)} labels for {len(dataset.rows)} rows")
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    capped = False
    keep_labels = set(counts)
    if max_classes is not None and len(counts) > int(max_classes):
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        keep_labels = {name for name, _ in ranked[: int(max_classes)]}
        capped = True
    mask = np.asarray([label in keep_labels for label in labels], dtype=bool)
    subset = dataset.subset(mask)
    kept = [label for label, take in zip(labels, mask, strict=True) if take]
    rows = tuple(replace(r, answer=label) for r, label in zip(subset.rows, kept, strict=True))
    info = {
        "n_labels_seen": len(counts),
        "n_labels_kept": len(keep_labels),
        "n_classes_capped": capped,
        "n_rows_kept": int(mask.sum()),
        "n_rows_dropped": int((~mask).sum()),
    }
    return replace(subset, rows=rows), info


def answer_labels(dataset: collect.ActivationDataset, mode: str, tok: Any) -> list[str]:
    if mode == "full":
        return [r.answer for r in dataset.rows]
    if mode == "family":
        return [r.family for r in dataset.rows]
    if mode == "first_token":
        cache: dict[str, str] = {}
        out = []
        for row in dataset.rows:
            if row.answer not in cache:
                cache[row.answer] = _first_answer_token(tok, row.answer)
            out.append(cache[row.answer])
        return out
    raise ValueError(f"unknown answer labelling {mode!r}; known: {list(ANSWER_LABELS)}")


def run_probes(
    dataset: collect.ActivationDataset,
    tok: Any,
    *,
    ridge_layers: Sequence[int],
    ridge_sites: Sequence[str],
    logistic_layers: Sequence[int],
    logistic_sites: Sequence[str],
    label_modes: Sequence[str],
    max_classes: int | None,
    n_permutations: int,
    test_fraction: float,
    seed: int,
) -> tuple[list[probes.ProbeResult], dict[str, Any]]:
    """Ridge probes on the raw dataset plus one logistic probe per answer labelling."""
    results = list(
        probes.run_probe_suite(
            dataset,
            tasks=RIDGE_TASKS,
            layers=ridge_layers,
            sites=ridge_sites,
            test_fraction=test_fraction,
            n_permutations=n_permutations,
            seed=seed,
        )
    )
    # The answer probes read the final query token only, so the class counts that decide
    # the cap are computed on exactly those rows. Counting over every role would rank
    # classes by how many control tokens they happen to carry.
    final_only = dataset.subset(dataset.role_mask(("final",)))
    relabelling: dict[str, Any] = {}
    for mode in label_modes:
        labels = answer_labels(final_only, mode, tok)
        # "full" is the brief's literal task and is expected to be skipped on this bank;
        # capping its classes would hide that, so it is passed through uncapped.
        cap = None if mode == "full" else max_classes
        derived, info = relabel_dataset(final_only, labels, max_classes=cap)
        relabelling[mode] = info
        task = probes.ProbeTask(
            f"answer_identity_{mode}", "logistic", "answer", roles=("final",)
        )
        results.extend(
            probes.run_probe_suite(
                derived,
                tasks=(task,),
                layers=logistic_layers,
                sites=logistic_sites,
                test_fraction=test_fraction,
                n_permutations=n_permutations,
                seed=seed,
            )
        )
    return results, relabelling


def probe_highlights(results: Sequence[probes.ProbeResult]) -> dict[str, Any]:
    """Best layer per (task, site), with its chance level and permutation baseline beside it.

    A score is never reported without the shuffled-label baseline that says what chance
    looks like for that design; ``probes.py`` computes it and this is where it is kept
    attached rather than summarised away.
    """
    best: dict[str, dict[str, Any]] = {}
    for res in results:
        if res.status != "ok" or not np.isfinite(res.score):
            continue
        key = f"{res.task}|{res.site}"
        current = best.get(key)
        if current is None or res.score > current["score"]:
            best[key] = {
                "layer": res.layer,
                "metric": res.metric,
                "score": round(float(res.score), 4),
                "chance": round(float(res.chance), 4),
                "permutation_mean": round(float(res.permutation_mean), 4),
                "permutation_p": round(float(res.permutation_p), 4),
                "n_permutations": res.n_permutations,
                "permutation_resolution": round(res.permutation_resolution, 4),
                "beats_chance": res.beats_chance,
                "n_test": res.n_test,
                "n_test_scenarios": res.n_test_scenarios,
                "n_classes": res.n_classes,
            }
    skipped = sorted({r.status for r in results if r.status != "ok"})
    return {"best_layer_per_task_site": best, "skip_reasons": skipped}


# --------------------------------------------------------------------------------------
# Patching summary
# --------------------------------------------------------------------------------------


def patch_summary(records: Sequence[patching.PatchRecord]) -> dict[str, Any]:
    """Mean delta gold logprob per (alignment, site, layer), with the control beside it."""
    buckets: dict[tuple[str, str, int], list[float]] = {}
    for rec in records:
        if rec.status != "ok" or not np.isfinite(rec.delta_gold_logprob):
            continue
        buckets.setdefault((rec.alignment, rec.site, rec.layer), []).append(
            float(rec.delta_gold_logprob)
        )
    out: dict[str, Any] = {}
    for (alignment, site, layer), values in sorted(buckets.items()):
        out[f"{alignment}|{site}|L{layer}"] = {
            "site_label": patching.site_label(site, alignment),
            "mean_delta_gold_logprob": round(float(np.mean(values)), 4),
            "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return out


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------


def cost_estimate(
    *,
    n_arms: int,
    stages: Sequence[str],
    n_items: int,
    n_pairs: int,
    n_head_items: int,
    patch_layers: Sequence[int],
    patch_sites: Sequence[str],
    alignments: Sequence[str],
    head_layers: Sequence[int],
    head_passes: int,
    length_bucket: str,
    n_ridge_fits: int,
    n_logistic_fits: int,
) -> dict[str, Any]:
    """GPU seconds per stage, plus the CPU probe cost, before anything is launched."""
    per_forward = FORWARD_SECONDS.get(length_bucket, FORWARD_SECONDS["4k"])
    lines: dict[str, Any] = {}
    gpu_seconds = 0.0

    if "probes" in stages:
        forwards = n_arms * n_items
        seconds = forwards * per_forward
        gpu_seconds += seconds
        lines["probes_collect"] = {
            "forwards": forwards, "seconds": round(seconds, 1), "estimated": False
        }
    if "patching" in stages:
        per_pair = 2 + len(patch_layers) * len(patch_sites) * len(alignments)
        forwards = n_arms * n_pairs * per_pair
        seconds = forwards * per_forward
        gpu_seconds += seconds
        lines["patching"] = {
            "forwards": forwards, "forwards_per_pair": per_pair,
            "seconds": round(seconds, 1), "estimated": False,
        }
    if "heads" in stages:
        forwards = n_arms * n_head_items * int(head_passes)
        # Capture layers run the chunked eager kernel; the rest keep the fast SDPA path.
        seconds = forwards * per_forward * CAPTURE_FORWARD_MULTIPLIER
        gpu_seconds += seconds
        lines["heads_capture"] = {
            "forwards": forwards, "capture_layers": len(head_layers),
            "seconds": round(seconds, 1), "estimated": True,
            "note": f"eager capture, {CAPTURE_FORWARD_MULTIPLIER}x forward, UNMEASURED",
        }

    cpu_seconds = (
        n_ridge_fits * PROBE_SECONDS["ridge"] + n_logistic_fits * PROBE_SECONDS["logistic"]
    )
    return {
        "length_bucket": length_bucket,
        "seconds_per_forward": per_forward,
        "per_stage": lines,
        "gpu_seconds": round(gpu_seconds, 1),
        "gpu_hours": round(gpu_seconds / 3600.0, 3),
        "gpu_hour_budget": GPU_HOUR_BUDGET,
        "exceeds_budget": gpu_seconds / 3600.0 > GPU_HOUR_BUDGET,
        "cpu_probe_fits": {"ridge": n_ridge_fits, "logistic": n_logistic_fits},
        "cpu_probe_seconds_estimated": round(cpu_seconds, 1),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _int_list(text: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if text is None:
        return tuple(default)
    if text.strip().lower() == "all":
        return ()  # empty means "every layer the model has"; resolved once arch is known
    return tuple(sorted({int(p) for p in text.split(",") if p.strip()}))


def _str_list(text: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if text is None:
        return tuple(default)
    return tuple(dict.fromkeys(p.strip() for p in text.split(",") if p.strip()))


def build_parser() -> argparse.ArgumentParser:
    default_arms = ",".join(arm_label(*a) for a in DEFAULT_ARMS)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="calibration", choices=("calibration", "validation"),
                    help="refused for sealed; that split is Phase 14 only")  # sealed-ok:
    ap.add_argument("--models", default=None,
                    help=f"comma-separated arm labels; default {default_arms}")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--records", default=None,
                    help="Phase-3 parquet supplying each arm's per-variant gold NLL")
    ap.add_argument("--length-bucket", default="4k",
                    help="SPEC 11.3's screening length; 'any' disables the filter")
    ap.add_argument("--family", default=None, help="restrict to one question family")
    ap.add_argument("--local-position", default="middle",
                    help="hold rhetorical position fixed; 'any' disables the filter")
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap items, for smoke runs")

    ap.add_argument("--probe-layers", default="all", help="layers for the ridge probes")
    ap.add_argument("--probe-sites", default=",".join(collect.SITES))
    ap.add_argument("--logistic-layers", default=None,
                    help="layers for the answer probes; default is the representative grid")
    ap.add_argument("--logistic-sites", default="resid")
    ap.add_argument("--answer-labels", default=",".join(ANSWER_LABELS))
    ap.add_argument("--probe-max-classes", type=int, default=16)
    ap.add_argument("--n-permutations", type=int, default=20,
                    help="below 20 the empirical p cannot reach 0.05 (probes.py)")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--max-evidence-tokens", type=int, default=8)
    ap.add_argument("--n-final-tokens", type=int, default=2)
    ap.add_argument("--n-control", type=int, default=8)
    ap.add_argument("--max-activation-gb", type=float, default=DEFAULT_MAX_ACTIVATION_GB)
    ap.add_argument("--save-activations", action="store_true",
                    help="also write each arm's activation store as npz (large)")

    ap.add_argument("--pair-mode", default="best_worst", choices=("best_worst", "edge_middle"))
    ap.add_argument("--min-nll-gap", type=float, default=0.25)
    ap.add_argument("--max-pairs", type=int, default=64)
    ap.add_argument("--patch-layers", default=None,
                    help="default is the representative grid (2,7,12,17,22,27 at 28 layers)")
    ap.add_argument("--patch-sites", default=",".join(patching.DEFAULT_SITES))
    ap.add_argument("--include-all-positions", action="store_true",
                    help="add the whole-residual alignment; an upper bound that localises nothing")
    ap.add_argument("--patch-final-tokens", type=int, default=1)

    ap.add_argument("--head-items", type=int, default=32)
    ap.add_argument("--head-layers", default="all")
    ap.add_argument("--head-bins", type=int, default=16)
    ap.add_argument("--head-final-rows", type=int, default=32,
                    help="0 disables the recency pass and recent_attn becomes an all-rows figure")

    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None, help="suffix for every output filename")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--device", default="cuda", help="'cpu' is for smoke tests only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915 - a driver, read top to bottom
    args = build_parser().parse_args(argv)

    if args.split == "sealed":  # defence in depth; argparse already refuses it  # sealed-ok:
        raise SystemExit("the sealed split is opened only by scripts/phase14_*.py")  # sealed-ok:

    stages = tuple(s for s in _str_list(args.stages, default=STAGES) if s)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(STAGES)}")

    arms = DEFAULT_ARMS
    if args.models:
        want = _str_list(args.models, default=())
        arms = tuple(a for a in ARMS if arm_label(*a) in want)
        missing = [w for w in want if w not in {arm_label(*a) for a in ARMS}]
        if missing:
            raise SystemExit(
                f"unknown arm(s) {missing}; known: {[arm_label(*a) for a in ARMS]}"
            )
    if not arms:
        raise SystemExit("no arms selected")

    items = filter_items(
        load_split(args.split),
        length_bucket=None if args.length_bucket == "any" else args.length_bucket,
        family=args.family,
        local_position=None if args.local_position == "any" else args.local_position,
        max_scenarios=args.max_scenarios,
        limit=args.limit,
    )
    if not items:
        raise SystemExit("no items survived the filters; loosen --family/--length-bucket")

    buckets = sorted({i.length_bucket for i in items})
    cost_bucket = buckets[0] if len(buckets) == 1 else "8k"  # price the worst case when mixed

    # Layer grids. The empty tuple from ``--...-layers all`` is resolved once the arch is
    # known; for the cost estimate the panel's 28 layers are the right stand-in and are
    # re-derived per arm below.
    n_layers_panel = 28
    coarse = representative_layers(n_layers_panel)
    probe_layers = _int_list(args.probe_layers, default=range(n_layers_panel))
    patch_layers = _int_list(args.patch_layers, default=coarse)
    head_layers = _int_list(args.head_layers, default=range(n_layers_panel))
    logistic_layers = _int_list(args.logistic_layers, default=coarse)
    probe_sites = _str_list(args.probe_sites, default=collect.SITES)
    logistic_sites = _str_list(args.logistic_sites, default=("resid",))
    patch_sites = _str_list(args.patch_sites, default=patching.DEFAULT_SITES)
    label_modes = _str_list(args.answer_labels, default=ANSWER_LABELS)
    bad_labels = [m for m in label_modes if m not in ANSWER_LABELS]
    if bad_labels:
        raise SystemExit(f"unknown answer labelling {bad_labels}; known: {list(ANSWER_LABELS)}")

    alignments = list(patching.DEFAULT_ALIGNMENTS)
    if args.include_all_positions:
        alignments.append("all_positions")

    policy = collect.PositionPolicy(
        max_evidence_tokens=args.max_evidence_tokens,
        n_final_tokens=args.n_final_tokens,
        n_control=args.n_control,
        seed=args.seed,
    )
    head_config = head_stats.HeadStatsConfig(
        n_bins=args.head_bins, final_rows=args.head_final_rows
    )

    n_head_items = min(int(args.head_items), len(items))
    n_probe_layers = len(probe_layers) or n_layers_panel
    fits_per_cell = 1 + int(args.n_permutations)
    n_ridge_fits = len(RIDGE_TASKS) * n_probe_layers * len(probe_sites) * fits_per_cell
    # "full" is skipped before any fit on this bank; count it as free rather than pretend.
    n_costed_labels = len([m for m in label_modes if m != "full"])
    n_logistic_fits = (
        n_costed_labels * (len(logistic_layers) or n_layers_panel)
        * len(logistic_sites) * fits_per_cell
    )

    # Pairs cannot be counted before the tokenizer and records are read, so the estimate
    # uses the cap. That over-states rather than under-states, which is the safe direction
    # for a budget gate.
    est = cost_estimate(
        n_arms=len(arms),
        stages=stages,
        n_items=len(items),
        n_pairs=int(args.max_pairs) if args.max_pairs else len(items) // 4,
        n_head_items=n_head_items,
        patch_layers=patch_layers or range(n_layers_panel),
        patch_sites=patch_sites,
        alignments=alignments,
        head_layers=head_layers or range(n_layers_panel),
        head_passes=2 if args.head_final_rows else 1,
        length_bucket=cost_bucket,
        n_ridge_fits=n_ridge_fits * len(arms),
        n_logistic_fits=n_logistic_fits * len(arms),
    )
    store_bytes = collect.estimate_bytes(
        len(items), n_probe_layers, len(probe_sites), policy.max_positions, 1024
    )

    print(f"{len(items)} items ({len({i.scenario_id for i in items})} scenarios, "
          f"buckets {buckets}) x {len(arms)} arm(s) {[arm_label(*a) for a in arms]}")
    print(f"stages: {list(stages)}")
    for name, line in est["per_stage"].items():
        flag = " (ESTIMATE)" if line.get("estimated") else ""
        print(f"  {name:16s} {line['forwards']:8d} forwards  {line['seconds'] / 60:8.1f} min{flag}")
    print(f"  GPU total {est['gpu_hours']:.2f} h (budget {GPU_HOUR_BUDGET} h); "
          f"CPU probe fits ridge={est['cpu_probe_fits']['ridge']} "
          f"logistic={est['cpu_probe_fits']['logistic']} "
          f"(~{est['cpu_probe_seconds_estimated'] / 60:.0f} min, estimate)")
    print(f"  activation store {store_bytes / 1e9:.2f} GB per arm "
          f"(ceiling {args.max_activation_gb} GB)")

    if args.dry_run:
        return 0
    if est["exceeds_budget"] and not args.yes_i_costed_this:
        raise SystemExit(
            f"estimate is {est['gpu_hours']:.2f} GPU-hours, over the {GPU_HOUR_BUDGET}-hour "
            "guard; re-run with --yes-i-costed-this once you have looked at that number"
        )
    if "probes" in stages and store_bytes > args.max_activation_gb * 1e9:
        raise SystemExit(
            f"the activation store would be {store_bytes / 1e9:.2f} GB per arm, over the "
            f"{args.max_activation_gb} GB ceiling. Lower --max-scenarios/--limit, cut "
            "--probe-layers, or raise --max-activation-gb if the host really has the RAM."
        )

    is_partial_run = bool(
        args.limit
        or args.max_scenarios
        or args.family
        or set(stages) != set(STAGES)
        or tuple(arms) != DEFAULT_ARMS
    )
    tag = args.tag or ("partial" if is_partial_run else None)
    if is_partial_run and args.tag is None:
        # Phase 3 learned this the hard way: a --limit smoke run silently overwrote the
        # real summary with a plausible-looking small one. A partial run cannot land on
        # the canonical filenames.
        print("NOTE: partial run; outputs are suffixed '_partial' so they cannot clobber "
              "a full run. Pass --tag to choose your own suffix.")

    out_dir = Path(args.out_dir) if args.out_dir else paths.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(stem: str, suffix: str) -> Path:
        return out_dir / f"{stem}{'_' + tag if tag else ''}{suffix}"

    records: list[dict[str, Any]] = []
    if "patching" in stages:
        records_path = (
            Path(args.records)
            if args.records
            else paths.RESULTS / f"baseline_position_results_{args.split}_tf.parquet"
        )
        records = load_records(records_path)

    tok = load_tokenizer("qwen3")  # one vocabulary serves the panel (SPEC 11.4)
    t_tok = time.perf_counter()
    pairs_all = prepare_items(tok, items)
    lookup = {tok_item.item_id: (item, tok_item) for item, tok_item in pairs_all}
    print(f"tokenized {len(pairs_all)} items in {time.perf_counter() - t_tok:.1f} s")

    backends = probe_backends() if args.device == "cuda" else {"cuda": False}

    probe_payload: dict[str, Any] = {}
    all_patch_records: list[patching.PatchRecord] = []
    head_frames: list[Any] = []
    per_arm: dict[str, Any] = {}

    for key, ungate in arms:
        label = arm_label(key, ungate)
        arm_info: dict[str, Any] = {
            "model_key": key,
            "ungated": ungate,
            "dtype_approximated": dtype_is_approximation(key),
        }
        model, arch = load_model(key, device=args.device, ungate=ungate)
        on_cuda = next(model.parameters()).device.type == "cuda"
        arm_layers = tuple(range(arch.num_layers))
        use_probe_layers = probe_layers or arm_layers
        use_patch_layers = patch_layers or arm_layers
        use_head_layers = head_layers or arm_layers
        use_logistic_layers = logistic_layers or arm_layers

        try:
            # ``required`` only when the run is long enough that a silent MATH fallback
            # would cost hours; a smoke run should not die on a machine without cuDNN.
            with efficient_attention(required=not is_partial_run, active=on_cuda) as pinned:
                if "probes" in stages:
                    t0 = time.perf_counter()
                    seen = [0]
                    n_collect = len(pairs_all)

                    # collect_activations calls progress(rows_in_this_item); the row count
                    # varies with how many evidence tokens survived, so items are counted.
                    def tick(_n_rows: int, _seen=seen, _k=label, _tot=n_collect) -> None:
                        before, _seen[0] = _seen[0], _seen[0] + 1
                        if before // 100 != _seen[0] // 100:
                            print(f"  {_k}/collect {_seen[0]}/{_tot}", flush=True)

                    dataset = collect.collect_activations(
                        model, pairs_all, layers=use_probe_layers, sites=probe_sites,
                        policy=policy, model_key=label, progress=tick,
                    )
                    print(f"{label}/collect {len(dataset)} rows, "
                          f"{dataset.nbytes / 1e9:.2f} GB, "
                          f"{(time.perf_counter() - t0) / 60:.1f} min (pinned={pinned})")
                    if args.save_activations:
                        dataset.save_npz(out_path(f"activations_{label}", ".npz"))
                    results, relabelling = run_probes(
                        dataset, tok,
                        ridge_layers=use_probe_layers, ridge_sites=probe_sites,
                        logistic_layers=use_logistic_layers, logistic_sites=logistic_sites,
                        label_modes=label_modes, max_classes=args.probe_max_classes,
                        n_permutations=args.n_permutations,
                        test_fraction=args.test_fraction, seed=args.seed,
                    )
                    probe_payload[label] = {
                        "dataset": dataset.describe(),
                        "relabelling": relabelling,
                        "highlights": probe_highlights(results),
                        "results": [r.to_json() for r in results],
                    }
                    arm_info["probes"] = probe_payload[label]["highlights"]
                    arm_info["answer_relabelling"] = relabelling
                    del dataset  # the arm's store is a gigabyte; do not carry it forward

                if "patching" in stages:
                    arm_rows = [r for r in records if str(r.get("arm")) == label]
                    if not arm_rows:
                        raise SystemExit(
                            f"no Phase-3 records for arm {label!r}; available: "
                            f"{sorted({str(r.get('arm')) for r in records})}"
                        )
                    pairs, info = pairs_for_arm(
                        arm_rows, lookup, mode=args.pair_mode,
                        min_nll_gap=args.min_nll_gap, max_pairs=args.max_pairs,
                    )
                    arm_info["pairs"] = info
                    print(f"{label}/pairs {info['n_pairs']} from {info['n_candidates']} "
                          f"candidates ({args.pair_mode})")
                    t0 = time.perf_counter()
                    total = len(pairs)
                    arm_records: list[patching.PatchRecord] = []
                    # Progress is per pair, not per patch: run_patch_pair's own callback
                    # fires once per (alignment, layer, site) and would print hundreds of
                    # lines per pair.
                    for index, pair in enumerate(pairs, start=1):
                        arm_records.extend(
                            patching.run_patch_pair(
                                model, pair, layers=use_patch_layers, sites=patch_sites,
                                alignments=alignments,
                                n_final_tokens=args.patch_final_tokens,
                                seed=args.seed, model_key=label,
                            )
                        )
                        if index % 8 == 0 or index == total:
                            print(f"  {label}/patch {index}/{total} pairs", flush=True)
                    print(f"{label}/patch {len(arm_records)} records in "
                          f"{(time.perf_counter() - t0) / 60:.1f} min")
                    all_patch_records.extend(arm_records)
                    arm_info["patch"] = patch_summary(arm_records)
                else:
                    arm_records = []

                if "heads" in stages:
                    t0 = time.perf_counter()
                    # Stride rather than head: the bank is ordered by scenario, so the
                    # first N items would be a handful of scenarios of one family.
                    stride = max(1, len(pairs_all) // max(1, n_head_items))
                    head_items = pairs_all[::stride][:n_head_items]
                    n_head = len(head_items)
                    seen_h = [0]

                    def head_tick(_n: int = 1, _seen=seen_h, _k=label, _tot=n_head) -> None:
                        _seen[0] += 1
                        if _seen[0] % 8 == 0 or _seen[0] == _tot:
                            print(f"  {_k}/heads {_seen[0]}/{_tot}", flush=True)

                    table = head_stats.capture_head_stats(
                        model, head_items, layers=use_head_layers, config=head_config,
                        arch=arch, model_key=label, progress=head_tick,
                    )
                    if arm_records:
                        table = table.with_patch_effect(
                            arm_records, site="attn_out", alignment="final_query"
                        )
                    head_frames.append(head_stats.head_stats_to_dataframe(table.rows()))
                    arm_info["heads"] = table.describe()
                    print(f"{label}/heads {len(head_items)} items in "
                          f"{(time.perf_counter() - t0) / 60:.1f} min")
        finally:
            free_model(model)
        per_arm[label] = arm_info

    # -- artifacts ---------------------------------------------------------------------
    config = {
        "split": args.split,
        "arms": [arm_label(*a) for a in arms],
        "stages": list(stages),
        "length_bucket": args.length_bucket,
        "family": args.family,
        "local_position": args.local_position,
        "n_items": len(items),
        "item_limit": args.limit,
        "max_scenarios": args.max_scenarios,
        "pair_mode": args.pair_mode,
        "min_nll_gap": args.min_nll_gap,
        "max_pairs": args.max_pairs,
        "alignments": alignments,
        "patch_layers": list(patch_layers) or "all",
        "patch_sites": list(patch_sites),
        "probe_layers": list(probe_layers) or "all",
        "probe_sites": list(probe_sites),
        "logistic_layers": list(logistic_layers) or "all",
        "logistic_sites": list(logistic_sites),
        "answer_labels": list(label_modes),
        "probe_max_classes": args.probe_max_classes,
        "n_permutations": args.n_permutations,
        "position_policy": policy.describe(),
        "head_config": head_config.describe(),
        "head_items": n_head_items,
        "seed": args.seed,
        "device": args.device,
        "is_partial_run": is_partial_run,
    }

    written: list[str] = []
    if "probes" in stages:
        path = out_path("activation_probe_results", ".json")
        write_result(
            {"backend": probes.BACKEND, "is_partial_run": is_partial_run,
             "config": config, "by_arm": probe_payload},
            path, config=config,
        )
        written.append(str(path))
    if "patching" in stages:
        path = out_path("activation_patch_results", ".parquet")
        patching.write_patch_records(all_patch_records, path, config=config)
        written.append(str(path))
    if "heads" in stages:
        import pandas as pd

        path = out_path("head_ablation_results", ".parquet")
        frame = pd.concat(head_frames, ignore_index=True) if head_frames else pd.DataFrame()
        write_result(frame, path, config=config)
        written.append(str(path))

    limitations = [
        "ablation_effect in head_ablation_results.parquet is all-NaN with "
        "ablation_scope='unset': Phases 4-5 have produced no SignificanceMatrix to attach.",
        "patch_effect is a LAYER quantity, not a head quantity: attn_out is captured after "
        "o_proj, where the heads have already been summed.",
        "span_aligned patches move the evidence's content and its RoPE phase together; a "
        "positive effect cannot be attributed to either alone. random_control bounds the "
        "amount-of-substitution part only.",
        "Probes are correlational. A high R^2 nominates a layer for patching; it is not "
        "evidence the forward pass uses that information.",
        "answer_identity_full is expected to report status='skipped' on this bank: answers "
        "are unique per scenario, so a by-scenario split leaves every test class unseen. "
        "answer_identity_first_token is near-degenerate for the same reason (measured on "
        "the calibration 4K bank: only two first-token classes appear in two or more "
        "scenarios, one of them covering ~70% of them), so read its accuracy against the "
        "'chance' column, which is the test-set majority rate. answer_identity_family is "
        "the balanced four-class control and is the one of the three that carries "
        "information -- about task identity at the final token, not about answer identity.",
    ]
    if args.include_all_positions:
        limitations.append(
            "all_positions was enabled: it replaces the whole computation up to the patched "
            "layer, so a near-total recovery is expected by construction and localises "
            "nothing. It is an upper bound, not a localisation."
        )
    else:
        limitations.append(
            "Whole-residual (all_positions) patching was NOT run. It needs the two runs to "
            "tokenize to the same length and, even then, position i holds a different token "
            "in the two runs wherever the evidence moved. --include-all-positions runs it as "
            "an explicit upper bound."
        )

    summary = {
        "config": config,
        "is_partial_run": is_partial_run,
        "cost_estimate": est,
        "backends": backends,
        "files": written,
        "n_patch_records": len(all_patch_records),
        "by_arm": per_arm,
        "limitations": limitations,
    }
    summary_path = out_path("phase6_summary", ".json")
    write_result(summary, summary_path, config=config)
    written.append(str(summary_path))

    for line in written:
        print(f"wrote {line}")
    if is_partial_run:
        print("NOTE: is_partial_run=true is recorded in every artifact written above")
    print(json.dumps({k: summary[k] for k in ("is_partial_run", "n_patch_records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
