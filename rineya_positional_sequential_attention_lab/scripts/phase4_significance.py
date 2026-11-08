"""Phase 4 - the layer x KV-head x position Significance Matrix (SPEC 11-12, brief 15).

Drives ``rineya_psa.significance.causal`` with SPEC 11.2's two-tier estimator, once per
panel checkpoint, on the calibration split:

* **Tier 1, attribution.** One forward + one backward per item yields every ``(l, h, b)``
  cell at once. Parameters are frozen (SPEC 12.4), batch size 1, ``logits_to_keep``
  narrowed to the answer window. Default length bucket is 2K because a frozen-parameter
  backward costs 4.2 GB there and 7.2 GB at 4K on a card with ~7.6 GB usable; ``--length
  4k`` is opt-in and prints that warning.
* **Tier 2, ablation.** Real ``eps = 0.25`` Mode-2 value attenuation, one cell per
  forward, on the top-K cells by ``|attribution|`` **plus a random control sample of the
  same size drawn from the remainder**. The control is not optional: without it the
  agreement statistic is computed only where attribution already said "big", and that
  range restriction inflates the correlation. This script has no flag to switch it off.
* **The gate.** ``validate_attribution_against_ablation`` on the union. Its verdict is
  printed as a banner and written to the summary json. When it is
  ``fallback_to_coarse_to_fine_ablation`` the Tier-1 matrix must not be quoted: SPEC 11.2
  says the study falls back to coarse-to-fine ablation on the six representative layers
  and says so in the report.

Artifacts, per arm, under ``<out-dir>/<arm>/`` (brief section 46 fixes the two names, so
each arm gets its own directory rather than a suffixed filename):

    kv_head_position_significance.npz    Tier-1 matrix + per-item evidence + provenance
    kv_head_position_significance.csv    one row per cell
    tier2_ablation_significance.npz/.csv Tier-2, NaN everywhere the ablation never went
    phase4_significance_summary.json     agreement, verdict, cost actually incurred

Defaults are ``exact_retrieval`` and ``multi_hop_guided``: Phase 3 found those two carry
the largest and the most *differently shaped* positional structure (a recency gradient
versus a genuine middle trough), so a significance matrix that explains both is the
interesting object. ``--families`` overrides.

The sealed split is not reachable from here. ``--split`` accepts only the two open
splits, and a ``--bank`` path that resolves inside ``data/`` must lie in one of them;
the third split is opened once, in Phase 14.

    PYTHONPATH=src python scripts/phase4_significance.py --dry-run
    PYTHONPATH=src python scripts/phase4_significance.py --yes-i-costed-this

DEVIATION: ``significance.causal`` has no public entry point for ablating an *explicit
list of cells*. ``estimate_causal_significance_ablation`` takes a layer x head x bin grid
plus a schedule (``full`` / ``ladder`` / ``grouped``), and the top-K-plus-controls set
SPEC 11.2 asks for is not a cross product of those axes - running it as one would cost
8x more forwards than the cells actually wanted. This script therefore composes
``causal._score_all`` (baseline) and ``causal._evaluate_groups`` (one single-cell group
per forward) itself and assembles the ``SignificanceResult``. Those two are private, so
their presence is asserted at import time rather than discovered mid-run. The missing
library function is ``estimate_causal_significance_cells(model, tok, items, cells, ...)``
and it is reported to the lead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.backend import efficient_attention, probe_backends  # noqa: E402
from rineya_psa.evaluation.runner import prepare_items  # noqa: E402
from rineya_psa.hooks.binning import AbsoluteBinning  # noqa: E402
from rineya_psa.loading import (  # noqa: E402
    dtype_is_approximation,
    free_model,
    load_model,
    load_tokenizer,
)
from rineya_psa.significance import causal  # noqa: E402
from rineya_psa.statistics.provenance import write_result  # noqa: E402
from rineya_psa.types import BenchmarkItem  # noqa: E402

# SPEC section 1: every mechanistic comparison runs Supa-A1.1 **ungated**. Its +5.15625
# offset on token 51378 shifts one vocabulary row, which moves the softmax normaliser and
# therefore every gold NLL; a significance matrix measured through it would be reading the
# abstention gate as if it were attention. The other three keys have no gate to ungate.
ARMS: tuple[tuple[str, bool], ...] = (
    ("qwen3", False),
    ("halo", False),
    ("nimbus", False),
    ("supa_a11", True),
)

#: The two splits this script may read. The third is opened once, in Phase 14, and is
#: deliberately not named here (``tests/test_split_discipline.py`` scans the AST).
OPEN_SPLITS: tuple[str, ...] = ("calibration", "validation")

#: Phase 3's finding, turned into a default: the recency gradient and the true middle
#: trough are different shapes, and a matrix that explains both is worth more than one
#: tuned on either.
DEFAULT_FAMILIES: tuple[str, ...] = ("exact_retrieval", "multi_hop_guided")

#: Nominal prompt length per bucket, for the cost estimate only. Attribution needs a
#: backward pass, which is 4.2 GB at 2K and 7.2 GB at 4K and OOMs at 8K, so the longer
#: buckets are not offered at all rather than offered and then failed into.
SEQ_LEN: dict[str, int] = {"2k": 2048, "4k": 4096}

#: SPEC 12.4's measured backward footprint, quoted in the --length 4k warning.
BACKWARD_GB: dict[str, float] = {"2k": 4.2, "4k": 7.2}

#: SPEC section 1: all four checkpoints are Qwen3-0.6B shaped, so the cost plan is the
#: same for every arm and can be priced before any model is on the card - which is what
#: makes --dry-run free.
PANEL_LAYERS: int = 28
PANEL_KV_HEADS: int = 8

# The two private helpers this script composes (see the DEVIATION note). Checked here so
# a library rename fails in the first second rather than after an hour of Tier-1 work.
for _needed in ("_score_all", "_evaluate_groups"):
    if not hasattr(causal, _needed):
        raise SystemExit(
            f"rineya_psa.significance.causal has no {_needed}; this script composes it to "
            "ablate an explicit cell list. See the DEVIATION note in the module docstring."
        )


# ======================================================================================
# items
# ======================================================================================


def arm_label(key: str, ungate: bool) -> str:
    return f"{key}_ungated" if ungate else key


def family_label(item: BenchmarkItem) -> str:
    """The family name Phase 3 reports under.

    ``multi_hop`` covers two constructions that behave nothing alike - the hard chain
    scored 0/36 in the Phase 2b pilot, the guided one does not - so they are separate
    labels here, exactly as ``scripts/phase3_analyze.py:group_label`` separates them.
    """
    if item.family == "multi_hop":
        return "multi_hop_guided" if (item.meta.get("example") or {}).get("guided") else (
            "multi_hop_hard"
        )
    return str(item.family)


def load_bank(split: str, bank: str | None) -> list[BenchmarkItem]:
    """Read the item bank, refusing any path inside ``data/`` that is not an open split."""
    if split not in OPEN_SPLITS:
        raise SystemExit(f"--split must be one of {list(OPEN_SPLITS)}; got {split!r}")
    path = Path(bank) if bank else paths.DATA / split / "position_examples.jsonl"
    resolved = path.resolve()
    data_root = paths.DATA.resolve()
    if data_root in resolved.parents:
        allowed = {(data_root / s).resolve() for s in OPEN_SPLITS}
        allowed.add(paths.DATA_GENERATED.resolve())
        if resolved.parent not in allowed:
            raise SystemExit(
                f"{resolved} is inside data/ but not in {sorted(p.name for p in allowed)}; "
                "the remaining split is opened exactly once, by scripts/phase14_*.py"
            )
    if not resolved.exists():
        raise SystemExit(f"no benchmark bank at {resolved}; run the Phase 2 generator first")
    rows = [
        BenchmarkItem.from_json(json.loads(line))
        for line in resolved.open(encoding="utf-8")
        if line.strip()
    ]
    # A --bank override may point at the pooled generated bank, which carries every split.
    return [r for r in rows if r.split == split] if bank else rows


def select_items(
    items: Sequence[BenchmarkItem],
    *,
    families: Sequence[str],
    length: str,
    fmt: str,
    limit: int | None,
    seed: int,
) -> list[BenchmarkItem]:
    """Filter to the design cells this phase measures, then subsample deterministically.

    ``local_position`` is pinned to ``middle`` for the same reason Phase 3 pins it: the
    local-rhetorical-position sweep only emits p in {0.02, 0.50, 0.98}, so pooling it puts
    a step at exactly the three positions that define the edge-middle gap.
    """
    wanted = set(families)
    kept = [
        it
        for it in items
        if family_label(it) in wanted
        and it.length_bucket == length
        and it.local_position == "middle"
        and (fmt == "all" or it.fmt == fmt)
    ]
    kept.sort(key=lambda it: it.item_id)
    if limit is not None and limit < len(kept):
        # Seeded permutation, not a head slice: the bank is written grouped by scenario
        # and by position, so ``kept[:limit]`` would silently cut the position grid down
        # to its first few points and make a smoke run look like a positional study.
        pick = np.random.default_rng(seed).permutation(len(kept))[:limit]
        kept = [kept[i] for i in sorted(int(x) for x in pick)]
    return kept


# ======================================================================================
# cost
# ======================================================================================


def tier2_cell_count(n_cells: int, top_k: int) -> tuple[int, int]:
    """``(n_top, n_control)`` for a grid of ``n_cells``. The control is never dropped."""
    n_top = max(0, min(int(top_k), int(n_cells)))
    n_control = max(0, min(n_top, int(n_cells) - n_top))
    return n_top, n_control


def plan_costs(
    *,
    n_items: int,
    n_ablation_items: int,
    n_layers: int,
    n_kv_heads: int,
    n_bins: int,
    top_k: int,
    seq_len: int,
) -> dict:
    """SPEC 11.3's estimate, from the library's own ``cost_estimate``, for one arm.

    Tier 2 is encoded as a ``mode="full"`` ablation over a degenerate ``n_cells x 1 x 1``
    grid. That is not a trick: ``cost_estimate`` prices ``mode="full"`` at one forward per
    cell per item, which is exactly what the selected-cell sweep does, and passing the
    real 28 x 8 x 8 grid would price the whole grid instead of the 128 cells visited.
    """
    n_cells = int(n_layers) * int(n_kv_heads) * int(n_bins)
    n_top, n_control = tier2_cell_count(n_cells, top_k)
    tier1 = causal.cost_estimate(
        n_items=n_items,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        n_bins=n_bins,
        seq_len=seq_len,
        tier="attribution",
    )
    tier2 = causal.cost_estimate(
        n_items=n_ablation_items,
        n_layers=max(1, n_top + n_control),
        n_kv_heads=1,
        n_bins=1,
        seq_len=seq_len,
        tier="ablation",
        mode="full",
        group_size=1,
    )
    seconds = float(tier1["estimated_seconds"]) + float(tier2["estimated_seconds"])
    return {
        "n_grid_cells": n_cells,
        "n_tier2_top": n_top,
        "n_tier2_control": n_control,
        "tier1": tier1,
        "tier2": tier2,
        "estimated_seconds": seconds,
        "estimated_gpu_hours": seconds / 3600.0,
    }


def print_plan(label: str, plan: dict) -> None:
    t1, t2 = plan["tier1"], plan["tier2"]
    print(
        f"  {label:16s} tier1 {t1['n_forwards']:6d} fwd + {t1['n_backwards']:6d} bwd "
        f"= {t1['estimated_seconds'] / 60:7.1f} min | "
        f"tier2 {t2['n_forwards']:7d} fwd over "
        f"{plan['n_tier2_top']}+{plan['n_tier2_control']} cells "
        f"= {t2['estimated_seconds'] / 60:7.1f} min | "
        f"total {plan['estimated_gpu_hours']:5.2f} GPU-h"
    )


# ======================================================================================
# tier 2 over an explicit cell list
# ======================================================================================


def nanmean_over_items(per_item: np.ndarray) -> np.ndarray:
    """Item-axis mean that keeps an all-NaN (never visited) cell NaN, and stays quiet.

    "Mean of empty slice" on every unvisited cell would be thousands of warnings on a run
    where NaN is the *intended* value, which is how a reader learns to ignore warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per_item, axis=0)


def ablate_cells(
    model,
    tok,
    pairs: Sequence[tuple],
    cells: Sequence[tuple[int, int, int]],
    *,
    arch,
    binning,
    eps: float,
    objective: str,
    device,
    model_key: str,
    selection: dict,
    progress=None,
) -> causal.SignificanceResult:
    """Real Mode-2 attenuation on ``cells`` only, one cell per forward.

    The returned matrix spans the **full** grid so its cell indices line up with the
    attribution result's; every cell the sweep did not visit stays NaN and
    ``evaluated=False``. Writing zeros there would claim "measured, and it did nothing",
    which is a different and false statement.
    """
    ls = tuple(range(arch.num_layers))
    hs = tuple(range(arch.num_kv_heads))
    bs = tuple(range(int(binning.n_bins)))
    li_of = {v: i for i, v in enumerate(ls)}
    hi_of = {v: i for i, v in enumerate(hs)}
    bi_of = {v: i for i, v in enumerate(bs)}

    per_item = np.full((len(pairs), len(ls), len(hs), len(bs)), np.nan, dtype=np.float64)
    evaluated = np.zeros((len(ls), len(hs), len(bs)), dtype=bool)

    started = time.perf_counter()
    # force_backend=False everywhere below: the SDPA pin is owned by the single
    # ``efficient_attention`` block in run_arm(), per SPEC 12.1 and this lab's rule that
    # one place owns it. Nesting the library's own pin inside it would be redundant.
    baseline = causal._score_all(
        model, tok, pairs, objective, device=device, ctx=None, force_backend=False
    )
    groups = [[tuple(int(x) for x in cell)] for cell in cells]
    deltas = causal._evaluate_groups(
        model,
        tok,
        pairs,
        groups,
        arch=arch,
        binning=binning,
        baseline=baseline,
        perturbation="value",
        eps=float(eps),
        objective=objective,
        device=device,
        force_backend=False,
        progress=progress,
    )
    for group, delta in zip(groups, deltas, strict=True):
        layer, head, b = group[0]
        per_item[:, li_of[layer], hi_of[head], bi_of[b]] = delta
        evaluated[li_of[layer], hi_of[head], bi_of[b]] = True
    wall = time.perf_counter() - started

    matrix = causal.SignificanceMatrix(
        values=nanmean_over_items(per_item),
        layers=ls,
        kv_heads=hs,
        bins=bs,
        binning_desc=binning.describe(),
        objective=objective,
        model_key=model_key,
        provenance={
            "tier": "ablation",
            "perturbation": "value",
            "eps": float(eps),
            "mode": "selected_cells",
            "n_items": len(pairs),
            "n_cells_visited": int(evaluated.sum()),
        },
    )
    return causal.SignificanceResult(
        matrix=matrix,
        per_item=per_item,
        item_ids=tuple(t.item_id for _, t in pairs),
        scenario_ids=tuple(i.scenario_id for i, _ in pairs),
        baseline=baseline,
        evaluated=evaluated,
        tier="ablation",
        perturbation="value",
        objective=objective,
        eps=float(eps),
        n_forwards=len(pairs) * (1 + len(groups)),
        wall_seconds=wall,
        notes={
            "mode": "selected_cells",
            "selection": {k: [list(c) for c in v] for k, v in selection.items()},
            "unvisited_cells_are_nan": True,
            "caveat": (
                "only the top-K plus the random control were ablated; every other cell "
                "is NaN because it was never measured, not because it measured zero"
            ),
        },
    )


def restrict_result_to_items(
    result: causal.SignificanceResult, keep: Sequence[int]
) -> causal.SignificanceResult:
    """Re-derive a result over a subset of its items, from the per-item evidence it kept.

    Needed when ``--ablation-limit`` runs Tier 2 on fewer items than Tier 1: comparing a
    128-item ablation mean against a 149-item attribution mean would put an item-set
    difference inside the Bland-Altman "difference in objective units". Slicing is exact
    and free, because Tier 1 already stored every item's contribution.
    """
    idx = np.asarray([int(i) for i in keep], dtype=np.int64)
    per_item = result.per_item[idx]
    prov = dict(result.matrix.provenance)
    prov["restricted_to_n_items"] = int(idx.size)
    notes = dict(result.notes)
    notes["restricted_to_n_items"] = int(idx.size)
    return dataclasses.replace(
        result,
        matrix=result.matrix.replace(values=nanmean_over_items(per_item), provenance=prov),
        per_item=per_item,
        item_ids=tuple(result.item_ids[int(i)] for i in idx),
        scenario_ids=tuple(result.scenario_ids[int(i)] for i in idx),
        baseline=result.baseline[idx],
        notes=notes,
    )


def mask_for_cells(
    result: causal.SignificanceResult, cells: Sequence[tuple[int, int, int]]
) -> np.ndarray:
    mask = np.zeros(result.evaluated.shape, dtype=bool)
    for cell in cells:
        li, hi, bi = result.cell_index(tuple(int(x) for x in cell))
        mask[li, hi, bi] = result.evaluated[li, hi, bi]
    return mask


def agreement_on(
    attribution: causal.SignificanceResult,
    ablation: causal.SignificanceResult,
    *,
    cells: Sequence[tuple[int, int, int]] | None,
    accept_rho: float,
    seed: int,
) -> dict:
    """The Tier-1/Tier-2 gate, optionally restricted to a subset of the ablated cells."""
    target = ablation
    if cells is not None:
        target = dataclasses.replace(ablation, evaluated=mask_for_cells(ablation, cells))
    return causal.validate_attribution_against_ablation(
        attribution, target, seed=seed, accept_rho=accept_rho
    )


# ======================================================================================
# one arm
# ======================================================================================


def _release_autograd_state(model: object) -> None:
    """Drop every gradient and cached block the Tier-1 backward left behind.

    ``empty_cache`` alone is not enough: parameter ``.grad`` tensors and any retained
    graph keep real allocations alive, so they are cleared first and the cache is
    released after.
    """
    import torch

    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
        param.requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_arm(
    key: str,
    ungate: bool,
    *,
    tok,
    items: Sequence[BenchmarkItem],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    import torch

    label = arm_label(key, ungate)
    binning = AbsoluteBinning(int(args.n_bins))
    model, arch = load_model(key, ungate=ungate)
    device = next(model.parameters()).device
    on_cuda = getattr(device, "type", str(device)) == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    try:
        pairs = prepare_items(tok, items)
        n_tokens = [len(t.input_ids) + len(t.answer_ids) for _, t in pairs]

        abl_index = list(range(len(pairs)))
        if args.ablation_limit is not None and args.ablation_limit < len(pairs):
            pick = np.random.default_rng(args.seed + 1).permutation(len(pairs))
            abl_index = sorted(int(i) for i in pick[: args.ablation_limit])
        abl_pairs = [pairs[i] for i in abl_index]

        # -- Tier 1 ------------------------------------------------------------------
        n_items = len(pairs)
        step = max(1, int(args.progress_every))
        seen = [0]

        def tier1_tick(n: int = 1) -> None:
            seen[0] += int(n)
            if seen[0] % (step * 8) == 0 or seen[0] == n_items:
                print(f"  {label} tier1 attribution {seen[0]}/{n_items}", flush=True)

        t0 = time.perf_counter()
        with efficient_attention(active=on_cuda, required=on_cuda) as pinned:
            attribution = causal.estimate_causal_significance_attribution(
                model,
                tok,
                pairs,
                None,
                None,
                int(args.n_bins),
                eps=float(args.eps),
                objective=args.objective,
                binning=binning,
                model_key=label,
                device=device,
                force_backend=False,  # the pin above owns it; see ablate_cells()
                progress=tier1_tick,
            )
            tier1_seconds = time.perf_counter() - t0
            print(
                f"  {label} tier1 done: {attribution.n_forwards} fwd + "
                f"{attribution.n_forwards} bwd in {tier1_seconds / 60:.1f} min "
                f"(pinned={pinned}, {attribution.matrix.values.size} cells)",
                flush=True,
            )

            # -- release Tier 1's autograd footprint before Tier 2 allocates ---------
            # Tier 1 runs a backward, which leaves ~4.2 GB of activation blocks in the
            # caching allocator plus fragmentation. Tier 2 then asks for fresh blocks and
            # OOMs on a 8.19 GB card even though its own steady-state need is ~1.3 GB.
            # Observed: Tier 1 completed in 1.1 min, Tier 2 died immediately after.
            _release_autograd_state(model)

            # -- cell selection: top-K plus the MANDATORY random control -------------
            selection = causal.select_validation_cells(
                attribution, k=int(args.top_k), seed=int(args.seed)
            )
            top_cells = [tuple(int(x) for x in c) for c in selection["top"]]
            control_cells = [tuple(int(x) for x in c) for c in selection["control"]]
            if not control_cells:
                raise SystemExit(
                    "the random control set is empty, so the agreement statistic would be "
                    "computed only where attribution already said 'big' - a range "
                    "restriction that inflates the correlation. Lower --top-k or raise "
                    "--n-bins so the grid has cells left over."
                )
            union = top_cells + control_cells

            # -- Tier 2 ---------------------------------------------------------------
            n_groups = len(union)
            done = [0]

            def tier2_tick(_rows: int) -> None:
                done[0] += 1
                if done[0] % step == 0 or done[0] == n_groups:
                    print(
                        f"  {label} tier2 ablation {done[0]}/{n_groups} cells "
                        f"({len(abl_pairs)} items each)",
                        flush=True,
                    )

            t0 = time.perf_counter()
            ablation = ablate_cells(
                model,
                tok,
                abl_pairs,
                union,
                arch=arch,
                binning=binning,
                eps=float(args.eps),
                objective=args.objective,
                device=device,
                model_key=label,
                selection={"top": top_cells, "control": control_cells},
                progress=tier2_tick,
            )
            tier2_seconds = time.perf_counter() - t0
        print(
            f"  {label} tier2 done: {ablation.n_forwards} fwd in "
            f"{tier2_seconds / 60:.1f} min",
            flush=True,
        )

        peak_gb = float(torch.cuda.max_memory_allocated() / 1e9) if on_cuda else None
    finally:
        free_model(model)

    # -- the gate --------------------------------------------------------------------
    attribution_for_gate = (
        attribution
        if len(abl_index) == len(pairs)
        else restrict_result_to_items(attribution, abl_index)
    )
    gate = agreement_on(
        attribution_for_gate,
        ablation,
        cells=None,
        accept_rho=float(args.accept_rho),
        seed=int(args.seed),
    )
    restricted: dict[str, object] = {}
    for name, subset in (("top_only", top_cells), ("control_only", control_cells)):
        # Reported because the contrast IS the point of the control: a rho that is high on
        # the union and near zero on the top-only subset is range restriction showing its
        # face, and a reader who only saw the union number could not tell.
        try:
            block = agreement_on(
                attribution_for_gate,
                ablation,
                cells=subset,
                accept_rho=float(args.accept_rho),
                seed=int(args.seed),
            )
            restricted[name] = {
                "n_cells": block["n_cells"],
                "spearman": block["spearman"],
                "spearman_ci": block["spearman_ci"],
            }
        except ValueError as exc:  # degenerate subset; say why, do not hide it
            restricted[name] = {"error": str(exc)}

    # -- artifacts -------------------------------------------------------------------
    arm_dir = out_dir / label
    verdict = str(gate["verdict"])
    usable = verdict == "attribution_usable"
    config = {
        "phase": 4,
        "arm": label,
        "model_key": key,
        "ungated": ungate,
        "split": args.split,
        "families": list(args.families),
        "length_bucket": args.length,
        "fmt": args.fmt,
        "objective": args.objective,
        "eps": float(args.eps),
        "n_bins": int(args.n_bins),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "accept_rho": float(args.accept_rho),
    }
    stamped = dataclasses.replace(
        attribution,
        matrix=attribution.matrix.replace(
            provenance={
                **dict(attribution.matrix.provenance),
                "agreement_verdict": verdict,
                "agreement_spearman": gate["spearman"],
                "usable_without_ablation": usable,
            }
        ),
        notes={
            **dict(attribution.notes),
            "agreement_verdict": verdict,
            "dtype_approximated": dtype_is_approximation(key),
        },
    )
    stamped.save(arm_dir / "kv_head_position_significance.npz", config=config)
    stamped.matrix.save_csv(arm_dir / "kv_head_position_significance.csv")
    ablation.save(arm_dir / "tier2_ablation_significance.npz", config=config)
    ablation.matrix.save_csv(arm_dir / "tier2_ablation_significance.csv")

    # The "unvisited stays NaN" promise, checked on the artifact rather than asserted in
    # prose: reloaded from disk, because that is what every downstream phase will read.
    reloaded = causal.SignificanceResult.load(arm_dir / "tier2_ablation_significance.npz")
    unvisited = ~reloaded.evaluated
    flat = reloaded.per_item.reshape(reloaded.n_items, -1)[:, unvisited.reshape(-1)]
    if flat.size and not np.isnan(flat).all():
        raise SystemExit(
            f"{label}: {int((~np.isnan(flat)).sum())} unvisited cell entries are not NaN "
            "in the saved tier-2 artifact; a never-measured cell must never read as zero"
        )
    if not np.isnan(reloaded.matrix.values[unvisited]).all():
        raise SystemExit(f"{label}: unvisited cells in the saved tier-2 matrix are not NaN")

    summary = {
        "arm": label,
        "model_key": key,
        "ungated": ungate,
        "dtype_approximated": dtype_is_approximation(key),
        "config": config,
        "n_items_tier1": len(pairs),
        "n_items_tier2": len(abl_pairs),
        "mean_prompt_tokens": float(np.mean(n_tokens)) if n_tokens else None,
        "grid": {
            "n_layers": arch.num_layers,
            "n_kv_heads": arch.num_kv_heads,
            "n_bins": int(args.n_bins),
            "n_cells": int(attribution.matrix.values.size),
        },
        "agreement": gate,
        "agreement_restricted": restricted,
        "verdict": verdict,
        "attribution_usable": usable,
        "fallback_to_coarse_to_fine_ablation": not usable,
        "cells": {
            "n_top": len(top_cells),
            "n_control": len(control_cells),
            "n_visited_by_ablation": int(ablation.evaluated.sum()),
            "n_never_visited": int((~ablation.evaluated).sum()),
            "never_visited_are_nan": True,
        },
        "cost_incurred": {
            "tier1_forwards": attribution.n_forwards,
            "tier1_backwards": attribution.n_forwards,
            "tier1_seconds": tier1_seconds,
            "tier2_forwards": ablation.n_forwards,
            "tier2_seconds": tier2_seconds,
            "total_seconds": tier1_seconds + tier2_seconds,
            "total_gpu_hours": (tier1_seconds + tier2_seconds) / 3600.0,
            "peak_vram_gb": peak_gb,
        },
        "tier1": stamped.to_json(),
        "tier2": ablation.to_json(),
        # Tier 1 runs with autograd on and Tier 2 under inference_mode. The same forward
        # should give the same loss; a large gap here means the two tiers are not scoring
        # the same thing and the Bland-Altman offset is an artefact, not a finding.
        "baseline_max_abs_gap": float(
            np.max(np.abs(attribution.baseline[abl_index] - ablation.baseline))
        ),
        "artifacts": {
            "tier1_npz": str(arm_dir / "kv_head_position_significance.npz"),
            "tier1_csv": str(arm_dir / "kv_head_position_significance.csv"),
            "tier2_npz": str(arm_dir / "tier2_ablation_significance.npz"),
            "tier2_csv": str(arm_dir / "tier2_ablation_significance.csv"),
        },
    }
    write_result(summary, arm_dir / "phase4_significance_summary.json", config=config)
    announce(label, summary)
    return summary


def announce(label: str, summary: dict) -> None:
    """Print the verdict, loudly when it is the fallback."""
    gate = summary["agreement"]
    ci = gate["spearman_ci"]
    lo = ci.get("ci_low")
    hi = ci.get("ci_high")
    line = (
        f"{label}: Spearman(tier1, tier2) = {gate['spearman']:+.3f} "
        f"[{lo:+.3f}, {hi:+.3f}] on {gate['n_cells']} cells "
        f"across {gate['n_layers']} layers; accept_rho={gate['accept_rho']}"
    )
    if summary["attribution_usable"]:
        print(f"\nVERDICT [{label}]: attribution_usable")
        print(f"  {line}\n", flush=True)
        return
    bar = "!" * 96
    print(f"\n{bar}")
    print(f"!! VERDICT [{label}]: fallback_to_coarse_to_fine_ablation")
    print(f"!! {line}")
    print("!! SPEC 11.2: Tier 1 does not track Tier 2, so the Tier-1 matrix is NOT usable")
    print("!! on its own. The study falls back to coarse-to-fine ablation on the six")
    print("!! representative layers, and the report must say so rather than proceed.")
    print(f"{bar}\n", flush=True)


# ======================================================================================
# entry point
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="calibration", choices=OPEN_SPLITS)
    ap.add_argument("--bank", default=None, help="override the item bank jsonl")
    ap.add_argument("--models", default=None, help="comma-separated subset of the panel")
    ap.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="comma-separated family labels (multi_hop_guided / multi_hop_hard are split)",
    )
    ap.add_argument("--length", default="2k", choices=tuple(SEQ_LEN))
    ap.add_argument("--fmt", default="natural", help="surface format, or 'all'")
    ap.add_argument("--objective", default="gold_nll", choices=("gold_nll", "margin"))
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--eps", type=float, default=causal.DEFAULT_EPS)
    ap.add_argument("--top-k", type=int, default=64, help="Tier-2 cells; controls match it")
    ap.add_argument("--accept-rho", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="cap items, for smoke runs")
    ap.add_argument(
        "--ablation-limit",
        type=int,
        default=None,
        help="cap the items Tier 2 uses; Tier 1 is re-derived on the same subset",
    )
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--progress-every", type=int, default=8, help="cells between log lines")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split not in OPEN_SPLITS:  # defence in depth; argparse already refuses it
        raise SystemExit(f"--split must be one of {list(OPEN_SPLITS)}")
    args.families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    if not args.families:
        raise SystemExit("--families is empty")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if args.n_bins < 2:
        raise SystemExit("--n-bins must be at least 2 for a positional matrix to mean anything")

    if args.length == "4k":
        print(
            "WARNING: --length 4k. Tier 1 needs a backward pass, measured at "
            f"{BACKWARD_GB['4k']} GB at 4K against ~7.6 GB usable on this card "
            f"(2K costs {BACKWARD_GB['2k']} GB). SPEC 12.4 calls 4K marginal: batch 1, "
            "nothing else resident, and it can still spill. 8K OOMs and is not offered.",
            flush=True,
        )

    arms = ARMS
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        unknown = want - {k for k, _ in ARMS}
        if unknown:
            raise SystemExit(f"unknown model keys {sorted(unknown)}; panel is {list(paths.PANEL)}")
        arms = tuple(a for a in ARMS if a[0] in want)

    bank = load_bank(args.split, args.bank)
    items = select_items(
        bank,
        families=args.families,
        length=args.length,
        fmt=args.fmt,
        limit=args.limit,
        seed=int(args.seed),
    )
    if not items:
        raise SystemExit(
            f"no items matched families={list(args.families)} length={args.length} "
            f"fmt={args.fmt} in the {args.split} split"
        )
    by_family: dict[str, int] = {}
    for it in items:
        by_family[family_label(it)] = by_family.get(family_label(it), 0) + 1

    n_ablation_items = min(len(items), args.ablation_limit or len(items))
    # The panel is shape-identical (SPEC section 1), so one plan prices every arm.
    plan = plan_costs(
        n_items=len(items),
        n_ablation_items=n_ablation_items,
        n_layers=PANEL_LAYERS,
        n_kv_heads=PANEL_KV_HEADS,
        n_bins=int(args.n_bins),
        top_k=int(args.top_k),
        seq_len=SEQ_LEN[args.length],
    )
    total_hours = plan["estimated_gpu_hours"] * len(arms)
    print(
        f"{len(items)} items ({by_family}) x {len(arms)} arms, {args.length}, "
        f"objective={args.objective}, eps={args.eps}, bins={args.n_bins}"
    )
    for key, ungate in arms:
        print_plan(arm_label(key, ungate), plan)
    # The comparison worth printing is against the SPEC 11.2 brute force -- every cell of
    # the real 28x8x8 grid ablated one at a time -- not against the 128 cells this plan
    # already narrowed to, which would compare tier 2 with itself and report a 1.0x win.
    brute_hours = plan["tier1"]["brute_force_gpu_hours"] * len(arms)
    print(
        f"  {'TOTAL':16s} {total_hours:5.2f} GPU-h estimated; brute-force ablation of all "
        f"{plan['n_grid_cells']} cells would be {brute_hours:.1f} GPU-h "
        f"({brute_hours / total_hours:.0f}x)"
    )
    if args.dry_run:
        print(json.dumps({"plan": plan, "n_arms": len(arms),
                          "total_gpu_hours": total_hours}, indent=2, default=str)[:4000])
        return 0
    if total_hours > causal.GPU_HOUR_BUDGET and not args.yes_i_costed_this:
        cells = plan["n_tier2_top"] + plan["n_tier2_control"]
        per_cell_hours = plan["tier2"]["estimated_gpu_hours"] / max(1, cells) * len(arms)
        raise SystemExit(
            f"estimate is {total_hours:.2f} GPU-hours, over the "
            f"{causal.GPU_HOUR_BUDGET}-hour guard of SPEC 11.3.\n"
            f"Tier 2 is {plan['tier2']['estimated_gpu_hours'] * len(arms):.2f} of it, at "
            f"~{per_cell_hours:.3f} GPU-h per ablated cell across all arms.\n"
            f"Knobs: --ablation-limit N scales Tier 2 linearly in N/{len(items)}; "
            f"--top-k K scales it in K/{args.top_k} (controls track K, so both halves "
            "shrink together and the control is never dropped).\n"
            "Re-run with --yes-i-costed-this once you have looked at that number."
        )

    import torch

    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir) if args.out_dir else paths.RESULTS / "phase4_significance"
    paths.ensure_dirs()
    try:
        backends = probe_backends()
    except Exception as exc:  # noqa: BLE001 - a probe failure must not kill the run
        backends = {"error": f"{type(exc).__name__}: {exc}"}

    tok = load_tokenizer("qwen3")  # one vocabulary serves the panel (SPEC 11.4)
    summaries: list[dict] = []
    for key, ungate in arms:
        label = arm_label(key, ungate)
        print(f"\n=== {label} ({len(items)} items) ===", flush=True)
        summaries.append(
            run_arm(key, ungate, tok=tok, items=items, args=args, out_dir=out_dir)
        )

    is_partial = bool(
        args.limit is not None
        or args.ablation_limit is not None
        or len(arms) != len(ARMS)
        or tuple(args.families) != DEFAULT_FAMILIES
    )
    fallback = [s["arm"] for s in summaries if not s["attribution_usable"]]
    run_summary = {
        "phase": 4,
        "split": args.split,
        "families": list(args.families),
        "family_counts": by_family,
        "length_bucket": args.length,
        "fmt": args.fmt,
        "objective": args.objective,
        "eps": float(args.eps),
        "n_bins": int(args.n_bins),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "n_items": len(items),
        "item_limit": args.limit,
        "ablation_item_limit": args.ablation_limit,
        "arms": [arm_label(k, u) for k, u in arms],
        "is_partial_run": is_partial,
        "estimated_gpu_hours": total_hours,
        "actual_gpu_hours": sum(
            s["cost_incurred"]["total_gpu_hours"] for s in summaries
        ),
        "any_fallback_to_coarse_to_fine_ablation": bool(fallback),
        "fallback_arms": fallback,
        "verdicts": {s["arm"]: s["verdict"] for s in summaries},
        "spearman": {s["arm"]: s["agreement"]["spearman"] for s in summaries},
        "n_cells_never_visited_by_ablation": {
            s["arm"]: s["cells"]["n_never_visited"] for s in summaries
        },
        "backends": backends,
        "arm_summaries": summaries,
    }
    write_result(
        run_summary,
        out_dir / "phase4_significance_run.json",
        config={"phase": 4, "arms": run_summary["arms"], "families": list(args.families)},
    )
    if is_partial:
        print(
            f"NOTE: partial run (limit={args.limit}, ablation_limit={args.ablation_limit}, "
            f"arms={len(arms)}/{len(ARMS)}, families={list(args.families)}); "
            "summary marked is_partial_run=true"
        )
    if fallback:
        bar = "!" * 96
        print(f"\n{bar}")
        print(f"!! {len(fallback)} of {len(summaries)} arms FELL BACK: {fallback}")
        print("!! Do not quote their Tier-1 matrices without the ablation beside them.")
        print(f"{bar}")
    print(f"\nwrote {out_dir / 'phase4_significance_run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
