"""Phase 3 — baseline positional fingerprint for the whole panel.

Answers brief questions 1-3 (does each model lose the middle, which has the best
worst-position robustness, which has the smallest edge-middle gap) and produces the
positional fingerprints ``merging/soup.py`` needs in Phase 8.

Runs the calibration and validation splits only. The sealed split is untouched here
and is opened once, in Phase 14.

    PYTHONPATH=src python scripts/phase3_baseline_fingerprint.py --split calibration

Cost: about 0.3 s per 4K item per model with the backend pinned (see
``rineya_psa.backend``), so a 2 000-item bank across four models is roughly 40
minutes. ``--dry-run`` prints the estimate and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_psa import paths  # noqa: E402
from rineya_psa.backend import efficient_attention, probe_backends  # noqa: E402
from rineya_psa.evaluation import metrics  # noqa: E402
from rineya_psa.evaluation.runner import run_items, write_parquet  # noqa: E402
from rineya_psa.loading import (  # noqa: E402
    dtype_is_approximation,
    free_model,
    load_model,
    load_tokenizer,
)
from rineya_psa.statistics.provenance import write_result  # noqa: E402
from rineya_psa.types import BenchmarkItem  # noqa: E402

# Supa-A1.1's abstention offset moves exactly one vocabulary row. That is a real part of
# the published model for accuracy, and a confound for anything log-prob based, so the
# panel is run in both modes and the two are reported separately rather than averaged.
ARMS: tuple[tuple[str, bool], ...] = (
    ("qwen3", False),
    ("halo", False),
    ("nimbus", False),
    ("supa_a11", False),   # gated, as published
    ("supa_a11", True),    # ungated, for the mechanistic comparison
)


def arm_label(key: str, ungate: bool) -> str:
    return f"{key}_ungated" if ungate else key


def load_split(split: str) -> list[BenchmarkItem]:
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


# Measured on this card, batch 1, backend pinned (see rineya_psa.backend).
_FWD_SECONDS = {"2k": 0.12, "4k": 0.29, "8k": 0.84, "16k": 2.27, "32k": 7.06}

# Greedy decoding pays one prefill plus 24 cached decode steps. The decode steps are
# cheap relative to the prefill at these lengths but not free; 1.6x the prefill is the
# measured ratio at 4K and is used as a flat multiplier.
_GREEDY_MULTIPLIER = 1.6


def estimate_seconds(items: Sequence[BenchmarkItem], n_arms: int, modes: Sequence[str]) -> float:
    prefill = sum(_FWD_SECONDS.get(it.length_bucket, 0.29) for it in items)
    per_arm = sum(prefill * (_GREEDY_MULTIPLIER if m == "greedy" else 1.0) for m in modes)
    return n_arms * per_arm


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="calibration", choices=("calibration", "validation"))
    ap.add_argument("--mode", default="both", choices=("teacher_forced", "greedy", "both"))
    ap.add_argument("--limit", type=int, default=None, help="cap items, for smoke runs")
    ap.add_argument("--models", default=None, help="comma-separated subset of the panel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-i-costed-this", action="store_true")
    args = ap.parse_args(argv)

    if args.split == "sealed":  # defence in depth; argparse already refuses it  # sealed-ok:
        raise SystemExit("the sealed split is opened only by scripts/phase14_*.py")  # sealed-ok:

    items = load_split(args.split)
    if args.limit:
        items = items[: args.limit]
    arms = ARMS
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        arms = tuple(a for a in arms if a[0] in want)

    modes = ("teacher_forced", "greedy") if args.mode == "both" else (args.mode,)
    est = estimate_seconds(items, len(arms), modes)
    print(f"{len(items)} items x {len(arms)} arms x {len(modes)} mode(s); "
          f"estimated {est / 60:.1f} min")
    if args.dry_run:
        return 0
    if est > 2 * 3600 and not args.yes_i_costed_this:
        raise SystemExit(
            f"estimate is {est / 3600:.1f} GPU-hours, over the 2-hour guard; "
            "re-run with --yes-i-costed-this once you have looked at that number"
        )
    backends = probe_backends()
    # RunRecord deliberately carries only run outcomes, so the design factors that are
    # not in it (local rhetorical position, which grid emitted the item, the multi-hop
    # guided flag) are joined back by item_id here. Phase 3 must be able to split on
    # local position: pooling it into the global-position axis puts a spurious step at
    # exactly the three positions that define the edge-middle gap.
    design = {
        it.item_id: {
            "local_position": it.local_position,
            "grid": it.meta.get("grid"),
            "guided": bool((it.meta.get("example") or {}).get("guided", False)),
            "n_answer_chars": len(it.answer),
            # THE position axis. For exact_retrieval and semantic_single_hop there is one
            # evidence sentence and this equals target_positions[0]. For multi_hop and
            # state_constraint it does NOT: those carry several evidence sentences, the
            # non-decisive ones are pinned by construction, and only the decisive one is
            # swept. Grouping by target_positions[0] collapses state_constraint onto the
            # single constant 0.25 and reads multi_hop off its wrong endpoint.
            "primary_target": float(it.meta["primary_target"]),
            "primary_achieved": float(
                it.meta.get("primary_achieved_char", it.meta["primary_target"])
            ),
            "decisive_index": int(it.meta["example"]["decisive_index"]),
        }
        for it in items
    }
    tok = load_tokenizer("qwen3")  # one vocabulary serves the panel; see Phase 1 §1
    records: list[dict] = []

    for key, ungate in arms:
        label = arm_label(key, ungate)
        model, arch = load_model(key, ungate=ungate)
        for mode in modes:
            t0 = time.perf_counter()
            # run_items calls progress(rows_in_this_batch), so accumulate here.
            seen = [0]

            total_rows = len(items)

            def tick(n_rows: int, _seen=seen, _k=label, _m=mode, _tot=total_rows) -> None:
                before, _seen[0] = _seen[0], _seen[0] + n_rows
                if before // 200 != _seen[0] // 200:
                    print(f"  {_k}/{_m} {_seen[0]}/{_tot}", flush=True)

            with efficient_attention(required=True) as pinned:
                out = run_items(
                    model, tok, items, mode=mode,
                    max_new_tokens=24,  # the project's canonical long-context budget
                    progress=tick,
                )
            dt = time.perf_counter() - t0
            print(f"{label}/{mode}: {len(out)} items in {dt / 60:.1f} min (pinned={pinned})")
            for rec in out:
                row = rec.to_json()
                row["arm"] = label
                row["model_key"] = key
                row["ungated"] = ungate
                row["dtype_approximated"] = dtype_is_approximation(key)
                row.update(design[row["item_id"]])
                records.append(row)
        free_model(model)

    # The summary path is DERIVED from --out below, not hardcoded. It used to be fixed,
    # which meant a `--limit 60` smoke run silently overwrote the real run's summary with
    # a 60-item one that looked entirely plausible. Anything that can quietly replace a
    # result with a smaller result is a correctness bug, not a convenience bug.
    default_out = paths.RESULTS / f"baseline_position_results_{args.split}.parquet"
    out_path = Path(args.out) if args.out else default_out
    paths.ensure_dirs()
    try:
        write_parquet(records, out_path)
    except Exception as exc:  # noqa: BLE001 - parquet is optional; the data is not
        print(f"parquet unavailable ({exc}); writing jsonl instead")
        out_path = out_path.with_suffix(".jsonl")
        with out_path.open("w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row) + "\n")

    summary = summarise(records)
    is_partial = bool(args.limit) or len(arms) != len(ARMS) or len(modes) != 2
    summary_path = out_path.with_name(
        out_path.name.replace("results", "summary").rsplit(".", 1)[0] + ".json"
    )
    write_result(
        {"summary": summary, "backends": backends, "n_records": len(records),
         "split": args.split, "modes": list(modes), "n_items": len(items),
         "item_limit": args.limit, "is_partial_run": is_partial,
         "arms": [arm_label(k, u) for k, u in arms]},
        summary_path,
    )
    if is_partial:
        print(f"NOTE: partial run (limit={args.limit}, arms={len(arms)}/{len(ARMS)}, "
              f"modes={list(modes)}); summary marked is_partial_run=true")
    print(json.dumps(summary, indent=2)[:3000])
    return 0


def summarise(records: Sequence[dict]) -> dict:
    """Per-arm, per-family positional summary: the table brief section 10 asks for.

    Two different position values are used deliberately. Cells are grouped by the
    **designed** target position, because that is the experimental condition and every
    item in a cell is a replicate of it. The **measured** position (evidence-span
    midpoint over prompt length, which differs from target by ~0.007 on average) is what
    the quadratic fit regresses on, because the fit is over a continuous axis and should
    use where the evidence actually landed. Grouping by the measured value instead --
    an earlier version of this function did -- shatters the seven-point grid into
    singleton bins and makes every aggregate meaningless.
    """
    import collections

    by = collections.defaultdict(list)
    for r in records:
        if r.get("gold_nll") is None and r.get("correct") is None:
            continue
        # Group on the DECISIVE evidence sentence, not the first one. See the note in
        # main() where primary_target is joined on.
        target = float(r.get("primary_target", r["target_positions"][0]))
        acts = r.get("actual_positions") or ()
        idx = int(r.get("decisive_index", 0))
        measured = float(acts[idx]) if idx < len(acts) else float(
            r.get("primary_achieved", target)
        )
        by[(r["arm"], r["family"], r["length_bucket"], r["local_position"])].append(
            (target, measured, r.get("correct"), r.get("gold_nll"))
        )

    out: dict = {}
    for (arm, family, length, local), rows in sorted(by.items()):
        cells = sorted({t for t, _, _, _ in rows})
        acc_by_pos, nll_by_pos, n_by_pos = {}, {}, {}
        for g in cells:
            sel = [(c, n) for t, _, c, n in rows if t == g]
            cs = [c for c, _ in sel if c is not None]
            ns = [n for _, n in sel if n is not None]
            n_by_pos[g] = len(sel)
            if cs:
                acc_by_pos[g] = sum(cs) / len(cs)
            if ns:
                nll_by_pos[g] = sum(ns) / len(ns)
        entry: dict = {"n": len(rows), "n_by_pos": n_by_pos,
                       "acc_by_pos": acc_by_pos, "nll_by_pos": nll_by_pos}

        meas = [m for _, m, _, _ in rows]
        if len(acc_by_pos) >= 3:
            entry["edge_middle_gap"] = metrics.edge_middle_gap(acc_by_pos)
            entry["worst_position_acc"] = metrics.worst_position(acc_by_pos)
            entry["worst_position_at"] = min(acc_by_pos, key=acc_by_pos.get)
            entry["pri"] = metrics.position_robustness_index(acc_by_pos)
            cs = [(m, c) for _, m, c, _ in rows if c is not None]
            if len(cs) >= 4:
                fit = metrics.ushape_fit([m for m, _ in cs], [float(c) for _, c in cs])
                entry["ushape"] = {"beta0": fit.beta0, "beta1": fit.beta1,
                                   "beta2": fit.beta2, "r_squared": fit.r_squared, "n": fit.n}
        if len(nll_by_pos) >= 3:
            # NLL is the primary metric and lower is better, so the sign is flipped to
            # keep "positive edge-middle gap means the middle is worse".
            entry["nll_edge_middle_gap"] = -metrics.edge_middle_gap(nll_by_pos)
            entry["nll_worst"] = max(nll_by_pos.values())
            entry["nll_worst_at"] = max(nll_by_pos, key=nll_by_pos.get)
            ns = [(m, n) for _, m, _, n in rows if n is not None]
            if len(ns) >= 4:
                fit = metrics.ushape_fit([m for m, _ in ns], [n for _, n in ns])
                entry["nll_ushape"] = {"beta0": fit.beta0, "beta1": fit.beta1,
                                       "beta2": fit.beta2, "r_squared": fit.r_squared, "n": fit.n}
        entry["measured_position_mean"] = sum(meas) / len(meas)
        out[f"{arm}|{family}|{length}|{local}"] = entry
    return out


if __name__ == "__main__":
    raise SystemExit(main())
