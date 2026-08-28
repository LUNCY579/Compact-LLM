#!/usr/bin/env python3
"""Stage 7: compute-matched answer-blind routing among Rineya arms.

Constraints that shape this stage
--------------------------------
* Qwen3-4B may NOT be an arm. At ~6.7x the cost it breaks the 6-pass budget, and
  using the comparison model inside the candidate is circular.
* Every routing decision must be answer-blind: it may read arm outputs and the
  model's own confidence, never the key.
* The probe gate costs no extra forward pass -- it reads the hidden state of a
  pass already being made -- so a probe-gated single arm is still 1 pass.

Policies are selected on DEV only. The winner is frozen before TEST is scored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = ["base", "paraphrase", "counterfactual", "unanswerable"]


def load_scorer(root: Path):
    spec = importlib.util.spec_from_file_location(
        "ev", root / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_jsonl(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def is_unknown(text: str, ev) -> bool:
    return ev.normalize(text) in {"unknown", "not determinable", "cannot be determined"}


def fit_probe_oof(signals: dict[str, dict], key: dict[str, dict],
                  ids: list[str], folds: int = 5) -> dict[str, bool]:
    """Out-of-fold probe fire decisions, grouped by scenario.

    Grouping matters: the four variants of a scenario share a context, so an
    ungrouped split would leak near-identical rows across the fold boundary.
    """
    X = np.asarray([
        np.concatenate([np.asarray(v, dtype=np.float32)
                        for _, v in sorted(signals[i]["hidden"].items())])
        for i in ids])
    y = np.asarray([0 if key[i]["answerable"] else 1 for i in ids])
    groups = np.asarray([key[i]["scenario_id"] for i in ids])
    fires = np.zeros(len(ids), dtype=bool)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=0.05))
        clf.fit(X[tr], y[tr])
        s_tr, s_te = clf.predict_proba(X[tr])[:, 1], clf.predict_proba(X[te])[:, 1]
        # Threshold chosen inside the training fold only.
        best_t, best = 0.5, -1
        for t in np.unique(np.round(s_tr, 3)):
            f = s_tr >= t
            ok = np.zeros(len(tr), dtype=bool)
            ok[f & (y[tr] == 1)] = True
            ok[~f & (y[tr] == 0)] = True
            if ok.sum() > best:
                best, best_t = ok.sum(), float(t)
        fires[te] = s_te >= best_t
    return {i: bool(f) for i, f in zip(ids, fires)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--arms", type=Path, required=True,
                    help="dir containing <arm>_dev256.jsonl files")
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    ev = load_scorer(args.root)
    key = load_jsonl(args.key)
    signals = load_jsonl(args.signals)
    ids = sorted(set(key) & set(signals))

    ARMS = ["supa_a1", "baya17", "nimbus16", "pulse16", "halo_unc15"]
    arms: dict[str, dict[str, dict]] = {}
    for a in ARMS:
        p = args.arms / f"{a}_dev256.jsonl"
        if p.is_file():
            d = load_jsonl(p)
            if all(i in d for i in ids):
                arms[a] = d
    print(f"rows={len(ids)}  arms available: {', '.join(arms)}")
    if "supa_a1" not in arms:
        print("supa_a1 missing; cannot proceed")
        return 2

    def correct(pred: str, rid: str) -> bool:
        acc = {ev.normalize(v) for v in key[rid]["accepted_answers"]}
        return ev.normalize(pred) in acc

    probe_fires = fit_probe_oof(signals, key, ids)

    def score(name: str, passes: float, chooser: Callable[[str], str]) -> dict[str, Any]:
        ok = {i: correct(chooser(i), i) for i in ids}
        per = {}
        for v in VARIANTS:
            sub = [i for i in ids if key[i]["variant"] == v]
            per[v] = f"{sum(ok[i] for i in sub)}/{len(sub)}"
        return {"policy": name, "passes": passes,
                "correct": sum(ok.values()), "n": len(ids), "per_variant": per,
                "eligible": passes <= 6}

    def probe_gate(base_arm: str):
        def f(i: str) -> str:
            if probe_fires[i]:
                return "UNKNOWN"
            return arms[base_arm][i]["prediction"]
        return f

    def unknown_gate(primary: str, fallback: str, use_probe: bool):
        """If the fallback arm abstains, adopt its abstention (family op)."""
        def f(i: str) -> str:
            if use_probe and probe_fires[i]:
                return "UNKNOWN"
            fb = arms[fallback][i]["prediction"]
            if is_unknown(fb, ev):
                return fb
            return arms[primary][i]["prediction"]
        return f

    def consensus(primary: str, left: str, right: str, use_probe: bool):
        """If two other arms agree on a non-abstention answer, adopt it."""
        def f(i: str) -> str:
            if use_probe and probe_fires[i]:
                return "UNKNOWN"
            cur = arms[primary][i]["prediction"]
            l, r = arms[left][i]["prediction"], arms[right][i]["prediction"]
            ln = ev.normalize(l)
            if ln and not is_unknown(l, ev) and ln == ev.normalize(r):
                return l
            return cur
        return f

    results: list[dict[str, Any]] = []
    results.append(score("supa_a1 alone", 1.0, lambda i: arms["supa_a1"][i]["prediction"]))
    results.append(score("supa_a1 + probe gate", 1.0, probe_gate("supa_a1")))
    for other in [a for a in arms if a != "supa_a1"]:
        results.append(score(f"supa_a1 + probe + unknown-gate({other})", 2.0,
                             unknown_gate("supa_a1", other, True)))
    pairs = [(l, r) for l in arms for r in arms if l < r and "supa_a1" not in (l, r)]
    for l, r in pairs[:4]:
        results.append(score(f"supa_a1 + probe + consensus({l},{r})", 3.0,
                             consensus("supa_a1", l, r, True)))

    # Oracle union across available arms, with the probe allowed to abstain.
    def oracle(i: str) -> str:
        if not key[i]["answerable"]:
            return "UNKNOWN" if (probe_fires[i] or any(
                is_unknown(arms[a][i]["prediction"], ev) for a in arms)) else arms["supa_a1"][i]["prediction"]
        for a in arms:
            if correct(arms[a][i]["prediction"], i):
                return arms[a][i]["prediction"]
        return arms["supa_a1"][i]["prediction"]
    results.append(score("ORACLE union (upper bound, not a policy)", float(len(arms)), oracle))

    results.sort(key=lambda r: (-r["correct"], r["passes"]))
    print()
    print(f"{'policy':46s} {'psx':>4s} {'correct':>9s} {'elig':>5s}  per-variant")
    print("-" * 116)
    for r in results:
        pv = " ".join(f"{k[:4]}={v}" for k, v in r["per_variant"].items())
        print(f"{r['policy'][:46]:46s} {r['passes']:>4.0f} {r['correct']:>4d}/{r['n']:<4d} "
              f"{'yes' if r['eligible'] else 'NO':>5s}  {pv}")

    best = next(r for r in results if r["eligible"] and "ORACLE" not in r["policy"])
    print()
    print(f"BEST ELIGIBLE on DEV: {best['policy']}  ({best['correct']}/{best['n']}, "
          f"{best['passes']:.0f} pass)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"rows": len(ids), "arms": sorted(arms), "results": results,
             "best_eligible": best}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
