"""HLCC-001 R2 (cold review F3): can the fit canary tell a METHOD VARIANT from the registered method?

On the tiny dry teacher (CPU, the queue's dry study and dry LC_ARMS), refit the 18-block analogue with
the registered method and with real variants -- ridge_rel 2e-3 and 5e-4 (the ridge doubled/halved),
and position 0 INCLUDED (the registered fit drops it) -- and compare each against the recorded dry
residuals at the canary tolerance. The registered method must reproduce (difference 0 on CPU) and every
variant must trip. The position-0 variant is a copy of fit_compensation with that one line removed
(the anchor is asserted to occur exactly once). Loads only the tiny random model.
usage: python design/hlcc_canary_variants.py --tiny <dir> --panels <LAYER_PANELS.jsonl> --study <dry study>
       --lc <dry LC_ARMS.json> [--max-rows-per-category 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import layer_ablation as la  # noqa: E402
import linear_compensation as lc  # noqa: E402
import run_hlcc  # noqa: E402
import run_layer_study as rls  # noqa: E402

ANCHOR = "            sel[:, 0] = False                                   # registered: drop position 0\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path)
    ap.add_argument("--study", required=True, type=Path)
    ap.add_argument("--lc", required=True, type=Path)
    ap.add_argument("--max-rows-per-category", type=int, default=3)
    a = ap.parse_args(argv)
    import tokenizers
    la.configure_runtime(17, "cpu")
    rows = [json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep, seen = [], {}
    for r in rows:
        k = (r["category"], r["split"])
        if seen.get(k, 0) < a.max_rows_per_category:
            keep.append(r)
            seen[k] = seen.get(k, 0) + 1
    sel = [r for r in keep if r["split"] == "select"]
    S1 = json.loads((a.study / "S1_teacher_outputs.json").read_text(encoding="utf-8"))["rows"]
    tok = tokenizers.Tokenizer.from_file(str(a.tiny / "tokenizer.json"))
    im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
    model = la.load_model(a.tiny, "cpu")
    ref = json.loads(a.lc.read_text(encoding="utf-8"))["arms"]["selected"]
    kept = ref["kept"]
    items = rls.build_items(sel, tok, S1, im_end)

    src = Path(lc.__file__).read_text(encoding="utf-8")
    if src.count(ANCHOR) != 1:
        raise SystemExit("position-0 anchor not found exactly once in linear_compensation.py")
    pos0 = types.ModuleType("lc_pos0")
    pos0.__file__ = lc.__file__
    exec(compile(src.replace(ANCHOR, ""), "lc_pos0", "exec"), pos0.__dict__)

    variants = {"registered": lambda: lc.fit_compensation(model, kept, items, eot, "cpu"),
                "ridge_rel_2e-3": lambda: lc.fit_compensation(model, kept, items, eot, "cpu", ridge_rel=2e-3),
                "ridge_rel_5e-4": lambda: lc.fit_compensation(model, kept, items, eot, "cpu", ridge_rel=5e-4),
                "position_0_included": lambda: pos0.fit_compensation(model, kept, items, eot, "cpu")}
    want = [(f["p"], f["fit_residual_rel_fitted"]) for f in ref["fits"] if f.get("compensated")]
    out = {}
    for name, fn in variants.items():
        _, fits = fn()
        got = [(f["p"], f["fit_residual_rel_fitted"]) for f in fits if f.get("compensated")]
        diff = max(abs(x[1] - y[1]) for x, y in zip(want, got))
        out[name] = {"max_abs_diff": diff, "canary_trips": not diff <= run_hlcc.CANARY_TOL}
    ok = (not out["registered"]["canary_trips"]) and all(v["canary_trips"] for k, v in out.items() if k != "registered")
    print(json.dumps({"tolerance": run_hlcc.CANARY_TOL, "variants": out,
                      "canary_discriminates": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
