"""SMALLVD-001 Amendment 2 -- exploratory arm H-LC: linear compensation of the dropped blocks.

Runs AFTER the layer study and its report, on the layer sets those already fixed (no new selection):
  LC@selected   the registered 18-block set (A2's set), plus the fitted folds
  LC@angular    the 18-block angular_consecutive set (one contiguous run -- the shape the method was
                designed for), plus its folds
For each: the PLAIN (bypass) student is recomputed in this process as the paired comparator; an
IDENTITY-fold control must reproduce the plain student bitwise or the run refuses; folds are fitted
on SELECT only; the compensated student is scored on SELECT2 and REPORT (teacher-forced KL, greedy
generation, answer bpb under V0 = A2-LC and under V2 = A3-LC).

Verdict (registered in SMALLVD001_PREREG.md Amendment 2 before any fit), on REPORT, LC@selected vs
plain@selected: r = 1 - obj(LC) / obj(plain), obj = the §4 category-mean answer KL; 95 % interval by
a category-stratified paired bootstrap over REPORT rows (2000 resamples, seed 17).
  SUPPORTED        r >= 0.20 and the interval's lower bound > 0
  REAL_BUT_SMALL   0 < r < 0.20 and the lower bound > 0
  HARMFUL          the interval's upper bound < 0
  INCONCLUSIVE     otherwise
  UNDERPOWERED     any category has fewer than 5 REPORT rows (the stratified bootstrap degenerates:
                   the tiny dry run, with 1 row per category, produced a zero-width interval)
Everything else is descriptive. Nothing here changes the registered A2/A3 or the §10 decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import gpu_lease  # noqa: E402
import hostbudget  # noqa: E402
import layer_ablation as la  # noqa: E402
import layer_select as ls  # noqa: E402
import lc_verdict as lv  # noqa: E402
import linear_compensation as lc  # noqa: E402
import param_accounting as pa  # noqa: E402
import run_layer_study as rls  # noqa: E402
import virtual_student as vs  # noqa: E402

# the verdict lives in lc_verdict.py (pure Python, so its power was simulated before any fit)
verdict = lv.verdict


def spearman(xs, ys):
    import report_layer_study as rep
    v = rep.spearman(xs, ys)
    return None if v != v else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path)
    ap.add_argument("--study", required=True, type=Path, help="the layer-study directory (S1, S3, S5)")
    ap.add_argument("--layer-report", required=True, type=Path)
    ap.add_argument("--vocab-dir", required=True, type=Path, help="the A3 vocabulary (V2)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--select-depth", type=int, default=18)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows-per-category", type=int, default=0, help="must equal the study's (dry runs)")
    ap.add_argument("--dry-run", action="store_true", help="TINY random teacher (< 50M params, asserted), CPU, no lease")
    a = ap.parse_args(argv)
    cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
    n0 = pa.analytic(cfg, cfg["vocab_size"], cfg["num_hidden_layers"])["unique_physical_parameters"]
    if a.dry_run:
        if n0 >= 50_000_000 or a.device != "cpu":
            raise SystemExit(f"--dry-run is only for tiny CPU models (got {n0:,} params on {a.device})")
        hostbudget.require(0.8, "SMALLVD-001 LC arms DRY RUN")
        token, guard_gib, chunk = None, 0.4, 128
    else:
        hostbudget.require(3.6, "SMALLVD-001 LC arms")
        token, guard_gib, chunk = gpu_lease.acquire(timeout_s=7200, label="smallvd001:lc-arms"), 1.5, 256
    if a.out.exists():                 # checked up front; CREATED only at the end, so a RAM stand-down
        raise SystemExit(f"{a.out} exists (create-only)")   # mid-run leaves nothing to block a retry
    try:
        import tokenizers
        import torch
        rt = la.configure_runtime(17, a.device)
        dev = a.device
        S1 = json.loads((a.study / "S1_teacher_outputs.json").read_text(encoding="utf-8"))
        S3 = json.loads((a.study / "S3_single_ablation.json").read_text(encoding="utf-8"))
        S5 = json.loads((a.study / "S5_depth_strategies.json").read_text(encoding="utf-8"))
        rep = json.loads(a.layer_report.read_text(encoding="utf-8"))
        T = S1["rows"]

        rows = [json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.max_rows_per_category:                         # the study's own subsampling, verbatim
            keep, seen = [], {}
            for r in rows:
                k = (r["category"], r["split"])
                if seen.get(k, 0) < a.max_rows_per_category:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            rows = keep
        sel = [r for r in rows if r["split"] == "select"]
        hold = [r for r in rows if r["split"] == "holdout"]
        if {r["id"] for r in hold} != {x["id"] for x in S3["layers"]["0"]["holdout_per_item"]}:
            raise SystemExit("HOLDOUT rows differ from the study's: pass the study's --max-rows-per-category")
        sub = ls.holdout_subsplit(hold)
        split_sha = hashlib.sha256(json.dumps(sorted(sub.items())).encode("utf-8")).hexdigest()
        for name, st in (("S1", S1), ("S5", S5)):
            if st.get("_holdout_subsplit_sha256") != split_sha:
                raise SystemExit(f"{name} was computed under a different HOLDOUT subsplit")

        tok = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))
        im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
        eos = [im_end, eot]
        model = la.load_model(a.teacher, dev)
        if next(model.parameters()).dtype != torch.float32:        # asserted, not a default argument
            raise SystemExit("H-LC requires an fp32 model: a small T - I can round away in bf16")
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        L = model.config.num_hidden_layers
        V0 = vs.Vocab(None, a.teacher / "tokenizer.json")
        V2 = vs.Vocab(a.vocab_dir, a.teacher / "tokenizer.json")
        items_sel = rls.build_items(sel, tok, T, im_end)
        items_hold = rls.build_items(hold, tok, T, im_end)
        chat_hold = [r for r in hold if r["kind"] == "chat"]
        ref_hold = [r for r in chat_hold if r.get("reference")]
        H_hold = la.collect_positions(model, items_hold, eot, dev)

        def parts(per_item, summ):
            out = {"holdout": {k: v for k, v in summ(per_item).items() if k != "per_item"}, "per_item": per_item}
            for part in ("select2", "report"):
                out[part] = {k: v for k, v in summ([x for x in per_item if sub[x["id"]] == part]).items()
                             if k != "per_item"}
            return out

        def student_H(kept):
            with la.keep_layers(model, kept):
                return la.collect_positions(model, items_hold, eot, dev)

        def gen(kept):
            g = vs.generate_rows(model, V0, chat_hold, kept, eos, eot, dev, a.gen_tokens)
            texts, ntok = [x["text"] for x in g], [x["new_tokens"] for x in g]
            ref = [T[r["id"]]["text"] for r in chat_hold]
            out = {}
            for part in ("select2", "report"):
                ix = [i for i, r in enumerate(chat_hold) if sub[r["id"]] == part]
                out[part] = la.generation_metrics([texts[i] for i in ix], [ref[i] for i in ix],
                                                  [chat_hold[i] for i in ix], [ntok[i] for i in ix], a.gen_tokens)
            out["per_row"] = [{"id": r["id"], "text": texts[i]} for i, r in enumerate(chat_hold)]
            return out

        def score(kept, H):
            tf = parts(la.compare(model.lm_head, H_hold, H, items_hold, chunk=chunk)["per_item"], la.summarise)
            return {"kl": tf, "generation": gen(kept),
                    "answer_bpb_V0": parts(vs.answer_bpb(model, V0, ref_hold, kept, im_end, dev, chunk=min(chunk, 256))["per_item"],
                                           vs.bpb_summary),
                    "answer_bpb_V2": parts(vs.answer_bpb(model, V2, ref_hold, kept, im_end, dev, chunk=min(chunk, 256))["per_item"],
                                           vs.bpb_summary)}

        sets = {"selected": rep["selected_layer_set"]["kept"],
                "angular_consecutive": S5["depths"][str(a.select_depth)]["angular_consecutive"]["kept"]}
        result = {"schema": 1, "record": "SMALLVD_LC_ARMS", "experiment": "SMALLVD-001", "amendment": 2,
                  "teacher_weight_sha256": rls.sha256_file(a.teacher / "model.safetensors"),
                  "holdout_subsplit_sha256": split_sha, "ridge_rel": lc.RIDGE_REL, "arms": {}}
        for name, kept in sets.items():
            t0 = time.time()
            H_plain = student_H(kept)
            ident = {r["p"]: torch.eye(model.config.hidden_size, device=dev)
                     for r in lc.runs(kept, L) if r["p"] is not None}
            with lc.folded(model, ident):
                H_id = student_H(kept)
            if not all(torch.equal(x, y) for x, y in zip(H_plain, H_id)):
                raise SystemExit(f"{name}: the IDENTITY fold does not reproduce the plain student bitwise")
            del H_id
            plain = score(kept, H_plain)
            del H_plain
            hostbudget.guard(guard_gib, f"LC {name} plain")
            folds, fits = lc.fit_compensation(model, kept, items_sel, eot, dev)
            hostbudget.guard(guard_gib, f"LC {name} fit")
            with lc.folded(model, folds):
                comp = score(kept, student_H(kept))
            hostbudget.guard(guard_gib, f"LC {name} scored")
            # review (a): is the L2 fit a proxy for KL? per run, the fit-residual drop against the
            # SELECT2 objective change when THAT fold is added to the ones before it (descriptive)
            s2 = lambda H: rls.category_mean_kl(la.summarise(
                [x for x in la.compare(model.lm_head, H_hold, H, items_hold, chunk=chunk)["per_item"]
                 if sub[x["id"]] == "select2"]))
            order = list(folds)
            objs = [rls.category_mean_kl(plain["kl"]["select2"])]
            for k in range(1, len(order)):
                with lc.folded(model, {p: folds[p] for p in order[:k]}):
                    objs.append(s2(student_H(kept)))
            objs.append(rls.category_mean_kl(comp["kl"]["select2"]))
            fitted = [f for f in fits if f.get("compensated")]
            proxy = [{"p": f["p"], "fit_residual_drop": f["fit_residual_rel_identity"] - f["fit_residual_rel_fitted"],
                      "select2_delta_obj": objs[k + 1] - objs[k]} for k, f in enumerate(fitted)]
            s5 = next(v for v in S5["depths"][str(a.select_depth)].values() if v["kept"] == kept)
            result["arms"][name] = {
                "kept": kept, "runs": lc.runs(kept, L), "fits": fits, "proxy_per_run": proxy,
                "identity_control": "bitwise equal to plain",
                "plain": plain, "lc": comp,
                "reproducibility_vs_S5": {"S5_report_objective": s5["report_objective"],
                                          "this_process_report_objective": rls.category_mean_kl(plain["kl"]["report"])},
                "select2_objective": {"plain": rls.category_mean_kl(plain["kl"]["select2"]),
                                      "lc": rls.category_mean_kl(comp["kl"]["select2"])},
                "seconds": round(time.time() - t0, 1)}
            print(f"[LC] {name}: REPORT obj plain {rls.category_mean_kl(plain['kl']['report']):.4f} -> "
                  f"LC {rls.category_mean_kl(comp['kl']['report']):.4f}; fits "
                  f"{[round(f.get('fit_residual_rel_fitted', float('nan')), 4) for f in fits]}", flush=True)
        pooled = [x for arm_ in result["arms"].values() for x in arm_["proxy_per_run"]]
        result["proxy_L2_vs_KL"] = {"runs": len(pooled),
                                    "spearman_fit_drop_vs_select2_delta": spearman([x["fit_residual_drop"] for x in pooled],
                                                                                   [x["select2_delta_obj"] for x in pooled])
                                    if len(pooled) >= 3 else None,
                                    "reading": "negative = a bigger L2 fit gain goes with a bigger KL reduction (a good proxy)"}
        report_ids = sorted(i for i, p in sub.items() if p == "report")
        arm = result["arms"]["selected"]
        # Amendment 5: the registered verdict is the CALIBRATED one; the raw fields stay in the record
        result["H_LC"] = lv.calibrated(verdict(arm["plain"]["kl"]["per_item"], arm["lc"]["kl"]["per_item"], report_ids))
        result["follow_up_choice_by_SELECT2"] = min(result["arms"], key=lambda k: result["arms"][k]["select2_objective"]["lc"])
        result["runtime"] = {"flags": rt, "python": sys.version.split()[0], "torch": torch.__version__,
                             "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                             "code_sha256": {p.name: rls.sha256_file(p) for p in sorted(HERE.glob("*.py"))}}
        result["quality_claim"], result["release_permitted"] = False, False
        a.out.mkdir(parents=True, exist_ok=False)
        la.dump(a.out / "LC_ARMS.json", result)
        print(f"[H-LC] {result['H_LC']['verdict']}: r = {result['H_LC']['relative_reduction_r']:+.3f} "
              f"95% [{result['H_LC']['bootstrap_95'][0]:+.3f}, {result['H_LC']['bootstrap_95'][1]:+.3f}]", flush=True)
        return 0
    finally:
        if token:
            gpu_lease.release(token)


if __name__ == "__main__":
    raise SystemExit(main())
