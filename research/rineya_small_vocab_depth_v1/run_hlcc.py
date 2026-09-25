"""HLCC-001 -- CONFIRM linear compensation on a FRESH panel (registered in HLCC001_PREREG.md).

  PRIMARY    the recommended student's depth set (SMALLVD-001 §5: V4_32768/20, removed
             {5,6,8,12,13,14,23,24}); KL per row (teacher vocabulary) and answer bpb per row under the
             V4 vocabulary; exact sign tests (hlcc_verdict)
  SECONDARY  the registered 18-block set (A2/A3's), same measures -- a replication, descriptive

The compensation is fitted EXACTLY as in SMALLVD-001 (calibration SELECT rows, the S1 teacher outputs,
ridge toward I at 1e-3 x median eigenvalue, position 0 excluded). Refusal controls: the identity fold
must reproduce the plain student bitwise, and the 18-block fits must reproduce SMALLVD-001's recorded
fit residuals (the same method, not a variant). The fresh panel's teacher continuations are generated
here (greedy, same settings as S1). Output created only at the end (create-only).

R2 (cold review): the REGISTERED verdict is ROW-level (pool-conditional H0, hlcc_verdict docstring); the
all-axis majority-sign unit test is the registered sensitivity analysis; the 267cec9b unit rule is
descriptive. PRIMARY is measured first. Any failure after the panel is read writes a create-only
attempts/ATTEMPT_*.json record (reason, traceback, every per-row value measured so far) before exiting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import gpu_lease  # noqa: E402
import hlcc_verdict as hv  # noqa: E402
import hostbudget  # noqa: E402
import layer_ablation as la  # noqa: E402
import lc_verdict as lv  # noqa: E402
import linear_compensation as lc  # noqa: E402
import param_accounting as pa  # noqa: E402
import run_layer_study as rls  # noqa: E402
import virtual_student as vs  # noqa: E402


CANARY_TOL = 1e-6    # on relative residuals of 0.2-0.7; the same batches in the same order reproduce bitwise


def parse_kept(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path, help="SMALLVD-001 LAYER_PANELS.jsonl (fit set = SELECT)")
    ap.add_argument("--study", required=True, type=Path, help="SMALLVD-001 layer_study dir (S1 teacher outputs)")
    ap.add_argument("--confirm", required=True, type=Path, help="CONFIRM_PANEL.jsonl (fresh)")
    ap.add_argument("--vocab-primary", required=True, type=Path, help="the recommended student's vocabulary (V4)")
    ap.add_argument("--vocab-secondary", required=True, type=Path, help="the registered A3 vocabulary (V2)")
    ap.add_argument("--primary-kept", required=True, type=parse_kept)
    ap.add_argument("--secondary-kept", required=True, type=parse_kept)
    ap.add_argument("--smallvd-lc", required=True, type=Path,
                    help="SMALLVD-001 LC_ARMS.json: the secondary set's fits must reproduce its recorded residuals")
    ap.add_argument("--allaxis-map", required=True, type=Path,
                    help="HLCC_ALLAXIS_UNIT_MAP.json (pinned): units of the registered sensitivity analysis")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows-per-category", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (create-only)")
    cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
    n0 = pa.analytic(cfg, cfg["vocab_size"], cfg["num_hidden_layers"])["unique_physical_parameters"]
    if a.dry_run:
        if n0 >= 50_000_000 or a.device != "cpu":
            raise SystemExit("--dry-run is only for tiny CPU models")
        hostbudget.require(0.8, "HLCC-001 DRY RUN")
        token, guard_gib, chunk = None, 0.4, 128
    else:
        hostbudget.require(3.6, "HLCC-001")
        token, guard_gib, chunk = gpu_lease.acquire(timeout_s=7200, label="hlcc001"), 1.5, 256
    panel_read, partial = False, {}
    try:
        import tokenizers
        import torch
        rt = la.configure_runtime(17, a.device)
        dev = a.device

        def subsample(rows):
            if not a.max_rows_per_category:
                return rows
            keep, seen = [], {}
            for r in rows:
                k = (r["category"], r["split"])
                if seen.get(k, 0) < a.max_rows_per_category:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            return keep
        panels = subsample([json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()])
        confirm = subsample([json.loads(l) for l in a.confirm.read_text(encoding="utf-8").splitlines() if l.strip()])
        if {r["id"] for r in confirm} & {r["id"] for r in panels}:
            raise SystemExit("confirm rows overlap the SMALLVD-001 panels")
        if any(r["split"] != "confirm" for r in confirm):
            raise SystemExit("the confirm file holds rows whose split is not 'confirm'")
        unit_by_id = {r["id"]: hv.unit_of(r) for r in confirm}
        amap = json.loads(a.allaxis_map.read_text(encoding="utf-8"))
        if amap["panel_sha256"] != rls.sha256_file(a.confirm):
            raise SystemExit("the all-axis unit map was built for a different panel")
        comp_kl, comp_bpb = amap["maps"]["kl_anchor"], amap["maps"]["bpb_anchor"]
        chat_c = [r for r in confirm if r["kind"] == "chat"]
        ref_c = [r for r in chat_c if r.get("reference")]
        missing = [r["id"] for r in confirm if r["id"] not in comp_kl] + [r["id"] for r in ref_c if r["id"] not in comp_bpb]
        if missing:
            raise SystemExit(f"{len(missing)} panel rows have no all-axis unit, e.g. {missing[:3]}")
        canary_ref = json.loads(a.smallvd_lc.read_text(encoding="utf-8"))["arms"]["selected"]
        if a.secondary_kept != canary_ref["kept"]:          # never skipped: a skipped canary passes vacuously
            raise SystemExit(f"--secondary-kept {a.secondary_kept} is not SMALLVD-001's LC@selected set {canary_ref['kept']}")
        S1 = json.loads((a.study / "S1_teacher_outputs.json").read_text(encoding="utf-8"))["rows"]
        sel = [r for r in panels if r["split"] == "select"]

        tok = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))
        im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
        eos = [im_end, eot]
        model = la.load_model(a.teacher, dev)
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("HLCC requires an fp32 model")
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        L = model.config.num_hidden_layers
        full = list(range(L))
        V0 = vs.Vocab(None, a.teacher / "tokenizer.json")
        Vp = vs.Vocab(a.vocab_primary, a.teacher / "tokenizer.json")
        Vs = vs.Vocab(a.vocab_secondary, a.teacher / "tokenizer.json")

        # EVERY refusal control runs BEFORE the real teacher touches the confirmation panel (cold review,
        # Rineya Delta): a refusal after the panel was read would spend its read budget and record nothing.
        # The controls need only SELECT rows: the identity fold is checked on them, and the fits use them.
        items_sel = rls.build_items(sel, tok, S1, im_end)
        sets = (("secondary", a.secondary_kept, "secondary_vocab"), ("primary", a.primary_kept, "primary_vocab"))
        prepared = {}
        for role, kept, _ in sets:            # the canary set FIRST: a failed canary costs one fit
            ident = {r["p"]: torch.eye(model.config.hidden_size, device=dev) for r in lc.runs(kept, L) if r["p"] is not None}
            with la.keep_layers(model, kept):
                Hp = la.collect_positions(model, items_sel, eot, dev)
            with lc.folded(model, ident), la.keep_layers(model, kept):
                Hi = la.collect_positions(model, items_sel, eot, dev)
            if not all(torch.equal(x, y) for x, y in zip(Hp, Hi)):
                raise SystemExit(f"{role}: the identity fold does not reproduce the plain student bitwise")
            del Hp, Hi
            folds, fits = lc.fit_compensation(model, kept, items_sel, eot, dev)
            hostbudget.guard(guard_gib, f"HLCC {role} fit")
            if role == "secondary":
                ref = [(f["p"], f["fit_residual_rel_fitted"]) for f in canary_ref["fits"] if f.get("compensated")]
                got = [(f["p"], f["fit_residual_rel_fitted"]) for f in fits if f.get("compensated")]
                diff = max((abs(x[1] - y[1]) for x, y in zip(ref, got)), default=float("inf"))
                if [x[0] for x in ref] != [y[0] for y in got] or not diff <= CANARY_TOL:
                    raise SystemExit(f"FIT CANARY FAILED: SMALLVD-001 {ref} vs now {got} -- not the same method")
                canary = {"smallvd001": ref, "now": got, "max_abs_diff": diff, "tolerance": CANARY_TOL,
                          "status": "reproduced"}
            prepared[role] = (folds, fits)
        # answer_bpb asserts every reference target is in the candidate vocabulary: check it on TEXT here,
        # before any panel forward pass, instead of refusing after the read (cold review F4)
        for nm, voc in (("primary", Vp), ("secondary", Vs)):
            if voc.retained is None:
                continue
            keep = set(voc.retained.tolist())
            bad = [r["id"] for r in ref_c if not set(voc.encode_old_ids(r["reference"]) + [im_end]) <= keep]
            if bad:
                raise SystemExit(f"{nm} vocabulary: {len(bad)} reference answers leave the candidate vocabulary, e.g. {bad[:3]}")
        panel_read = True
        print("[HLCC] refusal controls passed (identity x2, fit canary, vocabulary); the confirmation panel is read now", flush=True)

        # fresh teacher continuations (same settings as SMALLVD-001 S1)
        t0 = time.time()
        Tc = {g["id"]: g for g in vs.generate_rows(model, V0, chat_c, full, eos, eot, dev, a.gen_tokens)}
        hostbudget.guard(guard_gib, "HLCC teacher outputs")
        items_c = rls.build_items(confirm, tok, Tc, im_end)
        H_c = la.collect_positions(model, items_c, eot, dev)

        def student_H(kept):
            with la.keep_layers(model, kept):
                return la.collect_positions(model, items_c, eot, dev)

        def measure(kept):
            kl = la.compare(model.lm_head, H_c, student_H(kept), items_c, chunk=chunk)["per_item"]
            bp = {nm: vs.answer_bpb(model, voc, ref_c, kept, im_end, dev, chunk=min(chunk, 256))["per_item"]
                  for nm, voc in (("V0", V0), ("primary_vocab", Vp), ("secondary_vocab", Vs))}
            g = vs.generate_rows(model, V0, chat_c, kept, eos, eot, dev, a.gen_tokens)
            flips = {x["id"]: la.normalise_answer(x["text"]) != la.normalise_answer(Tc[x["id"]]["text"]) for x in g}
            return {"kl": kl, "bpb": bp, "flip": flips}

        result = {"schema": 1, "record": "HLCC_RESULT", "experiment": "HLCC-001",
                  "teacher_weight_sha256": rls.sha256_file(a.teacher / "model.safetensors"),
                  "confirm_panel_sha256": rls.sha256_file(a.confirm), "panels_sha256": rls.sha256_file(a.panels),
                  "teacher_outputs_seconds": round(time.time() - t0, 1), "fit_canary": canary,
                  "identity_control_rows": "SELECT (before the panel was read)",
                  "allaxis_map_sha256": rls.sha256_file(a.allaxis_map), "sets": {}}
        # PRIMARY first: a failure during the replication then never exposes the registered limb early
        for role, kept, pvocab in [s for s in sets if s[0] == "primary"] + [s for s in sets if s[0] != "primary"]:
            folds, fits = prepared[role]
            plain = measure(kept)
            partial[role] = {"plain": {"kl": plain["kl"], "bpb": plain["bpb"]}}
            hostbudget.guard(guard_gib, f"HLCC {role} plain")
            with lc.folded(model, folds):
                comp = measure(kept)
            partial[role]["lc"] = {"kl": comp["kl"], "bpb": comp["bpb"]}
            hostbudget.guard(guard_gib, f"HLCC {role} compensated")
            P, C = {x["id"]: x for x in plain["kl"]}, {x["id"]: x for x in comp["kl"]}
            bad = [i for i in P if not (P[i]["kl"] > 0 and C[i]["kl"] > 0)]
            if bad:                                   # log(plain/lc) is undefined; refuse rather than clip
                raise SystemExit(f"{role}: non-positive KL on {len(bad)} rows, e.g. {bad[:3]}")
            d_kl = [math.log(P[i]["kl"] / C[i]["kl"]) for i in P]
            Pb, Cb = ({x["id"]: x for x in m["bpb"][pvocab]} for m in (plain, comp))
            d_bpb = [math.log(Pb[i]["nats"] / Cb[i]["nats"]) for i in Pb]
            dk_by = {i: math.log(P[i]["kl"] / C[i]["kl"]) for i in P}
            db_by = {i: math.log(Pb[i]["nats"] / Cb[i]["nats"]) for i in Pb}
            sk, sb = hv.majority_sign_collapse(dk_by, comp_kl), hv.majority_sign_collapse(db_by, comp_bpb)
            uk, ub = hv.collapse(dk_by, unit_by_id), hv.collapse(db_by, unit_by_id)
            cats = {}
            for i in P:
                c = cats.setdefault(P[i]["category"], [0, 0])
                c[0] += P[i]["kl"] > C[i]["kl"]
                c[1] += 1
            lk = [math.log(max(P[i]["kl"], 1e-12)) for i in P]
            result["sets"][role] = {
                "kept": kept, "runs": lc.runs(kept, L), "fits": fits, "identity_control": "bitwise equal to plain",
                "VERDICT": hv.verdict(d_kl, d_bpb),                  # R2: row-level, pool-conditional H0
                "sensitivity_allaxis_majority_sign": {**hv.verdict(list(sk.values()), list(sb.values()), with_size=False),
                                                      "units": {"kl": len(sk), "bpb": len(sb)}},
                "descriptive_unit_rule_267cec9b": {**hv.verdict(list(uk.values()), list(ub.values())),
                                                   "units": {"kl": len(uk), "bpb": len(ub)}},
                "descriptive": {
                    "category_mean_kl": {"plain": lv.cat_mean_kl(plain["kl"]), "lc": lv.cat_mean_kl(comp["kl"])},
                    "kl_better_by_category": {c: f"{v[0]}/{v[1]}" for c, v in sorted(cats.items())},
                    "answer_bpb": {nm: {"plain": sum(x["nats"] for x in plain["bpb"][nm]) / math.log(2) / sum(x["bytes"] for x in plain["bpb"][nm]),
                                        "lc": sum(x["nats"] for x in comp["bpb"][nm]) / math.log(2) / sum(x["bytes"] for x in comp["bpb"][nm])}
                                   for nm in ("V0", "primary_vocab", "secondary_vocab")},
                    "flip_rate": {"plain": sum(plain["flip"].values()) / len(plain["flip"]),
                                  "lc": sum(comp["flip"].values()) / len(comp["flip"])},
                    # of log(LC/plain), exactly as SMALLVD-001's A5.8 check; corr < 0 = the rows the plain
                    # student gets most wrong gain the most
                    "shape_of_log_lc_over_plain": lv.shape_stats([-x for x in d_kl], lk)},
                "per_row": {"d_kl": {i: math.log(P[i]["kl"] / C[i]["kl"]) for i in P},
                            "d_bpb": {i: math.log(Pb[i]["nats"] / Cb[i]["nats"]) for i in Pb}},
                "per_unit_267cec9b": {"d_kl": uk, "d_bpb": ub},
                "per_allaxis_unit_sign": {"kl": sk, "bpb": sb}}
            print(f"[HLCC] {role}: {result['sets'][role]['VERDICT']['verdict']} | KL better "
                  f"{result['sets'][role]['VERDICT']['kl']['positive']}/{result['sets'][role]['VERDICT']['kl']['n']} | bpb better "
                  f"{result['sets'][role]['VERDICT']['bpb']['positive']}/{result['sets'][role]['VERDICT']['bpb']['n']}", flush=True)
        result["REGISTERED_VERDICT"] = result["sets"]["primary"]["VERDICT"]["verdict"]
        result["runtime"] = {"flags": rt, "python": sys.version.split()[0], "torch": torch.__version__,
                             "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                             "code_sha256": {p.name: rls.sha256_file(p) for p in sorted(HERE.glob("*.py"))}}
        result["quality_claim"], result["release_permitted"] = False, False
        a.out.mkdir(parents=True, exist_ok=False)
        la.dump(a.out / "HLCC_RESULT.json", result)
        print(f"[HLCC] REGISTERED VERDICT (primary): {result['REGISTERED_VERDICT']}", flush=True)
        return 0
    except BaseException as e:
        if panel_read:        # the read is spent: record the attempt before exiting (cold review F4)
            import traceback
            infra = (isinstance(e, (hostbudget.LowMemoryAbort, KeyboardInterrupt)) or "CUDA" in str(e)
                     or type(e).__name__ == "OutOfMemoryError")
            rec = {"schema": 1, "record": "HLCC_ATTEMPT", "experiment": "HLCC-001",
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                   "exception_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(),
                   "classification_hint": "infrastructure (rerun permitted, HLCC001_PREREG.md 9)" if infra
                   else "deterministic or unknown (no verdict; read spent; HLCC001_PREREG.md 9)",
                   "partial_per_row": partial,
                   "code_sha256": {q.name: rls.sha256_file(q) for q in sorted(HERE.glob("*.py"))}}
            d = a.out.parent / "attempts"
            d.mkdir(parents=True, exist_ok=True)
            fn = d / f"ATTEMPT_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json"
            with open(fn, "x", encoding="utf-8", newline="\n") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False, default=str)
            print(f"[HLCC] attempt record written: {fn}", flush=True)
        raise
    finally:
        if token:
            gpu_lease.release(token)


if __name__ == "__main__":
    raise SystemExit(main())
