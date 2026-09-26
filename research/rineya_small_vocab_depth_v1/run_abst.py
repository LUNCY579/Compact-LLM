"""ABST-001 -- does linear compensation lower the likelihood of the UNKNOWN reference on the recommended
V4_32768/20 student, and is that teacher-tracking or damage? (ABST001_PREREG.md)

Cells on the SAME fresh RTG1-dev rows (256 UNKNOWN + 256 known), one forward pass per row (B = 1):
  depth configs  S20 plain / S20 LC / S18 plain / S18 LC / teacher (28 blocks)
  input vocab    V0, V4, V2 -- each pass read out under every output support (V0, V4, V2)
  matched cells  A = S20/V4  B = S20/V2  C = S18/V4  D = S18/V2  E = S20/V0  F = S18/V0  T0 T4 T2
The compensation is SMALLVD-001's (fit on SELECT; no free parameter).

REFUSAL CONTROLS, all BEFORE the first forward pass of the real teacher on the panel:
  R1 data   panel/manifest hashes, strata, references, disjointness recomputed, reserve disjoint
  R2 vocab  UNKNOWN and 0-3 are single tokens with the same old id in V0/V4/V2; <|im_end|> retained;
            every reference + <|im_end|> inside V4 and V2 (checked on text)
  R3 identity fold x2 (S20, S18), bitwise, on SELECT
  R4 fit canary x2: S20 vs HLCC-001's recorded fits, S18 vs LC_ARMS's, <= 1e-6
  R5 measurement canary, tier 1 (EXACT, the recorded code path vs.answer_bpb on SMALLVD-001's spent
     HOLDOUT reference rows): A-plain vs S6 V4/20, D-plain vs S6 V2/18, T4 vs S6 V4/28,
     D-LC vs LC_ARMS lc V2, F-LC vs LC_ARMS lc V0 -- every per-row nats within 1e-6
  R5b tier 2 (the NEW B = 1 path on the same rows and cells): |nats - recorded| <= 1e-3 and the sign of
     d(D) agrees on every row with |d| > 1e-3
  R6 measured throughput: the projected panel time must be <= --runtime-cap-min
After the boundary: PRIMARY cells (A-plain, A-LC, T4) are measured FIRST and ABST_PRIMARY.json (steps 1-2)
is written create-only before any other cell; any failure writes attempts/ATTEMPT_*.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import abst_describe as ad  # noqa: E402
import abst_measure as am  # noqa: E402
import abst_verdict as av  # noqa: E402
import gpu_lease  # noqa: E402
import hostbudget  # noqa: E402
import layer_ablation as la  # noqa: E402
import linear_compensation as lc  # noqa: E402
import param_accounting as pa  # noqa: E402
import run_layer_study as rls  # noqa: E402
import virtual_student as vs  # noqa: E402

CANARY_TOL = 1e-6
TIER2_TOL = 1e-3
PRIMARY_ORDER = (("S20_plain", "V4"), ("S20_lc", "V4"), ("T", "V4"))
REST_ORDER = (("S20_plain", "V2"), ("S20_lc", "V2"), ("S18_plain", "V4"), ("S18_lc", "V4"),
              ("S18_plain", "V2"), ("S18_lc", "V2"), ("S20_plain", "V0"), ("S20_lc", "V0"),
              ("S18_plain", "V0"), ("S18_lc", "V0"), ("T", "V0"), ("T", "V2"))
GEN_CELLS = (("A_plain", "S20_plain", "V4"), ("A_lc", "S20_lc", "V4"), ("T4", "T", "V4"), ("T0", "T", "V0"))


def parse_kept(s):
    return [int(x) for x in s.split(",")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cells_for_verdict(records, ids, known_ids):
    g = lambda cfg, v, rows=ids: [records[i]["m"][cfg][v][v]["nats"] for i in rows]
    out = {"A": {"plain": g("S20_plain", "V4"), "lc": g("S20_lc", "V4")}, "T4": {"teacher": g("T", "V4")},
           "A_known": {"plain": g("S20_plain", "V4", known_ids), "lc": g("S20_lc", "V4", known_ids)},
           "T4_known": {"teacher": g("T", "V4", known_ids)}}
    have = lambda cfg, v: all(cfg in records[i]["m"] and v in records[i]["m"][cfg] for i in ids)
    for X, cfg, v in (("B", "S20", "V2"), ("C", "S18", "V4"), ("D", "S18", "V2")):
        if have(cfg + "_plain", v) and have(cfg + "_lc", v):
            out[X] = {"plain": g(cfg + "_plain", v), "lc": g(cfg + "_lc", v)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path, help="SMALLVD-001 LAYER_PANELS.jsonl (SELECT fit set, HOLDOUT canary rows)")
    ap.add_argument("--study", required=True, type=Path, help="SMALLVD-001 layer_study (S1 outputs, S6 recorded nats)")
    ap.add_argument("--smallvd-lc", required=True, type=Path, help="LC_ARMS.json (S18 fits + recorded LC nats)")
    ap.add_argument("--hlcc-result", required=True, type=Path, help="HLCC_RESULT.json (S20 recorded fits)")
    ap.add_argument("--abst-panel", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--confirm-panel", required=True, type=Path)
    ap.add_argument("--tokenizer-corpus", required=True, type=Path)
    ap.add_argument("--vocab-v4", required=True, type=Path)
    ap.add_argument("--vocab-v2", required=True, type=Path)
    ap.add_argument("--s20-kept", required=True, type=parse_kept)
    ap.add_argument("--s18-kept", required=True, type=parse_kept)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runtime-cap-min", type=float, default=60.0)
    ap.add_argument("--max-rows-per-stratum", type=int, default=0, help="dry runs only")
    ap.add_argument("--max-rows-per-category", type=int, default=0, help="must equal the study's (dry runs)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (create-only)")
    # READ-ONCE across launches (review): a durable BOUNDARY marker is written just before the panel is read.
    # One rerun is permitted only after an infrastructure failure (or a hard kill that left no record),
    # never after a deterministic failure, and never after a second crossing.
    att_dir = a.out.parent / "attempts"
    markers = sorted(att_dir.glob("BOUNDARY_*.json")) if att_dir.exists() else []
    done = sorted(a.out.parent.glob("*/ABST_PRIMARY.json")) if a.out.parent.exists() else []
    if done:                                   # a run already wrote its primary: the read is spent, whatever --out says
        raise SystemExit(f"READ BUDGET SPENT: a primary already exists at {done[0]}")
    if markers:
        prior = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(att_dir.glob("ATTEMPT_*.json"))]
        if len(markers) >= 2:
            raise SystemExit(f"READ BUDGET SPENT: {len(markers)} earlier boundary crossings")
        if any(p.get("classification_hint") != "infrastructure" for p in prior):
            raise SystemExit("READ BUDGET SPENT: the earlier crossing ended in a deterministic failure")
        print(f"[ABST] one earlier boundary crossing ({markers[0].name}) with no deterministic failure: "
              f"the single permitted rerun", flush=True)
    cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
    n0 = pa.analytic(cfg, cfg["vocab_size"], cfg["num_hidden_layers"])["unique_physical_parameters"]
    if a.dry_run:
        if n0 >= 50_000_000 or a.device != "cpu":
            raise SystemExit("--dry-run is only for tiny CPU models")
        hostbudget.require(0.8, "ABST-001 DRY RUN")
        token, guard_gib, chunk = None, 0.4, 128
    else:
        if a.max_rows_per_stratum or a.max_rows_per_category:
            raise SystemExit("subsampling is for dry runs only")
        hostbudget.require(3.6, "ABST-001")
        token, guard_gib, chunk = gpu_lease.acquire(timeout_s=7200, label="abst001"), 1.5, 256
    panel_read, partial, t_start = False, {}, time.time()
    try:
        import tokenizers
        import torch
        rt = la.configure_runtime(17, a.device)
        dev = a.device

        # ---------------- R1 data (CPU, no model)
        man = json.loads(a.manifest.read_text(encoding="utf-8"))
        if man["panel_sha256"] != sha(a.abst_panel):
            raise SystemExit("R1: the panel does not match its manifest")
        panel = jl(a.abst_panel)
        if a.max_rows_per_stratum:
            keep, seen = [], {}
            for r in panel:
                k = (r["stratum"], r["family"])
                if seen.get(k, 0) < a.max_rows_per_stratum:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            panel = keep
        strata = {}
        for r in panel:
            strata[(r["stratum"], r["family"])] = strata.get((r["stratum"], r["family"]), 0) + 1
        want = a.max_rows_per_stratum or 64
        if len(strata) != 8 or any(v != want for v in strata.values()):
            raise SystemExit(f"R1: strata {strata}")
        if any((r["reference"] == "UNKNOWN") != (r["stratum"] == "unk") for r in panel) or \
                any(r["stratum"] == "known" and r["reference"] not in ("0", "1", "2", "3") for r in panel):
            raise SystemExit("R1: a reference does not match its stratum")
        layer = jl(a.panels)
        used = {r["id"] for r in layer} | {r["id"] for r in jl(a.confirm_panel)} | {r["id"] for r in jl(a.tokenizer_corpus)}
        ids = [r["id"] for r in panel]
        reserve = {i for v in man["sealed_reserve_ids"].values() for i in v}
        if set(ids) & used or set(ids) & reserve or set(ids) & set(man["designer_seen_ids"]) or len(set(ids)) != len(ids):
            raise SystemExit("R1: panel rows overlap a used set, the sealed reserve or the designer-seen rows")
        # content-duplicate exclusion is the builder's (evidence + positive control in the manifest)
        if man["disjointness_evidence"]["content_duplicates_in_panel"] or man["content_dup_positive_control"] != "passed":
            raise SystemExit("R1: the manifest's content-duplicate evidence is not clean")
        if a.max_rows_per_category:
            keep, seen = [], {}
            for r in layer:
                k = (r["category"], r["split"])
                if seen.get(k, 0) < a.max_rows_per_category:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            layer = keep
        sel = [r for r in layer if r["split"] == "select"]
        ref_hold = [r for r in layer if r["split"] == "holdout" and r["kind"] == "chat" and r.get("reference")]
        S1 = json.loads((a.study / "S1_teacher_outputs.json").read_text(encoding="utf-8"))["rows"]
        S6 = json.loads((a.study / "S6_vocab_frontier.json").read_text(encoding="utf-8"))["configs"]
        LCA = json.loads(a.smallvd_lc.read_text(encoding="utf-8"))["arms"]["selected"]
        HL = json.loads(a.hlcc_result.read_text(encoding="utf-8"))["sets"]["primary"]
        if LCA["kept"] != a.s18_kept or HL["kept"] != a.s20_kept:
            raise SystemExit("R4: the recorded fits are for different kept sets than --s18-kept/--s20-kept")

        tok = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))
        im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
        eos = [im_end, eot]
        model = la.load_model(a.teacher, dev)
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("ABST requires an fp32 model")
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        L = model.config.num_hidden_layers
        full = list(range(L))
        V = {"V0": vs.Vocab(None, a.teacher / "tokenizer.json"), "V4": vs.Vocab(a.vocab_v4, a.teacher / "tokenizer.json"),
             "V2": vs.Vocab(a.vocab_v2, a.teacher / "tokenizer.json")}
        rows_v = model.lm_head.weight.shape[0]
        SUP = {k: am.support_index(v.retained, rows_v, dev) for k, v in V.items()}

        # ---------------- R2 vocabulary (text only)
        ans = am.answer_token_ids(V["V0"].teacher_tok, im_end)
        for k in ("V4", "V2"):
            keepset = set(V[k].retained.tolist())
            for s, i in ans.items():
                if i not in keepset:
                    raise SystemExit(f"R2: {s} (old id {i}) is not retained in {k}")
            for s in ("UNKNOWN", "0", "1", "2", "3"):
                if V[k].encode_old_ids(s) != [ans[s]]:
                    raise SystemExit(f"R2: {s!r} is not the single old id {ans[s]} under {k}")
            bad = [r["id"] for r in panel + ref_hold if not set(V[k].encode_old_ids(r["reference"]) + [im_end]) <= keepset]
            if bad:
                raise SystemExit(f"R2: {len(bad)} references leave {k}, e.g. {bad[:3]}")

        # ---------------- R3 identity folds, R4 fits + canaries (SELECT rows only)
        items_sel = rls.build_items(sel, tok, S1, im_end)
        sets = {"S20": a.s20_kept, "S18": a.s18_kept}
        recorded = {"S20": HL["fits"], "S18": LCA["fits"]}
        folds, fits, fit_canary = {}, {}, {}
        for nm in ("S18", "S20"):
            kept = sets[nm]
            ident = {r["p"]: torch.eye(model.config.hidden_size, device=dev) for r in lc.runs(kept, L) if r["p"] is not None}
            with la.keep_layers(model, kept):
                Hp = la.collect_positions(model, items_sel, eot, dev)
            with lc.folded(model, ident), la.keep_layers(model, kept):
                Hi = la.collect_positions(model, items_sel, eot, dev)
            if not all(torch.equal(x, y) for x, y in zip(Hp, Hi)):
                raise SystemExit(f"R3: {nm}: the identity fold does not reproduce the plain student bitwise")
            del Hp, Hi
            folds[nm], fits[nm] = lc.fit_compensation(model, kept, items_sel, eot, dev)
            hostbudget.guard(guard_gib, f"ABST {nm} fit")
            ref = [(f["p"], f["fit_residual_rel_fitted"]) for f in recorded[nm] if f.get("compensated")]
            got = [(f["p"], f["fit_residual_rel_fitted"]) for f in fits[nm] if f.get("compensated")]
            diff = max((abs(x[1] - y[1]) for x, y in zip(ref, got)), default=float("inf"))
            if [x[0] for x in ref] != [y[0] for y in got] or not diff <= CANARY_TOL:
                raise SystemExit(f"R4: FIT CANARY FAILED for {nm}: recorded {ref} vs now {got}")
            fit_canary[nm] = {"recorded": ref, "now": got, "max_abs_diff": diff}

        def configured(cfg_name):
            """(kept, folds or None) for a depth config name."""
            if cfg_name == "T":
                return full, None
            nm, kind = cfg_name.split("_")
            return sets[nm], (folds[nm] if kind == "lc" else None)

        # ---------------- R5 tier 1: the RECORDED code path on SMALLVD-001's spent HOLDOUT rows
        def rec_items(items):
            return {x["id"]: x["nats"] for x in items}
        L_name = f"{L}L"
        canary_cells = {
            "A_plain": ("S20_plain", "V4", rec_items(S6[f"{V['V4'].name}/{len(a.s20_kept)}L"]["answer_bpb"]["holdout_per_item"]),
                        S6[f"{V['V4'].name}/{len(a.s20_kept)}L"]["kept"]),
            "D_plain": ("S18_plain", "V2", rec_items(S6[f"{V['V2'].name}/{len(a.s18_kept)}L"]["answer_bpb"]["holdout_per_item"]),
                        S6[f"{V['V2'].name}/{len(a.s18_kept)}L"]["kept"]),
            "T4": ("T", "V4", rec_items(S6[f"{V['V4'].name}/{L_name}"]["answer_bpb"]["holdout_per_item"]), full),
            "D_lc": ("S18_lc", "V2", rec_items(LCA["lc"]["answer_bpb_V2"]["per_item"]), a.s18_kept),
            "F_lc": ("S18_lc", "V0", rec_items(LCA["lc"]["answer_bpb_V0"]["per_item"]), a.s18_kept),
        }
        tier1, tier2, t2_nats = {}, {}, {}
        t_tier2 = 0.0
        tok_count = 0
        for name, (cfg_name, v, recd, rec_kept) in canary_cells.items():
            kept, f = configured(cfg_name)
            if rec_kept != kept:
                raise SystemExit(f"R5: recorded {name} was for kept {rec_kept}, not {kept}")
            if set(recd) != {r["id"] for r in ref_hold}:
                raise SystemExit(f"R5: recorded {name} rows differ from the HOLDOUT reference rows")
            ctx = lc.folded(model, f) if f else _null()
            with ctx:
                now = rec_items(vs.answer_bpb(model, V[v], ref_hold, kept, im_end, dev, chunk=min(chunk, 256))["per_item"])
                t0 = time.time()
                new = am.measure(model, ref_hold, kept, V[v], {v: SUP[v]}, im_end, ans, dev)
                t_tier2 += time.time() - t0
            tok_count += sum(len(V[v].encode_old_ids(r["prompt_text"])) + 2 for r in ref_hold)
            d1 = max(abs(now[i] - recd[i]) for i in recd)
            if not d1 <= CANARY_TOL:
                raise SystemExit(f"R5: MEASUREMENT CANARY (tier 1) FAILED for {name}: max |diff| {d1}")
            t2_nats[name] = {i: new[i][v]["nats"] for i in recd}
            d2 = max(abs(t2_nats[name][i] - recd[i]) for i in recd)
            if not d2 <= TIER2_TOL:
                raise SystemExit(f"R5b: the B=1 path disagrees with the recorded nats for {name}: max |diff| {d2}")
            tier1[name], tier2[name] = d1, d2
            hostbudget.guard(guard_gib, f"ABST canary {name}")
        # reported, NOT gated (review: once |nats| <= 1e-3 holds, a sign flip needs |d| below ~1e-3/nats)
        rd = {i: av.log_ratio(canary_cells["D_plain"][2][i], canary_cells["D_lc"][2][i]) for i in canary_cells["D_lc"][2]}
        nd = {i: av.log_ratio(t2_nats["D_plain"][i], t2_nats["D_lc"][i]) for i in rd}
        sign_disagree = sum(1 for i in rd if av.sgn(rd[i]) != av.sgn(nd[i]))

        # ---------------- R6 measured throughput projection (generation priced as prefill + 4 decode
        # steps at 64 token-equivalents each: an upper bound on this card for B = 1)
        per_tok = t_tier2 / max(tok_count, 1)
        panel_tok = sum(len(V[v].encode_old_ids(r["prompt_text"])) + 2 for r in panel for v in V) * 5
        gen_tok = sum(len(V[v].encode_old_ids(r["prompt_text"])) + 4 * 64 for r in panel for _, _, v in GEN_CELLS)
        projected_min = (panel_tok + gen_tok) * per_tok / 60
        if projected_min > a.runtime_cap_min:
            raise SystemExit(f"R6: projected {projected_min:.1f} min > cap {a.runtime_cap_min}")
        controls = {"fit_canary": fit_canary, "tier1_max_abs_diff": tier1, "tier2_max_abs_diff": tier2,
                    "tier2_sign_disagreements_reported": sign_disagree, "projected_minutes": projected_min,
                    "panel_forward_tokens": panel_tok, "generation_token_equivalents": gen_tok}
        partial["refusal_controls"] = controls
        att_dir.mkdir(parents=True, exist_ok=True)
        with open(att_dir / f"BOUNDARY_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json", "x",
                  encoding="utf-8", newline="\n") as fh:
            json.dump({"record": "ABST_BOUNDARY", "refusal_controls": controls, "panel_sha256": sha(a.abst_panel),
                       "code_sha256": {q.name: rls.sha256_file(q) for q in sorted(HERE.glob("*.py"))}}, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        panel_read = True
        print(f"[ABST] refusal controls passed (R1-R6; projected {projected_min:.1f} min); the panel is read now", flush=True)

        # ---------------- measurement (PRIMARY first)
        records = {r["id"]: {"stratum": r["stratum"], "family": r["family"], "tier": r.get("tier"),
                             "reference": r["reference"], "m": {}, "gen": {},
                             "input_equal_V0": {v: V[v].encode_old_ids(r["prompt_text"]) == V["V0"].encode_old_ids(r["prompt_text"])
                                                for v in ("V4", "V2")}} for r in panel}
        partial["records"] = records

        def run_cfg(cfg_name, v):
            kept, f = configured(cfg_name)
            ctx = lc.folded(model, f) if f else _null()
            sink = {}
            partial["in_progress"] = {"cell": f"{cfg_name}/{v}", "rows": sink}   # rows land here as measured
            with ctx:
                res = am.measure(model, panel, kept, V[v], SUP, im_end, ans, dev, top5_for=v, sink=sink)
            for i, rec in res.items():
                records[i]["m"].setdefault(cfg_name, {})[v] = rec
            partial["in_progress"] = None
            hostbudget.guard(guard_gib, f"ABST {cfg_name}/{v}")

        for cfg_name, v in PRIMARY_ORDER:
            run_cfg(cfg_name, v)
        unk = [r["id"] for r in panel if r["stratum"] == "unk"]
        known_ids = [r["id"] for r in panel if r["stratum"] == "known"]
        primary = av.verdict(cells_for_verdict(records, unk, known_ids))
        a.out.mkdir(parents=True, exist_ok=False)
        prim = {"schema": 1, "record": "ABST_PRIMARY", "experiment": "ABST-001", "steps_1_2": {k: primary[k] for k in ("H1", "H2")},
                "panel_sha256": sha(a.abst_panel), "written_before_any_other_cell": True,
                "per_row_A_T4": {i: {"A_plain": records[i]["m"]["S20_plain"]["V4"]["V4"]["nats"],
                                     "A_lc": records[i]["m"]["S20_lc"]["V4"]["V4"]["nats"],
                                     "T4": records[i]["m"]["T"]["V4"]["V4"]["nats"]} for i in unk}}
        with open(a.out / "ABST_PRIMARY.json", "x", encoding="utf-8", newline="\n") as fh:
            json.dump(prim, fh, indent=1)
        print(f"[ABST] PRIMARY step 1: {primary['H1']['label']} ({primary['H1']['negative']} worse / "
              f"{primary['H1']['positive']} better) | step 2: {primary['H2']['label']}", flush=True)
        for cfg_name, v in REST_ORDER:
            run_cfg(cfg_name, v)
        for cell, cfg_name, v in GEN_CELLS:
            kept, f = configured(cfg_name)
            ctx = lc.folded(model, f) if f else _null()
            with ctx:
                g = vs.generate_rows(model, V[v], panel, kept, eos, eot, dev, 4, batch=1)
            for x in g:
                records[x["id"]]["gen"][cell] = x["text"]
            hostbudget.guard(guard_gib, f"ABST generation {cell}")
        full_v = av.verdict(cells_for_verdict(records, unk, known_ids))
        if {k: full_v[k] for k in ("H1", "H2")} != prim["steps_1_2"]:
            raise SystemExit("steps 1-2 changed between ABST_PRIMARY and the full verdict")
        result = {"schema": 1, "record": "ABST_RESULT", "experiment": "ABST-001",
                  "REGISTERED": {"H1": full_v["H1"]["label"], "H2": full_v["H2"]["label"], "H3": full_v["H3"]["label"]},
                  "verdict": full_v, "descriptive": ad.summarise(records),
                  "refusal_controls": controls, "earlier_boundary_markers": [p.name for p in markers],
                  "fits": fits, "answer_token_ids": ans, "kept": sets,
                  "teacher_weight_sha256": rls.sha256_file(a.teacher / "model.safetensors"),
                  "panel_sha256": sha(a.abst_panel), "manifest_sha256": sha(a.manifest),
                  "runtime": {"flags": rt, "python": sys.version.split()[0], "torch": torch.__version__,
                              "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                              "seconds": round(time.time() - t_start, 1),
                              "code_sha256": {p.name: rls.sha256_file(p) for p in sorted(HERE.glob("*.py"))}},
                  "records": records, "quality_claim": False, "release_permitted": False}
        la.dump(a.out / "ABST_RESULT.json", result)
        print(f"[ABST] REGISTERED: H1 {full_v['H1']['label']} | H2 {full_v['H2']['label']} | H3 {full_v['H3']['label']}", flush=True)
        return 0
    except BaseException as e:
        if panel_read:
            import traceback
            infra = (isinstance(e, (hostbudget.LowMemoryAbort, KeyboardInterrupt)) or "CUDA" in str(e)
                     or type(e).__name__ == "OutOfMemoryError")
            rec = {"schema": 1, "record": "ABST_ATTEMPT", "experiment": "ABST-001",
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                   "exception_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(),
                   "classification_hint": "infrastructure" if infra else "deterministic or unknown",
                   "primary_written": (a.out / "ABST_PRIMARY.json").exists(), "partial": partial,
                   "code_sha256": {q.name: rls.sha256_file(q) for q in sorted(HERE.glob("*.py"))}}
            d = a.out.parent / "attempts"
            d.mkdir(parents=True, exist_ok=True)
            fn = d / f"ATTEMPT_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json"
            with open(fn, "x", encoding="utf-8", newline="\n") as fh:
                json.dump(rec, fh, ensure_ascii=False, default=str)
            print(f"[ABST] attempt record written: {fn}", flush=True)
        raise
    finally:
        if token:
            gpu_lease.release(token)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
