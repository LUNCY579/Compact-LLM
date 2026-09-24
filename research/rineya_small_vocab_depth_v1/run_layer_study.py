"""SMALLVD-001 Phases 7-8 (+ the vocabulary frontier) -- one GPU job, staged, create-only outputs.

Registered in SMALLVD001_PREREG.md. Stages, each writing its own JSON into --out:

  S1 teacher_outputs      greedy teacher generations for every chat row (select + holdout)
  S2 diagnostics          residual / cosine / CKA / angular-distance diagnostics (teacher only)
  S3 single_ablation      28 leave-one-out bypasses: teacher-forced metrics on SELECT and HOLDOUT,
                          generation metrics on HOLDOUT
  S4 greedy_backward      remove one block at a time down to --min-layers, objective computed on
                          SELECT only (category-mean answer KL; registered)
  S5 depth_strategies     every strategy's layer set at every registered depth, scored on HOLDOUT
                          (teacher-forced + generation)
  S6 vocab_frontier       each candidate vocabulary at full depth, and the iso-parameter
                          frontier configurations, scored on HOLDOUT by answer bits-per-byte and
                          generation (tokenizer-independent)

AMENDMENT 1 (registered before any layer was scored): every HOLDOUT measurement is also summarised
separately on SELECT2 and REPORT (`layer_select.holdout_subsplit`), and the per-row records are
KEPT, so a subset or a bootstrap can be recomputed from the stage files. SELECT2 chooses among
strategies (the frontier's layer set at each depth); REPORT chooses nothing.

RESUMABILITY IS A FLAG, NOT A DEFAULT. Without --resume an existing stage file is an error. With it,
an existing stage is loaded and the manifest records `mode: resumed` for that stage, so a skipped
stage can never be mistaken for a verified one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import param_accounting as pa  # noqa: E402
import virtual_student as vs  # noqa: E402

DEPTHS = (24, 20, 19, 18, 16, 12)
# Nimbus/Supa lineage: blocks {6,7,8,9,10,14,18,23,24} are the ones the 19-layer donor DROPPED
# (lineage.md; claimed, not weight-verified). Included as a named hypothesis, not a prior.
LINEAGE_DONOR_DROPPED = [6, 7, 8, 9, 10, 14, 18, 23, 24]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


class Stages:
    def __init__(self, out: Path, resume: bool, split_sha: str):
        self.out, self.resume, self.modes, self.split_sha = out, resume, {}, split_sha

    def run(self, name: str, fn):
        p = self.out / f"{name}.json"
        if p.exists():
            if not self.resume:
                raise SystemExit(f"{p} exists; pass --resume to reuse it (recorded as resumed)")
            prev = json.loads(p.read_text(encoding="utf-8"))
            if prev.get("_holdout_subsplit_sha256") != self.split_sha:
                raise SystemExit(f"{p} was computed under a different HOLDOUT subsplit; refusing to resume it")
            self.modes[name] = "resumed"
            return prev
        t0 = time.time()
        res = fn()
        res["_stage_seconds"] = round(time.time() - t0, 1)
        res["_holdout_subsplit_sha256"] = self.split_sha     # the split each stage was computed under
        la.dump(p, res)
        self.modes[name] = "computed"
        return res


def build_items(rows, tok, teacher_out: dict, im_end: int):
    """Teacher-forced items. For chat rows the continuation is the TEACHER's own greedy output
    (what a distilled student should reproduce) plus <|im_end|>; for text rows every position."""
    items = []
    for r in rows:
        if r["kind"] == "chat":
            p = tok.encode(r["prompt_text"], add_special_tokens=False).ids
            c = tok.encode(teacher_out[r["id"]]["text"], add_special_tokens=False).ids + [im_end]
            ids = p + c
            items.append({"id": r["id"], "category": r["category"], "ids": ids,
                          "score_pos": list(range(len(p) - 1, len(ids) - 1)), "targets": c})
        else:
            ids = tok.encode(r["text"], add_special_tokens=False).ids[:512]
            items.append({"id": r["id"], "category": r["category"], "ids": ids,
                          "score_pos": list(range(0, len(ids) - 1)), "targets": ids[1:]})
    return items


def category_mean_kl(summary: dict) -> float:
    cats = summary["by_category"]
    return sum(c["kl_mean_pos"] for c in cats.values()) / len(cats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--panels", required=True, type=Path)
    ap.add_argument("--vocab-dirs", nargs="*", default=[], type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-layers", type=int, default=12)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stages", default="S1,S2,S3,S4,S5,S6")
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows-per-category", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="pipeline check on a TINY random model (< 50M params, asserted): CPU, no lease")
    a = ap.parse_args(argv)
    depths = tuple(int(x) for x in a.depths.split(","))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    want = set(a.stages.split(","))
    if a.dry_run:
        cfg0 = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
        n0 = pa.analytic(cfg0, cfg0["vocab_size"], cfg0["num_hidden_layers"])["unique_physical_parameters"]
        if n0 >= 50_000_000 or a.device != "cpu":
            raise SystemExit(f"--dry-run is only for tiny CPU models (got {n0:,} params on {a.device})")
        hostbudget.require(0.8, "SMALLVD-001 layer study DRY RUN")
        token, guard_gib = None, 0.4
    else:
        hostbudget.require(3.6, "SMALLVD-001 layer study")
        token, guard_gib = gpu_lease.acquire(timeout_s=7200, label="smallvd001:layer-study"), 1.5
    try:
        import torch
        import tokenizers
        import transformers
        rt = la.configure_runtime(17)
        dev = a.device
        chunk = 128 if dev == "cpu" else 512      # logits chunk: host RAM on CPU, VRAM on GPU
        rows = [json.loads(l) for l in a.panels.read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.max_rows_per_category:
            keep, seen = [], {}
            for r in rows:
                k = (r["category"], r["split"])
                if seen.get(k, 0) < a.max_rows_per_category:
                    keep.append(r)
                    seen[k] = seen.get(k, 0) + 1
            rows = keep
        sel = [r for r in rows if r["split"] == "select"]
        hold = [r for r in rows if r["split"] == "holdout"]
        sub = ls.holdout_subsplit(hold)       # Amendment 1: id -> select2 | report
        sub_record = {"salt": ls.SUBSPLIT_SALT,
                      "sha256": hashlib.sha256(json.dumps(sorted(sub.items())).encode("utf-8")).hexdigest(),
                      "counts": {f"{c}/{part}": sum(1 for r in hold if r["category"] == c and sub[r["id"]] == part)
                                 for c in sorted({r["category"] for r in hold}) for part in ("select2", "report")}}
        tok = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))
        im_end, eot = tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")
        eos = [im_end, eot]
        model = la.load_model(a.teacher, dev)
        if token:
            gpu_lease.set_workers(token, [os.getpid()])
        L = model.config.num_hidden_layers
        full = list(range(L))
        st = Stages(a.out, a.resume, sub_record["sha256"])
        V0 = vs.Vocab(None, a.teacher / "tokenizer.json")
        chat_sel = [r for r in sel if r["kind"] == "chat"]
        chat_hold = [r for r in hold if r["kind"] == "chat"]

        def teacher_outputs():
            out = {}
            for part in (chat_sel, chat_hold):
                gens = vs.generate_rows(model, V0, part, full, eos, eot, dev, a.gen_tokens)
                out.update({g["id"]: g for g in gens})
                hostbudget.guard(guard_gib, "layer study S1")
            return {"rows": out}
        T = st.run("S1_teacher_outputs", teacher_outputs)["rows"]

        items_sel = build_items(sel, tok, T, im_end)
        items_hold = build_items(hold, tok, T, im_end)
        H_sel = la.collect_positions(model, items_sel, eot, dev)
        H_hold = la.collect_positions(model, items_hold, eot, dev)

        def tf_metrics(kept, items, H_ref):
            with la.keep_layers(model, kept):
                H = la.collect_positions(model, items, eot, dev)
            res = la.compare(model.lm_head, H_ref, H, items, chunk=chunk)
            del H
            return res

        def gen_metrics(kept, vocab=V0):
            g = vs.generate_rows(model, vocab, chat_hold, kept, eos, eot, dev, a.gen_tokens)
            texts, ntok = [x["text"] for x in g], [x["new_tokens"] for x in g]
            ref = [T[r["id"]]["text"] for r in chat_hold]
            m = la.generation_metrics(texts, ref, chat_hold, ntok, a.gen_tokens)
            for part in ("select2", "report"):
                ix = [i for i, r in enumerate(chat_hold) if sub[r["id"]] == part]
                m[part] = la.generation_metrics([texts[i] for i in ix], [ref[i] for i in ix],
                                                [chat_hold[i] for i in ix], [ntok[i] for i in ix], a.gen_tokens)
            m["per_row"] = [{"id": r["id"], "text": texts[i], "new_tokens": ntok[i], "terminated": g[i]["terminated"]}
                            for i, r in enumerate(chat_hold)]
            return m

        def strip(res):
            return {k: v for k, v in res.items() if k != "per_item"}

        def split3(res, summ):
            """All-HOLDOUT summary, SELECT2 and REPORT summaries by the SAME aggregation code, and
            the per-row records (Amendment 1)."""
            pi = res["per_item"]
            out = {"holdout": strip(res), "holdout_per_item": pi}
            for part in ("select2", "report"):
                out[part] = strip(summ([x for x in pi if sub[x["id"]] == part]))
            return out

        if "S2" in want:
            st.run("S2_diagnostics", lambda: la.residual_diagnostics(model, items_sel, eot, dev))

        def single():
            out = {}
            for l in full:
                kept = [k for k in full if k != l]
                s, h = tf_metrics(kept, items_sel, H_sel), tf_metrics(kept, items_hold, H_hold)
                out[str(l)] = {"select": strip(s), **split3(h, la.summarise), "holdout_generation": gen_metrics(kept),
                               "select_objective": category_mean_kl(s)}
                print(f"[single] L{l}: select obj {out[str(l)]['select_objective']:.4f}", flush=True)
                hostbudget.guard(guard_gib, "layer study S3")
            return {"layers": out}
        S3 = st.run("S3_single_ablation", single) if "S3" in want else None

        def greedy():
            cache = {}

            def evaluate(kept):
                key = tuple(kept)
                if key not in cache:
                    cache[key] = category_mean_kl(tf_metrics(kept, items_sel, H_sel)) if len(kept) < L else 0.0
                    hostbudget.guard(guard_gib, "layer study S4")
                return cache[key]
            return {"trajectory": ls.greedy_backward(evaluate, L, a.min_layers), "evaluations": len(cache)}
        S4 = st.run("S4_greedy_backward", greedy) if "S4" in want else None

        def strategies():
            diag = json.loads((a.out / "S2_diagnostics.json").read_text(encoding="utf-8"))
            single_scores = {int(k): v["select_objective"] for k, v in S3["layers"].items()}
            bi = {b["layer"]: b["block_influence"] for b in diag["per_block"]}
            angular = {int(k): v for k, v in diag["angular_distance"].items()}
            sets = ls.strategy_sets(L, depths, single_scores, bi, angular, S4["trajectory"])
            if L == 28 and 19 in sets:
                sets[19]["lineage_donor"] = [k for k in full if k not in LINEAGE_DONOR_DROPPED]
            done, out = {}, {}
            for n, strat in sets.items():
                out[str(n)] = {}
                for name, kept in strat.items():
                    key = tuple(kept)
                    if key not in done:
                        h = split3(tf_metrics(kept, items_hold, H_hold), la.summarise)
                        done[key] = {"kept": kept, **h,
                                     "holdout_objective": category_mean_kl(h["holdout"]),
                                     "select2_objective": category_mean_kl(h["select2"]),
                                     "report_objective": category_mean_kl(h["report"]),
                                     "holdout_generation": gen_metrics(kept)}
                        hostbudget.guard(guard_gib, "layer study S5")
                    out[str(n)][name] = done[key]
                    print(f"[S5] {n}L {name}: select2 obj {done[key]['select2_objective']:.4f}", flush=True)
            return {"depths": out}
        S5 = st.run("S5_depth_strategies", strategies) if "S5" in want else None

        def vocab_frontier():
            vocabs = [V0] + [vs.Vocab(d, a.teacher / "tokenizer.json") for d in a.vocab_dirs]
            cfg = json.loads((a.teacher / "config.json").read_text(encoding="utf-8"))
            ref_hold = [r for r in chat_hold if r.get("reference")]
            # Amendment 1: the frontier's layer set at each depth is chosen on SELECT2, never REPORT
            best = {int(n): min(v.values(), key=lambda x: x["select2_objective"])["kept"]
                    for n, v in S5["depths"].items()}
            out = {}
            for voc in vocabs:
                rows_v = cfg["vocab_size"] if voc.retained is None else int(voc.retained.numel())
                for n in sorted({L} | set(best)):
                    kept = full if n == L else best[n]
                    params = pa.analytic(cfg, rows_v, n)["unique_physical_parameters"]
                    key = f"{voc.name}/{n}L"
                    out[key] = {"vocab": voc.name, "vocab_rows": rows_v, "n_layers": n, "kept": kept,
                                "unique_params": params,
                                "answer_bpb": split3(vs.answer_bpb(model, voc, ref_hold, kept, im_end, dev,
                                                                   chunk=min(chunk, 256)), vs.bpb_summary),
                                "holdout_generation": gen_metrics(kept, voc)}
                    print(f"[S6] {key}: {params:,} params, select2 bpb "
                          f"{out[key]['answer_bpb']['select2']['all']['bpb']:.4f}", flush=True)
                    hostbudget.guard(guard_gib, "layer study S6")
            return {"configs": out}
        if "S6" in want:
            st.run("S6_vocab_frontier", vocab_frontier)

        manifest = {"schema": 1, "record": "SMALLVD_LAYER_STUDY_MANIFEST", "experiment": "SMALLVD-001",
                    "teacher_dir": str(a.teacher), "teacher_weight_sha256": sha256_file(a.teacher / "model.safetensors"),
                    "panels": str(a.panels), "panels_sha256": sha256_file(a.panels),
                    "vocab_dirs": [str(d) for d in a.vocab_dirs],
                    "vocab_map_sha256": {str(d): sha256_file(d / "token_id_map.json") for d in a.vocab_dirs},
                    "stage_modes": st.modes, "runtime_flags": rt, "dry_run": a.dry_run,
                    "depths": depths, "max_rows_per_category": a.max_rows_per_category,
                    "holdout_subsplit": sub_record,
                    "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                                "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"},
                    "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))},
                    "quality_claim": False, "release_permitted": False}
        la.dump(a.out / f"MANIFEST_{int(time.time())}.json", manifest)
        return 0
    finally:
        if token:
            gpu_lease.release(token)


if __name__ == "__main__":
    raise SystemExit(main())
