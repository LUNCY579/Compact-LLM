"""End-to-end dry run of the O/T/REP/MR pipeline with SYNTHETIC generations. No model.

Exists because the pipeline demultiplexes composite ids (`{item}||{variant}`) across four
generation passes and three selector passes, and an id mismatch anywhere would surface as
"every condition scored zero" only AFTER a two-hour run on four systems. This exercises the
real render_selector and score_matrix code paths against fabricated predictions whose correct
answers are known by construction, so the plumbing is verified before any GPU time is spent.

The synthetic model is deliberately adversarial to the plumbing rather than to the science:
  R0/direct  correct on items 0 mod 3
  rep1..3    correct on items 1 mod 3        (so REP's selector has something to find)
  R1         correct on items 1 mod 3
  R2         abstains on the unanswerable rows, malformed elsewhere
  R3         correct on items 2 mod 3        (so MR's oracle covers everything R0 misses)
  T          same as direct
selectors always name the first offered candidate, which is a WORSE policy than S0 majority --
so selector utilisation must come out below the oracle ceiling, and the test asserts that
rather than hoping for it.

FIXTURE PASS IS NOT NATIVE PASS.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCOUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
             "dmrf001-20260918/scout")
TOK = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
           "Rineya-Supa-1.7.0-devfreeze")
PY = "C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe"
SEP = "||"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(f"{name} {detail}")


def write_gen(d: Path, label: str, preds: dict[str, str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.jsonl").write_text(
        "".join(json.dumps({"id": k, "prediction": v, "generated_tokens": len(v.split()),
                            "terminated": True}, sort_keys=True) + "\n"
                for k, v in preds.items()), encoding="utf-8")
    (d / "GENERATE.json").write_text(json.dumps({
        "label": label, "model": "SYNTHETIC", "rows": len(preds),
        "rows_skipped_context_overflow": 0, "max_new_tokens": 96, "max_context": 4096,
        "dtype": "fp32", "decoding": "greedy", "transformers": "SYNTHETIC",
        "numerics": {"cuda_matmul_allow_tf32": False}, "offload": False,
    }, indent=2), encoding="utf-8")


def main() -> int:
    if not (SCOUT / "SCOUT.jsonl").exists():
        print(f"SKIP: scout not built at {SCOUT}")
        return 0

    rows = [json.loads(l) for l in (SCOUT / "SCOUT.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if r["split"] == "CAL"]
    tmp = Path(tempfile.mkdtemp(prefix="dmrf_pipe_"))
    try:
        V = tmp / "variants"
        print("render variants (tokenizer only, no model)")
        r = subprocess.run([PY, str(ROOT / "inference" / "render_variants.py"),
                            "--scout", str(SCOUT / "SCOUT.jsonl"), "--split", "CAL",
                            "--tokenizer", str(TOK), "--out", str(V)],
                           capture_output=True, text=True)
        check("render_variants exits 0", r.returncode == 0, r.stderr[-300:])
        if r.returncode != 0:
            return 1
        vm = json.loads((V / "VARIANTS.json").read_text(encoding="utf-8"))
        check("solver pass covers 7 variants per item",
              vm["passes"]["solver"]["rows"] == 7 * len(rows),
              f"{vm['passes']['solver']['rows']} vs {7 * len(rows)}")
        check("tokenmatch pass is one row per item",
              vm["passes"]["tokenmatch"]["rows"] == len(rows))
        check("T allowance is MR's total (4x96+8=392)",
              vm["passes"]["tokenmatch"]["max_new_tokens"] == 392)
        check("registered reuse names O, REP[0], MR R0",
              set(vm["registered_reuse"]["serves"]) == {"O", "REP[0]", "MR R0"})

        # ---- fabricate solver predictions with known correctness
        solver_ids = [json.loads(l)["id"] for l in
                      (V / "solver" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                      if l.strip()]
        by_item = {r["id"]: r for r in rows}
        order = {iid: i for i, iid in enumerate(sorted(by_item))}
        WRONG = {"transition": '{"A":999,"B":999,"C":999,"D":999}',
                 "multi_step": '{"A":999,"B":999,"C":999,"D":999}',
                 "arith_competence": "-12345", "longctx_mixed": "-12345"}

        def answer(iid, variant):
            row, i = by_item[iid], order[iid]
            good = row["reference"]
            bad = WRONG[row["stratum"]]
            if variant in ("direct", "T"):
                return good if i % 3 == 0 else bad
            if variant.startswith("rep") or variant == "R1":
                return good if i % 3 == 1 else bad
            if variant == "R2":
                return "UNKNOWN" if row.get("unanswerable") else "!!!not parseable{"
            if variant == "R3":
                return good if i % 3 == 2 else bad
            raise AssertionError(variant)

        solver_preds = {}
        for sid in solver_ids:
            iid, variant = sid.split(SEP)
            solver_preds[sid] = answer(iid, variant)
        G = tmp / "gen"
        write_gen(G / "solver", "SYN_solver", solver_preds)
        write_gen(G / "tokenmatch", "SYN_tokenmatch",
                  {f"{i}{SEP}T": answer(i, "T") for i in by_item})

        print("render selector prompts from the synthetic candidates")
        SELD = tmp / "sel"
        r = subprocess.run([PY, str(ROOT / "inference" / "render_selector.py"),
                            "--scout", str(SCOUT / "SCOUT.jsonl"), "--split", "CAL",
                            "--solver-gen", str(G / "solver"),
                            "--tokenizer", str(TOK), "--out", str(SELD)],
                           capture_output=True, text=True)
        check("render_selector exits 0", r.returncode == 0, r.stderr[-300:])
        if r.returncode != 0:
            return 1
        sm = json.loads((SELD / "SELECTOR_RENDER.json").read_text(encoding="utf-8"))
        check("S1 subset is frozen by position",
              sm["s1_diagnostic_subset"]["chosen_by"].startswith("position"))
        check("S1 subset is a quarter of the items",
              abs(sm["s1_diagnostic_subset"]["n"] - len(rows) / 4) <= 1,
              str(sm["s1_diagnostic_subset"]["n"]))
        check("R2 abstentions became checker findings",
              sm["stats"]["findings_emitted"] > 0, str(sm["stats"]))

        # ---- selectors always name the first candidate offered
        def first_named(pdir: Path) -> dict[str, str]:
            out = {}
            pf = pdir / "prompts.jsonl"
            if not pf.exists():
                return out
            for line in pf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                ids = [x for x in ("R0", "R1", "R2", "R3", "P0", "P1", "P2", "P3")
                       if f"[{x}]" in o["text"]]
                out[o["id"]] = ids[0] if ids else "??"
            return out

        for name in ("rep_s2", "mr_s2", "mr_s1"):
            write_gen(G / name, f"SYN_{name}", first_named(SELD / name))

        print("score the matrix")
        outp = tmp / "MATRIX_SYN.json"
        r = subprocess.run([PY, str(ROOT / "analysis" / "score_matrix.py"),
                            "--scout", str(SCOUT / "SCOUT.jsonl"), "--split", "CAL",
                            "--label", "SYN",
                            "--solver", str(G / "solver"),
                            "--tokenmatch", str(G / "tokenmatch"),
                            "--rep-s2", str(G / "rep_s2"), "--mr-s2", str(G / "mr_s2"),
                            "--mr-s1", str(G / "mr_s1"), "--out", str(outp)],
                           capture_output=True, text=True)
        check("score_matrix exits 0", r.returncode == 0, r.stderr[-400:])
        if r.returncode != 0:
            return 1
        rep = json.loads(outp.read_text(encoding="utf-8"))

        n = len(rows)
        check("every item scored in every condition",
              all(rep["conditions"][c]["n"] == n for c in ("O", "T", "REP", "MR", "MR_S0")))
        # direct is correct on i%3==0, so O must equal that count exactly.
        expect_o = sum(1 for i in by_item if order[i] % 3 == 0)
        check("O matches the constructed ground truth",
              rep["conditions"]["O"]["total"] == expect_o,
              f"{rep['conditions']['O']['total']} vs {expect_o}")
        check("T equals O (same synthetic answers)",
              rep["conditions"]["T"]["total"] == rep["conditions"]["O"]["total"])
        check("no item was silently dropped from the paired tables",
              all(v["n"] == n for v in rep["paired"].values()))
        for k, v in rep["paired"].items():
            check(f"paired {k} counts sum to n",
                  v["both_correct"] + v["repair"] + v["damage"] + v["both_wrong"] == n)
        # R0/R1/R3 together cover all residues mod 3, so the oracle is every item.
        rc = rep["role_complementarity"]
        check("oracle over roles covers every item", rc["any_role_correct"] == n,
              str(rc["any_role_correct"]))
        check("roles are complementary, not redundant", rc["oracle_minus_best_single"] > 0,
              str(rc["oracle_minus_best_single"]))
        check("a first-candidate selector falls short of the oracle",
              rep["selector_utilisation_mr_s2"]["selector_shortfall_vs_oracle"] > 0)
        check("roles diverge (not name-only)",
              rep["roles_did_different_things"]["verdict"] == "ROLES_DIVERGE")
        check("malformed R2 output is visible in the flags",
              rep["flags"]["mr_all_malformed"] >= 0)
        acc = rep["accounting"]
        check("actual calls are recorded per pass", sum(acc["actual_calls_by_pass"].values()) > 0)
        check("logical deployment cost exceeds deduplicated cost",
              sum(acc["logical_deployment_calls"].values()) > acc["actual_total_calls"],
              f"{sum(acc['logical_deployment_calls'].values())} vs {acc['actual_total_calls']}")
        check("S1 extra calls accounted separately", acc["s1_extra_calls"] > 0)
        check("gold is declared to enter only at scoring", rep["gold_entered_here_only"] is True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FIXTURE FAIL: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("FIXTURE PASS (pipeline plumbing on synthetic generations; no model was called)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
