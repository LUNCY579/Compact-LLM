"""Build the REP and MR selector prompts from recorded candidates. No model is called here.

Runs between the solver pass and the selector pass. Candidates are parsed with the SAME
extractor the scorer uses (`analysis/answers.py`), because a selector that parses differently
from the scorer can be credited with choosing a candidate the scorer then marks wrong.

Three selector calls are rendered:
  `rep_s2`  -- REP candidates (direct + 3 paraphrases), checker findings shown (none exist for
               paraphrases, so S1 and S2 coincide for REP and only S2 is run)
  `mr_s2`   -- MR candidates (R0..R3), checker findings SHOWN. The main selector.
  `mr_s1`   -- MR candidates, checker findings WITHHELD. Diagnostic, on a prospectively fixed
               subset, so the value of the findings can be isolated. Its calls are accounted
               SEPARATELY: the experiment must not be described as five calls per item when
               part of it is six.

S0 needs no prompt -- it is majority/agreement over the same candidates, computed in the
scorer at zero model cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.answers import extract, finding_for       # noqa: E402
from inference.roles import assert_label_isolated, selector_prompt  # noqa: E402

SEP = "||"
#: The S1 diagnostic subset, fixed prospectively: every 4th item in bank order. Chosen by
#: position, never by outcome, and frozen before any selector ran.
S1_EVERY_NTH = 4


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    ap.add_argument("--solver-gen", type=Path, required=True,
                    help="gen dir from the solver pass (predictions.jsonl)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    import transformers
    from transformers import AutoTokenizer

    rows = {r["id"]: r for r in
            (json.loads(l) for l in a.scout.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["split"] == a.split}
    preds: dict[str, str] = {}
    for line in (a.solver_gen / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        preds[d["id"]] = d.get("prediction", "")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True)

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    rep_s2, mr_s2, mr_s1 = [], [], []
    s1_ids: list[str] = []
    stats = {"items": 0, "rep_all_malformed": 0, "mr_all_malformed": 0, "findings_emitted": 0}

    for pos, (iid, row) in enumerate(sorted(rows.items())):
        stats["items"] += 1
        direct = preds.get(f"{iid}{SEP}direct", "")
        direct_ans = extract(row, direct)

        # REP candidates: the direct call plus three paraphrases, labelled P0..P3.
        rep_c = [{"id": "P0", "answer": direct_ans}]
        for i in (1, 2, 3):
            rep_c.append({"id": f"P{i}", "answer": extract(row, preds.get(f"{iid}{SEP}rep{i}", ""))})
        rep_use = [c for c in rep_c if c["answer"] is not None]
        if not rep_use:
            stats["rep_all_malformed"] += 1
        else:
            m = selector_prompt(row["prompt"], rep_use, include_checker_findings=True)
            assert_label_isolated(m, row, allow=[c["answer"] for c in rep_use])
            rep_s2.append({"id": f"{iid}{SEP}rep_s2", "text": render(m)})

        # MR candidates: R0 (the direct call) plus R1..R3, with derived checker findings.
        mr_c = [{"id": "R0", "answer": direct_ans, "finding": None}]
        for rid in ("R1", "R2", "R3"):
            ans = extract(row, preds.get(f"{iid}{SEP}{rid}", ""))
            mr_c.append({"id": rid, "answer": ans,
                         "finding": finding_for(rid, ans, direct_ans)})
        mr_use = [c for c in mr_c if c["answer"] is not None]
        stats["findings_emitted"] += sum(1 for c in mr_use if c.get("finding"))
        if not mr_use:
            stats["mr_all_malformed"] += 1
            continue
        m2 = selector_prompt(row["prompt"], mr_use, include_checker_findings=True)
        assert_label_isolated(m2, row, allow=[c["answer"] for c in mr_use])
        mr_s2.append({"id": f"{iid}{SEP}mr_s2", "text": render(m2)})
        if pos % S1_EVERY_NTH == 0:
            s1_ids.append(iid)
            m1 = selector_prompt(row["prompt"],
                                 [{k: v for k, v in c.items() if k != "finding"} for c in mr_use],
                                 include_checker_findings=False)
            assert_label_isolated(m1, row, allow=[c["answer"] for c in mr_use])
            mr_s1.append({"id": f"{iid}{SEP}mr_s1", "text": render(m1)})

    def write(name, items):
        d = a.out / name
        d.mkdir()
        body = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in items)
        (d / "prompts.jsonl").write_text(body, encoding="utf-8")
        return {"dir": str(d), "rows": len(items), "prompts_sha256": h(body)}

    meta = {
        "schema": 1, "record": "DMRF_SELECTOR_RENDER",
        "split": a.split, "transformers": transformers.__version__,
        "enable_thinking": False,
        "passes": {"rep_s2": write("rep_s2", rep_s2), "mr_s2": write("mr_s2", mr_s2),
                   "mr_s1": write("mr_s1", mr_s1)},
        "s1_diagnostic_subset": {
            "rule": f"every {S1_EVERY_NTH}th item in sorted bank order",
            "frozen_prospectively": True, "chosen_by": "position, never by outcome",
            "n": len(s1_ids), "ids": s1_ids,
            "why": ("isolates the value of the checker findings by withholding them. Its calls "
                    "are accounted separately so the study is not described as five calls per "
                    "item when part of it is six."),
        },
        "rep_s1_not_run": ("paraphrases produce no checker findings, so S1 and S2 coincide for "
                           "REP and only S2 is rendered"),
        "s0_needs_no_prompt": "majority/agreement over the same candidates, 0 extra model calls",
        "candidate_parsing": "analysis/answers.py::extract -- the SAME extractor the scorer uses",
        "findings_are_derived_not_generated": (
            "This family emits one field, so a second 'findings' field would break the identical "
            "final-answer contract. Findings are derived from what the audit roles did -- R2 "
            "abstaining, R3 disagreeing with the direct solve -- by code, at zero extra model "
            "calls, without consulting the reference."),
        "stats": stats,
        "model_calls": 0,
    }
    (a.out / "SELECTOR_RENDER.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"rep_s2": len(rep_s2), "mr_s2": len(mr_s2), "mr_s1": len(mr_s1),
                      **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
