"""Render every O / T / REP / MR variant prompt for one split, as one generation file.

CALL ACCOUNTING AND REGISTERED REUSE, declared here rather than reconciled later
--------------------------------------------------------------------------------
The four conditions share prompts, and the sharing is exact rather than approximate:

  O      = build(prompt)                      -- the ordinary solve
  T      = build(prompt), IDENTICAL text      -- differs ONLY in the output-token allowance
  REP[0] = build(prompt), IDENTICAL text      -- the empty paraphrase is the ordinary solve
  MR R0  = build(prompt, role=R0), IDENTICAL  -- R0's instruction is empty by construction

So one `direct` generation at the ordinary allowance serves O, REP[0] and MR's R0. That reuse
is REGISTERED here and is legitimate only because checkpoint, prompt text, input ids, decoding,
seed, output limit and policy are all identical -- `tests/test_inference.py` asserts the
byte-identity of those three builds.

T is the same prompt at a DIFFERENT allowance, so it is a separate generation. It is
token-allowance matching, not compute matching, and the distinction is kept in the manifest.

The consequence for cost reporting, which must be stated both ways:
  * LOGICAL deployment cost   -- O 1, T 1, REP 5, MR 5 calls per item
  * ACTUAL deduplicated cost  -- 1 direct + 3 paraphrases + 3 roles + 1 T + 2 selectors = 10

Reporting only the second understates deployment; reporting only the first overstates what was
spent. Both go in the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.roles import (  # noqa: E402
    MR_SOLVER_ROLES, REP_PARAPHRASES, assert_label_isolated, build, role_roster,
)

SEP = "||"
#: Output allowance for every solver call. Matches the acquisition probe's 96.
ROLE_TOKENS = 96
#: Selector allowance. Small because the selector emits a candidate id, nothing else.
SELECTOR_TOKENS = 8


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    import transformers
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in a.scout.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    if not rows:
        raise SystemExit(f"no rows for split {a.split}")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True)

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    # variant -> list of rendered rows
    solver: list[dict] = []
    variants: dict[str, str] = {}
    for r in rows:
        # the shared direct call
        d = build(r["prompt"])
        assert_label_isolated(d, r)
        solver.append({"id": f"{r['id']}{SEP}direct", "text": render(d)})
        variants["direct"] = "O, REP[0] and MR R0 -- identical prompt, one generation"
        for i, para in enumerate(REP_PARAPHRASES):
            if i == 0:
                continue                       # REP[0] is the direct call
            m = build(r["prompt"], paraphrase=para)
            assert_label_isolated(m, r)
            solver.append({"id": f"{r['id']}{SEP}rep{i}", "text": render(m)})
            variants[f"rep{i}"] = f"REP paraphrase {i}"
        for role in MR_SOLVER_ROLES:
            if role.rid == "R0":
                continue                       # R0 is the direct call
            m = build(r["prompt"], role=role)
            assert_label_isolated(m, r)
            solver.append({"id": f"{r['id']}{SEP}{role.rid}", "text": render(m)})
            variants[role.rid] = f"MR role {role.rid}: {role.operation}"

    def write(name: str, items: list[dict]) -> dict:
        d = a.out / name
        d.mkdir()
        body = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in items)
        (d / "prompts.jsonl").write_text(body, encoding="utf-8")
        lens = [len(tok(x["text"], add_special_tokens=False)["input_ids"]) for x in items]
        return {"dir": str(d), "rows": len(items), "prompts_sha256": h(body),
                "prompt_tokens_max": max(lens), "prompt_tokens_p50": sorted(lens)[len(lens) // 2]}

    # Pass A: every solver call, one file, one model load, ordinary allowance.
    pass_a = write("solver", solver)
    # Pass B: T. Same prompts as `direct`, different allowance, so a separate generation.
    t_rows = [{"id": f"{x['id'].split(SEP)[0]}{SEP}T", "text": x["text"]}
              for x in solver if x["id"].endswith(f"{SEP}direct")]
    pass_b = write("tokenmatch", t_rows)

    mr_total = len(MR_SOLVER_ROLES) * ROLE_TOKENS + SELECTOR_TOKENS
    meta = {
        "schema": 1, "record": "DMRF_VARIANT_RENDER",
        "instrument": "DMRF-SCOUT-001", "split": a.split, "items": len(rows),
        "tokenizer": str(a.tokenizer), "transformers": transformers.__version__,
        "enable_thinking": False,
        "must_be_4_51_3": ("The qualified probes pin transformers 4.51.3. enable_thinking=False "
                            "is load-bearing for this family: without it the model opens its own "
                            "think block and spends the allowance reasoning. Measured -- both "
                            "cores scored 0/24 on the first CAL pass for exactly that reason."),
        "variants": variants,
        "passes": {
            "solver": {**pass_a, "max_new_tokens": ROLE_TOKENS,
                       "covers": "direct (= O, REP[0], MR R0), rep1-3, R1-3"},
            "tokenmatch": {**pass_b, "max_new_tokens": mr_total,
                           "covers": "T",
                           "why": ("same prompt as direct at MR's TOTAL output allowance "
                                   f"({len(MR_SOLVER_ROLES)} x {ROLE_TOKENS} + "
                                   f"{SELECTOR_TOKENS} = {mr_total}); token-allowance matching, "
                                   "NOT compute matching")},
            "selector": {"rendered_later": ("selector prompts depend on the candidates, so they "
                                            "are built by render_selector.py after this pass"),
                         "max_new_tokens": SELECTOR_TOKENS},
        },
        "registered_reuse": {
            "shared_generation": "direct",
            "serves": ["O", "REP[0]", "MR R0"],
            "identical_on": ["checkpoint", "prompt text", "input ids", "decoding", "seed",
                             "output limit", "output policy"],
            "asserted_by": "tests/test_inference.py (byte-identity of the three builds)",
        },
        "call_accounting": {
            "logical_deployment_calls_per_item": {"O": 1, "T": 1, "REP": 5, "MR": 5},
            "actual_deduplicated_calls_per_item": 10,
            "breakdown": "1 direct + 3 paraphrases + 3 roles + 1 T + 2 selectors",
            "note": ("Both figures are reported. The deduplicated count is what was spent; the "
                     "logical count is what a deployment would pay."),
        },
        "role_roster": role_roster(),
        "decoding": "greedy, do_sample=False, TF32 off",
    }
    (a.out / "VARIANTS.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(rows), "solver_rows": pass_a["rows"],
                      "tokenmatch_rows": pass_b["rows"],
                      "solver_prompt_tokens_max": pass_a["prompt_tokens_max"],
                      "T_max_new_tokens": mr_total,
                      "variants": sorted(variants)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
