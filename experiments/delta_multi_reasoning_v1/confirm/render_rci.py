"""Render an RCI-001 bank to prompt strings, and ENFORCE the token ceiling.

Rendering runs under the probes' pinned transformers 4.51.3 with `enable_thinking=False`, for
the two reasons this programme has already paid for: a different transformers version puts a
differently-rendered prompt in front of the model than the qualified table was measured with,
and without `enable_thinking=False` this family opens a think block and spends the whole
allowance reasoning (measured: 0/24 on a first CAL pass, while reasoning correctly).

Requirement 4 of `CONFIRMATION_INSTRUMENT_GAP.md` is enforced here rather than hoped for: every
rendered prompt must be <= 2048 tokens, and the script REFUSES rather than truncating. A
confirmation instrument that silently drops context is not measuring long context.

Refusing to render the SEALED bank is deliberate and is the point of the whole design: this
script will render it only with an explicit `--i-am-spending-the-seal` flag, which also
requires the read-budget record to be updated first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_PROMPT_TOKENS = 2048


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instrument", default="RCI-001",
                    help="recorded in RENDER.json; set it for successor banks")
    ap.add_argument("--i-am-spending-the-seal", action="store_true",
                    dest="spend_seal",
                    help="required to render RCI_SEALED; see RCI_BUILD.json read_budget")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    rows = [json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    parts = {r["partition"] for r in rows}
    if "SEALED" in parts:
        if not a.spend_seal:
            raise SystemExit(
                "REFUSING to render the SEALED bank.\n"
                "This bank carries a read budget of ONE, to be used after a candidate is frozen "
                "and never for selection. Rendering it here would spend it.\n"
                "If this really is the registered confirmation read, claim the budget first:\n"
                "  python confirm/seal_rci.py --root <dir> --claim-read '<reason>'\n"
                "then re-run with --i-am-spending-the-seal.")
        # The claim must ALREADY be recorded. Ordering is the whole point: RCI-001 incremented
        # the counter after rendering, leaving a window in which the record said `unspent`
        # while a read was in flight, and an audit that arrived in that window was too late.
        seal_p = a.bank.parent / "RCI_SEAL.json"
        if not seal_p.exists():
            raise SystemExit(f"REFUSING: no seal record at {seal_p}; this bank is not sealed")
        rb = json.loads(seal_p.read_text(encoding="utf-8"))["read_budget"]
        if rb["reads_used"] < 1:
            raise SystemExit(
                "REFUSING: the read budget has NOT been claimed.\n"
                f"  reads_used = {rb['reads_used']}\n"
                "Claim it BEFORE rendering, so the record can never say `unspent` while a read "
                "is in flight:\n"
                "  python confirm/seal_rci.py --root <dir> --claim-read '<reason>'")

    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True)

    out, lens, over = [], [], []
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        lens.append(n)
        if n > MAX_PROMPT_TOKENS:
            over.append((r["id"], n))
        out.append({"id": r["id"], "text": text})

    if over:
        raise SystemExit(
            f"REFUSING: {len(over)} rows exceed the {MAX_PROMPT_TOKENS}-token ceiling "
            f"(requirement 4), e.g. {over[:3]}. Not truncating -- a confirmation instrument "
            f"that drops context is not measuring long context. Shorten the generator instead.")

    body = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in out)
    (a.out / "prompts.jsonl").write_text(body, encoding="utf-8")
    meta = {
        "schema": 1, "record": "RCI_RENDER", "instrument": a.instrument,
        "partitions": sorted(parts), "rows": len(out),
        "tokenizer": str(a.tokenizer), "transformers": transformers.__version__,
        "enable_thinking": False, "add_generation_prompt": True,
        "max_prompt_tokens_allowed": MAX_PROMPT_TOKENS,
        "prompt_tokens": {"min": min(lens), "p50": sorted(lens)[len(lens) // 2],
                          "max": max(lens)},
        "headroom_at_4096": 4096 - max(lens),
        "rows_over_ceiling": 0,
        "prompts_sha256": h(body),
        "seal_spent_by_this_render": bool("SEALED" in parts),
        "model_calls": 0,
    }
    (a.out / "RENDER.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in
                      ("partitions", "rows", "transformers", "prompt_tokens",
                       "headroom_at_4096", "prompts_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
