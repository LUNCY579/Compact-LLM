"""Render DMRF-SCOUT-001 chat messages to prompt strings for `stage2_generate.py`.

Rendering is deliberately a separate stage, and it MUST run under the probes' pinned
transformers 4.51.3, for the reason QWEN-001's stage 1 records: if the chat template were
applied by a different transformers version than the qualified probes use, that version
difference would leak into the prompt text and every comparison to the existing table would be
against differently-rendered inputs. The generator can then be any version; the prompts cannot.

Emits `prompts.jsonl` ({"id", "text"}) plus a RENDER.json recording the template identity and
a hash over the rendered text, so a later run can prove it used the same prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", type=Path, required=True, help="SCOUT.jsonl")
    ap.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    ap.add_argument("--tokenizer", required=True, help="a directory with the family tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite an existing render: {a.out}")

    import transformers
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in a.scout.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    if not rows:
        raise SystemExit(f"no rows for split {a.split}")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True)

    out_rows, lengths = [], []
    for r in rows:
        # enable_thinking=False is NOT optional. It emits a pre-closed `<think>\n\n</think>`
        # block into the generation prefix, which is how this family is prompted everywhere in
        # this programme -- the prefill picks the container and the model emits the bare answer.
        # Without it the model opens its own <think> and spends the whole token allowance
        # reasoning: on the first CAL pass both cores scored 0/24 for exactly this reason while
        # reasoning correctly inside the think block.
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        out_rows.append({"id": r["id"], "text": text})

    body = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in out_rows)
    (a.out / "prompts.jsonl").write_text(body, encoding="utf-8")

    meta = {
        "schema": 1, "record": "DMRF_SCOUT_RENDER",
        "instrument": "DMRF-SCOUT-001", "split": a.split, "rows": len(out_rows),
        "tokenizer": str(a.tokenizer),
        "transformers": transformers.__version__,
        "must_be_4_51_3": ("The qualified probes pin transformers 4.51.3. Rendering under any "
                           "other version puts a different prompt string in front of the model "
                           "than the existing table was measured with."),
        "add_generation_prompt": True,
        "prompt_tokens": {"min": min(lengths), "max": max(lengths),
                          "p50": sorted(lengths)[len(lengths) // 2]},
        "max_context_headroom_at_4096": 4096 - max(lengths),
        "prompts_sha256": h(body),
        "model_calls": 0,
    }
    (a.out / "RENDER.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in
                      ("rows", "transformers", "prompt_tokens",
                       "max_context_headroom_at_4096", "prompts_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
