"""Render a probe's chat prompts to the flat `{id, text}` file the generator consumes.

    python probe/render_prompts.py --bank <records>/pres002-20260919/PRES2_PROBE.jsonl \
        --tokenizer <supa> --out <records>/pres002-20260919/render \
        [--verify-against <records>/alias001-20260918/render/prompts.jsonl]

**Closes the second half of the same reproducibility gap `probe/generate_alias.py` closed.**
ALIAS-001's `render/RENDER.json` records what was done but the script that did it was never
committed, so the prompts existed without their producer.

## The one setting that matters

`enable_thinking=False`. This family emits a `<think>\\n\\n</think>` block otherwise, and the
programme has already recorded that as an operational trap. The rendered tail must end

    <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

and this script **asserts** that rather than trusting the template, because a tokenizer upgrade
that changed it would silently alter every prompt while leaving the manifest looking identical.

## Why `--verify-against` exists

Pointed at an existing prompts file from a bank this script did not build, it re-renders that
bank's rows and checks the text is byte-identical. That is how `generate_alias.py` was shown to
reproduce the frozen source's chunk, and it is the same idea one layer up: a renderer that can
reproduce ALIAS-001's prompts byte-for-byte is rendering PRES-002's the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--verify-against", type=Path, default=None)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = read_jsonl(a.bank)

    out, lens = [], []
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        if not text.endswith(TAIL):
            raise SystemExit(f"rendered tail is not the expected no-think tail for {r['id']}; "
                             f"got {text[-80:]!r} -- refusing to write")
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        lens.append(n)
        out.append({"id": r["id"], "text": text})

    if a.verify_against is not None:
        ref = {x["id"]: x["text"] for x in read_jsonl(a.verify_against)}
        mine = {x["id"]: x["text"] for x in out}
        shared = [i for i in mine if i in ref]
        if not shared:
            raise SystemExit("no overlapping ids to verify against")
        diff = [i for i in shared if ref[i] != mine[i]]
        print(f"RENDER VERIFICATION on {len(shared)} shared rows")
        print(f"  identical: {len(shared) - len(diff)}/{len(shared)}")
        if diff:
            print(f"  DIVERGENT: {len(diff)}  e.g. {diff[:3]}")
            i = diff[0]
            print(f"    ref  tail {ref[i][-70:]!r}")
            print(f"    mine tail {mine[i][-70:]!r}")
            print("\nVERDICT: NOT REPRODUCED")
            return 2
        print("\nVERDICT: REPRODUCED byte-for-byte")
        return 0

    over = [i for i, n in enumerate(lens) if n > a.max_prompt_tokens]
    if over:
        raise SystemExit(f"{len(over)} prompts exceed {a.max_prompt_tokens} tokens")

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    s = sorted(lens)
    man = {
        "schema": 1, "record": "PROBE_RENDER", "bank": str(a.bank), "rows": len(out),
        "tokenizer": a.tokenizer, "transformers": transformers.__version__,
        "enable_thinking": False, "add_generation_prompt": True,
        "no_think_tail_asserted": True,
        "max_prompt_tokens_allowed": a.max_prompt_tokens,
        "prompt_tokens": {"min": s[0], "p50": s[len(s) // 2], "max": s[-1]},
        "headroom_at_4096": 4096 - s[-1],
        "rows_over_ceiling": 0,
        "quality_claim": False, "release_permitted": False,
    }
    (a.out / "RENDER.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}  ({len(out)} rows, tokens {s[0]}/{s[len(s)//2]}/{s[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
