"""Run stage2_generate in fixed-size chunks, each a fresh process, and merge the results.

WHY
---
`stage2_generate.py` leaks across rows on this host. Measured, by bisection, all at 7449 MiB
free VRAM and with the model loading fine:

    8 rows   OK        64 rows        OK
    16 rows  OK        128 rows       CUDA error: out of memory
    32 rows  OK        256 rows       CUDA error: out of memory
    64 rows  OK        896 rows       OOM, then "unknown error" on the next system

Two controls rule out the obvious alternatives. The eight LONGEST prompts in the bank (3452
chars) run fine on their own, so it is not prompt length. Rows 64:128 run fine as their own
file, so there is no single poisoned item -- the same rows that fail as part of 0:128 succeed
as 64:128. The failure is a function of how many rows one process has already done.

It is not my script to fix: `stage2_generate.py` belongs to the QWEN-001 lane and another
session depends on it. Editing it would also break the one thing that makes it worth using --
QWEN-001's path control, which reproduced all six qualified Supa-family values exactly on the
version as it stands. So this wraps it instead, at a chunk size well inside the measured safe
range.

WHAT CHUNKING DOES AND DOES NOT CHANGE
--------------------------------------
Each chunk is a fresh process with the SAME model, prompts, decoding, dtype, token limit and
context limit. The only difference is process boundaries, and a fresh process is exactly what
clears the leaked state. Nothing about the generation contract changes, so the merged output is
the output the unchunked run would have produced had it survived -- which is the claim this
file has to earn, and it earns it by recording the chunk boundaries and per-chunk receipts so
the merge can be audited rather than trusted.

The merged GENERATE.json aggregates faithfully: rows and terminated_rows summed, overflow
skips summed, wall_seconds summed, generated-token statistics RECOMPUTED from the merged
predictions rather than averaged from the chunks' summaries.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: Measured safe ceiling is between 64 and 128 rows. 64 is the largest proven-good size.
DEFAULT_CHUNK = 64


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator", required=True, help="path to stage2_generate.py")
    ap.add_argument("--python", required=True, help="interpreter for the generator")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True, dest="max_new_tokens")
    ap.add_argument("--max-context", type=int, default=4096, dest="max_context")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    a = ap.parse_args(argv)

    rows = [l for l in a.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no prompt rows in {a.prompts}")
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    work = a.out / "_chunks"
    work.mkdir()

    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    print(f"{a.label}: {len(rows)} rows in {len(chunks)} chunks of <= {a.chunk}", flush=True)

    merged: list[dict] = []
    receipts: list[dict] = []
    for ci, chunk in enumerate(chunks):
        cdir = work / f"c{ci:03d}"
        pf = work / f"c{ci:03d}.jsonl"
        pf.write_text("".join(x + "\n" for x in chunk), encoding="utf-8")
        r = subprocess.run(
            [a.python, a.generator, "--model", a.model, "--label", f"{a.label}_c{ci:03d}",
             "--prompts", str(pf), "--out", str(cdir),
             "--max-new-tokens", str(a.max_new_tokens), "--max-context", str(a.max_context),
             "--dtype", a.dtype],
            capture_output=True, text=True)
        pred = cdir / "predictions.jsonl"
        if r.returncode != 0 or not pred.exists() or not pred.stat().st_size:
            tail = (r.stderr or r.stdout or "")[-600:]
            print(f"CHUNK {ci} FAILED (rc={r.returncode}):\n{tail}", flush=True)
            raise SystemExit(f"chunk {ci} of {len(chunks)} failed; nothing merged")
        got = [json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(got) != len(chunk):
            raise SystemExit(
                f"chunk {ci} returned {len(got)} predictions for {len(chunk)} prompts; "
                f"refusing to merge a partial chunk")
        merged.extend(got)
        gm = json.loads((cdir / "GENERATE.json").read_text(encoding="utf-8"))
        receipts.append({"chunk": ci, "rows": len(chunk),
                         "wall_seconds": gm.get("wall_seconds"),
                         "rows_skipped_context_overflow": gm.get("rows_skipped_context_overflow"),
                         "terminated_rows": gm.get("terminated_rows")})
        print(f"  chunk {ci + 1}/{len(chunks)}: {len(got)} rows, "
              f"{gm.get('wall_seconds')}s", flush=True)

    ids = [m["id"] for m in merged]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids after merge; refusing to write")
    want = [json.loads(x)["id"] for x in rows]
    if ids != want:
        raise SystemExit("merged id ORDER differs from the prompts file; refusing to write")

    (a.out / "predictions.jsonl").write_text(
        "".join(json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" for m in merged),
        encoding="utf-8")

    toks = sorted(m.get("generated_tokens") or 0 for m in merged)
    last = json.loads((work / f"c{len(chunks) - 1:03d}" / "GENERATE.json").read_text(encoding="utf-8"))
    gen = {
        **{k: last[k] for k in ("model", "max_new_tokens", "max_context", "dtype", "offload",
                                "placement", "eos_ids", "decoding", "numerics", "transformers",
                                "serving_path") if k in last},
        "label": a.label,
        "rows": len(merged),
        "rows_skipped_context_overflow": sum(r["rows_skipped_context_overflow"] or 0
                                             for r in receipts),
        "terminated_rows": sum(r["terminated_rows"] or 0 for r in receipts),
        "wall_seconds": round(sum(r["wall_seconds"] or 0.0 for r in receipts), 1),
        "generated_tokens": {"min": toks[0], "p50": toks[len(toks) // 2], "max": toks[-1]},
        "prompts_file": str(a.prompts),
        "quality_claim": False, "release_permitted": False,
        "chunked_execution": {
            "chunk_size": a.chunk, "chunks": len(chunks),
            "why": ("stage2_generate.py leaks across rows on this host: 64 rows succeed and 128 "
                    "fail at 7449 MiB free VRAM. The eight longest prompts succeed alone and "
                    "rows 64:128 succeed as their own file, so it is neither prompt length nor "
                    "a poisoned item. Each chunk is a fresh process with identical model, "
                    "prompts, decoding, dtype and limits; only process boundaries differ."),
            "generator_unmodified": True,
            "per_chunk": receipts,
            "generated_token_stats": "RECOMPUTED from merged predictions, not averaged",
            "merge_checks": ["no duplicate ids", "merged id order equals the prompts file order",
                            "every chunk returned exactly as many predictions as prompts"],
        },
    }
    (a.out / "GENERATE.json").write_text(json.dumps(gen, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({k: gen[k] for k in ("label", "rows", "wall_seconds",
                                          "rows_skipped_context_overflow", "terminated_rows")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
