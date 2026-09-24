"""SMALLVD-001 gate G0b -- the pinned teacher loads through THIS track's runtime and reproduces a
recorded baseline, row for row.

Reference: qwen001-20260918/gen-supa_process/predictions.jsonl (384 SELECT-VIEWS rows, greedy,
fp32, max_new_tokens 96), which the teacher-lineage scout found TEXT-IDENTICAL on 384/384 rows to
the qualified DenseReference output (PROCESS_sv001_supa.json, score 14). The prompts are the
pre-rendered strings from qwen001-20260918/render-process/prompts.jsonl, so no chat template is
rendered here and the transformers version cannot leak into the prompt.

This is a LOAD CANARY, not an evaluation: SELECT-VIEWS is a development instrument, nothing is
selected on it, and the only quantity read is per-row text identity with a recorded file.
Batch size 1, like the reference, so no padding enters the comparison.
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

REC = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
PROMPTS = REC / "qwen001-20260918/render-process/prompts.jsonl"
REFERENCE = REC / "qwen001-20260918/gen-supa_process/predictions.jsonl"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit("create-only output exists")
    hostbudget.require(3.6, "SMALLVD-001 G0b")
    token = gpu_lease.acquire(timeout_s=3600, label="smallvd001:g0b")
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        rows = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
        ref = {r["id"]: r for r in map(json.loads, REFERENCE.read_text(encoding="utf-8").splitlines())}
        tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
        t0 = time.time()
        # device_map="cuda" streams tensors straight to the card, so the host never holds an fp32
        # copy of the model (host RAM, not VRAM, is the binding resource on this machine).
        model = Qwen3ForCausalLM.from_pretrained(str(a.teacher), torch_dtype=torch.float32,
                                                 attn_implementation="eager", device_map="cuda",
                                                 local_files_only=True).eval()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        gpu_lease.set_workers(token, [os.getpid()])
        load_s = time.time() - t0
        eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                      tok.convert_tokens_to_ids("<|endoftext|>")})
        per_row, match = [], 0
        t1 = time.time()
        for i, r in enumerate(rows, 1):
            if i % 32 == 0:
                hostbudget.guard(1.5, "SMALLVD-001 G0b")
            ids = tok(r["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
                                     eos_token_id=eos, pad_token_id=tok.eos_token_id,
                                     temperature=None, top_p=None, top_k=None)
            new = [int(t) for t in out[0][ids.shape[-1]:]]
            body = [t for t in new if t not in eos]
            pred = tok.decode(body, skip_special_tokens=True)
            same = pred == ref[r["id"]]["prediction"]
            match += same
            per_row.append({"id": r["id"], "match": same, "prediction": pred,
                            "generated_tokens": len(body)})
            if i % 64 == 0:
                print(f"  {i}/{len(rows)} match {match}/{i} | {time.time() - t1:.0f}s", flush=True)
        res = {
            "schema": 1, "record": "SMALLVD_G0B_BASELINE_REPRODUCTION", "experiment": "SMALLVD-001",
            "teacher_dir": str(a.teacher), "teacher_weight_sha256": sha256_file(a.teacher / "model.safetensors"),
            "prompts": str(PROMPTS), "prompts_sha256": sha256_file(PROMPTS),
            "reference": str(REFERENCE), "reference_sha256": sha256_file(REFERENCE),
            "rows": len(rows), "text_identical_rows": match, "all_identical": match == len(rows),
            "mismatched_ids": [p["id"] for p in per_row if not p["match"]],
            "decoding": {"greedy": True, "max_new_tokens": a.max_new_tokens, "batch": 1,
                         "dtype": "fp32", "attn_implementation": "eager", "tf32": False},
            "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0)},
            "load_seconds": round(load_s, 1), "wall_seconds": round(time.time() - t1, 1),
            "per_row": per_row, "quality_claim": False, "release_permitted": False,
        }
        with open(a.out, "x", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, ensure_ascii=False, allow_nan=False)
        print(f"G0b: {match}/{len(rows)} rows text-identical to the recorded baseline")
        return 0 if match == len(rows) else 1
    finally:
        gpu_lease.release(token)


if __name__ == "__main__":
    raise SystemExit(main())
