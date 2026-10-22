"""TRISKILL-001 C scorer: generate on one C bank for one arm and write SIF-format predictions.

    python score_c.py --arm-dir <dir> --bank {dev,cal,cfresh} --out <predictions.jsonl> [--no-lease]

Decoding is Phase 0's exactly: screen_source's self-tested render, the tokenizer's EOS, greedy,
FP32, TF32 off, batch 1, max_new_tokens 48. The output format is byte-identical to Phase 0's unit
files: one {"id","prompt_sha256","completion"} per line, json.dumps(ensure_ascii=False), LF.
BASE's DEV predictions can therefore be compared to Phase 0 byte for byte (the Stage A canary).
Scoring is SIF-001's own score_records, applied by the analysis, never here. Writes are atomic.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SIF = Path(r"C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1")
C_FRESH = HERE / "C_FRESH.jsonl"
C_FRESH_SHA = "52f921fc713898e06edf0ebf0241d89e1dd29e331d4c2ad65a094a79497d7ae3"
MAX_NEW = 48
sys.path.insert(0, str(EXP / "fresh_dual_skill_instrument_v1"))
sys.path.insert(0, str(HERE))
from score_arms import render, template_selftest  # noqa: E402


def load(bank: str):
    """Return (inputs, keys): inputs carry `messages`; keys are SIF key records for score_records."""
    if bank in ("dev", "cal"):
        inputs = [json.loads(l) for l in (SIF / "data/eval" / bank / "atomic_C.inputs.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
        keys = [json.loads(l) for l in gzip.open(SIF / "private_oracles" / bank / "atomic_C.keys.jsonl.gz", "rt", encoding="utf-8")]
    elif bank == "cfresh":
        if hashlib.sha256(C_FRESH.read_bytes()).hexdigest() != C_FRESH_SHA:
            raise SystemExit("C_FRESH.jsonl does not match its committed sha256")
        rows = [json.loads(l) for l in C_FRESH.read_text(encoding="utf-8").splitlines() if l.strip()]
        inputs = [{"id": r["id"], "messages": r["messages"]} for r in rows]
        keys = [{k: r[k] for k in ("id", "prompt_sha256", "answer", "response_format", "parent_id", "bank",
                                    "variant", "required_skills", "paired_with")} for r in rows]
    else:
        raise SystemExit(f"unknown bank {bank}")
    kid = {k["id"]: k for k in keys}
    if set(kid) != {r["id"] for r in inputs}:
        raise SystemExit(f"{bank}: inputs and keys differ")
    for r in inputs:
        if hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest() != kid[r["id"]]["prompt_sha256"]:
            raise SystemExit(f"{bank}: prompt hash mismatch {r['id']}")
    return inputs, keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--bank", required=True, choices=("dev", "cal", "cfresh"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-lease", action="store_true", help="caller holds the GPU lease")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    template_selftest()
    inputs, _ = load(a.bank)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def run():
        tok = AutoTokenizer.from_pretrained(str(a.arm_dir), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(a.arm_dir), torch_dtype=torch.float32, local_files_only=True,
                                                     low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("not FP32")
        preds = []
        for r in inputs:
            text = render(r["messages"][0]["content"], r["messages"][1]["content"])
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append({"id": r["id"], "prompt_sha256": hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest(),
                          "completion": gen})
        return preds

    if a.no_lease:
        preds = run()
    else:
        from gpu_lease import lease
        with lease(label="triskill001-score-c", job=f"C {a.bank} x 1 arm"):
            preds = run()
    tmp = a.out.with_suffix(".jsonl.tmp")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8", newline="\n")
    tmp.replace(a.out)
    print(f"wrote {a.out} ({len(preds)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
