"""Exploratory within-class mismatched-reply specificity control."""
from __future__ import annotations
import argparse, gc, hashlib, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "fresh_dual_skill_instrument_v1"))
from o_reply_nll import make_rows, EXPERIMENT, sha256  # noqa: E402

ARMS = ["BASE", "SPARSE", "OPALLO", "ADD-O75", "ADD-100"]


def mismatch_rows(rows):
    by = {}
    for r in rows:
        by.setdefault(r["reply_class"], []).append(r)
    out = []
    for r in rows:
        choices = by[r["reply_class"]]
        pick = next((x for x in choices if x["answer"] != r["answer"]), None)
        if pick is None:
            pick = choices[(choices.index(r) + 1) % len(choices)]
        out.append({**r, "mismatch_answer": pick["answer"], "mismatch_id": pick["id"]})
    return out


def score_arm(name, rows, model_dir, torch, AutoModelForCausalLM, AutoTokenizer):
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32,
        local_files_only=True, low_cpu_mem_usage=True, device_map={"": "cuda"})
    model.eval(); device = next(model.parameters()).device
    scored = []
    for ix, r in enumerate(rows, 1):
        pids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        tids = tok(r["mismatch_answer"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pids.tolist() + tids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
        pos = torch.arange(pids.numel() - 1, pids.numel() + len(tids) - 1, device=device)
        targets = torch.tensor(tids, dtype=torch.long, device=device)
        lp = torch.log_softmax(logits[pos], dim=-1).gather(1, targets[:, None]).squeeze(1)
        scored.append({"id": r["id"], "reply_class": r["reply_class"],
                       "mismatch_id": r["mismatch_id"], "tokens": len(tids),
                       "nll": float(-lp.sum().item()), "mean_nll": float(-lp.mean().item())})
        if ix % 1000 == 0:
            print(json.dumps({"arm": name, "rows": ix, "seconds": round(time.time() - started, 1)}), flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return {"arm": name, "seconds": round(time.time() - started, 1), "rows": scored}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
    rows = mismatch_rows(make_rows())
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    result = {"schema": 1, "instrument": "OPALLO-HELDOUT-REPLY-MISMATCH-CONTROL-001",
              "rows": len(rows), "arms": ARMS, "source_train_sha256": sha256(Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/outputs/splits_large/train.jsonl")),
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0), "dtype": "float32", "tf32": False, "use_cache": False},
              "results": [], "quality_claim": False, "release_permitted": False}
    from gpu_lease import lease
    with lease(label="opallo-heldout-mismatch", job="6741 mismatched replies x 5 arms"):
        for name in ARMS:
            print(f"scoring {name}", flush=True)
            result["results"].append(score_arm(name, rows, EXPERIMENT / "arms" / name, torch, AutoModelForCausalLM, AutoTokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__": main()
