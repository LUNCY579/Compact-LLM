"""Held-out Opallo-class reply-content NLL across the full arm matrix.

This is the O-side quantity that remains definable after the arithmetic
specificity audit: 6,741 never-consumed rows from the 24 reply classes touched
by Opallo's 41-row training draw. It is teacher-forced reply-content NLL, not
free generation and not a release/quality score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parent
FRESH = ROOT / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FRESH))
TRAIN = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/outputs/splits_large/train.jsonl")
ARM_DOSE = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/evidence/retention-design-v1/ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json")
ARMS = ["BASE", "SPARSE", "OPALLO", "AVG", "ADD-S75", "ADD-O75", "ADD-75", "ADD-100", "CONFLICT-LOCAL"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_reply_text(text: str) -> tuple[str, str]:
    prefix, final = text.rsplit("<|im_start|>assistant\n", 1)
    final = final.split("<|im_end|>", 1)[0]
    if "</think>\n\n" in final:
        answer = final.split("</think>\n\n", 1)[1]
    else:
        answer = final
    prompt = prefix + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt, answer


def make_rows() -> list[dict]:
    train = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    arm = json.loads(ARM_DOSE.read_text(encoding="utf-8"))
    consumed = set(arm["train_row_ids_consumed"])
    classes = {r["reply_class"] for r in train if r["id"] in consumed}
    rows = []
    for r in train:
        if r["id"] in consumed or r["reply_class"] not in classes:
            continue
        prompt, answer = parse_reply_text(r["text"])
        rows.append({"id": r["id"], "reply_class": r["reply_class"],
                     "prompt": prompt, "answer": answer})
    if len(rows) != 6741:
        raise RuntimeError(f"held-out row count changed: {len(rows)}")
    return rows


def score_arm(name, rows, model_dir, torch, AutoModelForCausalLM, AutoTokenizer):
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, local_files_only=True,
        low_cpu_mem_usage=True, device_map={"": "cuda"})
    model.eval()
    device = next(model.parameters()).device
    scored = []
    for ix, row in enumerate(rows, 1):
        pids = tok(row["prompt"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        tids = tok(row["answer"], add_special_tokens=False)["input_ids"]
        if not tids:
            continue
        ids = torch.tensor([pids.tolist() + tids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
        pos = torch.arange(pids.numel() - 1, pids.numel() + len(tids) - 1, device=device)
        targets = torch.tensor(tids, dtype=torch.long, device=device)
        selected = logits[pos]
        logp = torch.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
        gold = selected.gather(1, targets[:, None]).squeeze(1)
        other = selected.clone()
        other[torch.arange(other.shape[0], device=device), targets] = -torch.inf
        z = gold - torch.logsumexp(other, dim=-1)
        scored.append({"id": row["id"], "reply_class": row["reply_class"],
                       "tokens": len(tids), "nll": float(-logp.sum().item()),
                       "mean_nll": float(-logp.mean().item()),
                       "logodds": float(z.clamp(-10, 10).sum().item()),
                       "mean_logodds": float(z.clamp(-10, 10).mean().item())})
        if ix % 500 == 0:
            print(json.dumps({"arm": name, "rows": ix, "seconds": round(time.time() - started, 1)}), flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    by_class = {}
    for fam in sorted({r["reply_class"] for r in scored}):
        sub = [r for r in scored if r["reply_class"] == fam]
        by_class[fam] = {"rows": len(sub), "tokens": sum(r["tokens"] for r in sub),
                         "nll": sum(r["nll"] for r in sub) / sum(r["tokens"] for r in sub),
                         "logodds": sum(r["logodds"] for r in sub) / sum(r["tokens"] for r in sub)}
    return {"arm": name, "seconds": round(time.time() - started, 1),
            "rows": len(scored), "tokens": sum(r["tokens"] for r in scored),
            "nll_per_token": sum(r["nll"] for r in scored) / sum(r["tokens"] for r in scored),
            "logodds_per_token": sum(r["logodds"] for r in scored) / sum(r["tokens"] for r in scored),
            "by_class": by_class, "rows_detail": scored}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = make_rows()
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arm_hashes = {}
    for name in ARMS:
        arm_hashes[name] = json.loads((EXPERIMENT / "arms" / name / "BUILD.json").read_text())["weight_sha256"]
    result = {
        "schema": 1, "instrument": "OPALLO-HELDOUT-REPLY-CONTENT-NLL-001",
        "source_train_sha256": sha256(TRAIN), "source_train_rows": 10000,
        "consumed_arm_dose_sha256": sha256(ARM_DOSE), "heldout_rows": len(rows),
        "heldout_definition": "train rows outside Opallo's 41 consumed ids, restricted to its 24 reply classes",
        "arm_weight_hashes": arm_hashes,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
                    "dtype": "float32", "tf32": False, "use_cache": False,
                    "readout": "teacher-forced final assistant reply tokens"},
        "rows": len(rows), "results": [], "quality_claim": False, "release_permitted": False,
    }
    from gpu_lease import lease
    with lease(label="opallo-heldout-reply-nll", job="6741 heldout rows x 9 arms"):
        for name in ARMS:
            print(f"scoring {name}", flush=True)
            result["results"].append(score_arm(name, rows, EXPERIMENT / "arms" / name,
                                                 torch, AutoModelForCausalLM, AutoTokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
