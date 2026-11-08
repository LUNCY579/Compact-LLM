"""Exploratory CUDA probe for the isolated skill-vector arms.

The suite is newly authored for this experiment. This runner makes no claim of
formal qualification, does not read old evaluation banks, and writes only under
the new experiment directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def norm(value: str) -> str:
    return " ".join(value.lower().strip().strip(".,!?;:()[]{}").split())


def load_suite(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 24 or {r["bank"] for r in rows} != {"S", "O", "SxO"}:
        raise ValueError("fresh probe must contain 24 rows across S, O, and SxO")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate probe id")
    return rows


def prompt_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-root", type=Path, required=True)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this bounded probe")
    rows = load_suite(args.suite)
    arms = sorted(p for p in args.arms_root.iterdir() if p.is_dir())
    if len(arms) != 9:
        raise ValueError(f"expected nine arms, found {len(arms)}")
    result = {
        "schema": 1,
        "experiment": "TITAN-OPALLO-SKILL-VECTOR-001",
        "evidence_kind": "fresh_exploratory_functional_probe",
        "suite": str(args.suite.resolve()),
        "suite_sha256": __import__("hashlib").sha256(args.suite.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "dtype": "float32",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": 1,
            "thinking_disabled": True,
        },
        "arms": [],
        "native_authorized": "exploratory_local_probe_only",
        "quality_claim": False,
        "release_permitted": False,
    }
    for arm_dir in arms:
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(arm_dir), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(arm_dir), torch_dtype=torch.float32, local_files_only=True,
            low_cpu_mem_usage=True, device_map={"": "cuda"},
        )
        model.eval()
        device = next(model.parameters()).device
        predictions = []
        for row in rows:
            text = prompt_text(tokenizer, row["prompt"])
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False,
                                           max_new_tokens=args.max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.eos_token_id)
            prompt_len = encoded["input_ids"].shape[1]
            answer = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
            normalized = norm(answer)
            correct = any(norm(a) in normalized for a in row["answers"])
            predictions.append({"id": row["id"], "bank": row["bank"],
                                "answer": answer, "correct": correct})
        counts = defaultdict(lambda: [0, 0])
        for pred in predictions:
            counts[pred["bank"]][0] += int(pred["correct"])
            counts[pred["bank"]][1] += 1
        result["arms"].append({
            "arm": arm_dir.name,
            "seconds": time.time() - started,
            "by_bank": {key: {"correct": val[0], "rows": val[1],
                               "rate": val[0] / val[1]} for key, val in sorted(counts.items())},
            "predictions": predictions,
        })
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"arm": arm_dir.name, "by_bank": result["arms"][-1]["by_bank"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
