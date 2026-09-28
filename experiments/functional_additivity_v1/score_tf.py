"""FUNCTIONAL-ADDITIVITY-001 teacher-forced scorer.

This is the preregistered local function-space instrument.  It writes only into
this experiment directory and never changes the source arms or the prior
FRESH-DUAL-SKILL-001 receipts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRESH = HERE.parent / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FRESH))

import score_arms  # noqa: E402


EXPECTED_ARMS = ["BASE", "SPARSE", "OPALLO", "ADD-O75", "ADD-100"]
RECORDED_CAL = FRESH / "CALIBRATION_V2.json"
RECORDED_V2 = FRESH / "V2_ARMS.json"
EOS_ID = 151645


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recorded_sets(bank_rows: list[dict]) -> dict[str, dict[str, set[str]]]:
    refs = {row["id"]: row["reference"] for row in bank_rows}
    out: dict[str, dict[str, set[str]]] = {}
    for path in (RECORDED_CAL, RECORDED_V2):
        data = json.loads(path.read_text(encoding="utf-8"))
        for arm in data["arms"]:
            strict, tolerant = set(), set()
            for pred in arm["predictions"]:
                rid = pred["id"]
                if pred["generation"] == refs[rid]:
                    strict.add(rid)
                if pred["generation"] in (refs[rid], refs[rid] + "."):
                    tolerant.add(rid)
            out[arm["arm"]] = {"strict": strict, "period_tolerant": tolerant}
    return out


def logodds_and_logprob(logits, target_ids, positions):
    import torch
    selected = logits[positions]
    targets = selected.gather(1, target_ids[:, None]).squeeze(1)
    logsum_all = torch.logsumexp(selected, dim=1)
    masked = selected.clone()
    masked[torch.arange(masked.shape[0], device=masked.device), target_ids] = -torch.inf
    logsum_other = torch.logsumexp(masked, dim=1)
    z = targets - logsum_other
    lp = torch.log_softmax(selected, dim=1).gather(1, target_ids[:, None]).squeeze(1)
    p = lp.exp()
    argmax = selected.argmax(dim=1)
    return z, lp, p, argmax


def score_one(model, tok, row: dict):
    import torch
    rendered = score_arms.render(row["prompt"][0]["content"], row["prompt"][1]["content"])
    prompt_ids = tok(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    gold_ids = tok(row["reference"], add_special_tokens=False)["input_ids"]
    if not gold_ids:
        raise RuntimeError(f"empty gold tokenization: {row['id']}")
    ids = torch.tensor([list(prompt_ids.tolist()) + gold_ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    prompt_len = int(prompt_ids.numel())
    g = len(gold_ids)
    gold_pos = torch.arange(prompt_len - 1, prompt_len + g - 1, device=ids.device)
    gold_target = torch.tensor(gold_ids, dtype=torch.long, device=ids.device)
    eos_pos = torch.tensor([prompt_len + g - 1], dtype=torch.long, device=ids.device)
    eos_target = torch.tensor([EOS_ID], dtype=torch.long, device=ids.device)
    z, lp, p, argmax = logodds_and_logprob(logits, gold_target, gold_pos)
    ez, elp, ep, eargmax = logodds_and_logprob(logits, eos_target, eos_pos)
    zc = z.clamp(-10, 10)
    ezc = ez.clamp(-10, 10)
    strict = bool(torch.all(argmax == gold_target).item() and int(eargmax.item()) == EOS_ID)
    return {
        "id": row["id"], "family": row["family"],
        "content_len": max(0, g - 1), "gold_token_count": g,
        "gold_ids": gold_ids,
        "content": {
            "clipped_logodds": float(zc[:-1].sum().item()),
            "unclipped_logprob": float(lp[:-1].sum().item()),
            "token_logodds": [float(x) for x in zc[:-1].detach().cpu()],
            "token_logprob": [float(x) for x in lp[:-1].detach().cpu()],
            "token_probability": [float(x) for x in p[:-1].detach().cpu()],
            "argmax_ids": [int(x) for x in argmax[:-1].detach().cpu()],
        },
        "closing": {
            "clipped_logodds": float((zc[-1] + ezc[0]).item()),
            "unclipped_logprob": float((lp[-1] + elp[0]).item()),
            "token_logodds": [float(zc[-1].item()), float(ezc[0].item())],
            "token_logprob": [float(lp[-1].item()), float(elp[0].item())],
            "token_probability": [float(p[-1].item()), float(ep[0].item())],
            "argmax_ids": [int(argmax[-1].item()), int(eargmax.item())],
        },
        "strict_teacher_forced": strict,
        "gold_argmax_all": bool(torch.all(argmax == gold_target).item()),
        "eos_argmax": int(eargmax.item()),
    }


def score_arm(name: str, arm_dir: Path, rows: list[dict], torch, AutoModelForCausalLM, AutoTokenizer):
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(arm_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(arm_dir), torch_dtype=torch.float32, local_files_only=True,
        low_cpu_mem_usage=True, device_map={"": "cuda"})
    model.eval()
    records = [score_one(model, tok, row) for row in rows]
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return {"arm": name, "seconds": round(time.time() - started, 1), "rows": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=FRESH / "T_BANK_V2.json")
    ap.add_argument("--arms-root", type=Path, default=HERE.parent / "titan_opallo_skill_vector_v1" / "arms")
    ap.add_argument("--output", type=Path, default=HERE / "TF_RAW.json")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    print("startup: template self-test", flush=True)
    score_arms.template_selftest()
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    rows = bank["rows"]
    if len(rows) != 192:
        raise SystemExit(f"unexpected bank size: {len(rows)}")
    recorded = load_recorded_sets(rows)
    missing = set(EXPECTED_ARMS) - set(recorded)
    if missing:
        raise SystemExit(f"missing recorded canaries: {sorted(missing)}")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "schema": 1, "instrument": "FUNCTIONAL-ADDITIVITY-001",
        "bank": str(args.bank.resolve()), "bank_sha256": sha256(args.bank), "bank_n": len(rows),
        "code_sha256": sha256(Path(__file__)),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "device": torch.cuda.get_device_name(0), "dtype": "float32",
                    "tf32_matmul": False, "tf32_cudnn": False, "use_cache": False,
                    "tokenization": "score_arms.render; add_special_tokens=False",
                    "eos_id": EOS_ID, "sys_executable": sys.executable},
        "arms": [], "recorded_canary_sets": {
            arm: {k: sorted(v) for k, v in vals.items()} for arm, vals in recorded.items()
        },
        "gates": {}, "quality_claim": False, "release_permitted": False,
    }
    # G-A is completed before any model load.
    result["gates"]["G-A-template"] = {"passed": True, "detail": "score_arms.template_selftest passed"}
    from gpu_lease import lease
    with lease(label="functional-additivity-001", job="teacher-forced 5 arms x 192 rows"):
        for name in EXPECTED_ARMS:
            arm_dir = args.arms_root / name
            if not arm_dir.is_dir():
                raise SystemExit(f"missing arm: {arm_dir}")
            print(f"scoring {name}", flush=True)
            result["arms"].append(score_arm(name, arm_dir, rows, torch,
                                              AutoModelForCausalLM, AutoTokenizer))
            print(json.dumps({"arm": name, "seconds": result["arms"][-1]["seconds"]}), flush=True)
        sparse_dir = args.arms_root / "SPARSE"
        print("scoring SPARSE repeat for G-C", flush=True)
        result["sparse_repeat"] = score_arm("SPARSE_REPEAT", sparse_dir, rows, torch,
                                             AutoModelForCausalLM, AutoTokenizer)
    # G-B canary is computed from the full teacher-forced rows before analysis.
    gate_b = {}
    for arm in result["arms"]:
        actual = {r["id"] for r in arm["rows"] if r["strict_teacher_forced"]}
        expected = recorded[arm["arm"]]["strict"]
        diff = sorted(actual ^ expected)
        gate_b[arm["arm"]] = {"passed": len(diff) <= 2, "actual_n": len(actual),
                               "expected_n": len(expected), "symmetric_difference": diff}
    result["gates"]["G-B-strict-canary"] = {"passed": all(x["passed"] for x in gate_b.values()),
                                             "by_arm": gate_b}
    primary = next(a for a in result["arms"] if a["arm"] == "SPARSE")
    repeat = result["sparse_repeat"]
    max_content = max(abs(a["content"]["clipped_logodds"] - b["content"]["clipped_logodds"])
                      for a, b in zip(primary["rows"], repeat["rows"]))
    max_closing = max(abs(a["closing"]["clipped_logodds"] - b["closing"]["clipped_logodds"])
                      for a, b in zip(primary["rows"], repeat["rows"]))
    result["gates"]["G-C-determinism"] = {"passed": max(max_content, max_closing) <= 1e-3,
                                            "max_abs_delta_content": max_content,
                                            "max_abs_delta_closing": max_closing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
