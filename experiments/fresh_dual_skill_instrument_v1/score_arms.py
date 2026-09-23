"""Score arms on the FRESH-DUAL-SKILL-001 transition bank.

Decoding contract is copied from the recorded probe: greedy, batch 1, FP32, thinking disabled,
chat template rendered to the exact byte string the recorded bank used. The template is
SELF-TESTED at startup against a recorded prompt; a mismatch aborts before any model loads.

Scoring is exact string match on the stripped generation. No substring matching.
Writes only inside this experiment directory.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, platform, time
from collections import defaultdict
from pathlib import Path

RECORDED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/qwen001-20260918/render-process/prompts.jsonl")
RECORDED_VIEWS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/process-transition-probe-001/CANDIDATE_VIEWS.jsonl")


def render(system: str, user: str) -> str:
    """Exact byte form used by the recorded bank (Qwen3 chat template, thinking disabled)."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def template_selftest() -> None:
    """Re-render every recorded prompt from its own extracted parts and demand byte equality.

    This catches a wrong separator, a missing think block, or stray whitespace. A negative
    control is run too, so the test is known to be able to fail.
    """
    texts = [json.loads(l)["text"] for l in RECORDED.read_text(encoding="utf-8").splitlines() if l.strip()]
    checked = 0
    for text in texts:
        try:
            sysm = text.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]
            user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        except IndexError:
            raise SystemExit("TEMPLATE SELF-TEST FAILED: recorded prompt has an unexpected shape")
        if render(sysm, user) != text:
            raise SystemExit("TEMPLATE SELF-TEST FAILED: rendered prompt differs from the recorded bytes")
        checked += 1
    if checked < 100:
        raise SystemExit(f"TEMPLATE SELF-TEST INCONCLUSIVE: only {checked} recorded prompts")
    # negative control: the test must reject a template that drops the think block
    s, u = "sys", "user"
    broken = f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    if render(s, u) == broken:
        raise SystemExit("TEMPLATE SELF-TEST IS VACUOUS: it cannot distinguish a missing think block")
    print(f"  template self-test: {checked} recorded prompts reproduced byte-exactly "
          f"(negative control rejects a missing think block)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--arms-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm directory names")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    a = ap.parse_args()

    print("startup checks", flush=True)
    template_selftest()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lease import lease, LOCK
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    bank = json.loads(a.bank.read_text(encoding="utf-8"))
    rows = bank["rows"]
    bank_sha = hashlib.sha256(a.bank.read_bytes()).hexdigest()
    print(f"  bank {a.bank.name} n={len(rows)} sha256={bank_sha}", flush=True)

    names = a.arms or sorted(p.name for p in a.arms_root.iterdir() if p.is_dir())
    result = {
        "schema": 1, "instrument": "FRESH-DUAL-SKILL-001", "split": "T",
        "bank_sha256": bank_sha, "bank_n": len(rows),
        "scoring": "exact string match on stripped generation",
        "shortcut_floor_rows": bank["copy_baseline_rows"],
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
            "dtype": "float32", "decoding": "greedy", "batch_size": 1,
            "max_new_tokens": a.max_new_tokens, "thinking_disabled": True,
        },
        "arms": [],
        # Do not invent an authorization string. Record the actual basis and its scope.
        "authorization_basis": (
            "Operator instructed evaluation in-session on 2026-09-22 ('I will let you create "
            "evaluation scripts to evaluate those models'). This is a local forward pass on the "
            "operator's own machine and checkpoints."
        ),
        "is_soup_mini_001_CAL_or_CHECK": False,
        "soup_mini_001_protocol_unchanged": True,
        "note": (
            "SOUP-MINI-001's protocol.json still reads native_authorized:false and is untouched. "
            "This run is a separate exploratory instrument and does not consume, satisfy or "
            "modify that experiment's gates."
        ),
        "quality_claim": False, "release_permitted": False,
    }

    result["gpu_lease"] = {
        "path": str(LOCK),
        "held_for_the_whole_scoring_window": True,
        "note": ("Card 0 is shared by three sessions. The lease is taken before the first model "
                 "load and released after the last arm, so the window is contiguous and peers "
                 "can queue on it."),
    }

    with lease(label=f"fds1:{a.bank.stem}", job=f"score {len(names)} arm(s) x {len(rows)} rows"):
        _score_all(names, a, rows, result, torch, AutoModelForCausalLM, AutoTokenizer, gc)

    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.output}")
    return 0


def _score_all(names, a, rows, result, torch, AutoModelForCausalLM, AutoTokenizer, gc):
    for name in names:
        arm_dir = a.arms_root / name
        started = time.time()
        tok = AutoTokenizer.from_pretrained(str(arm_dir), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(arm_dir), torch_dtype=torch.float32, local_files_only=True,
            low_cpu_mem_usage=True, device_map={"": "cuda"})
        model.eval()
        dev = next(model.parameters()).device
        preds, by_fam = [], defaultdict(lambda: [0, 0])
        copy_like = 0
        for r in rows:
            text = render(r["prompt"][0]["content"], r["prompt"][1]["content"])
            enc = tok(text, return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ok = gen == r["reference"]
            if gen == r["copy_baseline"]:
                copy_like += 1
            by_fam[r["family"]][0] += int(ok)
            by_fam[r["family"]][1] += 1
            preds.append({"id": r["id"], "family": r["family"], "generation": gen, "correct": ok})
        total = sum(v[0] for v in by_fam.values())
        rec = {"arm": name, "seconds": round(time.time() - started, 1),
               "correct": total, "n": len(rows),
               "copied_input_verbatim": copy_like,
               "by_family": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(by_fam.items())},
               "predictions": preds}
        result["arms"].append(rec)
        print(json.dumps({k: rec[k] for k in ("arm", "correct", "n", "copied_input_verbatim", "by_family", "seconds")}), flush=True)
        del model, tok
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
