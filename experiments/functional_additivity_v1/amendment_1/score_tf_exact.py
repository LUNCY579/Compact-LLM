"""FUNCTIONAL-ADDITIVITY-001 Amendment 1 (POST-HOC): exact re-score.

Reuses the registered scorer's functions unchanged (imported from ../score_tf.py) and additionally
stores unclipped gold log-odds and the period-variant probabilities that TF_RAW.json does not keep.
Writes only TF_EXACT.json in this directory, and refuses to overwrite it. See PLAN.md.
"""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FA = HERE.parent
sys.path.insert(0, str(FA))
import score_tf  # noqa: E402  (registered scorer; imported, never edited)
from score_tf import score_arms, logodds_and_logprob, EOS_ID, EXPECTED_ARMS  # noqa: E402

PERIOD_ID = 13
ENDOFTEXT_ID = 151643
OUT = HERE / "TF_EXACT.json"
TF_RAW = FA / "TF_RAW.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_one(model, tok, row):
    import torch
    rendered = score_arms.render(row["prompt"][0]["content"], row["prompt"][1]["content"])
    prompt_ids = tok(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    gold_ids = tok(row["reference"], add_special_tokens=False)["input_ids"]
    dotted = tok(row["reference"] + ".", add_special_tokens=False)["input_ids"]
    merged_id = None
    if len(dotted) == len(gold_ids) and dotted[:-1] == gold_ids[:-1] and dotted[-1] != gold_ids[-1]:
        merged_id = int(dotted[-1])          # final piece + "." is one token, e.g. "}." = 7810
    ids = torch.tensor([list(prompt_ids.tolist()) + gold_ids], dtype=torch.long,
                       device=next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    P, g = int(prompt_ids.numel()), len(gold_ids)
    gold_pos = torch.arange(P - 1, P + g - 1, device=ids.device)
    gold_t = torch.tensor(gold_ids, dtype=torch.long, device=ids.device)
    eos_pos = torch.tensor([P + g - 1], dtype=torch.long, device=ids.device)
    eos_t = torch.tensor([EOS_ID], dtype=torch.long, device=ids.device)
    z, lp, p, am = logodds_and_logprob(logits, gold_t, gold_pos)
    ez, elp, ep, eam = logodds_and_logprob(logits, eos_t, eos_pos)
    lsm_final = torch.log_softmax(logits[P + g - 2], dim=0)
    lsm_eos = torch.log_softmax(logits[P + g - 1], dim=0)
    return {
        "id": row["id"], "family": row["family"], "gold_ids": gold_ids, "merged_id": merged_id,
        "z_unclipped": [float(x) for x in z.detach().cpu()],          # gold positions 0..G-1
        "z_clipped": [float(x) for x in z.clamp(-10, 10).detach().cpu()],
        "logprob": [float(x) for x in lp.detach().cpu()],
        "eos_z_unclipped": float(ez[0].item()), "eos_logprob": float(elp[0].item()),
        "final_pos_logprob_merged": (float(lsm_final[merged_id].item()) if merged_id is not None else None),
        "final_pos_logprob_period": float(lsm_final[PERIOD_ID].item()),
        "eos_pos_logprob_period": float(lsm_eos[PERIOD_ID].item()),
        "eos_pos_logprob_endoftext": float(lsm_eos[ENDOFTEXT_ID].item()),
        "strict_teacher_forced": bool(torch.all(am == gold_t).item() and int(eam.item()) == EOS_ID),
    }


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    print("startup: template self-test", flush=True)
    score_arms.template_selftest()
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bank_path = score_tf.FRESH / "T_BANK_V2.json"
    rows = json.loads(bank_path.read_text(encoding="utf-8"))["rows"]
    raw = json.loads(TF_RAW.read_text(encoding="utf-8"))
    raw_by_arm = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in raw["arms"]}
    arms_root = FA.parent / "titan_opallo_skill_vector_v1" / "arms"
    result = {
        "schema": 1, "instrument": "FUNCTIONAL-ADDITIVITY-001 Amendment 1 (post-hoc exact re-score)",
        "registered": False, "bank_sha256": sha256(bank_path), "tf_raw_sha256": sha256(TF_RAW),
        "code_sha256": sha256(Path(__file__)), "registered_scorer_sha256": sha256(FA / "score_tf.py"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
                    "dtype": "float32", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32, "use_cache": False,
                    "sys_executable": sys.executable, "torch_threads": torch.get_num_threads()},
        "authorization_basis": ("Operator authorised in-session 2026-09-23: 'Exact re-score, ~2 GPU-min' "
                                "(post-hoc, labelled; five existing arms; teacher-forced only)."),
        "arms": [], "canary": {}, "quality_claim": False, "release_permitted": False,
    }
    from gpu_lease import lease
    with lease(label="fa001-amendment1-exact", job="teacher-forced 5 arms x 192 rows (post-hoc)"):
        for name in EXPECTED_ARMS:
            avail = psutil.virtual_memory().available / 2**30
            free_vram = torch.cuda.mem_get_info()[0] / 2**30
            if avail < 3.0 or free_vram < 3.5:
                raise SystemExit(f"resource precheck failed before {name}: host {avail:.2f} GiB, "
                                 f"VRAM {free_vram:.2f} GiB")
            arm_dir = arms_root / name
            st = arm_dir / "model.safetensors"
            if st.stat().st_mtime >= TF_RAW.stat().st_mtime:
                raise SystemExit(f"{name} weights modified after TF_RAW (arm-overwrite trap)")
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(str(arm_dir), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(arm_dir), torch_dtype=torch.float32, local_files_only=True,
                low_cpu_mem_usage=True, device_map={"": "cuda"})
            model.eval()
            recs = [score_one(model, tok, r) for r in rows]
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            # canary against the registered run: clipped per-token log-odds and strict set
            ref = raw_by_arm[name]
            max_d, strict_diff = 0.0, 0
            for r in recs:
                o = ref[r["id"]]
                stored = o["content"]["token_logodds"] + [o["closing"]["token_logodds"][0]]
                mine = r["z_clipped"]
                if len(stored) != len(mine):
                    raise SystemExit(f"canary length mismatch {name} {r['id']}")
                max_d = max(max_d, max(abs(a - b) for a, b in zip(stored, mine)),
                            abs(o["closing"]["token_logodds"][1] - max(-10.0, min(10.0, r["eos_z_unclipped"]))))
                strict_diff += int(o["strict_teacher_forced"] != r["strict_teacher_forced"])
            result["canary"][name] = {"max_abs_delta_clipped_logodds": max_d, "strict_flag_mismatches": strict_diff,
                                      "passed": max_d <= 1e-5 and strict_diff == 0,
                                      "weights_mtime_precedes_tf_raw": True}
            result["arms"].append({"arm": name, "seconds": round(time.time() - t0, 1),
                                   "host_avail_gib_before": round(avail, 2),
                                   "vram_free_gib_before": round(free_vram, 2), "rows": recs})
            print(json.dumps({"arm": name, **result["canary"][name],
                              "seconds": result["arms"][-1]["seconds"]}), flush=True)
    result["canary_all_passed"] = all(c["passed"] for c in result["canary"].values())
    OUT.write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))
    print(f"wrote {OUT} sha256={sha256(OUT)} canary_all_passed={result['canary_all_passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
