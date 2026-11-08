"""Full nine-arm teacher-forced S-bank readout."""
from __future__ import annotations
import argparse, gc, hashlib, json, platform, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXPERIMENT = ROOT / "titan_opallo_skill_vector_v1"
FUNC = ROOT / "functional_additivity_v1"
FRESH = ROOT / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FUNC))
sys.path.insert(0, str(FRESH))
import score_tf  # noqa: E402
import score_arms  # noqa: E402

ARMS = ["BASE", "SPARSE", "OPALLO", "AVG", "ADD-S75", "ADD-O75", "ADD-75", "ADD-100", "CONFLICT-LOCAL"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    score_arms.template_selftest()
    bank_path = FRESH / "T_BANK_V2.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    rows = bank["rows"]
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arms_root = EXPERIMENT / "arms"
    result = {
        "schema": 1, "instrument": "FUNCTIONAL-ADDITIVITY-001-FULL-MATRIX",
        "bank": str(bank_path.resolve()), "bank_sha256": sha256(bank_path), "bank_n": len(rows),
        "arms": ARMS, "runtime": {"python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
        "dtype": "float32", "tf32": False, "use_cache": False,
        "tokenization": "score_arms.render; add_special_tokens=False", "eos_id": score_tf.EOS_ID},
        "quality_claim": False, "release_permitted": False,
    }
    from gpu_lease import lease
    with lease(label="functional-additivity-full-matrix", job="teacher-forced 9 arms x 192 rows"):
        for name in ARMS:
            print(f"scoring {name}", flush=True)
            rec = score_tf.score_arm(name, arms_root / name, rows, torch,
                                     AutoModelForCausalLM, AutoTokenizer)
            result.setdefault("results", []).append(rec)
            print(json.dumps({"arm": name, "seconds": rec["seconds"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
