from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.evaluation.runner import run_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen CPU-first candidate evaluation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--expected-candidate-hash")
    parser.add_argument("--abstain-logit-bias", type=float)
    args = parser.parse_args()
    summary = run_evaluation(
        model_path=args.model,
        dataset_path=args.dataset,
        output_directory=args.output_directory,
        split_name=args.split,
        dtype_name=args.dtype,
        threads=args.threads,
        max_context_tokens=args.max_context_tokens,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        case_limit=args.case_limit,
        sealed_authorization=args.sealed_authorization,
        expected_candidate_hash=args.expected_candidate_hash,
        abstain_logit_bias=args.abstain_logit_bias,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
