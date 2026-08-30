from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.fisher import estimate_diagonal_fisher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate a CPU diagonal empirical Fisher.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--abstain-logit-bias", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = estimate_diagonal_fisher(
        model_path=args.model,
        calibration_path=args.calibration,
        destination=args.output,
        limit=args.limit,
        max_length=args.max_length,
        threads=args.threads,
        trust_remote_code=args.trust_remote_code,
        abstain_logit_bias=args.abstain_logit_bias,
        overwrite=args.overwrite,
    )
    print(json.dumps({"examples": result["examples"], "tokens": result["tokens"], "loss_mean": result["loss_mean"], "output_fingerprint": result["output_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
