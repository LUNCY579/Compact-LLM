from __future__ import annotations

"""Long-horizon procedural-control curve for v2.3.1."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import discover_adapters, read_json, write_csv, write_json


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def real_rag(summary: dict[str, Any]) -> dict[str, Any]:
    return (summary.get("conditions") or {}).get("real_rag") or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--dev-benchmark", type=Path, required=True)
    ap.add_argument("--ood-benchmark", type=Path, required=True)
    ap.add_argument("--knowledge-db", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--maximum-evidence-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=180)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()
    for path in (args.dev_benchmark, args.ood_benchmark):
        if "sealed" in str(path).casefold():
            raise ValueError("procedure dynamics refuses sealed benchmarks")
    checkpoints = discover_adapters(args.training_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in checkpoints:
        step = int(item["step"])
        row: dict[str, Any] = {"step": step}
        for label, bench in (("dev", args.dev_benchmark), ("ood", args.ood_benchmark)):
            out = args.output_root / f"step_{step:06d}" / label
            summary_path = out / "summary.json"
            if not (args.reuse_existing and summary_path.exists()):
                cmd = [
                    sys.executable,
                    "scripts/phase_9_student_distillation/evaluate_rag.py",
                    "--model", args.base_model,
                    "--benchmark", str(bench),
                    "--output-dir", str(out),
                    "--knowledge-db", str(args.knowledge_db),
                    "--gate-calibration", str(args.calibration),
                    "--conditions", "real_rag",
                    "--retrieval-selection-policy", "query_aware_v23",
                    "--maximum-evidence-tokens", str(args.maximum_evidence_tokens),
                    "--max-new-tokens", "32",
                    "--minimum-cases-per-condition", "1",
                    "--minimum-supported-cases", "1",
                    "--minimum-unsupported-cases", "1",
                    "--limit", str(args.limit),
                    "--smoke-test",
                    "--resume",
                ]
                if item.get("adapter"):
                    cmd += ["--adapter", str(item["adapter"])]
                if args.require_cuda:
                    cmd.append("--require-cuda")
                run(cmd)
            summary = read_json(summary_path)
            real = real_rag(summary)
            row[f"{label}_procedure_correctness"] = real.get("procedure_correctness")
            row[f"{label}_supported_answer_correctness"] = real.get("supported_answer_correctness")
            row[f"{label}_evidence_faithfulness"] = real.get("evidence_faithfulness")
            row[f"{label}_correct_abstention_rate"] = real.get("correct_abstention_rate")
        rows.append(row)
    payload = {"schema_version": "2.3.1", "sealed_data_used": False, "rows": rows}
    write_json(args.output_root / "procedure_time_curve.json", payload)
    write_csv(args.output_root / "procedure_time_curve.csv", rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
