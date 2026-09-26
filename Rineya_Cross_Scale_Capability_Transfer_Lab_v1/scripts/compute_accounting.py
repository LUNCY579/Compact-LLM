from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


def bucket(path: Path) -> str:
    value = path.as_posix().lower()
    if "attempt_" in value or "/smoke/" in value:
        return "debugging_or_superseded"
    return "frozen_valid"


def main() -> int:
    evidence = LAB_ROOT / "cross_scale_evidence"
    summaries = []
    for path in sorted(evidence.glob("06_dev_results/**/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "path": path.relative_to(LAB_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bucket": bucket(path),
                "items": int(payload.get("items", 0)),
                "generated_tokens": int(payload.get("generated_tokens", 0)),
                "wall_seconds": float(payload.get("wall_seconds", 0.0)),
                "elapsed_seconds_including_load": float(
                    payload.get("elapsed_seconds_including_load", 0.0)
                ),
                "load_seconds": float(payload.get("load_seconds", 0.0)),
                "rss_final_bytes": payload.get("rss_final_bytes"),
                "model_hash": payload.get("model_hash"),
                "dataset_sha256": payload.get("dataset_sha256"),
            }
        )
    aggregates = {}
    for name in ("frozen_valid", "debugging_or_superseded", "all"):
        rows = summaries if name == "all" else [row for row in summaries if row["bucket"] == name]
        aggregates[name] = {
            "runs": len(rows),
            "items": sum(row["items"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "model_execution_seconds": sum(row["wall_seconds"] for row in rows),
            "elapsed_seconds_including_load": sum(
                row["elapsed_seconds_including_load"] for row in rows
            ),
            "model_load_seconds": sum(row["load_seconds"] for row in rows),
            "peak_recorded_rss_bytes": max(
                (int(row["rss_final_bytes"]) for row in rows if row["rss_final_bytes"]),
                default=None,
            ),
        }
    report = {
        "schema_version": "1.0",
        "cpu": "AMD Ryzen 9 7940HX, 16 cores / 32 logical processors",
        "gpu_hours": 0.0,
        "gpu_note": "No model construction or evaluation used GPU; installed Torch reported CUDA unavailable.",
        "evaluation": aggregates,
        "runs": summaries,
        "final_candidate": {
            "effective_parameter_count": 596049920,
            "serialized_weight_bytes": 1192134784,
            "storage_dtype": "float16",
            "inference_device_tested": "cpu",
        },
        "instrumentation_gaps": [
            "Projection and merge wall time was not recorded in early manifests; commands completed in-session but no retrospective exact CPU-hour claim is made.",
            "Peak RAM is available only as process RSS after evaluation, not a sampled peak during projection or merge.",
            "TTFT was not separately instrumented; per-case end-to-end latency and aggregate tokens/s are preserved in predictions/summaries."
        ],
    }
    output_json = evidence / "01_environment" / "compute_accounting.json"
    output_md = evidence / "01_environment" / "compute_accounting.md"
    atomic_write_json(output_json, report)
    valid = aggregates["frozen_valid"]
    lines = [
        "# Compute Accounting",
        "",
        f"- Frozen valid evaluation runs: {valid['runs']}",
        f"- Frozen valid evaluated rows: {valid['items']}",
        f"- Frozen valid model-execution time: {valid['model_execution_seconds'] / 3600:.3f} CPU-hours wall-clock",
        f"- Frozen valid elapsed time including loads: {valid['elapsed_seconds_including_load'] / 3600:.3f} CPU-hours wall-clock",
        f"- Generated tokens: {valid['generated_tokens']}",
        f"- Peak recorded post-run RSS: {(valid['peak_recorded_rss_bytes'] or 0) / 1024**3:.3f} GiB",
        "- GPU hours: 0",
        "- Final candidate: 596,049,920 parameters; 1,192,134,784 weight bytes; FP16 storage",
        "",
        "## Instrumentation gaps",
        "",
    ]
    lines.extend(f"- {value}" for value in report["instrumentation_gaps"])
    atomic_write_text(output_md, "\n".join(lines))
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
