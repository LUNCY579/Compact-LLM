#!/usr/bin/env python3
"""Apply a frozen abstention gate to one confidence capture and report the result.

The gate spec is read from disk, never re-tuned here. The written prediction
JSONL is scored by the published ``evaluate_predictions.py`` so the headline
number is produced by the suite's own scorer, not by this lab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def gated_rows(
    spec: core.GateSpec,
    rows: list[core.GateRow],
    source_by_id: dict[str, dict[str, Any]],
    run_name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        fired = spec.decide(row)
        source = source_by_id[row.row_id]
        payload = {
            "id": row.row_id,
            "status": "ok",
            "prediction": core.ABSTAIN_TEXT if fired else row.prediction,
            "baseline_prediction": row.prediction,
            "gate_fired": bool(fired),
            "gate_rule": core.describe(spec),
            "signal_values": {
                spec.signal: row.signals[spec.signal],
                **(
                    {spec.signal_b: row.signals[spec.signal_b]}
                    if spec.signal_b is not None
                    else {}
                ),
            },
            "input_tokens": source.get("input_tokens"),
            "generated_tokens": source.get("generated_tokens"),
            "seed": source.get("seed"),
            "decoding": source.get("decoding"),
            "run": run_name,
        }
        output.append(payload)
    return output


def run_published_scorer(
    key: Path, predictions: Path, json_report: Path, markdown_report: Path
) -> dict[str, Any]:
    scorer = core.SCORER_PATH
    command = [
        sys.executable,
        str(scorer),
        "--key",
        str(key),
        "--predictions",
        str(predictions),
        "--json-report",
        str(json_report),
        "--markdown-report",
        str(markdown_report),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return json.loads(json_report.read_text(encoding="utf-8"))


def markdown(
    *,
    tag: str,
    split: str,
    spec: core.GateSpec,
    baseline: core.GateOutcome,
    gated: core.GateOutcome,
    ceiling: int,
    scorer_strict_exact: float,
) -> str:
    rows = gated.rows
    lines = [
        f"# Abstention gate result: {tag}",
        "",
        f"- Split: `{split}` ({rows} rows, {gated.unanswerable_rows} unanswerable)",
        f"- Gate: `{core.describe(spec)}`",
        "",
        "| metric | nimbus_low baseline | gated |",
        "| --- | --- | --- |",
        f"| correct | {baseline.baseline_correct}/{rows} | {gated.gated_correct}/{rows} |",
        f"| abstentions | {baseline.abstentions} | {gated.abstentions} |",
        f"| abstention precision | {baseline.abstention_precision:.3f} "
        f"({baseline.abstention_true_positives}/{baseline.abstentions}) | "
        f"{gated.abstention_precision:.3f} "
        f"({gated.abstention_true_positives}/{gated.abstentions}) |",
        f"| abstention recall | {baseline.abstention_recall:.3f} "
        f"({baseline.abstention_true_positives}/{baseline.unanswerable_rows}) | "
        f"{gated.abstention_recall:.3f} "
        f"({gated.abstention_true_positives}/{gated.unanswerable_rows}) |",
        "",
        f"- Delta vs baseline: **{gated.delta:+d}**",
        f"- Discordant pairs (gate-only correct / baseline-only correct): "
        f"{gated.mcnemar_b} vs {gated.mcnemar_c}",
        f"- Exact McNemar two-sided p: **{gated.mcnemar_p:.4f}**",
        f"- Oracle perfect-abstention ceiling on this split: {ceiling}/{rows}",
        f"- Published scorer strict_exact: {scorer_strict_exact:.6f} "
        f"({round(scorer_strict_exact * rows)}/{rows})",
        "",
        f"Rows gained: {', '.join(gated.gained_ids) or 'none'}",
        "",
        f"Rows lost: {', '.join(gated.lost_ids) or 'none'}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-spec", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec_payload = json.loads(args.gate_spec.read_text(encoding="utf-8"))
    spec = core.GateSpec.from_dict(spec_payload["gate"])

    key_rows = core.load_jsonl(args.key)
    split = sorted({str(row.get("split")) for row in key_rows})
    confidence_rows = core.load_jsonl(args.confidence)
    rows = core.build_rows(confidence_rows, key_rows)
    source_by_id = {str(row["id"]): row for row in confidence_rows}

    baseline = core.evaluate_gate(core.IDENTITY_GATE, rows)
    gated = core.evaluate_gate(spec, rows)
    ceiling = core.oracle_ceiling(rows)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = out / f"{args.tag}.jsonl"
    core.write_jsonl(
        predictions_path, gated_rows(spec, rows, source_by_id, args.tag)
    )

    scorer_report = run_published_scorer(
        args.key,
        predictions_path,
        out / f"{args.tag}_scorer_report.json",
        out / f"{args.tag}_scorer_report.md",
    )
    strict_exact = float(scorer_report["runs"][0]["strict_exact"])
    scorer_correct = round(strict_exact * len(rows))
    if scorer_correct != gated.gated_correct:
        raise RuntimeError(
            "Published scorer disagrees with the lab bookkeeping: "
            f"{scorer_correct} vs {gated.gated_correct}"
        )

    report = {
        "schema_version": "1.0",
        "kind": "abstention_gate_application",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "split": split,
        "gate": spec.to_dict(),
        "gate_spec_source": {
            "path": str(args.gate_spec.resolve()),
            "sha256": sha256_file(args.gate_spec),
            "tuning_split": spec_payload.get("tuning_split"),
        },
        "inputs": {
            "confidence": str(args.confidence.resolve()),
            "confidence_sha256": sha256_file(args.confidence),
            "key": str(args.key.resolve()),
            "key_sha256": sha256_file(args.key),
        },
        "baseline": baseline.to_dict(),
        "gated": gated.to_dict(),
        "oracle_abstention_ceiling": ceiling,
        "published_scorer_strict_exact": strict_exact,
        "implementation": {
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "gate_core_sha256": sha256_file(Path(core.__file__).resolve()),
            "scorer": str(core.SCORER_PATH),
            "scorer_sha256": sha256_file(core.SCORER_PATH),
        },
        "output": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "rows": len(rows),
        },
    }
    core.write_json(out / f"{args.tag}_gate_report.json", report)
    text = markdown(
        tag=args.tag,
        split=", ".join(split),
        spec=spec,
        baseline=baseline,
        gated=gated,
        ceiling=ceiling,
        scorer_strict_exact=strict_exact,
    )
    (out / f"{args.tag}_gate_report.md").write_text(
        text, encoding="utf-8", newline="\n"
    )
    print(text)


if __name__ == "__main__":
    main()

