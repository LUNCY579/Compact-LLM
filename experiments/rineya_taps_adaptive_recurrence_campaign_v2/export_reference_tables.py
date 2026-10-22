"""Export reviewable tables from the approved native report/raw journal only."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from .run_apparatus import PACKAGE, sha


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(name):
    if Path(name).name != name:
        raise ValueError("directory name only")
    report_path = PACKAGE / "analyses/gpu-reference-002/REPORT.json"
    raw_path = PACKAGE / "runs/native-gpu-reference-002/result/outcomes.jsonl"
    report = json.loads(report_path.read_text())
    if not report["complete"] or report["errors"]:
        raise RuntimeError("complete approved report required")
    out = PACKAGE / "analyses" / name
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "policy_accuracy.csv", [{"policy": row["policy"], "accuracy": row["accuracy"],
        "items": row["case_count"], "inherited_format_score": row["format_score"],
        "format_caution": "counterfactualformat is gold-coupled; not independent syntax",
        "literal_UNKNOWN_rate": row["literal_UNKNOWN_rate"]} for row in report["tables"]])
    contrasts = []
    pairs = []
    for name, comparison in report["pairs"].items():
        contrasts.append({"contrast": name, "treatment": comparison["treatment"],
            "comparator": comparison["comparator"], "score_delta": comparison["score_delta"],
            "wins": len(comparison["wins"]), "losses": len(comparison["losses"]),
            "source_correct": comparison["source_correct_cases"], "retained": comparison["source_correct_retained"],
            "confirmatory_p_value": None, "confidence_interval": None,
            "uncertainty": comparison["uncertainty"]})
        pairs.extend({"contrast": name, **row} for row in comparison["per_case"])
    write_csv(out / "paired_contrasts.csv", contrasts)
    write_csv(out / "paired_cases.csv", pairs)
    alpha = report["features"]["prefix_weighted"]["alpha"]
    write_csv(out / "alpha_features.csv", [{"feature": name, **values} for name, values in report["features"]["prefix_weighted"].items()])
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    grouped = {}
    for row in raw:
        if row["split"] != "DEV" or not row["physical_run"]:
            continue
        policy = f"A2:{row['fixed_alpha']:g}" if row["arm"] == "A2" else row["arm"]
        grouped.setdefault(policy, []).append(row)
    costs = []
    for policy, rows in sorted(grouped.items()):
        costs.append({"policy": policy, "physical_repeats": len(rows), "unique_DEV_cases": len({row['case_id'] for row in rows}),
            "independence_caution": "16 exposed cases; at most13 provisional scenario clusters; independent units unestablished",
            "mean_case_wall_seconds": statistics.mean(row["wall_seconds"] for row in rows),
            "full_model_calls": sum(row["full_model_calls_including_gold_nll"] for row in rows),
            "prefix_positions": sum(row["processed_token_positions"] for row in rows),
            "generated_tokens": sum(len(row["token_ids"]) for row in rows),
            "timing_scope": "synchronized detailed instrumentation; source/model/device same; not deployable speed"})
    write_csv(out / "instrumented_costs.csv", costs)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "report_sha256": sha(report_path),
        "raw_journal_sha256": sha(raw_path), "source_sha256": sha(Path(__file__)),
        "headroom": report["constant_oracle"]["headroom"], "capture": report["constant_oracle"]["capture"],
        "alpha_mean": alpha["mean"], "broad_inference": "UNSUPPORTED_SMALL_EXPOSED_CLUSTERED_BANK",
        "files": {p.name: sha(p) for p in sorted(out.glob("*.csv"))}}
    with (out / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(out), "tables": len(manifest["files"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="reference-tables-001")
    main(parser.parse_args().name)
