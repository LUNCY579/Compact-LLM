"""Derive a new audit/plot directory without modifying completed run evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
from armr.provenance import sha256, write_json
from armr.reporting import csv_file, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "evidence/01_middle_reapplication_screen").resolve()):
        parser.error("only an ARMR middle-screen evidence directory may be analyzed")
    manifest = json.loads((run / "run_manifest.json").read_text())
    atlas = json.loads((run / "atlas.json").read_text())
    raw = [json.loads(s) for s in (run / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    out = ROOT / "evidence/analysis" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    base = {r["case_id"]: r for r in raw if r["arm"] == "intact"}
    by_arm = defaultdict(list)
    for r in raw:
        by_arm[r["arm"]].append(r)
    diagnostics = []
    checks = {"canonical_unchanged": manifest["canonical_unchanged"],
              "all_arms_present": len(by_arm) == 1 + len(manifest["config"]["candidate_layers"]) * len(manifest["config"]["applications"])}
    for name, rows in by_arm.items():
        checks[f"{name}_all_case_ids"] = len(rows) == len(base) and {r["case_id"] for r in rows} == set(base)
        if name == "intact":
            continue
        apps = rows[0]["applications"]
        checks[f"{name}_hidden_pass_counts"] = all(len(r["hidden"]) == apps for r in rows)
        h2 = [h for r in rows for h in r["hidden"] if h["application"] == 2]
        h3 = [h for r in rows for h in r["hidden"] if h["application"] == 3]
        diagnostics.append({"layer": rows[0]["layer"], "applications": apps,
                            "corrected_smoke_cases": sum(r["score"] > base[r["case_id"]]["score"] for r in rows),
                            "regressed_smoke_cases": sum(r["score"] < base[r["case_id"]]["score"] for r in rows),
                            "identical_output_cases": sum(r["token_ids"] == base[r["case_id"]]["token_ids"] for r in rows),
                            "mean_second_delta_l2": mean([h["delta_l2"] for h in h2]),
                            "mean_third_delta_l2": mean([h["delta_l2"] for h in h3]) if h3 else None,
                            "mean_rho_2": mean([h["rho"] for h in h2]),
                            "mean_rho_3": mean([h["rho"] for h in h3]) if h3 else None,
                            "mean_delta_cosine_3": mean([h["delta_cosine"] for h in h3]) if h3 else None,
                            "mean_state_cosine_to_normal": mean([r["hidden"][-1]["cosine_to_normal"] for r in rows]),
                            "total_dense_matmul_flops": sum(r["generation_dense_matmul_flops_estimate"] for r in rows),
                            "total_prefill_seconds": sum(r["prefill_seconds"] for r in rows),
                            "total_decode_seconds": sum(r["decode_seconds"] for r in rows)})
    csv_file(out / "dynamics_and_paired_smoke.csv", diagnostics)
    write_json(out / "integrity_checks.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    write_json(out / "source_manifest.json", {"source_run": str(run),
                "source_files": {str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob("*")) if p.is_file()},
                "analysis_script_sha256": sha256(Path(__file__))})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, apps in zip(axes, (2, 3)):
        for r in atlas:
            if r["applications"] != apps:
                continue
            color = "#177d69" if r["go_fixed_smoke"] else "#b4413c"
            ax.scatter(r["skip_cosine"], r["mean_logit_cosine"], c=color, s=55)
            ax.annotate(str(r["layer"]), (r["skip_cosine"], r["mean_logit_cosine"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axhline(.9, color="#777777", linestyle="--", linewidth=.8)
        ax.set(xlabel="Removal: logit cosine after identity skip", ylabel="Reuse: logit cosine after reapplication",
               title=f"{apps} total applications ({apps-1} extra)", xlim=(.68,1.005), ylim=(0,1.015))
        ax.grid(alpha=.18)
    fig.suptitle("Supa-A1.1 necessity × repeatability: 8-prompt mechanical screen\nGreen = all frozen smoke gates pass; red = at least one fails", fontsize=12)
    fig.savefig(out / "necessity_repeatability_atlas.png", dpi=180)
    fig.savefig(out / "necessity_repeatability_atlas.svg")
    plt.close(fig)
    (out / "report.md").write_text("# Derived evidence audit\n\nStatus: " + ("PASS" if all(checks.values()) else "FAIL") +
        ".\n\nQuestion: are all prespecified layer/depth/case results present, and what do paired smoke responses and hidden trajectories show? "
        "No model was loaded. The source manifest hashes every immutable input file; integrity_checks.json checks coverage and pass counts. "
        "See dynamics_and_paired_smoke.csv and necessity_repeatability_atlas.png.\n\n"
        "Rho below one describes diminishing successive deltas along this trajectory; it is not a local Jacobian bound or proof of contraction. "
        "Positive successive-delta cosine suggests persistent directional drift, not a reasoning process. "
        "Corrected/regressed smoke cases remain descriptive. No new gate is licensed by this derived analysis.\n", encoding="utf-8")
    print(out)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
