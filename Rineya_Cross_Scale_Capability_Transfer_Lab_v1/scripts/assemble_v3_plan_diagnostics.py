"""Assemble plan-index deliverables under 13_v3_transport_diagnostics/.

Reuses executed 13_v3_transport_diagnosis artifacts. Does not reload 4B/8B
weights and does not open sealed evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.consensus_target_basis import (  # noqa: E402
    analyze_consensus_vs_target,
)
from rineya_cross_scale.functional_atlas import assemble_plan_atlas  # noqa: E402
from rineya_cross_scale.functional_localization import (  # noqa: E402
    run_functional_localization,
)
from rineya_cross_scale.functional_translator import (  # noqa: E402
    curate_translator_results,
)
from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402
from rineya_cross_scale.registry import ExperimentRegistry  # noqa: E402
from rineya_cross_scale.resource_trace import resource_trace  # noqa: E402
from rineya_cross_scale.transport_sanity import run_transport_sanity  # noqa: E402
from rineya_cross_scale.v2 import verify_cached_artifacts  # noqa: E402

DIAGNOSIS = LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnosis"
INDEX = LAB_ROOT / "cross_scale_evidence" / "13_v3_transport_diagnostics"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=INDEX)
    args = parser.parse_args()
    root: Path = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    with resource_trace("v3_plan_index_assemble", root / "report" / "resource_assemble.json") as trace:
        artifacts = verify_cached_artifacts(LAB_ROOT)
        atomic_write_json(root / "protocol" / "artifact_verification.json", artifacts)
        if artifacts.get("status") != "ok":
            print("artifact verification failed", artifacts.get("mismatches"))
            return 1

        e4 = DIAGNOSIS / "e4_functional_atlas" / "functional_atlas.json"
        e5 = DIAGNOSIS / "e5_cross_scale_atlas" / "cross_scale_atlas_last.json"
        e1 = DIAGNOSIS / "e1_channel_capacity" / "channel_capacity.json"
        e1c = DIAGNOSIS / "e1_channel_capacity" / "shape_matched_donor_null.json"
        e2 = DIAGNOSIS / "e2_target_basis_alignment" / "target_basis_alignment.json"
        e3 = DIAGNOSIS / "e3_localization" / "localization.json"
        e6 = DIAGNOSIS / "e6_functional_translator" / "translator.json"

        atlas = assemble_plan_atlas(e4_path=e4, e5_path=e5 if e5.is_file() else None, output_dir=root / "atlas")
        sanity = run_transport_sanity(
            e1_path=e1,
            e1c_path=e1c if e1c.is_file() else None,
            e2_path=e2,
            output_dir=root / "transport_sanity",
        )
        localization = run_functional_localization(e3_path=e3, output_dir=root / "localization")
        consensus = analyze_consensus_vs_target(e2, root / "consensus_analysis")
        translator = curate_translator_results(e6, root / "functional_translator")

        inject_gate = (
            sanity.get("overall_verdict") == "PASS"
            and localization.get("verdict") == "PASS"
            and consensus.get("verdict") == "PASS"
            and translator.get("suggest_micro_injection") is True
        )
        micro = {
            "schema_version": "3.0-diagnostics-index",
            "hypothesis": "Single-region injection can beat both controls",
            "gate_inputs": {
                "transport_sanity": sanity.get("overall_verdict"),
                "localization": localization.get("verdict"),
                "consensus": consensus.get("verdict"),
                "translator_suggests_injection": translator.get("suggest_micro_injection"),
            },
            "authorised": inject_gate,
            "candidates_created": 0,
            "verdict": "PASS" if inject_gate else "FAIL",
            "reason": (
                "Upstream diagnostics failed the plan's injection gate "
                "(sanity/localization/consensus/translator). Recorded as NO_GO."
            ),
            "escalation": {
                "DEV_FULL": False,
                "sealed": False,
                "sealed_status": "CLOSED",
            },
        }
        atomic_write_json(root / "candidates" / "NO_GO.json", micro)

        registry = ExperimentRegistry(root / "candidates" / "registry.jsonl")
        for experiment_id, hypothesis, transformation, verdict, payload in (
            ("V3IDX_ATLAS", "functional atlas", "E4+E5", "PASS" if e5.is_file() else "INCONCLUSIVE", atlas),
            ("V3IDX_SANITY", "transport sanity", "P4/P5", sanity.get("overall_verdict", "INCONCLUSIVE"), sanity),
            ("V3IDX_LOCAL", "localization", "E3 depth/module", localization.get("verdict", "INCONCLUSIVE"), localization),
            ("V3IDX_CONSENSUS", "consensus vs target", "E2", consensus.get("verdict", "INCONCLUSIVE"), consensus),
            ("V3IDX_TRANSLATOR", "functional translator", "E6", translator.get("verdict", "INCONCLUSIVE"), translator),
            ("V3IDX_MICRO", "micro-injection", "none", micro["verdict"], micro),
        ):
            registry.record_diagnostic(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                transformation=transformation,
                verdict=verdict if verdict in {"PASS", "FAIL", "INCONCLUSIVE"} else "INCONCLUSIVE",
                quantitative_result={"keys": list(payload.keys())[:20]},
                interpretation=str(payload.get("interpretation") or payload.get("reason") or ""),
                next_action=str(payload.get("next_action") or payload.get("stop_rule") or ""),
                source_hashes={},
                resource_usage={"model_load_count": 0, "note": "curated from existing evidence"},
            )

        outcome = "D"
        if e5.is_file() and translator.get("verdict") == "FAIL":
            # Functional structure exists (E5) but donor is redundant / T empty (E1/E6).
            outcome = "B_partial_then_D_for_tested_T"
        summary = {
            "plan_outcome": outcome,
            "atlas_ok": e4.is_file() and e5.is_file(),
            "transport_sanity": sanity.get("overall_verdict"),
            "localization": localization.get("verdict"),
            "consensus": consensus.get("verdict"),
            "translator": translator.get("verdict"),
            "micro_injection": micro["verdict"],
            "DEV_FULL_justified": False,
            "sealed_justified": False,
            "hypothesis_strength": (
                "Weaker for spectral P4/P5 parameter transport; "
                "functional cross-scale correspondence exists (E5) but does not "
                "yield a tiny portable update for Supa under tested maps (E6)."
            ),
        }
        atomic_write_json(root / "report" / "decision_summary.json", summary)
        trace.sample_rss()
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
