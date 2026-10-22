"""Create-only, model-free preparation receipt. This command cannot launch a model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .resources import LIMITS, blockers, snapshot
from .statistics import damage_power, paired

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new external session directory")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("receipt must use a new external directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = json.loads((PACKAGE / "protocol.json").read_text(encoding="utf-8"))
    if protocol["native_launch_enabled"] is not False:
        raise ValueError("this preparation implementation cannot enable native launch")
    asset = json.loads((PACKAGE / "asset_audit.json").read_text(encoding="utf-8"))
    vault = json.loads((PACKAGE / "vault_availability.json").read_text(encoding="utf-8"))
    resources = snapshot(ROOT)
    resource_reasons = blockers(resources)
    source = asset["source"]
    source_matches_now = file_sha(Path(source["path"])) == source["expected_sha256"]
    sidecar_matches_now = {name: file_sha(Path(info["path"])) == info["sha256"]
                           for name, info in source["sidecars"].items()}
    # Missing paths come from bound local deletion audits. Never estimate their
    # inference cost by substituting another checkpoint.
    endpoints_now = []
    for info in asset["blockers"]:
        if info["id"] in ("ASSET_T_MISSING", "ASSET_H_MISSING"):
            path = Path(info["path"])
            exists = path.is_file()
            actual = file_sha(path) if exists else None
            endpoints_now.append({"path": str(path), "required_sha256": info["required_sha256"],
                                  "present": exists, "sha256": actual,
                                  "matches": exists and actual == info["required_sha256"]})
    hashes = {p.name: file_sha(p) for p in sorted(PACKAGE.iterdir())
              if p.is_file() and p.suffix in (".py", ".md", ".json")}
    write_new(out / "resources.json", {"snapshot": resources, "limits": LIMITS,
                                         "resource_blockers": resource_reasons,
                                         "qualification_of_resource_envelope": "NOT_RUN"})
    n = protocol["data_reservation"]["corpus_counts"]["T"]
    equal = paired([True] * (n // 2) + [False] * (n // 2),
                   [False] * (n // 2) + [True] * (n // 2), claims=10)
    write_new(out / "statistical_sensitivity.json", {
        "kind": "MODEL_FREE_DESIGN_CALCULATION_NOT_OUTCOMES",
        "worst_discordance_equal_aggregate_counts": equal,
        "conditional_damage_planning": damage_power(900),
        "historical_47_zero_losses_upper": damage_power(47),
        "warning": "power assumptions do not certify a future bank's validity or endpoint sensitivity"})
    reasons = ["EXACT_ENDPOINT_ASSETS_UNAVAILABLE", "FRESH_BANK_NOT_REGISTERED",
               "SCORER_UNBOUND", "NATIVE_QUALIFICATION_NOT_RUN", "BOUNDED_COLLECTOR_NOT_IMPLEMENTED"]
    if not source_matches_now or not all(sidecar_matches_now.values()):
        reasons.append("SOURCE_ASSET_DRIFT")
    reasons.extend(resource_reasons)
    record = {"schema": 1, "experiment": protocol["experiment"],
              "time_utc": datetime.now(timezone.utc).isoformat(),
              "time_zone_for_report": "America/Chicago", "status": "HOLD",
              "worktree": str(ROOT), "branch": subprocess.check_output(
                  ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "base_commit": protocol["base_commit"],
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "package_file_sha256": hashes, "protocol_sha256": hashes["protocol.json"],
              "asset_audit_sha256": hashes["asset_audit.json"],
              "exact_source_reverified": source_matches_now,
              "exact_sidecars_reverified": sidecar_matches_now, "endpoints_rechecked": endpoints_now,
              "blockers": reasons, "data_registered": False,
              "native_executed": False, "native_qualification": "NOT_RUN",
              "engineering_mechanics_tested": "see test receipt; synthetic only",
              "independent_confirmation": False, "hypothesis_supported": None,
              "promotion_eligible": False, "model_worker_calls": 0, "gpu_model_seconds": 0,
              "generated_tokens": 0, "paid_cost": 0,
              "original_restoration_attempts": vault["restoration_attempts"],
              "completed_original_endpoint_restorations": 0,
              "downloaded_weight_bytes": None,
              "partial_original_payload_logical_bytes": vault["observed_incomplete_file_logical_bytes"],
              "replacement_weights_downloaded": False,
              "remote_original_custody_metadata_verified": True,
              "expected_inference_cost": None, "expected_cost_reason": "original endpoints not restored locally; resource/data/native prerequisites blocked",
              "preflight_wall_seconds": time.monotonic() - start,
              "python": sys.version, "session_directory": str(out)}
    write_new(out / "PREPARATION_RECEIPT.json", record)
    write_new(out / "PAIRED_RESULTS_STATUS.json", {
        "expected_registered_confirmation_calls": protocol["budgets"]["max_confirmation_model_calls"],
        "completed_calls": 0, "paired_rows": 0, "results": "NOT_RUN",
        "reason": "prerequisites blocked; no outputs exist to pair or score",
        "engineering_feasible": "UNQUALIFIED", "exploratory_new_model_evidence": False,
        "genuinely_independent_confirmation": False})
    print(json.dumps({"status": "HOLD", "out": str(out), "blockers": reasons,
                      "protocol_sha256": record["protocol_sha256"], "native_executed": False}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
