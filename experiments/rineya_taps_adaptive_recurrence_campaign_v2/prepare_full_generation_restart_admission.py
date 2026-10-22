"""Freeze a request-only same-science full restart; no model or actual grant."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .run_apparatus import PACKAGE, ROOT, sha


def main(name):
    if Path(name).name != name:
        raise ValueError("filename only")
    original = PACKAGE / "bulk_admission_v1.json"
    result = json.loads(original.read_text(encoding="utf-8"))
    for relative, expected in result["dispatch_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("original scientific/source dependency drift: "+relative)
    names = [
        "bulk_restart_native.py", "bulk_restart_entry.py", "bulk_restart_supervisor.py",
        "full_generation_restart_authority.py", "prepare_full_generation_restart_admission.py",
        "full_generation_restart_readiness.py", "bulk_restart_analysis.py",
        "tests/test_full_generation_restart_authority.py", "tests/test_bulk_restart_parent_gates.py",
        "tests/test_bulk_restart_job.py", "tests/test_bulk_restart_native.py",
        "tests/test_bulk_restart_analysis.py", "BULK_RESTART_REGISTRATION_001.md",
        "BULK_RESTART_NATIVE_DRAFT_001.md", "BULK_RESTART_ANALYSIS_PLAN_001.md",
        "FULL_GENERATION_RESTART_ALLOWANCE_TEMPLATE_001.json", "FULL_GENERATION_RESTART_REQUEST_001.md",
        "audits/full_generation_restart_engineering_001.json",
        "audits/full_generation_restart_engineering_addendum_002.json",
        "audits/full_generation_restart_feasibility_001.json",
        "reviews/bulk_restart_native_review_001.json", "reviews/bulk_restart_parent_review_001.json",
        "reviews/bulk_restart_analysis_review_001.json",
    ]
    for value in names:
        path = PACKAGE / value
        result["dispatch_hashes"][path.relative_to(ROOT).as_posix()] = sha(path)
    template = PACKAGE / "FULL_GENERATION_RESTART_ALLOWANCE_TEMPLATE_001.json"
    result.update(id="FULL-GENERATION-RESTART-001", status="READY_FOR_BULK_RESTART_FP32_DISPATCH",
                  lane="GPU_FP32_FULL_GENERATION_RESTART", hard_deadline_seconds=16000,
                  review_receipt="reviews/bulk_restart_dispatch_receipt_001.json",
                  registered_utc=datetime.now(timezone.utc).isoformat(),
                  requires_additional_human_authorization=True,
                  human_extension_path=str(PACKAGE / "HUMAN_FULL_GENERATION_RESTART_ALLOWANCE_001.json"),
                  human_extension_sha256=sha(template), infrastructure_retry_available=False,
                  data_exposure="All DEV exposed;58 old validation deterministic cases exposed; shared generators; no independent confirmation",
                  original_admission="bulk_admission_v1.json", original_admission_sha256=sha(original),
                  original_run="runs/native-bulk-001", controller_revisions=0, optimizer_steps=0,
                  internal_finalization_reserve_seconds=1800, global_finalization_reserve_seconds=1800,
                  integrity_checkpoints=["1728 deterministic DEV before fixed/pool freeze", "2112 complete DEV and A5 qualification before validation"],
                  unchanged_scientific_configuration=True, new_native_permission="REQUIRED_DIRECT_USER_APPROVAL")
    assert result["max_new_tokens"] == 8 and result["qualified_maximum_prefix_tokens"] == 357
    assert result["forecast"]["physical_case_evaluations"] == 3296
    assert result["forecast"]["maximum_model_API_attempts"] == 28436
    with (PACKAGE / name).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"admission": name, "sha256": sha(PACKAGE / name),
                      "dispatch_files": len(result["dispatch_hashes"]), "native_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="full_generation_restart_admission_v1.json")
    main(parser.parse_args().name)
