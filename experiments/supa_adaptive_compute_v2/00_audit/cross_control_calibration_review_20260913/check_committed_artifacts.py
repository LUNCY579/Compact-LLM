"""Supplement the independent audit with post-run evidence commitment checks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import audit_calibration as audit

BASE = Path(__file__).resolve().parent
first = json.loads((BASE / "attempt_001/receipt.json").read_text(encoding="utf-8"))
report_path = Path(first["report_path"])
report = audit.read(report_path)
commit = audit.git("rev-parse", "HEAD").decode().strip()
paths = [report_path, report_path.parent / "fit_scale.json", report_path.parent / "run_manifest.json",
         report_path.parent / "started_manifest.json"]
paths.extend(report_path.parent / "cases" / name for name in report["case_files"])
audit.committed_at(commit, [p.relative_to(audit.REPO).as_posix() for p in paths])
output = BASE / "commitment_receipt.json"
with output.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump({"status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "commit": commit, "file_count": len(paths), "check_count": len(audit.checks),
               "checks": audit.checks, "auditor_sha256": audit.digest(__file__),
               "main_audit_sha256": audit.digest(BASE / "attempt_001/receipt.json"),
               "report_sha256": audit.digest(report_path)}, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"status": "PASS", "commit": commit, "files": len(paths), "checks": len(audit.checks)}))
