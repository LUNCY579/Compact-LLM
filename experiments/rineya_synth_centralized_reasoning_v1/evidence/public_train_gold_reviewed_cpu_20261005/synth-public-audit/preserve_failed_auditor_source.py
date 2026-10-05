"""Preserve the first failed auditor bytes only if the recorded hash matches."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
receipt = json.loads((here / "FAILED_AUDIT_ATTEMPT_001.json").read_text(encoding="utf-8"))
text = (here / "audit_student_visible.py").read_text(encoding="utf-8")
text = text.replace(
    'require(sorted(row["row_id"] for row in by_arm[arm]) == enrollment["record_ids"][arm], "exact sorted approved record membership")',
    'require([row["row_id"] for row in by_arm[arm]] == enrollment["record_ids"][arm], "exact approved record enrollment/order")')
remove = (
    '                            "independent_oracle_UNKNOWN_rows": sum(row.get("status") == "PASS_ACTUAL_PUBLIC_TARGET" and row.get("all_certificate_count") == 0 for row in rows),',
    '                            "provided_UNKNOWN_supervised_targets": sum(parse(public["target"])["answer"] == "UNKNOWN" for arm in pools for public in pools[arm]),')
text = "\n".join(line for line in text.split("\n") if line not in remove and not line.startswith('        "answerability_scope":'))
raw = text.encode("utf-8")
if hashlib.sha256(raw).hexdigest() != receipt["auditor_source_sha256"]:
    raise ValueError("failed auditor reconstruction does not match preserved source hash")
with (here / "FAILED_AUDITOR_SOURCE_001.py").open("xb") as stream:
    stream.write(raw)
print("Exact failed auditor source bytes preserved: " + receipt["auditor_source_sha256"])
