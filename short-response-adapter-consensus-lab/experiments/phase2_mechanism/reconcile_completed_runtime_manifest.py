from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "results/runs/phase2_selected_committee_dev_free_run_dev_full.jsonl"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".runtime.json")
DEFAULT_EXPECTED_ROWS = 48_600
KEY_FIELDS = ("dataset", "committee", "method", "example_id")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_is_running(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def audit_output(path: Path, *, expected_identity_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result JSONL is missing: {path}")
    keys: set[tuple[str, ...]] = set()
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"Blank JSONL record at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL record at line {line_number}: {exc}") from exc
            missing = [field for field in KEY_FIELDS if row.get(field) is None]
            if missing:
                raise RuntimeError(f"Missing key fields at line {line_number}: {missing}")
            if row.get("runtime_identity_sha256") != expected_identity_sha256:
                raise RuntimeError(
                    f"Runtime identity drift at line {line_number}: "
                    f"{row.get('runtime_identity_sha256')!r}"
                )
            key = tuple(str(row[field]) for field in KEY_FIELDS)
            if key in keys:
                raise RuntimeError(f"Duplicate result key at line {line_number}: {key}")
            keys.add(key)
            row_count += 1
    return {
        "rows": row_count,
        "unique_keys": len(keys),
        "sha256": sha256(path),
        "runtime_identity_sha256": expected_identity_sha256,
    }


def _write_json_atomic(path: Path, packet: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def reconcile_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    expected_rows: int,
    process_checker: Callable[[int], bool] = pid_is_running,
    observed_at_utc: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Close a stale manifest only after strict, source-bound JSONL validation.

    This records an unclean final invocation; it never relabels that invocation as
    a clean return and never alters the raw result file.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_sha256 = str(manifest.get("identity_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", identity_sha256):
        raise RuntimeError("Runtime manifest has no valid identity SHA-256")
    recorded_output = Path(str(manifest.get("output_path", ""))).resolve()
    if recorded_output != output_path.resolve():
        raise RuntimeError(
            f"Runtime manifest points at a different output: {recorded_output} != {output_path.resolve()}"
        )
    output_audit = audit_output(output_path, expected_identity_sha256=identity_sha256)
    if output_audit["rows"] != expected_rows or output_audit["unique_keys"] != expected_rows:
        raise RuntimeError(
            "Result is not complete and unique: "
            f"rows={output_audit['rows']}, unique={output_audit['unique_keys']}, "
            f"expected={expected_rows}"
        )

    if manifest.get("status") == "COMPLETE":
        if (
            manifest.get("observed_rows") != expected_rows
            or manifest.get("expected_rows") != expected_rows
        ):
            raise RuntimeError("Manifest says COMPLETE but its row counters disagree")
        return manifest, False

    invocations = manifest.get("invocations", [])
    if not invocations or invocations[-1].get("status") != "STARTED":
        raise RuntimeError("Only a manifest with a stale final STARTED invocation can be reconciled")
    final_invocation = invocations[-1]
    pid = int(final_invocation.get("pid", -1))
    if pid > 0 and process_checker(pid):
        raise RuntimeError(f"Refusing to reconcile while recorded PID {pid} is still running")

    timestamp = observed_at_utc or utc_now()
    rows_before = int(final_invocation.get("rows_before", 0))
    final_invocation.update(
        {
            "status": "INTERRUPTED_OR_UNCLEAN_EXIT",
            "observed_at_reconciliation_utc": timestamp,
            "rows_observed_at_reconciliation": expected_rows,
            "rows_added_before_unclean_exit": expected_rows - rows_before,
        }
    )
    script_path = Path(__file__).resolve()
    reconciliation = {
        "status": "POST_HOC_COMPLETE_RESULT_VERIFIED",
        "observed_at_utc": timestamp,
        "reason": (
            "The final worker appended the remaining durable rows but exited before "
            "finish_runtime_invocation closed the manifest."
        ),
        "output_sha256": output_audit["sha256"],
        "rows": expected_rows,
        "unique_keys": expected_rows,
        "runtime_identity_sha256": identity_sha256,
        "script_path": str(script_path.relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256(script_path),
    }
    manifest.setdefault("reconciliations", []).append(reconciliation)
    manifest.update(
        {
            "status": "COMPLETE",
            "completion_basis": "POST_HOC_VERIFIED_COMPLETE",
            "observed_rows": expected_rows,
            "expected_rows": expected_rows,
            "verified_output": output_audit,
            "updated_at_utc": timestamp,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile a complete JSONL whose final worker did not close its runtime manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    args = parser.parse_args()
    packet, changed = reconcile_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "status": packet["status"],
                "changed": changed,
                "observed_rows": packet["observed_rows"],
                "output_sha256": packet.get("verified_output", {}).get("sha256"),
                "manifest": str(args.manifest.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
