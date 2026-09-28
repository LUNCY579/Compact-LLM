"""Freeze CALIBRATION, DEV or TEST scope receipts before any donor read (COMP-RS-001 Amendment 1, section 1; C8).

python -m experiments.maxrl_comp002_pretraining_v1.scope_receipt --data DATA --role calibration --out NEW.json

Generalises q1_dev_scope.py: core filter, per-signature rows/parents/ids_sha256/parents_sha256 with
preflight.digest over sorted lists, shared-parent checks, create-only. Only id/parent_id/tag/skills are
retained from canonical rows and only id from policy rows. Role 'test' also requires --spec-only-ack.
Role 'dev' covers R, S, P and RS (C8: the H1 P gate needs P, which Q1_DEV_SCOPE.json lacks).
Guard (preservation) banks have no scope receipt yet: the scorer binds them by their manifest-checked
bank sha (C8), and a guard_test scope receipt is future work.

The code map (this file and preflight.py) is hashed at import and re-checked just before the receipt
is written; a mismatch refuses the write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1 import preflight
from experiments.maxrl_comp002_pretraining_v1.preflight import MANIFEST, digest

ROLES = ("calibration", "dev", "test")
SIGNATURES = ("R", "S", "P", "RS")
FIELDS = ("id", "parent_id", "tag", "skills")
EXPECTED = {"calibration": {"signatures": ("R", "S", "P"), "rows": 150},
            "dev": {"signatures": ("R", "S", "P", "RS"), "rows": 250},
            "test": {"signatures": ("R", "S", "P", "RS"), "rows": 250}}
# Every source file the receipt's values depend on: hashed at import, re-checked before writing.
CODE_FILES = {"scope_receipt.py": Path(__file__).resolve(),
              "preflight.py": Path(preflight.__file__).resolve()}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    """sha256 of every file in CODE_FILES, keyed by name."""
    return {name: sha256_bytes(path.read_bytes()) for name, path in CODE_FILES.items()}


_IMPORT_CODE = code_sha256()


def project(line: str, fields=FIELDS) -> dict:
    """Parse one JSONL row and keep only the named fields."""
    row = json.loads(line)
    if type(row) is not dict or any(key not in row for key in fields):
        raise ValueError("row lacks scope fields")
    return {key: row[key] for key in fields}


def signature_scope(rows, signatures=SIGNATURES) -> dict:
    """Core-filtered per-signature scope over projected rows; one row per parent per signature."""
    seen = set()
    groups = {sig: {"ids": [], "parents": []} for sig in signatures}
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate id")
        seen.add(row["id"])
        sig = "".join(row["skills"])
        if row["tag"] == "core" and sig in groups:
            groups[sig]["ids"].append(row["id"])
            groups[sig]["parents"].append(row["parent_id"])
    present = {sig: g for sig, g in groups.items() if g["ids"]}
    out = {}
    for sig, group in present.items():
        parents = set(group["parents"])
        if len(parents) != len(group["parents"]):
            raise ValueError("more than one core row per parent: " + sig)
        out[sig] = {"rows": len(group["ids"]), "parents": len(parents),
                    "ids_sha256": digest(sorted(group["ids"])), "parents_sha256": digest(sorted(parents))}
    parent_sets = [set(group["parents"]) for group in present.values()]
    shared = len(parent_sets[0]) if parent_sets and all(s == parent_sets[0] for s in parent_sets) else None
    return {"rows": len(seen), "signatures": out, "absent_signatures": [s for s in signatures if s not in present],
            "shared_parent_worlds": shared}


def verified_lines(data: Path, relative: str, manifest: dict) -> tuple[list[str], str]:
    raw = (data / relative).read_bytes()
    value = sha256_bytes(raw)
    if relative not in manifest["files"] or value != manifest["files"][relative]["sha256"]:
        raise SystemExit("data file drift: " + relative)
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines, value


def write_receipt(receipt: dict, out: Path) -> None:
    """Create-only LF write, refused if any CODE_FILES source changed since import."""
    if code_sha256() != _IMPORT_CODE:
        raise SystemExit("scope-receipt source changed since import")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--spec-only-ack", action="store_true",
                    help="required for role test: only id/parent_id/tag/skills are retained")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    role = args.role
    if role == "test" and not args.spec_only_ack:
        raise SystemExit("role test requires --spec-only-ack")
    if args.out.exists():
        raise SystemExit("scope receipt exists (create-only)")
    data = args.data.resolve(strict=True)
    if args.out.resolve().is_relative_to(data):
        raise SystemExit("protected output path")
    manifest_raw = (data / "manifest.json").read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST:
        raise SystemExit("dataset manifest drift")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    canonical_lines, canonical_sha = verified_lines(data, "canonical/" + role + ".jsonl", manifest)
    rows = [project(line) for line in canonical_lines]
    del canonical_lines
    policy_lines, policy_sha = verified_lines(data, "policy/" + role + ".jsonl", manifest)
    policy_ids = [project(line, ("id",))["id"] for line in policy_lines]
    del policy_lines
    try:
        scope = signature_scope(rows)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if scope["rows"] != manifest["counts"][role]:
        raise SystemExit("row count mismatch: " + role)
    if len(policy_ids) != len(set(policy_ids)) or set(policy_ids) != {r["id"] for r in rows}:
        raise SystemExit("policy/canonical ID mismatch: " + role)
    expected = EXPECTED[role]
    if tuple(scope["signatures"]) != expected["signatures"]:
        raise SystemExit("signature set mismatch: " + role)
    if any(v["rows"] != expected["rows"] or v["parents"] != expected["rows"] for v in scope["signatures"].values()):
        raise SystemExit("per-signature row/parent count mismatch: " + role)
    if scope["shared_parent_worlds"] != expected["rows"]:
        raise SystemExit("shared-parent alignment mismatch: " + role)
    receipt = {"experiment": "COMP-RS-001", "stage": "S0PRE_" + role.upper() + "_SCOPE",
               "amendment": "COMP_RS_001_AMENDMENT_1.md section 1; clarification C8", "role": role,
               "dataset_manifest_sha256": MANIFEST,
               "canonical_" + role + "_sha256": canonical_sha, "policy_" + role + "_sha256": policy_sha,
               "core_filter": "tag == 'core'", "fields_retained": list(FIELDS),
               "signatures": scope["signatures"], "absent_signatures": scope["absent_signatures"],
               "shared_parent_worlds": scope["shared_parent_worlds"], "rows_in_bank": scope["rows"],
               "code_sha256": dict(_IMPORT_CODE), "code_bundle_sha256": digest(_IMPORT_CODE),
               "spec_only_ack": bool(args.spec_only_ack), "test_challenge_read": role == "test",
               "model_outputs_read": False,
               "note": "Scope identities only. Every scorer asserts this receipt's digest."}
    write_receipt(receipt, args.out)
    print(json.dumps({"role": role, "shared_parents": receipt["shared_parent_worlds"],
                      "rows_per_signature": {k: v["rows"] for k, v in receipt["signatures"].items()},
                      "absent": receipt["absent_signatures"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
