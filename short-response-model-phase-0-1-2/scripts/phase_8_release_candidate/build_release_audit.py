from __future__ import annotations

"""Build the Phase 8 artifact and gate audit for the selected checkpoint.

The audit records every gate, including inserted Phase 4B, and hashes the exact
model directory passed by the runner. It does not approve release; the report
ends with a separate human decision section.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import sha256, write_json
from scripts.common.reference_selection import DEFAULT_REFERENCE_MANIFEST

GATE_LABELS = ("1", "2", "3", "4", "4B", "5", "6", "7")


def gate_status(path: Path) -> str:
    """Return PASS only for an explicit human approval line."""

    if not path.exists():
        return "MISSING"
    text = path.read_text(encoding="utf-8")
    return (
        "PASS"
        if re.search(
            r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$",
            text,
        )
        else "NOT_APPROVED"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the release-candidate audit.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local checkpoint selected by the Phase 4B reference manifest.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "RELEASE_CANDIDATE_AUDIT.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists() or not args.model_dir.is_dir():
        raise SystemExit(
            f"Release model must be a local checkpoint directory: {args.model_dir}"
        )
    if not args.reference_manifest.exists():
        raise SystemExit(
            f"Missing Phase 4B selected-reference manifest: {args.reference_manifest}"
        )

    gates = {
        label: gate_status(ROOT / "reports" / f"PHASE_{label}_STOP_GATE.md")
        for label in GATE_LABELS
    }

    files: list[dict[str, object]] = []
    for path in sorted(args.model_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(args.model_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    required_ablations = [
        "A base",
        "B all-layer SFT",
        "C best single",
        "D best 2-4",
        "E random control",
        "F selective + DPO",
        "G selective + DPO + soup",
        "4B dense vs structural candidate controls",
        "H all-layer + DPO + sparse",
        "I full combined selected system",
    ]
    status = (
        "PASS_PENDING_HUMAN_RELEASE_DECISION"
        if all(value == "PASS" for value in gates.values())
        else "BLOCKED"
    )

    lines = [
        "# RELEASE CANDIDATE AUDIT",
        "",
        "## Status",
        "",
        f"`{status}`",
        "",
        "## Selected reference",
        "",
        f"- Model directory: `{args.model_dir}`",
        f"- Phase 4B manifest: `{args.reference_manifest}`",
        f"- Phase 4B manifest SHA-256: `{sha256(args.reference_manifest)}`",
        "",
        "## Gate chain",
        "",
        *[f"- Phase {label}: `{value}`" for label, value in gates.items()],
        "",
        "## Required ablation matrix",
        "",
        *[f"- [ ] {item}" for item in required_ablations],
        "",
        "## Artifact hashes",
        "",
        *[
            f"- `{item['path']}` — `{item['sha256']}` ({item['bytes']} bytes)"
            for item in files
        ],
        "",
        "## Human release decision",
        "",
        "- Decision: `PENDING`",
        "- Reviewer:",
        "- Date:",
        "- Notes:",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        args.out.with_suffix(".json"),
        {
            "status": status,
            "model_dir": str(args.model_dir),
            "reference_manifest": str(args.reference_manifest),
            "gates": gates,
            "files": files,
        },
    )
    print(args.out)


if __name__ == "__main__":
    main()
