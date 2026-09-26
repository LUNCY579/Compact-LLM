"""Independent verification of the gate-off ablation runs.

Closes the two gaps an adversarial review raised against run_gate_off_eval.py:

  1. The gate-ON self-validation was asserted in prose but never implemented or
     recorded. A literal byte comparison can never pass, because every record
     carries its own ``inference_seconds``. This compares every field EXCEPT
     ``inference_seconds``, joined on (evaluation, example_id), and records the
     outcome.

  2. Nothing checked that the adapter under test was the right adapter, or was
     attached at all. This hashes the adapter tensors, cross-checks the core
     against the adapter's own recorded base model, and applies a positive
     control: an adapter arm's gate-ON result must differ from the frozen
     baseline, otherwise the adapter silently failed to load.

Read-only with respect to the lab; writes one JSON report into this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
EXPERIMENT = ABLATION_DIR.parents[1]
RESULTS = ABLATION_DIR / "results"

CORE_FOR_STAGE = {"phase1": "a11", "phase1_a1": "a1"}
VOLATILE_FIELDS = {"inference_seconds"}


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[(r["evaluation"], r["example_id"])] = r
    return rows


def compare_rows(new_path: Path, stored_path: Path) -> dict:
    """Field-exact comparison ignoring per-run timing only."""
    new, stored = load_rows(new_path), load_rows(stored_path)
    same_keys = set(new) == set(stored)
    mismatched: list[str] = []
    fields_seen: set[str] = set()
    for key in sorted(set(new) & set(stored)):
        a, b = new[key], stored[key]
        fields = (set(a) | set(b)) - VOLATILE_FIELDS
        fields_seen |= fields
        for f in fields:
            if str(a.get(f)) != str(b.get(f)):
                mismatched.append(f"{key[0]}/{key[1]}:{f}")
                break
    return {
        "stored": str(stored_path),
        "n_new": len(new),
        "n_stored": len(stored),
        "same_example_ids": same_keys,
        "compared_fields": sorted(fields_seen),
        "ignored_fields": sorted(VOLATILE_FIELDS),
        "n_mismatched_rows": len(mismatched),
        "mismatched": mismatched[:20],
        "reproduced_stored_rows": same_keys and not mismatched,
    }


def adapter_identity(manifest: dict) -> dict | None:
    d = manifest.get("adapter_dir")
    if not d:
        return None
    p = Path(d)
    weights = p / "adapter_model.safetensors"
    cfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
    declared = cfg.get("base_model_name_or_path", "")
    return {
        "adapter_dir": str(p),
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_bytes": weights.stat().st_size,
        "declared_base_model": declared,
        "core_path_used": manifest["core_path"],
        "base_model_matches_core": Path(declared).resolve() == Path(manifest["core_path"]).resolve(),
        "target_modules": sorted(cfg.get("target_modules") or []),
        "layers_to_transform": cfg.get("layers_to_transform"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "expected_core_for_stage": CORE_FOR_STAGE.get(manifest.get("adapter_stage") or ""),
        "core_stage_pairing_correct": (
            CORE_FOR_STAGE.get(manifest.get("adapter_stage") or "") == manifest["core"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ABLATION_DIR / "results" / "verification.json"))
    args = ap.parse_args()

    runs = {}
    for d in sorted(RESULTS.iterdir()):
        mf = d / "manifest.json"
        if d.is_dir() and mf.is_file():
            runs[d.name] = json.loads(mf.read_text(encoding="utf-8"))

    report: dict = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": {},
    }

    for label, m in runs.items():
        entry = {
            "core": m["core"],
            "arm": m["arm"],
            "gate": m["gate"],
            "abstain_logit_bias_used": m["abstain_logit_bias_used"],
            "adapter_stage": m.get("adapter_stage"),
            "canonical_checkpoint_unchanged": m["canonical_checkpoint_unchanged"],
            "frozen_base_unchanged": m["frozen_base_unchanged"],
            "summary": m["summary"],
            "adapter": adapter_identity(m),
        }
        # gate-ON runs of an arm that has stored history must reproduce it exactly
        if m["gate"] == "on" and m.get("adapter_stage"):
            stored = EXPERIMENT / "results" / m["adapter_stage"] / f"arm_{m['arm']}" / "per_example.jsonl"
            if stored.is_file():
                entry["reproduction"] = compare_rows(d_path(label) / "per_example.jsonl", stored)
        elif m["gate"] == "on" and m["arm"] == "A":
            for stage, core in CORE_FOR_STAGE.items():
                if core == m["core"]:
                    stored = EXPERIMENT / "results" / stage / "arm_A" / "per_example.jsonl"
                    if stored.is_file():
                        entry["reproduction"] = compare_rows(d_path(label) / "per_example.jsonl", stored)
        report["runs"][label] = entry

    # Positive control: an adapter arm must not score identically to the frozen
    # arm under the same gate, or the adapter silently failed to attach.
    frozen = {
        (m["core"], m["gate"]): m["summary"]["target_total"]["correct"]
        for m in runs.values()
        if m["arm"] == "A"
    }
    for label, m in runs.items():
        if m["arm"] == "A":
            continue
        ref = frozen.get((m["core"], m["gate"]))
        got = m["summary"]["target_total"]["correct"]
        report["runs"][label]["adapter_attachment_positive_control"] = {
            "frozen_same_gate_target_correct": ref,
            "this_arm_target_correct": got,
            "differs_from_frozen": None if ref is None else ref != got,
            "note": "identical scores would suggest the adapter did not attach",
        }

    problems = []
    for label, e in report["runs"].items():
        if not e["canonical_checkpoint_unchanged"]:
            problems.append(f"{label}: canonical checkpoint changed")
        if not e["frozen_base_unchanged"]:
            problems.append(f"{label}: frozen base changed")
        a = e.get("adapter")
        if a:
            if not a["base_model_matches_core"]:
                problems.append(f"{label}: adapter declares a different base model than the core used")
            if not a["core_stage_pairing_correct"]:
                problems.append(f"{label}: --core does not match --adapter-stage")
            if a["layers_to_transform"] != [17, 18, 19]:
                problems.append(f"{label}: unexpected layers {a['layers_to_transform']}")
        r = e.get("reproduction")
        if r and not r["reproduced_stored_rows"]:
            problems.append(f"{label}: did NOT reproduce stored rows ({r['n_mismatched_rows']} mismatched)")
        pc = e.get("adapter_attachment_positive_control")
        if pc and pc["differs_from_frozen"] is False:
            problems.append(f"{label}: scores identical to frozen; adapter may not have attached")

    report["problems"] = problems
    report["status"] = "PASS" if not problems else "REVIEW_REQUIRED"

    Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")
    print(f"status: {report['status']}   runs: {report['n_runs']}")
    for label, e in sorted(report["runs"].items()):
        r = e.get("reproduction")
        rep = "n/a" if not r else ("reproduced" if r["reproduced_stored_rows"]
                                   else f"MISMATCH {r['n_mismatched_rows']}")
        pc = e.get("adapter_attachment_positive_control")
        if not pc:
            att = "n/a"
        elif pc["differs_from_frozen"] is None:
            # No frozen run at this (core, gate) to compare against yet.
            att = "no-reference"
        else:
            att = "attached" if pc["differs_from_frozen"] else "SUSPECT"
        print(f"  {label:34} core={e['core']:4} arm={e['arm']} gate={e['gate']:3} "
              f"target={e['summary']['target_total']['correct']:>3}/256  "
              f"repro={rep:14} adapter={att}")
    for p in problems:
        print("  PROBLEM:", p)


def d_path(label: str) -> Path:
    return RESULTS / label


if __name__ == "__main__":
    main()
