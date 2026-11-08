"""RRP2-QUARANTINE: the registry of banks barred from RRP-002 claims, and the guard that reads it.

Why this is code and not a paragraph in a document. Three precedents on this machine:
`model_soup.py` already refuses to run while `splits_onpanel/CONTAMINATED.md` exists;
`SOURCE_AUDIT.md` states splits_onpanel is excluded from the clean input path and that no
quarantine has been lifted; and the exposed forge rows are *length-confounded*, so reaching for
them costs a second look at a selection surface AND a biased sample. A documented exclusion that
nothing checks is an exclusion that gets forgotten at 3am.

The registry is NOT a blocklist of files. Every entry carries `bars`, the set of uses it is
barred from, because the scopes genuinely differ: `residual_capability_v23` dev is barred from
TRAIN and CONFIRMATION but is the legitimate source of the preservation DEV bank, while both
`fresh_sealed` splits are barred from ANY_READ because another programme's pre-registration owns
them. Collapsing those two into one "contaminated" flag would either leak a sealed split or
refuse the bank we are entitled to build.

Torch-free by contract: the foreground controller imports this before any native runtime exists.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, read_json, seal, write_new

from .sources import stable_hash

# Every use an entry can be barred from. ANY_READ subsumes the rest: it means do not open the
# file at all, because reading it would consume something that is not ours to spend.
USES = ("ANY_READ", "TRAIN", "SELECTION", "DEV", "CONFIRMATION")

REASONS = (
    "SELECTION_EXPOSED",            # read by a completed selection/dose-calibration loop
    "PANEL_LEAKAGE",                # measured verbatim overlap with an evaluation panel
    "SUPERSEDED",                   # replaced by a later build; carries SUPERSEDED.txt
    "CONSUMED_SEALED",              # a sealed TEST that has already been scored once
    "POLICY_LOCKED_OTHER_PROGRAMME",  # another programme's pre-registration owns it
    "SPENT",                        # scored once, so no longer a held-out surface
    "RESERVED_ELSEWHERE",           # sealed for a named future stage of another programme
)

ALL_USES = ("TRAIN", "SELECTION", "DEV", "CONFIRMATION")
NOT_HELD_OUT = ("TRAIN", "SELECTION", "CONFIRMATION")

# The registered specification. `root` selects which checkout the artifact lives in, because
# splits_onpanel exists only in the original ced-native-loader-qualification checkout and this
# is a worktree. `rows` is the count asserted by RRP002_INVENTORY.json; build() MEASURES the
# real count and records both, so a drifted assertion shows up as a mismatch instead of being
# silently trusted.
SPEC = (
    {"key": "forge_v1/test_iid/exposed_ids", "root": "worktree", "kind": "id_set",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 200, "bars": NOT_HELD_OUT,
     "evidence": "experiments/architecture_benchmark_001/evidence/retention-design-v1",
     "note": "Scored by 19 architecture_benchmark_001 floor probes. The survivors skew LONG "
             "(measured 256:137 512:156 1024:115 2048:112 4096:80), so the unexposed remainder "
             "is length-confounded as well as small."},
    {"key": "forge_v1/test_transfer/exposed_ids", "root": "worktree", "kind": "id_set",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 200, "bars": NOT_HELD_OUT,
     "evidence": "experiments/architecture_benchmark_001/evidence/retention-design-v1"},
    {"key": "splits_onpanel/train", "root": "origin", "kind": "jsonl",
     "path": "experiments/architecture_benchmark_001/outputs/splits_onpanel/train.jsonl",
     "reason": "PANEL_LEAKAGE", "asserted_rows": 845, "bars": ALL_USES,
     "evidence": "experiments/architecture_benchmark_001/outputs/splits_onpanel/CONTAMINATED.md",
     "note": "Measured 47/150 verbatim and 102/150 digit-blinded collisions with the panel."},
    {"key": "splits_onpanel/dev", "root": "origin", "kind": "jsonl",
     "path": "experiments/architecture_benchmark_001/outputs/splits_onpanel/dev.jsonl",
     "reason": "PANEL_LEAKAGE", "asserted_rows": 120, "bars": ALL_USES,
     "evidence": "experiments/architecture_benchmark_001/outputs/splits_onpanel/CONTAMINATED.md"},
    {"key": "residual_capability_v23/direct_reasoning/fresh_sealed", "root": "worktree",
     "kind": "jsonl", "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/direct_reasoning/splits/fresh_sealed.reviewed.jsonl",
     "reason": "POLICY_LOCKED_OTHER_PROGRAMME", "asserted_rows": 3240, "bars": ("ANY_READ",),
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/direct_reasoning/SEALED_DATA_POLICY.md",
     "note": "The v2.3 programme evaluates this exactly once after its own winner is frozen. "
             "RRP-002 declines it even as confirmation: spending it here would destroy it there."},
    {"key": "residual_capability_v23/procedural_rag/fresh_sealed", "root": "worktree",
     "kind": "jsonl", "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/procedural_rag/splits/fresh_sealed.reviewed.jsonl",
     "reason": "POLICY_LOCKED_OTHER_PROGRAMME", "asserted_rows": 720, "bars": ("ANY_READ",),
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/SEALED_REGISTRY.json"},
    {"key": "residual_capability_v23/direct_reasoning/dev", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/direct_reasoning/splits/dev.reviewed.jsonl",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 4860, "bars": ("TRAIN", "CONFIRMATION"),
     "evidence": "research/reasoning_posttraining_v1/RRP002_INSTRUMENT_PREREG.md",
     "note": "Used for dose calibration and DEV/OOD selection by the completed v2.3 run. Still "
             "the legitimate source of RRP2-PRES-DEV, which is why this bars TRAIN and "
             "CONFIRMATION but not DEV."},
    {"key": "residual_capability_v23/direct_reasoning/ood_dev", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/direct_reasoning/splits/ood_dev.reviewed.jsonl",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 4860, "bars": ("TRAIN", "CONFIRMATION"),
     "evidence": "research/reasoning_posttraining_v1/RRP002_INSTRUMENT_PREREG.md"},
    {"key": "residual_capability_v23/procedural_rag/dev", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/procedural_rag/splits/dev.reviewed.jsonl",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 1080, "bars": ("TRAIN", "CONFIRMATION"),
     "evidence": "research/reasoning_posttraining_v1/RRP002_INSTRUMENT_PREREG.md"},
    {"key": "residual_capability_v23/procedural_rag/ood_dev", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "residual_capability_v23/procedural_rag/splits/ood_dev.reviewed.jsonl",
     "reason": "SELECTION_EXPOSED", "asserted_rows": 1080, "bars": ("TRAIN", "CONFIRMATION"),
     "evidence": "research/reasoning_posttraining_v1/RRP002_INSTRUMENT_PREREG.md"},
    {"key": "rag_reasoning_v3_large/test", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "rag_reasoning_v3_large/reviewed_splits/test.reviewed.jsonl",
     "reason": "CONSUMED_SEALED", "asserted_rows": 738, "bars": NOT_HELD_OUT,
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/SEALED_REGISTRY.json"},
    {"key": "procedural_recovery_v211/test", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "procedural_recovery_v211/reviewed_splits/test.reviewed.jsonl",
     "reason": "CONSUMED_SEALED", "asserted_rows": 792, "bars": NOT_HELD_OUT,
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/SEALED_REGISTRY.json"},
    {"key": "general_reasoning_v212/test", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "general_reasoning_v212/reviewed_splits/test.reviewed.jsonl",
     "reason": "CONSUMED_SEALED", "asserted_rows": 1320, "bars": NOT_HELD_OUT,
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/SEALED_REGISTRY.json"},
    {"key": "multidomain_reasoning_v22/test", "root": "worktree", "kind": "jsonl",
     "path": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
     "multidomain_reasoning_v22/reviewed_splits/test.reviewed.jsonl",
     "reason": "CONSUMED_SEALED", "asserted_rows": 4050, "bars": NOT_HELD_OUT,
     "evidence": "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
                 "residual_capability_v23/SEALED_REGISTRY.json"},
    {"key": "Rineya_Long_Context_Demo_v1/all", "root": "worktree", "kind": "jsonl",
     "path": "Rineya_Long_Context_Demo_v1/data/all_answer_key.jsonl",
     "reason": "SPENT", "asserted_rows": 256, "bars": NOT_HELD_OUT,
     "evidence": "research/reasoning_posttraining_v1/evidence/inspection-20260915/"
                 "RRP002_INVENTORY.json"},
    {"key": "Rineya_Invisible_Limit_Lab_v1/test", "root": "worktree", "kind": "jsonl",
     "path": "Rineya_Invisible_Limit_Lab_v1/data/test_answer_key.jsonl",
     "reason": "SPENT", "asserted_rows": 1024, "bars": NOT_HELD_OUT,
     "evidence": "research/reasoning_posttraining_v1/evidence/inspection-20260915/"
                 "RRP002_INVENTORY.json"},
    {"key": "Rineya_Invisible_Limit_Lab_v1/test2", "root": "worktree", "kind": "jsonl",
     "path": "Rineya_Invisible_Limit_Lab_v1/data/test2_answer_key.jsonl",
     "reason": "RESERVED_ELSEWHERE", "asserted_rows": 1024, "bars": ("ANY_READ",),
     "evidence": "research/reasoning_posttraining_v1/evidence/inspection-20260915/"
                 "RRP002_INVENTORY.json",
     "note": "Sealed for that programme's Stage 6. Not ours to open."},
    # Asserted by the inventory but NOT located in either checkout. Registered anyway, with
    # present=false, so the absence is on the record instead of quietly dropping an exclusion.
    {"key": "mq-data-001", "root": "worktree", "kind": "jsonl",
     "path": "experiments/model_quality_gpu_v1/outputs/mq-data-001/dev.jsonl",
     "reason": "SUPERSEDED", "asserted_rows": None, "bars": ALL_USES,
     "evidence": "research/reasoning_posttraining_v1/evidence/inspection-20260915/"
                 "RRP002_INVENTORY.json",
     "note": "RRP002_INVENTORY.json cites a SUPERSEDED.txt for this bank. No such file was "
             "found in the worktree or the origin checkout, so no digest is recorded and the "
             "entry stands on the inventory's assertion alone."},
)


def forge_exposed_ids(worktree: Path, split: str) -> tuple[set[str], list[str]]:
    """Read the exposed ids back out of the floor-probe receipts, rather than trusting a count."""
    root = Path(worktree) / "experiments/architecture_benchmark_001/evidence/retention-design-v1"
    pattern = str(root / f"FORGE_{split}_*.json")
    ids: set[str] = set()
    files = sorted(glob.glob(pattern))
    for path in files:
        receipt = read_json(Path(path))
        if receipt.get("split") != split:
            raise ValueError(f"{path} is labelled {receipt.get('split')!r}, not {split!r}")
        ids.update(record["id"] for record in receipt["records"])
    return ids, [Path(p).name for p in files]


def _rows(path: Path) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build(worktree: Path, origin: Path) -> dict:
    """Measure every registered entry. Never opens an ANY_READ artifact's contents."""
    worktree, origin = Path(worktree), Path(origin)
    roots = {"worktree": worktree, "origin": origin}
    entries = []
    for spec in SPEC:
        entry = {"key": spec["key"], "reason": spec["reason"], "bars": list(spec["bars"]),
                 "asserted_rows": spec["asserted_rows"], "evidence_path": spec["evidence"],
                 "lifted": False}
        if spec.get("note"):
            entry["note"] = spec["note"]
        if spec["reason"] not in REASONS:
            raise ValueError(f"unregistered reason code {spec['reason']!r}")
        if not set(spec["bars"]) <= set(USES):
            raise ValueError(f"unregistered use in {spec['key']!r}")

        if spec["kind"] == "id_set":
            split = spec["key"].split("/")[1]
            ids, files = forge_exposed_ids(worktree, split)
            entry.update(path=spec["evidence"], present=True, rows=len(ids),
                         ids=sorted(ids), id_set_sha256=digest(sorted(ids)),
                         derived_from=files)
        else:
            path = roots[spec["root"]] / spec["path"]
            entry["path"] = spec["path"]
            entry["root"] = spec["root"]
            if not path.exists():
                entry.update(present=False, rows=None, sha256=None)
            elif "ANY_READ" in spec["bars"]:
                # A digest is a whole-file read, but it reveals nothing and consumes nothing.
                # The labels stay unread: no row is ever parsed out of a sealed split here.
                entry.update(present=True, sha256=stable_hash(path), rows=None,
                             rows_not_counted="ANY_READ: file not parsed, only digested")
            else:
                entry.update(present=True, sha256=stable_hash(path), rows=_rows(path))
        if entry.get("rows") is not None and spec["asserted_rows"] is not None:
            entry["rows_match_assertion"] = entry["rows"] == spec["asserted_rows"]
        entries.append(entry)

    return seal({
        "record": "RRP2_QUARANTINE",
        "schema": 1,
        "what_this_is": ("Banks barred from RRP-002 claims, with the scope of each bar. Built to "
                         "be checked against by code, not read by a human at decision time."),
        "worktree": str(worktree),
        "origin_checkout": str(origin),
        "uses": list(USES),
        "reasons": list(REASONS),
        "entries": entries,
        "totals": {
            "entries": len(entries),
            "present": sum(1 for e in entries if e["present"]),
            "absent": sum(1 for e in entries if not e["present"]),
            "any_read_barred": sum(1 for e in entries if "ANY_READ" in e["bars"]),
            "rows_barred_where_counted": sum(e["rows"] or 0 for e in entries
                                             if e["reason"] != "SELECTION_EXPOSED"
                                             or "/exposed_ids" not in e["key"]),
            "assertion_mismatches": sorted(e["key"] for e in entries
                                           if e.get("rows_match_assertion") is False),
        },
        "no_quarantine_is_lifted": all(not e["lifted"] for e in entries),
        "model_calls": 0,
    })


def load(path: Path) -> dict:
    registry = read_json(Path(path))
    if registry.get("record") != "RRP2_QUARANTINE":
        raise ValueError("not a quarantine registry")
    if not registry["no_quarantine_is_lifted"]:
        raise ValueError("a quarantine entry is marked lifted; that needs a human decision")
    return registry


def barred_ids(registry: dict, use: str) -> dict[str, str]:
    """id -> the key that bars it, for every entry that bars `use` and enumerates ids."""
    if use not in USES:
        raise ValueError(f"unknown use {use!r}")
    out: dict[str, str] = {}
    for entry in registry["entries"]:
        bars = entry["bars"]
        if use not in bars and "ANY_READ" not in bars:
            continue
        for value in entry.get("ids", ()):
            out.setdefault(value, entry["key"])
    return out


def refuse(candidate_ids, registry: dict, *, use: str) -> None:
    """Raise if any candidate id is barred from `use`. Called before a bank is admitted."""
    barred = barred_ids(registry, use)
    hits = sorted({str(i) for i in candidate_ids} & set(barred))
    if hits:
        where = sorted({barred[h] for h in hits})
        raise ValueError(f"{len(hits)} candidate id(s) barred from {use} by {where}: "
                         f"{hits[:5]}{'...' if len(hits) > 5 else ''}")


def assert_readable(registry: dict, relative_path: str) -> None:
    """Raise if a path is ANY_READ-barred. The guard for 'do not open someone else's seal'."""
    wanted = relative_path.replace("\\", "/")
    for entry in registry["entries"]:
        if "ANY_READ" in entry["bars"] and entry.get("path", "").replace("\\", "/") == wanted:
            raise ValueError(f"{entry['key']} is ANY_READ-barred ({entry['reason']}): "
                             f"{entry.get('note', entry['evidence_path'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = build(args.worktree, args.origin)
    write_new(args.out, registry)
    print(json.dumps(registry["totals"], indent=2))
    for entry in registry["entries"]:
        flag = "" if entry["present"] else "   [ARTIFACT NOT FOUND]"
        mismatch = "" if entry.get("rows_match_assertion", True) else "   [ROW COUNT MISMATCH]"
        print(f"  {entry['reason']:30} {entry['key']:52} rows={entry['rows']}{flag}{mismatch}")


if __name__ == "__main__":
    main()
