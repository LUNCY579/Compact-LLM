from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs/phase2_selected_dev_free_run.yaml"
TEACHER = ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
SELECTED_MANIFEST = ROOT / "artifacts/phase2_mechanism/selected_dev_free_run_manifest.json"
CONFIG = ROOT / "configs/phase2_selected_committee_dev_free_run.yaml"
MANIFEST = ROOT / "artifacts/phase2_mechanism/selected_committee_dev_manifest.json"
OUTPUT = ROOT / "results/runs/phase2_selected_committee_dev_free_run_dev_full.jsonl"
EXPERTS = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def members(mask: int) -> list[str]:
    return [expert for index, expert in enumerate(EXPERTS) if mask & (1 << index)]


def main() -> None:
    if OUTPUT.is_file() and OUTPUT.stat().st_size:
        raise RuntimeError("Selected committee DEV output already exists; refusing to refreeze")
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER.read_text(encoding="utf-8-sig"))
    selected_manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
    if teacher.get("status") != "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS":
        raise RuntimeError("Frozen teacher-forced analysis is not complete")
    if selected_manifest.get("test_data_read_for_selection") is not False:
        raise RuntimeError("The source full-K7 selection was not DEV-only")
    hard_scaling = {
        int(row["k"]): row
        for row in teacher["committee_scaling"]
        if row["method"] == "hard_plurality"
    }
    if set(hard_scaling) != set(range(1, 8)):
        raise RuntimeError("Hard committee-size inventory is incomplete")

    selected_masks = [int(hard_scaling[k]["best_mask"]) for k in range(1, 6)]
    loo_masks = [127 ^ (1 << index) for index in range(7)]
    masks = selected_masks + [mask for mask in loo_masks if mask not in selected_masks]
    if len(masks) != 12 or len(set(masks)) != 12:
        raise RuntimeError(f"Unexpected targeted committee inventory: {masks}")
    committees = []
    roles: dict[str, Any] = {}
    for mask in masks:
        size = len(members(mask))
        best_k = next(
            (k for k, row in hard_scaling.items() if int(row["best_mask"]) == mask), None
        )
        removed = [expert for expert in EXPERTS if expert not in members(mask)] if size == 6 else []
        name = f"best_k{size}_m{mask}" if best_k is not None and size < 6 else f"loo_without_{removed[0]}"
        role_parts = []
        if best_k is not None:
            role_parts.append(f"teacher_forced_best_k{best_k}")
        if removed:
            role_parts.append(f"leave_one_out_{removed[0]}")
        role = "+".join(role_parts)
        committees.append({"name": name, "experts": members(mask), "role": role})
        roles[name] = {
            "mask": mask,
            "members": members(mask),
            "k": size,
            "teacher_forced_best_at_k": best_k,
            "removed_expert": removed[0] if removed else None,
            "role": role,
        }

    config = {
        "name": "phase2_selected_committee_dev_free_run",
        "backbone": source["backbone"],
        "experts": source["experts"],
        "committees": committees,
        "methods": ["hard_plurality"],
        "datasets": source["datasets"],
        "max_sequence_length": source["max_sequence_length"],
        "max_new_tokens": source["max_new_tokens"],
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": source["seed"],
    }
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "DEV_SELECTED_COMMITTEE_PANEL_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MODEL-SELECTION / TARGETED FREE-RUNNING COMMITTEE PANEL",
        "freeze_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "source_teacher_analysis": str(TEACHER),
        "source_teacher_analysis_sha256": sha256(TEACHER),
        "source_selected_manifest": str(SELECTED_MANIFEST),
        "source_selected_manifest_sha256": sha256(SELECTED_MANIFEST),
        "selection_used_test_outcomes": False,
        "selection_rule": {
            "best_k1_to_k5": "teacher-forced maximum all-gold-token example accuracy at K; frozen best_mask",
            "leave_one_out": "all seven K=6 complements of the full K=7 expert set",
            "best_k6": "already included among the seven leave-one-out committees",
            "full_k7": "reuse full_k7/hard_plurality from phase2_selected_dev_free_run; do not recompute",
            "no_switching": "do not change committees, aggregation, EOS, decoder, or reporting after panel output",
        },
        "committee_count": len(committees),
        "expected_rows": EXPECTED_ROWS,
        "committees": roles,
        "config": {
            "path": str(CONFIG),
            "file_sha256": sha256(CONFIG),
            "canonical_sha256": canonical_hash(config),
        },
        "output": str(OUTPUT),
        "same_split_full_k7_partial_output_existed_at_freeze": (
            ROOT / "results/runs/phase2_selected_dev_free_run_dev_full.jsonl"
        ).is_file(),
        "partial_dev_outcome_boundary": (
            "Same-split DEV full-K7 rows already existed and some row-level outcome data had been visible. "
            "This panel is therefore explicitly DEV selection evidence; committee identities are determined "
            "only by the frozen teacher-forced best-at-K rule and exhaustive leave-one-out structure."
        ),
        "notes": [
            "No TEST outcome is used to choose this panel.",
            "The panel measures trajectory, answer/abstain, latency, and forward effects omitted by cached counterfactuals.",
            "Running every one of 127 committees autoregressively was rejected as disproportionate; all 127 remain covered under teacher forcing.",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "committee_count": len(committees),
                "expected_rows": EXPECTED_ROWS,
                "config": manifest["config"],
                "output_exists": OUTPUT.exists(),
            },
            indent=2,
        )
    )


EXPECTED_ROWS = 12 * 4050


if __name__ == "__main__":
    main()
