from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experiments.phase2_mechanism.watch_selected_resume_history import merge_once
except ModuleNotFoundError:  # Direct ``python experiments/...`` execution.
    from watch_selected_resume_history import merge_once


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/runs/phase2_selected_dev_free_run_dev_full.jsonl"
PARITY = ROOT / "results/phase2_mechanism/rank_topk_parity_verification.json"
OUTPUT = ROOT / "results/phase2_mechanism/selected_dev_runtime_boundaries.json"
REPORT = ROOT / "reports/PHASE2_SELECTED_RUNTIME_BOUNDARIES.md"
RESUME_HISTORY = ROOT / "artifacts/phase2_mechanism/selected_dev_resume_history.json"
RESUME_WATCHER = ROOT / "experiments/phase2_mechanism/watch_selected_resume_history.py"
EXPECTED_CONFIG = "a887a4874f9717584aacb3a0f7019224d4a1571c8b76f28a92c04b965e56552e"
EXPECTED_ROWS = 16_200
RANK_FIRST_LINE = 12_151
PARTIAL_TOPK_FIRST_LINE = 12_239
POST_EXAMPLE_CACHE_RELEASE_FIRST_LINE = 12_765
OOM_RETRY_TELEMETRY_FIRST_LINE = 12_783


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest() -> dict[str, Any]:
    # This is a historical runtime record, not a live status file. Once it is
    # captured against the complete raw run, later reporting/runtime hardening
    # must not rewrite the source hashes that actually closed the run.
    if OUTPUT.is_file():
        captured = load_json(OUTPUT)
        if captured.get("status") != "SELECTED_DEV_RUNTIME_BOUNDARIES_CAPTURED":
            raise RuntimeError("Existing selected runtime-boundary artifact has an invalid status")
        if captured.get("run", {}).get("sha256") != sha256(RUN):
            raise RuntimeError("Completed selected run changed after runtime-boundary capture")
        if captured.get("run", {}).get("rows") != EXPECTED_ROWS:
            raise RuntimeError("Existing runtime-boundary artifact does not bind 16,200 rows")
        return captured
    parity = load_json(PARITY)
    if parity.get("status") != "RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED":
        raise RuntimeError("Rank implementation parity is not verified")
    merge_once()
    resume_history = load_json(RESUME_HISTORY)
    if resume_history.get("status") != "SELECTED_DEV_RESUME_HISTORY_COMPLETE":
        raise RuntimeError("Selected DEV resume history is not complete")
    if resume_history.get("observed_rows") != EXPECTED_ROWS:
        raise RuntimeError("Selected DEV resume history does not bind the complete row count")
    if resume_history.get("generator", {}).get("sha256") != sha256(RESUME_WATCHER):
        raise RuntimeError("Selected DEV resume history does not bind the current watcher")
    counts: collections.Counter[str] = collections.Counter()
    keys: set[tuple[str, str, str, str]] = set()
    config_hashes: set[str] = set()
    selected_lines: dict[int, dict[str, Any]] = {}
    telemetry_lines: list[int] = []
    retry_total = 0
    retry_examples = 0
    row_count = 0
    with RUN.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            row_count = line_number
            method = str(row["method"])
            counts[method] += 1
            key = (
                str(row["dataset"]),
                str(row["committee"]),
                method,
                str(row["example_id"]),
            )
            if key in keys:
                raise RuntimeError(f"Duplicate selected-run key at line {line_number}: {key}")
            keys.add(key)
            config_hashes.add(str(row["config_sha256"]))
            if line_number in {
                RANK_FIRST_LINE,
                PARTIAL_TOPK_FIRST_LINE - 1,
                PARTIAL_TOPK_FIRST_LINE,
                POST_EXAMPLE_CACHE_RELEASE_FIRST_LINE,
                OOM_RETRY_TELEMETRY_FIRST_LINE,
                EXPECTED_ROWS,
            }:
                selected_lines[line_number] = {
                    "method": method,
                    "example_id": str(row["example_id"]),
                }
            if "cuda_oom_retry_count" in row:
                telemetry_lines.append(line_number)
                retries = int(row["cuda_oom_retry_count"])
                retry_total += retries
                retry_examples += int(retries > 0)
    if row_count != EXPECTED_ROWS or len(keys) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Selected run must be complete and unique before boundary capture: "
            f"rows={row_count}, unique={len(keys)}, expected={EXPECTED_ROWS}"
        )
    expected_counts = {
        "hard_plurality": 4050,
        "probability_mean": 4050,
        "calibrated_probability_mean": 4050,
        "reciprocal_rank_top5": 4050,
    }
    if dict(counts) != expected_counts:
        raise RuntimeError(f"Unexpected selected-run method counts: {dict(counts)}")
    if config_hashes != {EXPECTED_CONFIG}:
        raise RuntimeError(f"Selected-run config hash drift: {sorted(config_hashes)}")
    if selected_lines[RANK_FIRST_LINE]["method"] != "reciprocal_rank_top5":
        raise RuntimeError("Reciprocal-rank block does not begin at its frozen method boundary")
    if telemetry_lines and telemetry_lines[0] != OOM_RETRY_TELEMETRY_FIRST_LINE:
        raise RuntimeError(
            f"OOM retry telemetry begins at line {telemetry_lines[0]}, "
            f"expected {OOM_RETRY_TELEMETRY_FIRST_LINE}"
        )
    rules = ROOT / "src/consensus_lab/voting/rules.py"
    direct = ROOT / "src/consensus_lab/evaluation/direct.py"
    decoder = ROOT / "src/consensus_lab/decoding/decoder.py"
    packet = {
        "schema_version": "1.0",
        "status": "SELECTED_DEV_RUNTIME_BOUNDARIES_CAPTURED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "path": str(RUN),
            "sha256": sha256(RUN),
            "rows": row_count,
            "unique_keys": len(keys),
            "config_sha256": EXPECTED_CONFIG,
            "method_counts": dict(counts),
        },
        "boundaries": {
            "rank_method_first_line": RANK_FIRST_LINE,
            "legacy_full_stable_sort_last_line": PARTIAL_TOPK_FIRST_LINE - 1,
            "stable_partial_topk_first_line": PARTIAL_TOPK_FIRST_LINE,
            "post_example_cache_release_first_line": POST_EXAMPLE_CACHE_RELEASE_FIRST_LINE,
            "oom_retry_telemetry_first_line": (
                telemetry_lines[0] if telemetry_lines else None
            ),
            "selected_line_identities": {
                str(line_number): value
                for line_number, value in sorted(selected_lines.items())
            },
        },
        "rank_equivalence": {
            "legacy_operation": "torch.argsort(scores, descending=True, stable=True)[:, :k]",
            "current_operation": "consensus_lab.voting.rules._stable_topk_indices(scores, k)",
            "parity_artifact": str(PARITY),
            "parity_artifact_sha256": sha256(PARITY),
            "all_checks_exact": parity["equivalence"]["all_checks_exact"],
        },
        "oom_retry_telemetry": {
            "rows_with_field": len(telemetry_lines),
            "examples_with_one_or_more_retries": retry_examples,
            "total_retries": retry_total,
            "semantic_boundary": (
                "failed allocation attempts are discarded before append; only successful "
                "recomputation is stored and scored"
            ),
        },
        "current_runtime": {
            "voting_rules_sha256": sha256(rules),
            "direct_evaluator_sha256": sha256(direct),
            "reference_decoder_sha256": sha256(decoder),
        },
        "resume_history": {
            "path": str(RESUME_HISTORY),
            "sha256": sha256(RESUME_HISTORY),
            "generator_sha256": sha256(RESUME_WATCHER),
            "restart_event_count": resume_history["restart_event_count"],
            "failed_attempt_count": resume_history["failed_attempt_count"],
            "clean_batch_count": resume_history["clean_batch_count"],
            "incomplete_event_count": resume_history["incomplete_event_count"],
        },
        "interpretation": (
            "The rank implementation transition is result-preserving by exact parity evidence. "
            "Cache release and clean process batching alter allocator lifetime only. OOM retries "
            "occur before durable append and do not mix partial generations into the raw file."
        ),
    }
    REPORT.write_text(
        "\n".join(
            [
                "# Selected DEV runtime boundaries",
                "",
                "Status: **SELECTED_DEV_RUNTIME_BOUNDARIES_CAPTURED**",
                "",
                f"- Complete unique rows: **{row_count:,}**",
                f"- Reciprocal-rank block begins at line: **{RANK_FIRST_LINE:,}**",
                f"- Legacy stable full-sort rows end at line: **{PARTIAL_TOPK_FIRST_LINE - 1:,}**",
                f"- Proven-equivalent stable partial top-k begins at line: **{PARTIAL_TOPK_FIRST_LINE:,}**",
                f"- Post-example CUDA cache release begins at line: **{POST_EXAMPLE_CACHE_RELEASE_FIRST_LINE:,}**",
                f"- OOM retry telemetry begins at line: **{telemetry_lines[0] if telemetry_lines else 'none'}**",
                f"- Stored examples with OOM retry: **{retry_examples:,}** ({retry_total:,} total retries)",
                f'- Archived supervisor attempts: **{resume_history["restart_event_count"]:,}** '
                f'({resume_history["failed_attempt_count"]:,} failed; '
                f'{resume_history["clean_batch_count"]:,} clean batch returns; '
                f'{resume_history["incomplete_event_count"]:,} without recorded terminal event)',
                "",
                "The two ranking implementations return exactly identical stable top-k indices "
                "under the source-bound parity artifact. Memory cleanup and bounded worker "
                "lifetime do not alter logits or scoring. A failed allocation attempt is never "
                "appended; only a successful exact recomputation can become a result row.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    packet["report"] = {"path": str(REPORT), "sha256": sha256(REPORT)}
    packet["generator"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def main() -> None:
    packet = build_manifest()
    print(
        json.dumps(
            {"status": packet["status"], "output": str(OUTPUT), "report": str(REPORT)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
