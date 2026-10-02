"""Build the section H membership files, CPU validation receipt and freeze (CPU only).

    python freeze_sectionh.py membership   # screen_sectionh.json, development_sectionh.json
    python freeze_sectionh.py validate     # CPU_VALIDATION_SECTIONH.json (reads exposed banks only)
    python freeze_sectionh.py freeze       # refuses on uncommitted code; runs validate, then
                                           # FREEZE_SECTIONH.json pinning the validation receipt

Every output is create-only. No model is loaded and no protected bank is opened: the
collision checks read T_BANK_V3X, T_BANK_V4, delta_T's training rows and the exposed
HSKILL_VIEWS_V2 contract only. Run with the hermes interpreter (pinned runtime packages).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import benchmark  # noqa: E402
import dependency_binding  # noqa: E402
import geometry  # noqa: E402
import worker  # noqa: E402

RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
EXPOSED_T_BANKS = (REPO / "experiments/composition_confirmation_v1/T_BANK_V3X.json",
                   REPO / "experiments/composition_confirmation_v2/T_BANK_V4.json")
DELTA_T_TRAINING = (RECORDS / "rrp015-20260917/control-prepared/microfit-train.jsonl",
                    "0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1")
BOUNDARY_RULE = RECORDS / "hskill-views-002/boundary_rule.py"
HSKILL_VIEWS = RECORDS / "hskill-views-002/HSKILL_VIEWS_V2.jsonl"   # exposed selection bank
VALIDATION_NAME = "CPU_VALIDATION_SECTIONH.json"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt")
PACKAGES = ("numpy", "tokenizers", "psutil", "torch", "transformers", "safetensors")
FROZEN_LOCAL = (
    "worker.py", "geometry.py", "benchmark.py", "dependency_binding.py", "supervise.py", "launch_wmi.py",
    "freeze_sectionh.py", worker.GEOMETRY_NPZ, worker.GEOMETRY_JSON, worker.SCREEN_JSON, worker.DEVELOPMENT_JSON,
    "AMENDMENT_SECTIONH.md", "PREREG.md", "BENCHMARK_MATRIX.md", "FOLLOWUP_PREREG.md", "FUTURE_PROTOCOL.md",
    "SOURCES_VERIFIED.md", "geometry.npz", "geometry.json", "screen.json", "development.json",
    "format_development.json", "conjunction_development.json", "test_benchmark.py", "test_worker.py",
    "test_geometry.py", "test_dependency_binding.py", "test_future_power.py", "test_native_tiny.py",
    "test_supervise.py", "test_native_e2e.py", "future_power.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def membership() -> None:
    write_json(HERE / worker.SCREEN_JSON, [row.to_dict() for row in benchmark.build_screen()])
    write_json(HERE / worker.DEVELOPMENT_JSON, [row.to_dict() for row in benchmark.build_development()])


def _t_payloads():
    """(family, payload) from the exposed T banks and delta_T's training rows."""
    for bank in EXPOSED_T_BANKS:
        for row in json.loads(bank.read_text(encoding="utf-8"))["rows"]:
            content = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else row["prompt"]
            yield bank.name, row["family"], json.loads(content)
    path, pin = DELTA_T_TRAINING
    if sha256(path) != pin:
        raise ValueError("delta_T training rows changed")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        content = next(message["content"] for message in row["prompt"] if message["role"] == "user")
        yield path.name, row["family"], json.loads(content)


def t_collisions(rows) -> dict:
    """Exact payload, operative event (tick removed) with before-state, and (before, answer)."""
    def operative(payload):
        event = {key: value for key, value in payload["event"].items() if key != "tick"}
        return benchmark.canonical({"before": payload["before"], "event": event})
    keys = {}
    for row in rows:
        payload = json.loads(row.user)
        keys[row.row_id] = {"payload": benchmark.canonical(payload), "operative": operative(payload),
                            "before_answer": benchmark.canonical([payload["before"], json.loads(row.expected)])}
    hits = {row_id: [] for row_id in keys}
    scanned, skipped = 0, {}
    for source, family, payload in _t_payloads():
        if family not in benchmark.T_SUPPORT:  # e.g. prerequisite_plan: a bitmask, not a section H family
            skipped[family] = skipped.get(family, 0) + 1
            continue
        scanned += 1
        historical = {"payload": benchmark.canonical(payload), "operative": operative(payload),
                      "before_answer": benchmark.canonical([payload["before"],
                                                            benchmark.transition_oracle(family, payload)])}
        for row_id, key in keys.items():
            for kind in ("payload", "operative", "before_answer"):
                if historical[kind] == key[kind]:
                    hits[row_id].append([source, kind])
    return {"scanned_payloads": scanned, "skipped_other_families": skipped, "hits": hits,
            "collision_free": scanned > 0 and not any(hits.values())}


def validate() -> None:
    sys.path.insert(0, str(REPO / "experiments/fresh_dual_skill_instrument_v1"))
    from score_arms import render
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(geometry.SOURCES["BASE"][0].parent / "tokenizer.json"))
    rows, development = benchmark.build_screen(), benchmark.build_development()
    tokens = {}
    for row in (*rows, *development):
        count = lambda system: len(tokenizer.encode(render(system, row.user), add_special_tokens=False).ids)
        reply = row.expected if row.output_kind == "json_object" else row.metadata["strict_reply"]
        spaced = json.dumps(json.loads(reply)) if row.output_kind == "json_object" else reply
        tokens[row.row_id] = {"input": count(row.system),
                              "input_with_prompt_prefix": count(worker.prompt_system(row, "PROMPT")),
                              "reply": len(tokenizer.encode(reply).ids), "spaced_reply": len(tokenizer.encode(spaced).ids)}
    rule_text = BOUNDARY_RULE.read_text(encoding="utf-8")
    namespace = {}
    exec(compile(rule_text, str(BOUNDARY_RULE), "exec"), namespace)
    views_systems = set()
    for line in HSKILL_VIEWS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("row_type") == "answerable":
            views_systems.update(m["content"] for m in record["prompt"] if m["role"] == "system")
    receipt = {
        "experiment": "TRISKILL-GEO-001-SECTIONH", "model_loads": 0, "model_forwards": 0,
        "protected_banks_opened": [],
        "membership": benchmark.validate_local_membership(),
        "screen_n": len(rows), "development_n": len(development),
        "t_support_violations": {row.row_id: benchmark.t_support_violations(row)
                                 for row in (*rows, *development) if row.task_kind == "transition_parent"},
        "t_collisions": t_collisions([row for row in (*rows, *development) if row.task_kind == "transition_parent"]),
        "floors": benchmark.screen_floors(rows),
        "token_counts": tokens,
        "max_input_tokens": max(max(v["input"], v["input_with_prompt_prefix"]) for v in tokens.values()),
        "max_reply_tokens": max(max(v["reply"], v["spaced_reply"]) for v in tokens.values()),
        "slots": {"input": worker.INPUT_SLOTS, "output": worker.OUTPUT_SLOTS, "readout": worker.READOUT_SLOTS},
        "allocation": worker.native_allocation(rows, development),
        "boundary_rule": {"sha256": sha256(BOUNDARY_RULE), "pinned": benchmark.BOUNDARY_RULE_SHA256,
                          "patterns_equal": tuple(namespace["ABSTAIN_PATTERNS"]) == benchmark.ABSTAIN_PATTERNS},
        "answerability_contract_verbatim_in_views_v2": benchmark.ANSWERABILITY_SYSTEM_H in views_systems,
    }
    receipt["passed"] = bool(
        receipt["membership"]["local_disjoint"] and receipt["membership"]["legacy_rows_disjoint"]
        and not any(receipt["t_support_violations"].values()) and receipt["t_collisions"]["collision_free"]
        and receipt["max_input_tokens"] <= worker.INPUT_SLOTS and receipt["max_reply_tokens"] <= worker.OUTPUT_SLOTS
        and receipt["allocation"]["fits"] and receipt["boundary_rule"]["sha256"] == benchmark.BOUNDARY_RULE_SHA256
        and receipt["boundary_rule"]["patterns_equal"] and receipt["answerability_contract_verbatim_in_views_v2"])
    write_json(HERE / VALIDATION_NAME, receipt)
    print(json.dumps({"passed": receipt["passed"], "max_input_tokens": receipt["max_input_tokens"],
                      "max_reply_tokens": receipt["max_reply_tokens"], "allocation": receipt["allocation"],
                      "t_collisions": receipt["t_collisions"]["collision_free"], "floors": receipt["floors"]}, indent=2))


def _git(*args) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def freeze(implementation_parent: str) -> None:
    """Build gate: committed code, then a fresh passing CPU validation, then the freeze."""
    dirty = [line for line in _git("status", "--porcelain", "--", str(HERE)).splitlines() if not line.startswith("??")]
    if dirty:
        raise SystemExit(f"tracked pilot files have uncommitted changes: {dirty}")
    validate()
    receipt = json.loads((HERE / VALIDATION_NAME).read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise SystemExit("CPU validation did not pass; refusing to freeze")
    gpu_lease = REPO / "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py"
    base_dir = geometry.SOURCES["BASE"][0].parent
    payload = {
        "experiment": "TRISKILL-GEO-001-SECTIONH", "parent_commit": implementation_parent,
        "implementation_commit": _git("rev-parse", "HEAD"),
        "cpu_validation": {"file": VALIDATION_NAME, "sha256": sha256(HERE / VALIDATION_NAME), "passed": True},
        "source_commit": "01f47694988a85bdedab7831d1652f0b9bf07e9f",
        "files": {name: sha256(HERE / name) for name in FROZEN_LOCAL},
        "dependency_binding": dependency_binding.build_dependency_manifest(),
        "supervisor_dependencies": {"gpu_lease": {"path": "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py",
                                                  "sha256": sha256(gpu_lease)}},
        "source_sidecars": {name: sha256(base_dir / name) for name in SIDECARS},
        "source_weights": {name: pin for name, (_, pin) in geometry.SOURCES.items()},
        "runtime_packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "native_budget": {"screen_generations_max": worker.MAX_GENERATIONS, "model_forwards_max": worker.MAX_FORWARDS,
                          "token_positions_including_padding_max": worker.MAX_TOKEN_POSITIONS,
                          "worker_wall_seconds_max": worker.WALL_SECONDS,
                          "supervisor_hard_wall_seconds_max": 600, "model_loads_max": 1,
                          "input_slots": worker.INPUT_SLOTS, "output_slots": worker.OUTPUT_SLOTS,
                          "readout_slots": worker.READOUT_SLOTS, "eos_token_ids": list(worker.EOS_TOKEN_IDS),
                          "gpu_memory_fraction": worker.GPU_MEMORY_FRACTION},
        "native_allocation": worker.native_allocation(benchmark.build_screen(), benchmark.build_development()),
        "native_execution_status": "REQUIRES_EXPLICIT_OPERATOR_GRANT",
        "native_execution_authorized": False, "confirmation": False, "quality_claim": False,
        "construction_C_training_examples": 0, "model_loads": 0, "model_forwards": 0,
        "evidence_not_self_pinned": ["FREEZE_SECTIONH.json", "CPU_TESTS_SECTIONH.log", "REPORT_SECTIONH.md"],
    }
    write_json(HERE / worker.FREEZE_NAME, payload)
    print(sha256(HERE / worker.FREEZE_NAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("membership", "validate", "freeze"))
    parser.add_argument("--parent-commit", default="332b7f9b")
    args = parser.parse_args()
    {"membership": membership, "validate": validate}.get(args.step, lambda: freeze(args.parent_commit))()
