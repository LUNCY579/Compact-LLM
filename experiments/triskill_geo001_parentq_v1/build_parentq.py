"""Build PARENTQ membership, CPU validation receipt and freeze (CPU only; no model).

    python build_parentq.py membership   # parentq_bank.json, parentq_canary.json (create-only)
    python build_parentq.py validate     # CPU_VALIDATION_PARENTQ.json (create-only)
    python build_parentq.py freeze       # refuses on uncommitted code; validates; FREEZE_PARENTQ.json

Reads only exposed sources: GEO's frozen code and the T exposure scan it already uses,
delta_H's training rows (prompt text only), HSKILL_VIEWS_V2 (the exposed selection bank)
and CONFIRM-002's stored predictions on it. No protected bank is opened. Run with the
hermes interpreter (the GEO freeze pins its runtime packages).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import geo_frozen  # noqa: E402
import parentq_decide  # noqa: E402
import parentq_rows  # noqa: E402
import parentq_worker  # noqa: E402

RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
VIEWS = (RECORDS / "hskill-views-002/HSKILL_VIEWS_V2.jsonl",
         "ba1f2982fd5d08aac861d4547b1dadb33d20e6022ff9aa06934e1ad2e7154723")
STORED = ((REPO / "experiments/composition_confirmation_v2/STAGE_A_H_VIEWS_BASE_SPARSE.json",
           "dc34d901ac46665dabbdc63bff3dceb65371c8efc7f012105b887f499d8a0a55"),
          (REPO / "experiments/composition_confirmation_v2/STAGE_A_H_VIEWS_NEW.json",
           "4cd95350228fff39e56de1baa643e391d58c0bfd118e2ce28565571e09269bd5"))
H_TRAINING = RECORDS / "compose001-20260918/H/H-prepared/microfit-train.jsonl"
CANARY_SOURCES = ("BASE", "SPARSE", "H25")
CANARY_PER_TYPE = 4
VALIDATION_NAME = "CPU_VALIDATION_PARENTQ.json"
FROZEN_LOCAL = ("parentq_worker.py", "parentq_rows.py", "parentq_decide.py", "parentq_supervise.py",
                "parentq_launch_wmi.py", "geo_frozen.py", "build_parentq.py", "derive_from_geo.py", parentq_worker.BANK_JSON,
                parentq_worker.CANARY_JSON, "PREREG_PARENTQ.md", "test_parentq_rows.py", "test_parentq_decide.py",
                "test_parentq_worker.py", "test_parentq_supervise.py")
PACKAGES = ("numpy", "tokenizers", "psutil", "torch", "transformers", "safetensors")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"source changed: {path}")
    return data


def write_json(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def bank_rows():
    benchmark = geo_frozen.load("benchmark")
    return benchmark, parentq_rows.build_bank(benchmark)


def build_canary(tokenizer=None) -> dict:
    """Registered selection: VIEWS rows whose stored BASE, SPARSE and H25 texts are pairwise
    different (so a wrong arm cannot pass), each <= 31 generated tokens (fits 32 slots with its
    EOS), prompt exactly [system, user]; 4 answerable and 4 boundary rows in sha256(id) order."""
    views = [json.loads(line) for line in pinned(*VIEWS).decode("utf-8").splitlines() if line.strip()]
    stored = {}
    for path, pin in STORED:
        for arm in json.loads(pinned(path, pin))["arms"]:
            stored[arm["arm"]] = {p["id"]: p for p in arm["predictions"]}
    eligible = []
    for row in views:
        roles = [message["role"] for message in row["prompt"]]
        texts = {source: stored[source][row["id"]]["prediction"] for source in CANARY_SOURCES}
        lengths = {source: stored[source][row["id"]]["generated_tokens"] for source in CANARY_SOURCES}
        if roles == ["system", "user"] and len(set(texts.values())) == 3 and max(lengths.values()) <= 31:
            eligible.append((hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), row, texts, lengths))
    chosen = []
    for row_type in ("answerable", "boundary"):
        pool = sorted((item for item in eligible if item[1]["row_type"] == row_type), key=lambda item: item[0])
        chosen += pool[:CANARY_PER_TYPE]
    rows = [{"row_id": f"canary:{row['id']}", "views_id": row["id"], "row_type": row["row_type"],
             "system": row["prompt"][0]["content"], "user": row["prompt"][1]["content"],
             "stored": texts, "stored_generated_tokens": lengths} for _, row, texts, lengths in chosen]
    return {"rows": rows, "eligible": len(eligible), "sources": {
        "views": {"path": str(VIEWS[0]), "sha256": VIEWS[1]},
        "stored_predictions": [{"path": path.relative_to(REPO).as_posix(), "sha256": pin} for path, pin in STORED],
        "H25_is": "BASE + 0.25 * P16(delta_H) (CONFIRM-002 build_arms_c2.py); the worker's H arm is "
                  "FP32(Supa + TT - Sparse), equal up to float rounding"}}


def membership() -> None:
    benchmark, bank = bank_rows()
    write_json(HERE / parentq_worker.BANK_JSON, [row.to_dict() for row in parentq_rows.all_rows(bank)])
    write_json(HERE / parentq_worker.CANARY_JSON, build_canary())


def _geo_rows(benchmark):
    return (*benchmark.build_screen(), *benchmark.build_development(), *benchmark.build_legacy_screen(),
            *benchmark.build_legacy_development(), *benchmark.build_conjunction_controls(),
            *benchmark.build_format_variants())


def validate() -> None:
    from transformers import AutoTokenizer
    benchmark, bank = bank_rows()
    geo = geo_frozen.load("worker")
    geometry = geo_frozen.load("geometry")
    rows = parentq_rows.all_rows(bank)
    frozen_bank = json.loads((HERE / parentq_worker.BANK_JSON).read_text(encoding="utf-8"))
    canary = json.loads((HERE / parentq_worker.CANARY_JSON).read_text(encoding="utf-8"))
    geo_rows = _geo_rows(benchmark)
    signatures = {}
    for field in ("row_id", "world_signature", "state_signature", "rule_signature"):
        ours = [getattr(row, field) for row in (*bank["T"], *bank["H_ON"])]
        theirs = {getattr(row, field) for row in geo_rows}
        signatures[field] = {"disjoint_from_geo_rows": not (set(ours) & theirs)}
    # One operative event may recur on different states (a small ledger event space); a row
    # must not recur, so uniqueness is on row_id, world and the (state, event) pair.
    for name, key in (("row_id", lambda r: r.row_id), ("world_signature", lambda r: r.world_signature),
                      ("state_and_rule", lambda r: (r.state_signature, r.rule_signature))):
        keys = [key(row) for row in (*bank["T"], *bank["H_ON"])]
        signatures.setdefault(name, {})["unique_within_bank"] = len(set(keys)) == len(keys)
    geo_words = set(re.findall(r"[a-z]+", " ".join(f"{r.system} {r.user} {r.row_id}" for r in geo_rows).lower()))
    training = [json.loads(line) for line in H_TRAINING.read_bytes().decode("utf-8").splitlines() if line.strip()]
    training_sha = sha256(H_TRAINING)
    training_words = set(re.findall(r"[a-z]+", " ".join(m["content"] for r in training for m in r["prompt"]).lower()))
    native_counts = {label: {"rows": sum(1 for r in training if r["prompt"][0]["content"] == prompt),
                             "exact_abstention_replies": sum(1 for r in training if r["prompt"][0]["content"] == prompt
                                                             and r["reference"] == benchmark.ABSTAIN_PHRASE)}
                     for label, prompt in parentq_rows.H_NATIVE_PROMPTS.items()}
    contract_rows = [r for r in training if r["prompt"][0]["content"] == benchmark.ANSWERABILITY_SYSTEM_H]
    freeze_module = geo_frozen.load("freeze_sectionh")
    collisions = freeze_module.t_collisions(list(bank["T"]))
    tokenizer = AutoTokenizer.from_pretrained(str(geometry.SOURCES["BASE"][0].parent), local_files_only=True)
    sys.path.insert(0, str(REPO / "experiments/fresh_dual_skill_instrument_v1"))
    renderer_path = REPO / "experiments/fresh_dual_skill_instrument_v1/score_arms.py"
    geo_freeze = geo_frozen.geo_freeze()
    renderer_pinned = sha256(renderer_path) == geo_freeze["dependency_binding"]["roles"]["renderer"]["sha256"]
    from score_arms import render
    count = lambda system, user: len(tokenizer(render(system, user), add_special_tokens=False)["input_ids"])
    prompt_tokens = {row.row_id: count(row.system, row.user) for row in rows}
    prompt_tokens.update({row["row_id"]: count(row["system"], row["user"]) for row in canary["rows"]})
    reply_tokens = {}
    for row in rows:
        reply = row.expected if row.output_kind == "json_object" else row.metadata["strict_reply"]
        spaced = json.dumps(json.loads(reply)) if row.output_kind == "json_object" else reply
        reply_tokens[row.row_id] = max(len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in (reply, spaced))
    template_equal = all(
        render(row["system"], row["user"]) == tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for row in canary["rows"])
    units = parentq_worker.schedule_units(bank, canary["rows"])
    caps = parentq_worker.allocation(sum(len(ids) for _, _, ids in units), geo.INPUT_SLOTS, geo.OUTPUT_SLOTS)
    tables = parentq_decide.gate_tables(benchmark)
    receipt = {
        "experiment": "TRISKILL-GEO-001-PARENTQ", "model_loads": 0, "model_forwards": 0,
        "protected_banks_opened": [], "geo_freeze_sha256": geo_frozen.GEO_FREEZE_SHA256,
        "bank_matches_generator": frozen_bank == [row.to_dict() for row in rows],
        "canary_matches_builder": canary == build_canary(),
        "counts": {block: len(block_rows) for block, block_rows in bank.items()}, "canary_n": len(canary["rows"]),
        "canary_eligible_rows": canary["eligible"],
        "signatures": signatures,
        "t_support_violations": {row.row_id: benchmark.t_support_violations(row) for row in bank["T"]},
        "t_collisions": collisions,
        "worlds_absent_from_geo_rows": sorted(w for w in parentq_rows.WORLDS if w.lower() in geo_words) == [],
        "worlds_absent_from_delta_H_training_prompts": sorted(w for w in parentq_rows.WORLDS if w.lower() in training_words) == [],
        "delta_H_training_rows": {"path": str(H_TRAINING), "sha256": training_sha,
                                  "sha256_registered": parentq_rows.H_TRAINING_ROWS_SHA256,
                                  "native_prompt_counts": native_counts,
                                  "screen_contract_rows": len(contract_rows),
                                  "screen_contract_rows_with_abstention_gold": sum(
                                      1 for r in contract_rows if r["reference"] == benchmark.ABSTAIN_PHRASE)},
        "renderer_is_geo_pinned": renderer_pinned,
        "renderer_equals_chat_template_thinking_disabled_on_canary": template_equal,
        "max_prompt_tokens": max(prompt_tokens.values()), "max_reply_tokens": max(reply_tokens.values()),
        "slots": {"input": geo.INPUT_SLOTS, "output": geo.OUTPUT_SLOTS},
        "allocation": caps,
        "gate_tables": {"T_subtypes": list(tables["T"]["subtypes"]), "T_passing_combinations": sum(tables["T"]["passed"]),
                        "H_kinds": list(tables["H"]["subtypes"]), "H_passing_combinations": sum(tables["H"]["passed"])},
    }
    receipt["passed"] = bool(
        receipt["bank_matches_generator"] and receipt["canary_matches_builder"]
        and receipt["counts"] == {"T": 24, "H_ON": 32, "H_OFF": 32} and receipt["canary_n"] == 8
        and all(signatures[field]["disjoint_from_geo_rows"]
                for field in ("row_id", "world_signature", "state_signature", "rule_signature"))
        and all(signatures[name]["unique_within_bank"] for name in ("row_id", "world_signature", "state_and_rule"))
        and not any(receipt["t_support_violations"].values()) and collisions["collision_free"]
        and receipt["worlds_absent_from_geo_rows"] and receipt["worlds_absent_from_delta_H_training_prompts"]
        and training_sha == parentq_rows.H_TRAINING_ROWS_SHA256
        and all(c["rows"] == 28 and c["exact_abstention_replies"] == 28 for c in native_counts.values())
        and receipt["delta_H_training_rows"]["screen_contract_rows_with_abstention_gold"] == 0
        and renderer_pinned and template_equal
        and receipt["max_prompt_tokens"] <= geo.INPUT_SLOTS and receipt["max_reply_tokens"] < geo.OUTPUT_SLOTS
        and tuple(tables["T"]["subtypes"]) == parentq_decide.T_SUBTYPES
        and sorted(set(tables["H"]["subtypes"])) == sorted(parentq_decide.H_KINDS))
    write_json(HERE / VALIDATION_NAME, receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "counts", "canary_n", "canary_eligible_rows",
                                                    "max_prompt_tokens", "max_reply_tokens", "allocation")}, indent=2))


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def freeze() -> None:
    dirty = [line for line in _git("status", "--porcelain", "--", str(HERE)).splitlines() if not line.startswith("??")]
    if dirty:
        raise SystemExit(f"tracked PARENTQ files have uncommitted changes: {dirty}")
    validate()
    receipt = json.loads((HERE / VALIDATION_NAME).read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise SystemExit("CPU validation did not pass; refusing to freeze")
    geo = geo_frozen.load("worker")
    gpu_lease = REPO / "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py"
    payload = {
        "experiment": "TRISKILL-GEO-001-PARENTQ", "implementation_commit": _git("rev-parse", "HEAD"),
        "geo_freeze": {"path": "experiments/triskill_geometric_pilot_v1/FREEZE_SECTIONH.json",
                       "sha256": geo_frozen.GEO_FREEZE_SHA256},
        "cpu_validation": {"file": VALIDATION_NAME, "sha256": sha256(HERE / VALIDATION_NAME), "passed": True},
        "files": {name: sha256(HERE / name) for name in FROZEN_LOCAL},
        "supervisor_dependencies": {"gpu_lease": {"path": "experiments/fresh_dual_skill_instrument_v1/gpu_lease.py",
                                                  "sha256": sha256(gpu_lease)}},
        "runtime_packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "native_allocation": receipt["allocation"],
        "native_budget": {"worker_wall_seconds_max": parentq_worker.WALL_SECONDS,
                          "supervisor_hard_wall_seconds_max": 600, "model_loads_max_per_attempt": 1,
                          "loaded_attempts_max": 3, "input_slots": geo.INPUT_SLOTS,
                          "max_output_slots": geo.OUTPUT_SLOTS, "eos_token_ids": list(geo.EOS_TOKEN_IDS),
                          "gpu_memory_fraction": geo.GPU_MEMORY_FRACTION},
        "native_execution_status": "REQUIRES_EXPLICIT_OPERATOR_GRANT", "native_execution_authorized": False,
        "confirmation": False, "model_loads": 0, "model_forwards": 0,
        "evidence_not_self_pinned": ["FREEZE_PARENTQ.json", "CPU_TESTS_PARENTQ.log", "REPORT_PARENTQ.md"],
    }
    write_json(HERE / parentq_worker.FREEZE_NAME, payload)
    print(sha256(HERE / parentq_worker.FREEZE_NAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("membership", "validate", "freeze"))
    args = parser.parse_args()
    {"membership": membership, "validate": validate, "freeze": freeze}[args.step]()
