"""Write-once TRAIN-only inference qualification registration; no model imports.

This independently registered four-prompt scope reuses the immutable Native7
compute/resource profile. Neither its registration nor a mechanics pass admits
DEV collection, training, scoring, or protected-bank access.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess

import native_contract as native
from inference_component import ROW_FIELDS, validate_row
from native_loader import ASSET_PINS, MODEL_DIR
from rs_contract import canonical, digest, require

PACKAGE, ROOT = native.PACKAGE, native.ROOT
VERSION = "synth-train-inference-qualification-contract-1"
ID = "SYNTH-INFQ-001"
PROFILE = "BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1"
NATIVE7_CONFIG = PACKAGE / "configs" / "SYNTH_NQ001_PROC7.json"
NATIVE7_AUDIT = PACKAGE / "evidence" / "nq001_attempt7_independent_audit_20261003.json"
NATIVE7_EVIDENCE = PACKAGE / "evidence" / "nq001_attempt7_20261003"
BUNDLE = PACKAGE / "data" / "crossed_source2_v1_20261003"
HELPERS = ("native_contract.py", "native_loader.py", "native_precision.py", "native_worker.py",
           "native_supervisor.py", "native_nvml.py", "native_process.py", "native_mechanics.py")
AUTHORIZATION = {"scope": "inference_native_qualification_only",
                 "owner_authority": "user_goal_and_delegated_decisions_2026-10-03",
                 "production_training": False, "DEV_collection": False, "scoring": False,
                 "protected_bank_access": False, "paid_services": False}
FORWARD_CAP, POSITION_CAP = 262, 134144  # 4*(2*32+1)+2; every call bounded by512.
DATA_NAMES = ("preparation_receipt.json", "plan.json", "enrollment.json", "candidate_records.jsonl",
              "pilot_decisions.json", "schedule_result.json", "token_pool_M1.jsonl", "token_pool_M2.jsonl")
EVIDENCE_NAMES = ("worker_result.json", "SUPERVISOR_RECEIPT.json", "RESOURCE_OBSERVATIONS.json",
                  "preservation_receipt.json")


def select_prompts(records, pools):
    """Freeze longest complete TRAIN R/S prompts per arm before model outputs.

    Selection reads only row ID, arm, split, skill and prompt length. Gold
    completion/input/label arrays and candidate targets are discarded. Lexically
    smallest row ID breaks equal-length ties; no candidate is dropped to fit.
    """
    require(type(records) is list and 0 < len(records) <= 8192 and type(pools) is dict and
            set(pools) == {"M1", "M2"}, "bounded TRAIN source records/two token pools")
    metadata = {}
    for record in records:
        require(type(record) is dict and type(record.get("row_id")) is str and
                record.get("arm") in ("M1", "M2") and record.get("split") == "train" and
                record.get("skill") in ("R", "S"), "only TRAIN atomic R/S source metadata allowed")
        identity = (record["arm"], record["row_id"])
        require(identity not in metadata, "duplicate source row")
        metadata[identity] = {"arm": record["arm"], "skill": record["skill"], "split": "train"}
    choices, bindings = [], []
    for arm in ("M1", "M2"):
        require(type(pools[arm]) is list and 0 < len(pools[arm]) <= 4096, "bounded token pool")
        candidates, seen = {"R": [], "S": []}, set()
        for source in pools[arm]:
            require(type(source) is dict and (arm, source.get("row_id")) in metadata and source["row_id"] not in seen,
                    "unknown or duplicate token source row")
            seen.add(source["row_id"])
            prompt = source.get("prompt_ids")
            require(type(prompt) is list and bool(prompt) and all(type(x) is int and 0 <= x < 151936 for x in prompt)
                    and 151643 not in prompt and type(source.get("prompt_tokens")) is int
                    and source["prompt_tokens"] == len(prompt), "complete typed source prompt")
            row = {"row_id": source["row_id"], "prompt_sha256": source.get("prompt_text_sha256"),
                   "prompt_ids": list(prompt), "prompt_tokens": len(prompt), "attention_mask": [1] * len(prompt)}
            # Deliberately do not inspect any teacher continuation or its mask.
            candidates[metadata[(arm, source["row_id"])]["skill"]].append(row)
        require(seen == {rid for a, rid in metadata if a == arm}, "token pool/source inventory incomplete")
        for skill in ("R", "S"):
            require(candidates[skill], "missing TRAIN arm/skill stratum")
            chosen = sorted(candidates[skill], key=lambda r: (-r["prompt_tokens"], r["row_id"]))[0]
            validate_row(chosen)  # An overlong longest prompt blocks the scope.
            choices.append(chosen)
            bindings.append({"arm": arm, "skill": skill, "split": "train", "row_id": chosen["row_id"],
                             "source_prompt_tokens": chosen["prompt_tokens"], "prompt_ids_sha256": digest(chosen["prompt_ids"]),
                             "selection": "longest_prompt_then_lexically_smallest_row_id"})
    require(len(choices) == 4 and len({r["row_id"] for r in choices}) == 4, "four distinct complete TRAIN prompts")
    return {"rows": choices, "selection_bindings": bindings, "source_packet_sha256": digest(choices)}


def _sources():
    pins = native.package_sources()
    for path in sorted((PACKAGE / "tests").glob("*.py")):
        pins[str(path.resolve())] = native.filehash(path)
    return dict(sorted(pins.items()))


def _native7():
    prior, audit = native.read_json(NATIVE7_CONFIG), native.read_json(NATIVE7_AUDIT)
    require(prior.get("sha256") == digest({k: v for k, v in prior.items() if k != "sha256"}), "Native7 config checksum")
    require(audit.get("status") == "PASS_SHORT_SCOPED_NATIVE_MECHANICS" and
            audit.get("supervisor_status") == "SUPERVISED_WORKER_EXITED_ZERO" and
            prior.get("profile") == PROFILE and prior.get("resources") == native.RESOURCE_BOUNDS and
            prior.get("tolerances") == native.TOLERANCES and prior.get("runtime_versions") == native.RUNTIME,
            "qualified Native7 profile/resource/numerical binding required")
    helper_pins = {}
    for name in HELPERS:
        filename = str((PACKAGE / name).resolve())
        require(prior["source_pins"].get(filename) == native.filehash(Path(filename)), "qualified helper source drift: " + name)
        helper_pins[filename] = prior["source_pins"][filename]
    require(audit["hash_closure"]["config_sha256_verified"] == native.filehash(NATIVE7_CONFIG), "Native7 audit/config binding")
    pins = {str(path.resolve()): native.filehash(path) for path in
            (NATIVE7_CONFIG, NATIVE7_AUDIT, *(NATIVE7_EVIDENCE / name for name in EVIDENCE_NAMES))}
    for name in EVIDENCE_NAMES[:3]:
        require(audit["hash_closure"]["audit_input_hashes"][name] == pins[str((NATIVE7_EVIDENCE / name).resolve())],
                "Native7 authoritative evidence changed: " + name)
    return prior, {"profile": PROFILE, "status": audit["status"], "evidence_pins": pins,
                   "qualified_helper_pins": helper_pins, "scope_transfer": "compute_helpers_only_new_scope_requires_supervised_pass"}


def build_scope():
    prior, qualification = _native7()
    selected = select_prompts(native.read_jsonl(BUNDLE / "candidate_records.jsonl"),
                              {arm: native.read_jsonl(BUNDLE / ("token_pool_%s.jsonl" % arm)) for arm in ("M1", "M2")})
    data_pins = {str((BUNDLE / name).resolve()): native.filehash(BUNDLE / name) for name in DATA_NAMES}
    require(data_pins == prior["data_pins"], "registered TRAIN corpus differs from qualified corpus")
    runtime = native.runtime_files()
    require(runtime == prior["runtime_pins"], "qualified runtime source/binary closure drift")
    model_pins = dict(ASSET_PINS)
    for name, pin in model_pins.items():
        require(native.filehash(MODEL_DIR / name) == pin, "pinned model/tokenizer asset drift: " + name)
    lengths = [r["prompt_tokens"] for r in selected["rows"]]
    return {"schema_version": 1, "id": ID, "contract": VERSION, "profile": PROFILE,
            "authorization": deepcopy(AUTHORIZATION), "resources": deepcopy(native.RESOURCE_BOUNDS),
            "tolerances": deepcopy(native.TOLERANCES), "runtime_versions": deepcopy(native.RUNTIME),
            "offline_environment": deepcopy(native.OFFLINE_ENV), "precision": deepcopy(prior["precision"]),
            "source_pins": _sources(), "runtime_pins": runtime, "data_pins": data_pins,
            "model_dir": str(MODEL_DIR.resolve()), "model_pins": model_pins, "data_bundle": str(BUNDLE.resolve()),
            "native7": qualification, "prompt_hash_kind": "bound_source_rendered_text_sha256", **selected,
            "dose": {"context_cap": 512, "maximum_generated_tokens": 32, "qualification_rows": 4,
                     "capture_repeats_per_row": 2, "bulk_forwards_per_row": 1, "parity_forwards": 2,
                     "maximum_forward_calls": FORWARD_CAP, "maximum_forward_input_positions": POSITION_CAP,
                     "maximum_parity_input_positions": 2 * lengths[0],
                     "maximum_capture_input_positions": sum(65 * length + 1023 for length in lengths),
                     "maximum_qualification_forwards": 260, "optimizer_steps": 0, "backwards": 0},
            "initial_model": "fresh_unchanged_dense_M0_no_STATE1_or_adapter_checkpoint",
            "adapter_state": "new_rank8_qv_alpha16_zeroB_no_optimizer_no_backward",
            "seed": 101, "production_training_authorized": False, "DEV_collection_authorized": False,
            "scoring_authorized": False, "protected_bank_access": False, "full_DEV_collection_qualified": False,
            "quality_improvement_established": False}


def _committed_checkout(require_clean):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if require_clean:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        require(not status.strip(), "inference qualification requires clean committed sources/config")
    return head


def write_scope(path):
    """Create only a new owned config after source bytes have been committed."""
    target = Path(path).absolute()
    require(target.parent.resolve() == (PACKAGE / "configs").resolve() and target.suffix == ".json" and
            not target.exists() and not target.is_symlink(), "new write-once configs/*.json required")
    _committed_checkout(True)
    body = build_scope()
    body["sha256"] = digest(body)
    raw = (canonical(body) + "\n").encode("ascii")
    require(len(raw) <= native.MAX_JSON, "bounded frozen config")
    with target.open("xb", buffering=0) as stream:
        require(stream.write(raw) == len(raw), "short frozen config write")
        os.fsync(stream.fileno())
    return body


def preflight(path, *, require_clean=True):
    filename = Path(path).absolute()
    require(filename.parent.resolve() == (PACKAGE / "configs").resolve() and filename.is_file() and
            not filename.is_symlink(), "owned inference qualification config required")
    body = native.read_json(filename)
    require(type(body) is dict and body.get("sha256") == digest({k: v for k, v in body.items() if k != "sha256"}), "scope checksum")
    # Reconstruct every field, source closure and deterministic selection. A
    # re-signed permissive amendment cannot pass by updating its checksum alone.
    require(canonical({k: v for k, v in body.items() if k != "sha256"}) == canonical(build_scope()),
            "scope differs from current frozen TRAIN/profile/source/runtime/model/data bindings")
    head = _committed_checkout(require_clean)
    return {**body, "worker_path": str(PACKAGE / "inference_qual_worker.py"), "config_path": str(filename.resolve()),
            "config_sha256": native.filehash(filename), "source_commit": head,
            "bindings": {"source": body["source_pins"], "runtime": body["runtime_pins"],
                         "model": body["model_pins"], "data": body["data_pins"], "native7": body["native7"]}}
