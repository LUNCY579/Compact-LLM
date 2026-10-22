"""Create-only offline packet preparation. No native execution or approval grant."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from .data import TEACHER_TEMPLATE
from .plan import Arm, digest_json, hash_file, write_json
from .proposal_data import build_candidate_bank

SOURCE_CORE = "rineya.core.supa.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
SOURCE_SHA = "495dee5eef1eb5365188ee589961da75909feb4a7446ac4f2d0760e48602d6b4"
SOURCE_WEIGHT_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
START_COMMIT = "e1737660cd0feacd6efd4177ea1d1579bf0cd865"
BRANCH = "research/rineya-self-distillation-axes-v1"
SELECTED = "model.layers.12.self_attn.q_proj.weight"
TINY_ARMS = (Arm("sd", "student", "forward", 0.),
             Arm("sd", "teacher", "forward", 0.), Arm("sft", None, None, 0.))
SEEDS = (17, 29)
FIXED_LIMITS = {"total_updates": 72, "max_response_tokens": 32, "max_sequence_tokens": 512,
    "qualification_updates": 0, "qualification_forward_calls": 2048,
    "qualification_processed_input_tokens": 1048576, "qualification_generated_tokens": 2048,
    "per_sd_cell_forward_calls": 5528, "per_sft_cell_forward_calls": 5132,
    "per_cell_processed_input_tokens": 3000000, "per_cell_generated_tokens": 5504,
    "comparison_processed_input_tokens": 16576512, "comparison_generated_tokens": 32256,
    "comparison_sft_reference_tokens": 768, "qualification_wall_seconds": 1800,
    "per_cell_wall_seconds": 1800, "minimum_free_host_gib_before": 6.,
    "minimum_free_host_gib_during": 1.5, "minimum_free_cuda_gib_before": 4.,
    "minimum_free_cuda_gib_during": .5, "minimum_free_disk_gib": 5., "automatic_retries": 0}
RUNTIME_RECIPE = {"python": "3.11.15", "torch": "2.10.0+cu126", "transformers": "4.57.1",
                  "safetensors": "0.7.0", "tokenizers": "0.22.2"}


def external_destination(path, workspace, business_root):
    """Require external BusinessProject custody; resolve junctions before checks."""
    target, workspace, root = map(lambda p: Path(p).resolve(), (path, workspace, business_root))
    if (not target.is_relative_to(root) or target == root or
            target.is_relative_to(workspace) or any(p.lower() == ".codex" for p in target.parts)):
        raise ValueError("New artifacts must be external to source under BusinessProject, never .codex")
    if any((ancestor / ".git").exists() for ancestor in (target, *target.parents)):
        raise ValueError("New artifacts cannot be placed inside another Git checkout")
    return target


def validate_packet(packet):
    """Validate prospective fixed choices, never interpret this as permission."""
    if packet.get("status") != "PROSPECTIVE_REVIEW_PACKET" or packet.get("execution_authorized") is not False:
        raise ValueError("Preparation is not an execution permit")
    if packet["source"]["core_id"] != SOURCE_CORE or packet["source"]["checkpoint_sha256"] != SOURCE_SHA:
        raise ValueError("Exact ungated Supa source required; no substitute or foreign permit")
    if packet["source"]["weight_sha256"] != SOURCE_WEIGHT_SHA:
        raise ValueError("Registered source weight identity changed")
    if packet["design"]["arms"] != [asdict(a) for a in TINY_ARMS] or packet["design"]["seeds"] != list(SEEDS):
        raise ValueError("Only the frozen single-axis contrast and SFT control are in scope")
    if (packet["design"]["steps_per_cell"] != 12 or packet["design"]["cells"] != 6
            or packet["design"]["learning_rate"] != 1e-5 or packet["design"]["selected_parameter"] != SELECTED
            or packet["design"]["selected_numel"] != 2097152 or packet["design"]["acquisition_steps"] != [10, 11, 12]):
        raise ValueError("Tiny update ceiling changed")
    if any(type(packet["limits"].get(k)) is not type(v) or packet["limits"][k] != v for k, v in FIXED_LIMITS.items()):
        raise ValueError("Frozen sample/token/call/time/resource limit changed")
    if packet.get("runtime_recipe") != RUNTIME_RECIPE:
        raise ValueError("Runtime requires its own prospective qualification")
    if any(packet["teacher_gate"].get(k) != v for k, v in {"rows": 8, "min_correct": 7,
            "min_paired_wins": 2, "max_paired_harms": 0}.items()):
        raise ValueError("Frozen teacher competence threshold changed")
    counts = {"train": 16, "iid_acquisition": 12, "harder_transfer": 12,
              "retention_arithmetic": 8, "retention_spatial": 8, "teacher_gate": 8}
    cohorts = packet["data"]["cohorts"]
    if set(cohorts) != set(counts) or any(len(cohorts[k]) != n for k, n in counts.items()):
        raise ValueError("Frozen sample/cohort counts changed")
    ids = [i for cohort in cohorts.values() for i in cohort]
    if len(set(ids)) != 64:
        raise ValueError("Cohort IDs overlap")
    if packet["data"]["status"] != "CANDIDATE_NOT_ADMITTED":
        raise ValueError("Do not invent independent data admission")
    if packet["stages"]["native_qualification"]["status"] != "NOT_EXECUTED":
        raise ValueError("Offline packet cannot claim native qualification")
    return True


def prepare(output, custody_path, inspection_path, workspace, business_root):
    output = external_destination(output, workspace, business_root)
    custody = json.loads(Path(custody_path).read_text(encoding="utf-8"))
    inspection = json.loads(Path(inspection_path).read_text(encoding="utf-8"))
    if output.is_relative_to(Path(custody["checkpoint_path"]).resolve()):
        raise ValueError("Do not create preparation artifacts inside the immutable checkpoint")
    if custody["sha256"] != SOURCE_SHA or inspection["checkpoint_manifest_sha256"] != SOURCE_SHA:
        raise ValueError("Static source receipts do not bind the exact eligible checkpoint")
    if inspection["source_identity"]["core_id"] != SOURCE_CORE or not all(inspection["eligibility"].values()):
        raise ValueError("Source static eligibility is incomplete")
    output.mkdir(parents=True, exist_ok=False)
    bank = build_candidate_bank()
    for name, rows in (("screen", bank["screen_rows"]), ("teacher-gate", bank["teacher_gate_rows"])):
        with (output / (name + ".jsonl")).open("x", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "data-custody.json", {"manifest": bank["cohort_manifest"], "audit": bank["audit"]})
    train_ids = [r["id"] for r in bank["screen_rows"] if r["split"] == "train"]
    streams, config_digests = {}, {}
    for seed in SEEDS:
        order = list(train_ids)
        random.Random(seed).shuffle(order)
        streams[str(seed)] = order[:12]
        config = {"checkpoint": custody["checkpoint_path"], "checkpoint_sha256": SOURCE_SHA,
                  "dataset": str(output / "screen.jsonl"), "dataset_sha256": hash_file(output / "screen.jsonl"),
                  "task": "modular_state", "seed": seed, "steps": 12, "learning_rate": 1e-5,
                  "trainable_names": [SELECTED], "max_trainable_parameters": 2097152,
                  "max_new_tokens": 32, "max_sequence_tokens": 512, "device": "cuda",
                  "min_free_cuda_before_gib": 4., "min_free_cuda_during_gib": .5}
        write_json(output / f"native-seed{seed}.json", config)
        config_digests[str(seed)] = digest_json(config)
    bindings = {name: {"path": str(output / name), "sha256": hash_file(output / name)}
                for name in ("screen.jsonl", "teacher-gate.jsonl", "data-custody.json", "native-seed17.json", "native-seed29.json")}
    code = {str(p.relative_to(Path(__file__).parent)): hash_file(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}
    packet = {
        "schema_version": 1, "experiment": "DSD-001", "status": "PROSPECTIVE_REVIEW_PACKET",
        "execution_authorized": False, "start_commit": START_COMMIT, "branch": BRANCH,
        "source": {"core_id": SOURCE_CORE, "checkpoint": custody["checkpoint_path"],
                   "checkpoint_sha256": SOURCE_SHA, "weight_sha256": SOURCE_WEIGHT_SHA,
                   "custody_path": str(Path(custody_path).resolve()), "custody_file_sha256": hash_file(custody_path),
                   "inspection_path": str(Path(inspection_path).resolve()), "inspection_file_sha256": hash_file(inspection_path)},
        "code_sha256": digest_json(code), "code_files": code, "runtime_recipe": dict(RUNTIME_RECIPE),
        "teacher_template_sha256": digest_json(TEACHER_TEMPLATE),
        "data": {"status": "CANDIDATE_NOT_ADMITTED", "bindings": bindings,
                 "cohorts": {k: v["ids"] for k, v in bank["cohort_manifest"]["cohorts"].items()}, "screen_rows": 56, "gate_rows": 8,
                 "row_identity": {r["id"]: {k: r[k] for k in ("parent_id", "task", "purpose", "split")}
                                  for r in bank["screen_rows"]},
                 "native_tokenization": "NOT_EXECUTED", "admission_review": "PENDING"},
        "design": {"arms": [asdict(a) for a in TINY_ARMS], "seeds": list(SEEDS),
                   "cells": 6, "steps_per_cell": 12, "train_row_streams": streams,
                   "config_sha256_by_seed": config_digests,
                   "acquisition_steps": [10, 11, 12], "learning_rate": 1e-5,
                   "selected_parameter": SELECTED, "selected_numel": 2097152,
                   "rationale": "Isolate rollout source with forward KL and frozen EMA0; test privileged trajectories versus clean deployment prefixes. No controlled/native outcome selected this contrast.",
                   "matching": "Same per-seed ordered rows, source, trainables, optimizer, LR, references and updates. Tokens/FLOPs and actual update sizes measured separately.",
                   "seed_unit": "Two paired ordering/sampling seeds on one candidate bank; not two independent training draws or confirmation."},
        "teacher_gate": {"rows": 8, "min_correct": 7, "min_paired_wins": 2, "max_paired_harms": 0,
                         "generation": "greedy, thinking disabled, 32 response tokens; source teacher, no update",
                         "failure": "Stop comparison if gate fails, source clean ceiling prevents two wins, prompt budget fails or reference verification fails; preserve all outputs, no easier replacement.",
                         "privilege_boundary": "References only in TRAIN objective and separate gate-only teacher context; never in student inference, IID/transfer/retention generation or selection."},
        "limits": {"total_updates": 72, "max_response_tokens": 32, "max_sequence_tokens": 512,
                   "qualification_updates": 0, "qualification_forward_calls": 2048,
                   "qualification_processed_input_tokens": 1048576, "qualification_generated_tokens": 2048,
                   "per_sd_cell_forward_calls": 5528, "per_sft_cell_forward_calls": 5132,
                   "per_cell_processed_input_tokens": 3000000, "per_cell_generated_tokens": 5504,
                   "comparison_processed_input_tokens": 16576512, "comparison_generated_tokens": 32256,
                   "comparison_sft_reference_tokens": 768,
                   "qualification_wall_seconds": 1800, "per_cell_wall_seconds": 1800,
                   "minimum_free_host_gib_before": 6., "minimum_free_host_gib_during": 1.5,
                   "minimum_free_cuda_gib_before": 4., "minimum_free_cuda_gib_during": .5,
                   "minimum_free_disk_gib": 5., "automatic_retries": 0,
                   "derivation": "40 DEV rows x baseline+3 checkpoints x32 generation calls=5120 calls/cell; SD12x(32 rollout+2 scoring)=408; SFT12 scoring. Upper bounds, not measured cost or fit."},
        "stages": {
            "offline": {"status": "PREPARED", "native_loads": 0},
            "native_qualification": {"status": "NOT_EXECUTED", "approval": "NEW_DSD_BYTE_BOUND_REVIEW_REQUIRED",
                "checks": ["exact source pre/post custody and full-load keys", "raw BF16 source versus selected FP32-master unprivileged full-logit parity (exact; no relaxed tolerance)",
                           "representative and 512-token maximum shape forward/backward without optimizer initialization or update",
                           "both rollout sources x both KL directions plus SFT; math SDPA profiler dispatch",
                           "source/teacher/frozen parameters and buffers unchanged; exception swap restoration",
                           "gate-only teacher competence with paired clean student generation",
                           "complete native tokenization with no truncation and privilege sentinel invariance",
                           "load/stage host, CUDA allocated+reserved/free, disk, token/call/time ceilings and failure receipts"]},
            "comparison": {"status": "NOT_AUTHORIZED", "approval": "SEPARATE_EXPLICIT_TRAINING_REVIEW_AFTER_QUALIFICATION",
                           "reload_exact_source_each_cell": True, "no_sweeps": True}},
        "analysis": {"primary_acquisition": "IID gain averaged over steps10,11,12, harder cohort reported separately",
                     "harder_transfer": "source/final row transitions on 5-7 operations; no posthoc row exclusions or pooled primary endpoint",
                     "retention": "final macro-task change plus each task source-correct retained/total, damages, repairs; zero denominator undefined",
                     "minimum_source_correct_per_retention_task": 4,
                     "retention_margin": "No more than one source-success damage per8-row retention task; no repairs hiding damages",
                     "acquisition_condition": "Positive IID gain and at least2 newly correct IID rows averaged over final3 checkpoints; harder transfer gains reported independently",
                     "interpretation": "A weaker update or null acquisition is not retention success. If less damage accompanies materially smaller cumulative update and no acquisition match, report attenuation-confounded/inconclusive. No SFT dominance from one LR; later matched-dose/LR controls need new preregistration.",
                     "uncertainty": "Row transitions and paired seed contrasts descriptive only; no pooled repeated-row p-values, no promotion."},
        "failure_rules": ["Any identity/custody/privilege/budget/resource/nonfinite/missing-gradient/frozen mutation failure stops the stage with create-only receipt.",
                          "Gate or parity failure bars comparison; no source, tolerance, precision, template, task or optimizer substitution.",
                          "Cell failure remains part of attempted6; stop campaign without automatic rerun, shorter rows or dropped cases.",
                          "No source editing, native execution here, sealed/confirmation reads, foreign permit, download, serving change or promotion."],
        "limitations": ["Candidate semantic/rights/admission review and native tokenizer budgets pending.",
                        "The reviewed qualification entry point is prospective and fixture-verified only; legacy qualify/train commands do not enforce this packet. Comparison needs later reviewed bounded execution integration and explicit training approval.",
                        "Exact runtime proposed: Python3.11.15, torch2.10.0+cu126, Transformers4.57.1, safetensors0.7.0, tokenizers0.22.2. Compatibility/native fit unexecuted; shared runtime must not be modified.",
                        "Small modular-state and arithmetic/spatial cohorts are DEV implementation probes, not broad reasoning or independent-family confirmation."],
        "next_decision": "Review exact candidate data/design/runtime and approve only the byte-bound bounded zero-update qualification; comparison execution integration and training require a later explicit decision after qualification."
    }
    validate_packet(packet)
    write_json(output / "proposal.json", packet)
    write_json(output / "packet-receipt.json", {"status": "OFFLINE_PACKET_PREPARED", "proposal_sha256": hash_file(output / "proposal.json"),
        "native_executed": False, "training_updates": 0, "execution_authorized": False, "bindings": bindings})
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "custody", "inspection", "workspace", "business-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.output, args.custody, args.inspection, args.workspace, args.business_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
