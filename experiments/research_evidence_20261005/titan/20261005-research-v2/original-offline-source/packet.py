"""Build one create-only offline quality packet under the owned CPU supervisor.

New final membership is compiled and mechanically audited prospectively, without
model outputs. It is quarantined in an archive, never used for recipe selection.
"""
from __future__ import annotations
import collections
import hashlib
import gzip
import io
import json
from pathlib import Path
import zipfile

from . import custody, exclusion, generator, registry, schema, token_audit
from .replay import load_replay
from .statistics import ENDPOINT_REGISTRY, ENDPOINT_ALPHA
from .native_proposal import proposal


def _jsonl(rows):
    return "".join(schema.canonical(row) + "\n" for row in rows).encode("utf-8")


def _token_summary(audit):
    # The per-row hashes remain committed by audit_sha256. Avoid writing 32,000
    # repetitive token summaries into a lightweight worker's output allowance.
    return {k: v for k, v in audit.items() if k != "row_audits"}


def _fixture_target_policy(rows):
    natural = [r for r in rows if r["variant_id"] == "natural" and not r["answerable"]]
    if any(generator.target_reply(r) == "UNKNOWN" for r in natural):
        raise ValueError("STOP_UNKNOWN_IMPOSED_ON_NATURAL_RESPONSE")
    return {"natural_unanswerable_rows": len(natural), "natural_unknown_targets": 0}


def prepare_packet(root):
    root = custody.external(root)
    recipe_registry = registry.build_registry()
    plan = registry.make_plan(recipe_registry)
    identity = custody.source_identity()
    custody.write_json(root / "SOURCE_IDENTITY.json", identity)
    keys, exposure = exclusion.exposed_keys()
    replay_rows, replay_receipt = load_replay()
    custody.write_json(root / "EXPOSED_SOURCE_AUDIT.json", exposure)
    custody.write_json(root / "REPLAY_PROVENANCE.json", replay_receipt)
    tokenizer = token_audit.load_tokenizer(custody.CORE / "tokenizer.json", identity["sidecar_sha256"]["tokenizer.json"])
    dev = generator.build_split("dev", keys)
    final = generator.build_split("final", keys)
    # A compiler may verify new deterministic gold. No model or prediction bank
    # participates and no final payload is returned in CLI output.
    leakage = exclusion.audit_partition({"dev": dev, "final": final, "replay": replay_rows}, keys)
    if not leakage["passed"] or len(dev) != 512 or len(final) != 4096:
        raise ValueError("STOP_EVALUATION_MEMBERSHIP_AUDIT")
    dev_tokens = _token_summary(token_audit.audit_rows(dev, tokenizer))
    final_tokens = _token_summary(token_audit.audit_rows(final, tokenizer))
    final_bytes = _jsonl(final)
    archive = root / "FINAL_MEMBERSHIP.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        stream.writestr("final.jsonl", final_bytes)
    final_manifest = {"archive": str(archive), "archive_sha256": custody.sha_bytes(archive.read_bytes()),
        "payload_sha256": custody.sha_bytes(final_bytes), "member": "final.jsonl",
        "rows": len(final), "canonical_worlds": len({r["world_id"] for r in final}),
        "generation_policy": "FIXED_PROSPECTIVE_SEED_ORACLE_VALIDATED_WITH_ZERO_MODEL_OUTPUTS",
        "new_final_payload_compiled_offline": True, "final_model_evaluations": 0,
        "final_access": "BLOCKED_NATIVE_OPERATOR_PERMIT_SELECTION_AND_EXPORT_RECHECK_REQUIRED",
        "archive_is_security_boundary": False}
    custody.write_json(root / "FINAL_MEMBERSHIP_MANIFEST.json", final_manifest)
    with (root / "DEVELOPMENT.jsonl").open("xb") as stream:
        stream.write(_jsonl(dev))
    dataset_receipts, max_prompt, max_target, max_sequence = {}, 0, 0, 0
    designs = {"generator": generator.design_metadata(), "endpoint_registry": list(ENDPOINT_REGISTRY),
               "endpoint_alpha": ENDPOINT_ALPHA, "variants_are_independent_n": False}
    train_folder = root / "training"
    train_folder.mkdir(exist_ok=False)
    for recipe in recipe_registry["recipes"]:
        for seed in schema.SEEDS:
            rows = generator.training_rows(recipe, seed, replay_rows)
            # Repeated historical replay is shared training material, not new
            # independent observations. World and arithmetic leakage into eval
            # is audited for every exact prospective training sequence.
            partition = exclusion.audit_partition({"train": rows, "dev": dev, "final": final}, keys)
            if not partition["passed"]:
                raise ValueError(f"STOP_TRAINING_MEMBERSHIP_AUDIT:{recipe['recipe_id']}:{seed}")
            audit = token_audit.audit_rows(rows, tokenizer)
            _fixture_target_policy(rows)
            trial_id = f"TQR-{recipe['recipe_id']}-S{seed}"
            path = train_folder / (trial_id + ".jsonl.gz")
            content = _jsonl(rows)
            compressed = gzip.compress(content, compresslevel=6, mtime=0)
            with path.open("xb") as stream:
                stream.write(compressed)
            actual_replay = [r for r in rows if r.get("is_replay")]
            source_ids = {r["provenance"]["source_id"] for r in actual_replay}
            expected_replay = 400 if recipe["replay_fraction"] else 0
            if len(actual_replay) != expected_replay or len(rows) != 2000:
                raise ValueError("STOP_REALIZED_TRAINING_MIXTURE")
            replay_mass = 0
            for row, row_audit in zip(rows, audit["row_audits"], strict=True):
                if row.get("is_replay"):
                    replay_mass += row_audit["loss_tokens"]
            dataset_receipts[trial_id] = {"path": str(path), "sha256": custody.sha_bytes(compressed),
                "encoding": "gzip-jsonl", "payload_sha256": custody.sha_bytes(content),
                "payload_bytes": len(content), "artifact_bytes": len(compressed),
                "recipe_id": recipe["recipe_id"], "seed": seed, "examples": 2000,
                "repair_examples": 2000 - len(actual_replay), "replay_examples": len(actual_replay),
                "unique_replay_source_rows": len(source_ids), "replay_occurrences_beyond_unique": len(actual_replay) - len(source_ids),
                "example_replay_fraction": len(actual_replay) / 2000,
                "replay_loss_tokens": replay_mass, "replay_loss_token_fraction": replay_mass / audit["total_loss_tokens"],
                "family_example_counts": dict(collections.Counter(r["family"] for r in rows)),
                "prompt_style_counts": dict(collections.Counter(r["variant_id"] for r in rows if not r.get("is_replay"))),
                "independent_oracles_agree": True, "leakage_passed": True, "token_audit": _token_summary(audit)}
            max_prompt = max(max_prompt, audit["max_prompt_tokens"])
            max_target = max(max_target, audit["max_target_tokens_including_eos"])
            max_sequence = max(max_sequence, audit["max_sequence_tokens"])
    source_files = custody.source_manifest()
    receipt = {"experiment": schema.EXPERIMENT, "status": "OFFLINE_VALIDATED_NATIVE_BLOCKED",
        "offline_validation_passed": True, "source_metadata_verified": identity["metadata_verified"],
        "fresh_weight_digest_verified": False, "old_sealed_banks_accessed": False,
        "registry_sha256": recipe_registry["registry_sha256"], "recipes": 8, "trials": 16,
        "development": {"prompts": 512, "worlds": 448, "token_audit": dev_tokens,
            "path": str(root / "DEVELOPMENT.jsonl"), "sha256": custody.sha_bytes(_jsonl(dev)),
            "qualification_row_ids": [r["row_id"] for f in schema.REPAIR_FAMILIES
                for r in [r for r in dev if r["family"] == f and r["variant_id"] == "canonical"][:2]],
            "reload_readout_row_id": next(r["row_id"] for r in dev if r["family"] == "ledger" and r["variant_id"] == "canonical")},
        "final": {"prompts": 4096, "worlds": 3584, "token_audit": final_tokens,
                  "model_evaluations": 0, "access_blocked": True},
        "training_rows_prepared": 32000, "datasets": dataset_receipts, "leakage": leakage,
        "max_training_prompt_tokens": max_prompt, "max_training_target_tokens_with_eos": max_target,
        "max_training_sequence_tokens": max_sequence, "source_manifest": source_files,
        "current_model_loads": 0, "current_model_forwards": 0, "current_training_updates": 0,
        "quality_claim": False, "native_execution_authorized": False,
        "limitations": ["Full HF template equivalence and fresh checkpoint digest verification require separate qualified access.",
            "Replay uses 195 independently checked historical transition rows; broader conversation replay remains unqualified.",
            "Ledger balance-domain support is unestablished; expanded operand cases and balance uncertainty are disclosed.",
            "No measured native memory, throughput, optimizer or dense-export qualification.",
            "Local/exposed exclusions cannot establish global pretraining or protected-bank disjointness."]}
    custody.write_json(root / "CPU_VALIDATION.json", receipt)
    custody.write_json(root / "DESIGN_METADATA.json", designs)
    custody.write_json(root / "RESOURCE_PLAN.json", plan)
    custody.write_json(root / "NATIVE_QUALIFICATION_PROPOSAL.json", proposal(receipt))
    custody.write_json(root / "FROZEN_RECIPE_REGISTRY.json", {"registry": recipe_registry,
        "source_manifest": source_files, "source_manifest_sha256": schema.digest(source_files),
        "validation_sha256": custody.sha_bytes((root / "CPU_VALIDATION.json").read_bytes()),
        "design_fixed_before_any_model_outcomes": True,
        "status": "IMMUTABLE_OFFLINE_REGISTRY_NOT_A_NATIVE_EXECUTION_FREEZE",
        "native_execution_authorized": False})
    print(json.dumps({"offline_validation_passed": True, "recipes": 8, "trials": 16,
        "development_prompts": 512, "final_prompts": 4096, "training_examples": 32000,
        "max_prompt_tokens": max_prompt, "max_target_tokens": max_target,
        "native_model_loads": 0, "native_execution_authorized": False}, indent=2))
    return receipt


def read_training_dataset(descriptor):
    """Verify both compressed artifact and original payload; never read final data."""
    path = custody.external(descriptor["path"])
    if path.suffixes[-2:] != [".jsonl", ".gz"] or descriptor.get("encoding") != "gzip-jsonl":
        raise ValueError("STOP_TRAINING_DATA_ENCODING")
    compressed = custody.read_pinned(path, descriptor["sha256"])
    if len(compressed) > 50 * 1024 ** 2 or type(descriptor.get("payload_bytes")) is not int or not 0 < descriptor["payload_bytes"] <= 50 * 1024 ** 2:
        raise ValueError("STOP_TRAINING_PAYLOAD_SIZE")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        payload = stream.read(descriptor["payload_bytes"] + 1)
    if custody.sha_bytes(payload) != descriptor["payload_sha256"] or len(payload) != descriptor["payload_bytes"]:
        raise ValueError("STOP_TRAINING_PAYLOAD_DIGEST")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(rows) != 2000 or descriptor.get("examples") != 2000:
        raise ValueError("STOP_TRAINING_DATASET_DOSE")
    for row in rows:
        if row["split"] not in {"train", "replay"}:
            raise ValueError("STOP_NONTRAIN_ROWS_IN_TRAINING_DATASET")
        schema.validate_row(row)
        generator.target_reply(row)
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("STOP_DUPLICATE_TRAINING_ROWS")
    if sum(bool(r.get("is_replay")) for r in rows) != descriptor["replay_examples"]:
        raise ValueError("STOP_TRAINING_REPLAY_DOSE")
    return rows


def verify_packet(root):
    """Require entire supervised attempt success and unchanged source, not a child flag."""
    root = custody.external(root)
    supervisor = json.loads((root / "OFFLINE_SUPERVISOR_RECEIPT.json").read_bytes())
    validation_bytes = (root / "CPU_VALIDATION.json").read_bytes()
    validation = json.loads(validation_bytes)
    if (supervisor.get("experiment") != schema.EXPERIMENT or supervisor.get("stage") != "prepare"
            or supervisor.get("status") != "PASS"
            or supervisor.get("source_unchanged_during_worker") is not True
            or validation.get("offline_validation_passed") is not True):
        raise ValueError("STOP_NO_PASSING_COMPLETE_SUPERVISED_PACKET")
    current = custody.source_manifest()
    if (validation.get("source_manifest") != current
            or supervisor.get("source_admission_sha256") != current
            or supervisor.get("source_completion_sha256") != current):
        raise ValueError("STOP_OFFLINE_PACKET_SOURCE_DRIFT")
    frozen = json.loads((root / "FROZEN_RECIPE_REGISTRY.json").read_bytes())
    registry.validate_registry(frozen["registry"])
    if (frozen.get("source_manifest") != current or frozen.get("source_manifest_sha256") != schema.digest(current)
            or frozen.get("validation_sha256") != custody.sha_bytes(validation_bytes)):
        raise ValueError("STOP_FROZEN_OFFLINE_PACKET_BINDING")
    if validation.get("registry_sha256") != frozen["registry"]["registry_sha256"] or validation.get("native_execution_authorized") is not False:
        raise ValueError("STOP_OFFLINE_REGISTRY_OR_AUTHORITY")
    for descriptor in validation["datasets"].values():
        custody.read_pinned(custody.external(descriptor["path"]), descriptor["sha256"])
    custody.read_pinned(custody.external(validation["development"]["path"]), validation["development"]["sha256"])
    final = json.loads((root / "FINAL_MEMBERSHIP_MANIFEST.json").read_bytes())
    custody.read_pinned(custody.external(final["archive"]), final["archive_sha256"])
    # Do not decompress/open final members while verifying custody.
    return {"verified": True, "status": "OFFLINE_PACKET_VERIFIED_NATIVE_BLOCKED",
            "source_manifest_sha256": schema.digest(current), "registry_sha256": validation["registry_sha256"],
            "final_archive_opened": False, "native_execution_authorized": False}
