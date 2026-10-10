"""Executable factorial training with fixed teacher, exposure, and immutable runs."""
from __future__ import annotations

import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .backend import Rollout, tensor_hash
from .contracts import Arm, Plan, digest, read_json, require, sealed, step_seed, unseal, validate_arm, write_new
from .data import Task, require_roles
from .metrics import evaluate, paired


def schedule(tasks, n, seed, step, purpose):
    require(0 <= n <= len(tasks), "insufficient unique tasks for the registered batch")
    return random.Random(step_seed(seed, "schedule", step, purpose)).sample(sorted(tasks, key=lambda t: t.key), n)


def teacher_bank(backend, tasks: list[Task], seed: int, path: Path, identity: str, *, consumer=None):
    backend.bind_prompt_catalog(tasks, identity)
    train = [t for t in tasks if t.role == "train"]
    expected = {"identity": identity, "plan": backend.plan.fingerprint, "seed": seed,
                "prompts": digest([asdict(t.public()) for t in sorted(train, key=lambda t: t.key)])}
    if path.exists():
        saved = unseal(read_json(path))
        require(saved["binding"] == expected, "teacher bank identity/configuration mismatch")
        require(set(saved["rollouts"]) == {t.key for t in train}, "teacher-bank task coverage")
        require("collection_manifest" in saved and saved.get("created_by"),
                "legacy teacher bank lacks complete consumed-prompt evidence; retain it and register a new bank")
        backend.inherit_prompt_manifest(saved["collection_manifest"], relationship="reused_teacher_bank_collection")
        bank = {}
        for task in train:
            row = dict(saved["rollouts"][task.key])
            for key in ("prompt_ids", "response_ids", "token_logps"):
                row[key] = tuple(row[key])
            r = Rollout(**row)
            backend.check_rollout(task, r, "teacher")
            backend.record_prompt_consumption(task.public(), role="teacher", purpose="teacher_bank_reuse_validation",
                                              kind="teacher_bank_reuse", rendered_prompt_ids=r.prompt_ids)
            bank[task.key] = r
        return bank, saved["collection_cost"], False
    previous_phase = backend.meter.phase
    backend.meter.phase = "teacher_bank"
    before = backend.meter.snapshot()
    bank = {t.key: backend.sample(t.public(), "teacher", step_seed(seed, t.key, 0, "fixed-teacher-bank"),
                                 purpose="fixed_teacher_bank_generation") for t in train}
    after = backend.meter.snapshot()
    cost = {k: after[k]-before.get(k, 0) for k in after if after[k]-before.get(k, 0)}
    write_new(path, sealed({"binding": expected, "collection_cost": cost,
                            "created_by": str(consumer or path.resolve()),
                            "collection_manifest": backend.prompt_manifest(current_only=True),
                            "rollouts": {k: asdict(v) for k, v in bank.items()}}))
    backend.meter.phase = previous_phase
    return bank, cost, True


def run_arm(backend, tasks: list[Task], arm: Arm, output: Path, *, identity: str, bank_path: Path):
    plan = backend.plan
    validate_arm(plan, arm)
    require_roles(tasks, plan.replay_tasks > 0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend.reset()
    backend.bind_prompt_catalog(tasks, identity)
    frozen = backend.frozen_hashes()
    initial_hash = tensor_hash(backend.initial)
    write_new(output / "STARTED.json", sealed({"arm": asdict(arm), "plan": plan.fingerprint,
              "identity": identity, "initial_adapter": initial_hash, "evidence_kind": backend.evidence_kind}))
    start = time.monotonic()
    try:
        train = [t for t in tasks if t.role == "train"]
        replay = [t for t in tasks if t.role == "retention_train"]
        dev = [t for t in tasks if t.role.startswith("dev_")]
        schedule(train, plan.batch_tasks, arm.seed, 0, "train")
        schedule(replay, plan.replay_tasks, arm.seed, 0, "replay")
        bank, bank_cost, new_bank = ({}, {}, False)
        if arm.source == "teacher":
            bank, bank_cost, new_bank = teacher_bank(backend, tasks, arm.seed, Path(bank_path), identity,
                                                     consumer=str(output.resolve()))
        backend.meter.phase = "baseline_eval"
        baseline = evaluate(backend, dev, "student", arm.seed)
        write_new(output / "baseline.json", baseline)
        optimizer = torch.optim.AdamW(list(backend.params.values()), lr=arm.learning_rate,
                                      betas=(.9, .999), eps=1e-8, weight_decay=plan.weight_decay)
        checkpoints = []
        for step in range(plan.steps):
            backend.meter.phase = "training"
            backend.guard()
            optimizer.zero_grad(set_to_none=True)
            losses, exposures = [], []
            for t in schedule(train, plan.batch_tasks, arm.seed, step, "train"):
                if arm.objective == "ce":
                    loss = backend.supervised(t)
                    exposures.append({"key": t.key, "source": "verified_solution", "policy_step": None})
                else:
                    r = bank[t.key] if arm.source == "teacher" else backend.sample(
                        t.public(), "student", step_seed(arm.seed, t.key, step, "on-policy"),
                        purpose="current_student_rollout")
                    loss = backend.distill(t, r, arm.objective, arm.source)
                    exposures.append({"key": t.key, "source": r.source, "policy_step": r.policy_step,
                                      "rollout": asdict(r)})
                require(bool(torch.isfinite(loss)), "nonfinite acquisition loss")
                losses.append(float(loss.detach()))
                # Consume graph before moving models between devices for the next example.
                (loss / plan.batch_tasks).backward()
            replay_keys = []
            for t in schedule(replay, plan.replay_tasks, arm.seed, step, "replay"):
                loss = backend.supervised(t)
                require(bool(torch.isfinite(loss)), "nonfinite retention loss")
                (loss * plan.replay_weight / plan.replay_tasks).backward()
                replay_keys.append(t.key)
            norm = backend.gradient_step(optimizer)
            write_new(output / "steps" / f"{step+1:06d}.json", sealed({
                "step": step+1, "acquisition_sequence_mean_loss": sum(losses)/len(losses),
                "unclipped_gradient_norm": norm, "exposures": exposures, "replay_keys": replay_keys,
                "prompt_consumption_sha256": digest(backend.prompt_manifest()),
                "prompt_consumption_events": backend.prompt_manifest()["event_count"],
                "meter": backend.meter.snapshot()}))
            if (step+1) % plan.checkpoint_every == 0 or step+1 == plan.steps:
                receipt = backend.save(output / "checkpoints" / f"{step+1:06d}")
                backend.meter.phase = "development_eval"
                predictions = evaluate(backend, dev, "student", arm.seed)
                summary = paired(baseline, predictions)
                write_new(output / "eval" / f"{step+1:06d}.json", sealed({
                    "checkpoint": receipt, "predictions": predictions, "summary": summary}))
                manifest = backend.write_prompt_manifest(output / "checkpoints" / f"{step+1:06d}" / "CONSUMED_PROMPTS.json",
                    complete=backend.evidence_kind == "qualified_native_hf")
                checkpoints.append({"step": step+1, "receipt": receipt, "summary": summary,
                                    "consumed_prompt_manifest": manifest,
                                    "geometry": backend.update_geometry(), "meter": backend.meter.snapshot()})
        require(backend.frozen_hashes() == frozen, "frozen teacher or base changed")
        manifest = backend.write_prompt_manifest(output / "CONSUMED_PROMPTS.json",
            complete=backend.evidence_kind == "qualified_native_hf")
        result = {"schema": 1, "status": "COMPLETE", "identity": identity, "plan": plan.fingerprint,
                  "arm": asdict(arm), "initial_adapter": initial_hash, "frozen": frozen,
                  "evidence_kind": backend.evidence_kind, "checkpoints": checkpoints,
                  "teacher_bank_collection_cost": bank_cost, "teacher_bank_created_here": new_bank,
                  "teacher_bank_path": str(Path(bank_path).resolve()) if arm.source == "teacher" else None,
                  "consumed_prompt_manifest": manifest,
                  "meter": backend.meter.snapshot(), "wall_seconds": time.monotonic()-start,
                  "resume_supported": False, "native_efficacy_claim": False}
        write_new(output / "RESULT.json", sealed(result))
        return result
    except BaseException as exc:
        # Preserve the primary failure even if partial consumption cannot be reconciled.
        failure_manifest = None
        manifest_error = None
        try:
            failure_manifest = backend.write_prompt_manifest(output / "FAILED_CONSUMED_PROMPTS.json")
        except BaseException as manifest_exc:
            manifest_error = {"type": type(manifest_exc).__name__, "message": str(manifest_exc)}
        write_new(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc),
                                           "completed_steps": backend.version, "meter": backend.meter.snapshot(),
                                           "consumed_prompt_manifest": failure_manifest,
                                           "manifest_error": manifest_error})
        raise
    finally:
        try:
            backend.restore(backend.initial)
            backend.version = 0
        except BaseException as exc:
            write_new(output / "RESTORE_FAILED.json", {"type": type(exc).__name__, "message": str(exc),
                                                        "primary_failure_retained": (output / "FAILED.json").exists()})
            if not (output / "FAILED.json").exists():
                write_new(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc),
                                                    "stage": "initial_adapter_restore"})
            raise
