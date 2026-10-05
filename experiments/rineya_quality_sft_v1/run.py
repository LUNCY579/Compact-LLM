"""Owner export -> exposure review -> supervised qualification/baseline/two arms."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import csv
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from .bank import owner_request_template, read_owner_bundle
from .compiler import compile_plan, validate_plan
from .contract import ARMS, DATASET_OWNER, Design, SOURCE_PATH
from .integrity import (PACKAGE, ROOT, canonical, checkout_identity, code_identity, digest,
    external_caches, file_hash, gpu_lease, new_dir, read_json, regular, seal, unseal, write_new)
from .supervisor import bounded_child, claim_worker


def exposure_review_template(plan):
    return {"experiment": EXPERIMENT, "plan_sha256": plan["sha256"], "decision": "PENDING",
        "reviewer": "", "rationale": "", "native_masks_and_eos_checked": False,
        "all_rows_and_parent_groups_checked": False, "actual_token_and_update_accounting_checked": False,
        "owner_admission_and_source_binding_checked": False, "public_dev_no_protected_outputs": False}


def validate_exposure_review(review, plan):
    expected = exposure_review_template(plan)
    if (not isinstance(review, dict) or set(review) != set(expected)
            or review["experiment"] != EXPERIMENT or review["plan_sha256"] != plan["sha256"]
            or review["decision"] != "APPROVED" or not isinstance(review["reviewer"], str)
            or not review["reviewer"].strip() or not isinstance(review["rationale"], str) or not review["rationale"].strip()
            or any(review[key] is not True for key, value in expected.items() if type(value) is bool)):
        raise PermissionError("Actual independent RQSFT exposure review remains missing/stale/pending")


def admission_preflight(plan_value, review=None):
    plan = validate_plan(plan_value)
    if plan["kind"] != "native_plan" or plan["admission"]["fixture_only"] is not False:
        raise PermissionError("Fixture/EXCLUDED data cannot authorize native Rineya")
    if review is not None:
        validate_exposure_review(review, plan_value)
    path = regular(plan["inputs"]["owner_manifest_path"])
    if file_hash(path) != plan["inputs"]["owner_manifest_sha256"]:
        raise ValueError("Owner export changed after preparation")
    bank = read_owner_bundle(path)
    if bank["export_binding_sha256"] != plan["inputs"]["owner_export_binding_sha256"]:
        raise ValueError("Actual owner admission/export binding differs")
    if code_identity() != plan["code_identity"]:
        raise ValueError("Reviewed consumer/dependency code changed")
    return plan, bank


def prepare(owner_manifest, output, source=SOURCE_PATH):
    directory = new_dir(output)
    try:
        # Actual owner metadata/receipts pass BEFORE tokenizer or model dependencies.
        bank = read_owner_bundle(owner_manifest)
        source = Path(source)
        if source.resolve() != Path(SOURCE_PATH).resolve():
            raise ValueError("Exact registered local source path required")
        with external_caches(directory):
            # Cache locations must be set before HF/Torch modules capture them.
            from .native import source_custody, runtime, RUNTIME_PINS
            from transformers import AutoTokenizer
            custody = source_custody(source)
            versions = runtime()
            if any(versions["libraries"][n].split("+")[0] != v for n, v in RUNTIME_PINS.items()):
                raise ValueError("Pinned native tokenizer runtime required")
            tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True,
                trust_remote_code=False, use_fast=True)
            if not tokenizer.is_fast or tokenizer.eos_token_id != 151645:
                raise ValueError("Exact fast source tokenizer/EOS required")
            source_identity = {"core_id": SOURCE_CORE, "weight_sha256": SOURCE_WEIGHT_SHA256,
                "parameters": 596049920, "custody": custody}
            plan = compile_plan(bank, tokenizer, source_identity, code_identity())
            if source_custody(source) != custody:
                raise ValueError("Immutable source changed during tokenizer preparation")
        write_new(directory / "plan.json", plan)
        write_new(directory / "EXPOSURE_REVIEW_PENDING.json", exposure_review_template(plan))
        write_new(directory / "preparation.json", {"experiment": EXPERIMENT, "status": "OWNER_BANK_COMPILED_REVIEW_PENDING",
            "plan_sha256": plan["sha256"], "owner": DATASET_OWNER, "all_train_rows": 200,
            "dev_rows": 512, "native_model_loaded": False, "native_training_updates": 0,
            "processed_tokens": plan["processed_tokens"], "updates_per_arm": len(plan["groups"])})
        return plan
    except BaseException as exc:
        write_new(directory / "FAILED.json", {"experiment": EXPERIMENT, "phase": "prepare",
            "status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc),
            "native_model_loaded": False, "native_training_updates": 0})
        raise


def require_model_exclusivity(launch):
    legacy = Path(launch["legacy_mq_lock"])
    if legacy.exists():
        raise RuntimeError("Another/stale legacy GPU lease exists; preserve it and reuse its owner")
    probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if probe.returncode:
        raise RuntimeError("Cannot establish local GPU process ownership; no launch fallback")
    owners = {os.getpid(), os.getppid(), launch["supervisor_pid"]}
    unknown = []
    for fields in csv.reader(probe.stdout.splitlines()):
        if not fields or not any(f.strip() for f in fields):
            continue
        try:
            pid = int(fields[0].strip())
        except ValueError:
            if "No running processes" in ",".join(fields):
                continue
            raise RuntimeError("GPU owner inventory is opaque; cannot assume idle")
        if pid not in owners:
            unknown.append({"pid": pid, "process": fields[1].strip() if len(fields) > 1 else "unknown"})
    if unknown:
        raise RuntimeError("Another GPU compute owner is active: " + str(unknown))


class WorkGuard:
    """Charge logical input/forward work; retain actual host/CUDA boundary peaks."""
    def __init__(self, phase, directory, groups, snapshot=None):
        self.phase, self.directory, self.started = phase, Path(directory), time.monotonic()
        self.groups, self.samples, self.calls, self.phase_calls = groups, [], 0, 0
        self.counts = {"forward_calls": 0, "processed_input_tokens": 0, "generated_tokens": 0}
        self.snapshot_fn = snapshot

    def __call__(self, stage="boundary", **cost):
        if self.snapshot_fn is None:
            from experiments.model_quality_gpu_v1.resources import guard
            value = guard(before_load=stage == "load_before", phase="train" if self.phase == "train" else "qualify")
            import psutil, torch
            value["process_peak_working_set_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
            value["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        else:
            value = self.snapshot_fn(stage)
        for src, dst in (("forward_calls", "forward_calls"), ("input_tokens", "processed_input_tokens"),
                         ("generated_tokens", "generated_tokens")):
            amount = cost.get(src, 0)
            if type(amount) is not int or amount < 0:
                raise ValueError("Nonnegative integer logical work charges required")
            self.counts[dst] += amount
        max_input = 25000 if self.phase == "train" else 2 * 1024 * 1024 if self.phase == "qualify" else 1048576
        if (self.counts["processed_input_tokens"] > max_input or self.counts["forward_calls"] > 32768
                or self.counts["generated_tokens"] > 32768):
            raise RuntimeError("Prospective native work ceiling exceeded before further operation")
        disk = shutil.disk_usage(self.directory).free
        checkpoint_count = math.ceil(self.groups / 4) if self.phase == "train" else 1
        minimum_disk = (314618880 * 12 * checkpoint_count if self.phase != "baseline" else 0) + 1024**3
        if stage == "load_before" and disk < minimum_disk or disk < 1024**3:
            raise RuntimeError("Disk cannot retain every required checkpoint/receipt; no cleanup fallback")
        elapsed = time.monotonic() - self.started
        if elapsed > (1800 if self.phase == "qualify" else 3600):
            raise RuntimeError("Native stage wall ceiling exceeded")
        self.calls += 1
        self.phase_calls += 1
        sample = dict(value, stage=stage, elapsed_seconds=elapsed, disk_free_bytes=disk,
            counts=dict(self.counts), forecasts={k: v for k, v in cost.items() if k not in {"input_tokens", "forward_calls", "generated_tokens"}})
        # Preserve full samples on disk; no growing unbounded in-memory journal.
        # At most three boundary samples per64-token generation forward, plus
        # bounded stage overhead. The first20k cap could not cover512 legal rows.
        sample_ceiling = 3 * 32768 + 1024 if self.phase == "baseline" else 20000
        if self.phase_calls > sample_ceiling:
            raise RuntimeError("Resource sample ceiling exceeded; preserve samples")
        folder = self.directory / "resources"
        folder.mkdir(exist_ok=True)
        write_new(folder / f"{self.calls:05d}.json", sample)
        return sample

    def report(self):
        return {**self.counts, "wall_seconds": time.monotonic() - self.started,
            "resource_samples": self.calls, "counter_scope": "Logical top-level calls/input tokens; checkpoint recomputation FLOPs not measured"}


def binding_for(plan_value, review, source_info, checkout):
    from .evaluation import RUBRIC_SHA256
    from .native import GENERATION_RECIPE
    recipe = dict(GENERATION_RECIPE, eos_ids=source_info["eos_ids"])
    return {"experiment": EXPERIMENT, "source_core": SOURCE_CORE, "source_weight_sha256": SOURCE_WEIGHT_SHA256,
        "compiled_plan_sha256": plan_value["sha256"], "code_sha256": plan_value["code_identity"]["sha256"],
        "runtime": source_info["runtime"], "initial_parameters_sha256": source_info["initial_parameters_sha256"],
        "initial_frozen_state_sha256": source_info["initial_frozen_state_sha256"],
        "generation_sha256": digest(recipe), "scorer_sha256": digest({"rubric": RUBRIC_SHA256,
            "evaluation": file_hash(PACKAGE / "evaluation.py"), "analysis": file_hash(PACKAGE / "analysis.py")}),
        "checkout": checkout, "review_sha256": digest(review)}


def completed_stage(path, identity, phase):
    """Complete predictions/inner PASS remain partial evidence until outer success."""
    path = regular(path)
    folder = path.parent
    stage = unseal(read_json(folder / "stage.json"))
    supervisor = read_json(folder / "supervisor.json")
    if (stage.get("experiment") != EXPERIMENT or stage.get("phase") != phase or stage.get("identity") != identity
            or stage.get("status") != "NATIVE_STAGE_COMPLETED_QUALITY_UNDECIDED"
            or stage.get("source_unchanged") is not True or stage.get("inputs_unchanged") is not True
            or stage.get("native_model_loaded") is not True or (folder / "FAILED.json").exists()
            or supervisor.get("status") != "EXITED" or supervisor.get("returncode") != 0):
        raise PermissionError("Producing attempt failed/incomplete/unsupervised; inner artifacts cannot authorize work")
    witnesses = stage.get("artifact_hashes", {})
    if witnesses.get(path.name) != file_hash(path):
        raise PermissionError("Inner output bytes differ from completed supervised stage witness")
    return stage


def completed_qualification(path, identity):
    from .native import verify_qualification
    completed_stage(path, identity, "qualify")
    return verify_qualification(read_json(path), identity)


def capture_final(model, tokenizer, plan, identity, role, directory, guard, updates):
    from .native import generate
    records, started = [], time.monotonic()
    measured = {"forward_calls": 0, "processed_input_tokens": 0, "generated_tokens": 0}
    cases = directory / "cases"; cases.mkdir()
    for ordinal, row in enumerate(plan["dev"]):
        try:
            result = generate(model, row, tokenizer=tokenizer, eos_ids=identity["eos_ids"], guard=guard)
            record = {"id": row["id"], "input_sha256": result["prompt_sha256"],
                "prediction": result["prediction"], "generated_ids": result["response_ids"],
                "stop_reason": result["stop_reason"], "context_tokens": len(row["prompt_ids"]),
                "generated_tokens": len(result["response_ids"])}
            measured["forward_calls"] += result["cost"]["forward_calls"]
            measured["processed_input_tokens"] += result["cost"]["processed_input_tokens"]
            measured["generated_tokens"] += record["generated_tokens"]
            write_new(cases / f"{ordinal:04d}.json", record)
            records.append(record)
        except BaseException as exc:
            write_new(cases / f"{ordinal:04d}-FAILED.json", {"id": row["id"], "status": "FAILED",
                "reason": str(exc), "partial_ids": getattr(exc, "partial_ids", None),
                "input_sha256": digest(row["prompt_ids"]), "partial_outputs_preserved": True})
            raise
    measured["wall_seconds"] = time.monotonic() - started
    envelope = {"experiment": EXPERIMENT, "role": role, "identity": {k: v for k, v in identity.items() if k != "eos_ids"},
        "records": records, "costs": measured, "updates": updates, "fixture_only": False,
        "promotion_allowed": False, "eos_ids": identity["eos_ids"]}
    write_new(directory / "predictions.json", seal(envelope))
    write_blind_packet(plan, envelope, directory)
    return envelope


def write_blind_packet(plan, envelope, directory):
    """Public task/gold/prediction packet omits model role and native job metadata."""
    from .evaluation import ADJUDICATION_RUBRIC, RUBRIC_SHA256, prediction_hash
    import secrets
    salt = secrets.token_hex(32)
    lookup, tasks = {}, []
    rows = {r["id"]: r for r in plan["dev"]}
    for record in envelope["records"]:
        key = digest([salt, record["id"], envelope["role"]])
        raw = rows[record["id"]]["metadata"]
        lookup[key] = {"row_id": record["id"], "prediction_sha256": prediction_hash(record["prediction"])}
        tasks.append({"opaque_id": key, "messages": raw["messages"][:-1],
            "evaluation": raw["evaluation"], "prediction": record["prediction"],
            "prediction_sha256": prediction_hash(record["prediction"])})
    tasks.sort(key=lambda task: task["opaque_id"])
    write_new(directory / "BLINDED_REVIEW_PACKET.json", {"rubric": ADJUDICATION_RUBRIC,
        "rubric_sha256": RUBRIC_SHA256, "tasks": tasks, "model_role_visible": False,
        "reviews": "PENDING_TWO_INDEPENDENT_REVIEWERS"})
    write_new(directory / "REVIEW_LINK_PRIVATE.json", {"lookup": lookup,
        "prediction_envelope_sha256": digest(envelope), "do_not_supply_to_reviewers": True})


def reviewed_source(plan_path, prediction_path, adjudication_path, output):
    from .analysis import summarize_source
    value = read_json(plan_path)
    plan = validate_plan(value)
    if plan["kind"] != "native_plan":
        raise PermissionError("Source review needs actual owner-admitted native plan")
    envelope = unseal(read_json(prediction_path))
    if (envelope.get("role") != "source" or envelope.get("fixture_only") is not False
            or envelope.get("identity", {}).get("compiled_plan_sha256") != value["sha256"]):
        raise PermissionError("Actual same-plan source predictions required")
    if envelope["identity"]["code_sha256"] != plan["code_identity"]["sha256"]:
        raise ValueError("Source predictions do not bind the reviewed code")
    completed_stage(prediction_path, envelope["identity"], "baseline")
    summary = summarize_source([r["metadata"] for r in plan["dev"]], envelope["records"],
        identity=envelope["identity"], adjudications=read_json(adjudication_path), costs=envelope["costs"],
        eos_token_ids=envelope["eos_ids"], input_hashes={r["id"]: digest(r["prompt_ids"]) for r in plan["dev"]})
    summary.update(experiment=EXPERIMENT, prediction_file_sha256=file_hash(prediction_path),
        compiled_plan_sha256=value["sha256"], source_review_file_sha256=file_hash(adjudication_path))
    directory = new_dir(output)
    write_new(directory / "source-report.json", seal(summary))
    return summary


def worker(args):
    directory = Path(args.output).absolute()
    launch = claim_worker(directory)
    model = guard = None
    before = after = info = identity = None
    loaded, failure = False, None
    try:
        needed = {"qualify": "execute_qualification", "baseline": "execute_evaluation", "train": "execute_training"}[args.command]
        if getattr(args, needed) is not True or getattr(args, "resume", None):
            raise PermissionError("Worker requires explicit phase opt-in; scientific resume remains unapproved")
        plan_value, review = read_json(args.plan), read_json(args.review)
        plan, bank = admission_preflight(plan_value, review)
        checkout = checkout_identity(require_clean=True)
        require_model_exclusivity(launch)
        from . import native
        import torch
        native.configure_runtime()
        guard = WorkGuard(args.command, directory, len(plan["groups"]))
        before = native.source_custody(Path(args.source))
        if before != plan["source_identity"]["custody"]:
            raise ValueError("Actual immutable source differs from compiled plan")
        guard("load_before")
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer, info = native.bind_model(args.source, training=args.command != "baseline")
        loaded = True
        guard("load_after")
        # Rebuild ALL IDs/masks against actual runtime; no source/data permit transplant.
        if compile_plan(bank, tokenizer, plan["source_identity"], code_identity()) != plan_value:
            raise ValueError("Actual native tokenizer/history/EOS plan differs")
        identity = binding_for(plan_value, review, info, checkout)
        write_new(directory / "binding.json", seal(identity))
        if args.command == "qualify":
            native.qualify(model, tokenizer, plan_value, identity, guard, directory=directory)
        else:
            completed_qualification(args.qualification, identity)
            if args.command == "baseline":
                capture_final(model, tokenizer, plan, dict(identity, eos_ids=info["eos_ids"]), "source", directory, guard, 0)
            else:
                source_envelope = unseal(read_json(args.baseline))
                if (source_envelope.get("identity") != identity or source_envelope.get("role") != "source"
                        or source_envelope.get("fixture_only") is not False or len(source_envelope.get("records", [])) != 512):
                    raise PermissionError("Actual same-binding complete source baseline required before training")
                completed_stage(args.baseline, identity, "baseline")
                source_report = unseal(read_json(args.source_report))
                if (source_report.get("status") != "COMPLETE" or source_report.get("identity") != identity
                        or source_report.get("compiled_plan_sha256") != plan_value["sha256"]
                        or source_report.get("prediction_file_sha256") != file_hash(args.baseline)
                        or source_report.get("fixture_only") is not False or source_report.get("captured_rows") != 512):
                    raise PermissionError("Reviewed complete source/error report required; missing judgments cannot authorize training")
                from .engine import train_cell
                trained = unseal(train_cell(model, plan_value, args.arm, directory, identity,
                    guard=guard, resume=Path(args.resume) if args.resume else None))
                updates = [{"step": r["updates"], "loss": r["objective_ce"],
                    "preclip_gradient_norm": r["preclip_gradient_l2"], "update_l2": r["update_l2"]} for r in trained["trace"]]
                write_new(directory / "training-costs.json", guard.report())
                evaluation_guard = WorkGuard("baseline", directory, len(plan["groups"]))
                evaluation_guard.started = guard.started
                # Shared resource filenames retain both phase streams without overwrite.
                evaluation_guard.calls = guard.calls
                capture_final(model, tokenizer, plan, dict(identity, eos_ids=info["eos_ids"]), args.arm, directory, evaluation_guard, updates)
                write_new(directory / "evaluation-costs.json", evaluation_guard.report())
                guard = evaluation_guard
        guard("stage_complete")
    except BaseException as exc:
        failure = exc
    finally:
        if before is not None:
            try:
                from .native import source_custody
                after = source_custody(Path(args.source))
                if after != before:
                    raise RuntimeError("Source custody changed, including failed stage")
            except BaseException as exc:
                if failure is None: failure = exc
                else: failure.add_note("Source final custody: " + str(exc))
        inputs_unchanged = False
        if identity is not None:
            try:
                admission_preflight(read_json(args.plan), read_json(args.review))
                inputs_unchanged = True
            except BaseException as exc:
                if failure is None: failure = exc
                else: failure.add_note("Final input/code custody: " + str(exc))
        receipt = {"experiment": EXPERIMENT, "phase": args.command, "native_model_loaded": loaded,
            "source_before": before, "source_after": after, "source_unchanged": before is not None and before == after,
            "inputs_unchanged": inputs_unchanged,
            "identity": identity, "costs": guard.report() if guard else None,
            "automatic_retry": False, "promotion_allowed": False}
        receipt["artifact_hashes"] = {name: file_hash(directory / name) for name in
            ("qualification.json", "predictions.json", "train-result.json") if (directory / name).is_file()}
        if failure is not None:
            receipt.update(status="FAILED", error_type=type(failure).__name__, reason=str(failure),
                notes=getattr(failure, "__notes__", []))
            write_new(directory / "FAILED.json", seal(receipt))
        else:
            receipt["status"] = "NATIVE_STAGE_COMPLETED_QUALITY_UNDECIDED"
            write_new(directory / "stage.json", seal(receipt))
        model = None
        gc.collect()
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    return 0


def public_native(args):
    needed = {"qualify": "execute_qualification", "baseline": "execute_evaluation", "train": "execute_training"}[args.command]
    if getattr(args, needed) is not True:
        raise PermissionError("Explicit --" + needed.replace("_", "-") + " required")
    if getattr(args, "resume", None):
        raise PermissionError("Scientific resume needs separately reviewed complete attempted-cost/update lineage; recovery mechanics alone are insufficient")
    if not args.plan or not args.review:
        raise PermissionError("Actual admitted plan and independent exposure review required")
    plan_value, review = read_json(args.plan), read_json(args.review)
    plan, _ = admission_preflight(plan_value, review)
    checkout_identity(require_clean=True)
    if args.command != "qualify" and not args.qualification:
        raise PermissionError("Matching native zero-update qualification required")
    if args.command == "train" and (args.arm not in ARMS or not args.baseline or not args.source_report):
        raise PermissionError("Exactly one frozen arm and source baseline must be selected")
    command = [sys.executable, "-B", "-m", "experiments.rineya_quality_sft_v1", args.command]
    for name in ("source", "plan", "review", "qualification", "baseline", "source_report", "resume", "output"):
        value = getattr(args, name, None)
        if value: command += ["--" + name.replace("_", "-"), str(Path(value).absolute())]
    if args.arm: command += ["--arm", args.arm]
    command += ["--" + needed.replace("_", "-"), "--_worker"]
    with gpu_lease():
        from .campaign import claim, settle
        reservation = claim(plan_value["sha256"], digest(review), args.command, args.arm, args.output)
        code = None
        try:
            code = bounded_child(command, args.output, 1800 if args.command == "qualify" else 3600)
            return code
        finally:
            settle(reservation, code, args.output)


def analyze(plan_path, source_path, candidate_path, source_reviews, candidate_reviews, output):
    from .analysis import compare_quality
    plan_value = read_json(plan_path)
    plan = validate_plan(plan_value)
    if plan["kind"] != "native_plan":
        raise PermissionError("Native analysis cannot consume fixture plan")
    source, candidate = unseal(read_json(source_path)), unseal(read_json(candidate_path))
    if any(envelope.get("identity", {}).get("compiled_plan_sha256") != plan_value["sha256"]
           or envelope.get("identity", {}).get("code_sha256") != plan["code_identity"]["sha256"]
           for envelope in (source, candidate)):
        raise ValueError("Both measured envelopes must bind this exact reviewed training/evaluation plan")
    completed_stage(source_path, source["identity"], "baseline")
    completed_stage(candidate_path, candidate["identity"], "train")
    rows = [r["metadata"] for r in plan["dev"]]
    hashes = {r["id"]: digest(r["prompt_ids"]) for r in plan["dev"]}
    report = compare_quality(rows, source, candidate, expected_identity=source["identity"],
        baseline_adjudications=read_json(source_reviews), candidate_adjudications=read_json(candidate_reviews),
        input_hashes=hashes, eos_token_ids=source["eos_ids"])
    directory = new_dir(output)
    write_new(directory / "comparison.json", seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    request = subs.add_parser("owner-request"); request.add_argument("--output", required=True)
    prep = subs.add_parser("prepare"); prep.add_argument("--owner-manifest", required=True)
    prep.add_argument("--source", default=SOURCE_PATH); prep.add_argument("--output", required=True)
    for name in ("qualify", "baseline", "train"):
        p = subs.add_parser(name)
        for field in ("plan", "review", "qualification", "baseline", "source-report", "resume", "output"):
            p.add_argument("--" + field, required=field in {"plan", "review", "output"})
        p.add_argument("--source", default=SOURCE_PATH); p.add_argument("--arm", choices=tuple(ARMS))
        p.add_argument("--execute-qualification", action="store_true")
        p.add_argument("--execute-evaluation", action="store_true")
        p.add_argument("--execute-training", action="store_true")
        p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p = subs.add_parser("review-source")
    for field in ("plan", "source-predictions", "source-reviews", "output"):
        p.add_argument("--" + field, required=True)
    p = subs.add_parser("analyze")
    for field in ("plan", "source-predictions", "candidate-predictions", "source-reviews", "candidate-reviews", "output"):
        p.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    if args.command == "owner-request":
        directory = new_dir(args.output); write_new(directory / "OWNER_EXPORT_CONTRACT.json", owner_request_template()); return 0
    if args.command == "prepare":
        prepare(args.owner_manifest, args.output, args.source); return 0
    if args.command == "analyze":
        analyze(args.plan, args.source_predictions, args.candidate_predictions, args.source_reviews, args.candidate_reviews, args.output); return 0
    if args.command == "review-source":
        reviewed_source(args.plan, args.source_predictions, args.source_reviews, args.output); return 0
    return worker(args) if args._worker else public_native(args)
