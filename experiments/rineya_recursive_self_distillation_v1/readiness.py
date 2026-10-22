"""Offline readiness packet and prospective three-arm design; never run a model."""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .contracts import BASE_SHA256, Plan, digest, file_hash, read_json, require
from .evidence import write_new

BRANCH = "research/rineya-recursive-self-distillation-v1"
PACKAGE = "experiments/rineya_recursive_self_distillation_v1"
CENTRAL = "experiments/post_training_mimo_centralized_v1"
SELECTED_ARMS = ("sft", "frozen", "evolving")


def external_output(path, root):
    """Keep generated packets in BusinessProject, outside any tracked checkout."""
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root.parent) and not path.is_relative_to(root)
            and ".codex" not in {part.casefold() for part in path.parts},
            "outputs must stay under BusinessProject outside tracked source and .codex")
    require(not any((ancestor / ".git").exists() for ancestor in (path, *path.parents)),
            "output cannot be inside another tracked checkout")
    require(not path.exists(), "readiness output exists; preserve it and choose a new directory")
    return path


def tiny_design(plan):
    require(plan.rounds == 2 and plan.eval_every == 1 and plan.seed == 101
            and plan.batch_tasks == 4 and plan.replay_tasks == 1,
            "this proposal selects exactly two updates, seed101, 4 acquisition and 1 replay")
    return {"status": "PROSPECTIVE_DESIGN_INPUT_IDENTITIES_UNRESOLVED",
            "experiment": "RSD-001", "selected_arms": list(SELECTED_ARMS),
            "srcl_selected": False, "srcl_followup": "independent justification and prospective amendment required",
            "primary_contrast": {"left": "frozen", "right": "evolving", "evaluated_step": 2},
            "continued_sft_control": {"left": "sft", "right": "evolving", "evaluated_step": 2},
            "why_two_updates": "Both guidance arms score with parent version0 at update1; teacher refresh first differs at update2.",
            "plan": asdict(plan), "plan_digest": plan.fingerprint,
            "common_parent": "same exact independently reviewed completed ordinary-SFT adapter; fresh optimizer per arm",
            "teacher": {"frozen": "initial common parent adapter", "evolving": "current pre-update student adapter",
                        "prefix": "identical student response prefixes with identical privileged placement",
                        "loss": "detached full-vocabulary sequence-mean KL(teacher || student)"},
            "serial_order": list(SELECTED_ARMS), "max_updates_per_invocation": 1,
            "clean_resume": "resume only PAUSED after exact source/plan/runtime/parent/input checks; INFLIGHT is quarantined",
            "catalog_minimum_parents": {"train": 32, "train_selection": 8, "calibration": 16,
                                        "retention_train_per_source": 16, "retention_eval_per_source": 16},
            "retention": "fixed independent CE exposure/weight and separate retention-source stop rules; report source limitations",
            "panel_size": "freeze actual source names and rows within token limits before qualification; no automatic expansion",
            "qualification": "unchanged exact E0 plus source-bound parent/catalog and one independent disposable mechanics receipt per arm",
            "qualification_optimizer_calls": 9,
            "treatment_optimizer_ceiling": 6,
            "parent_and_e0_work": "separate prerequisite proposals; costs never counted as treatment efficacy",
            "probe": "after parent qualification, eight TRAIN_SELECTION parents; no-update diagnostic with separately recorded costs",
            "primary_metric": "paired parent-macro calibration accuracy difference at step2; adaptive exploratory bootstrap only",
            "practical_effect": "0.02 is a prospective screen target, not a powered efficacy or significance claim",
            "report": ["correctness", "repairs/damages", "termination/truncation", "length", "every retention source",
                       "attempted generation", "full-prefix processed-token proxy", "model calls", "all invocation wall/load costs"],
            "compute_matched": False, "native_execution_authorized": False,
            "priority": "secondary to CE8/pruning/matched_mlp; no native slot requested while prerequisites are unresolved"}


def prepare(root, plan_path, e0_config, e0_result, binding_path=None):
    root = Path(root).resolve()
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    require(git("branch", "--show-current") == BRANCH, "wrong RSD branch")
    plan = Plan.load(plan_path)
    from .native import checked_file, inputs, source_identity
    from .e0_qualification import validate_e0_chain
    sources = source_identity(root)
    inventory = {str(p.relative_to(root).as_posix()): {"sha256": file_hash(p), "bytes": p.stat().st_size}
                 for p in sorted((root / PACKAGE).rglob("*"))
                 if p.is_file() and p.suffix in (".py", ".md", ".json") and "__pycache__" not in p.parts}
    receipts = {}
    for path in (Path(e0_config), Path(e0_result), Path(e0_result).parent / "SOURCE_RECEIPT.json",
                 Path(e0_result).parent / "RUNTIME_RECEIPT.json",
                 Path(e0_result).parent.with_name(Path(e0_result).parent.name + "_supervisor") / "SUPERVISOR.json",
                 root / CENTRAL / "receipts/CONTINUATION_007_BLOCKED_AUDIT.json"):
        if path.is_file():
            receipts[str(path.resolve())] = {"sha256": file_hash(path), "status": read_json(path).get("status")}
    checks = {}
    def check(name, action):
        try:
            value = action()
            checks[name] = {"status": "OFFLINE_METADATA_CHECKED", "evidence": value}
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            checks[name] = {"status": "UNQUALIFIED", "reason": str(exc)}
    check("exact_e0", lambda: validate_e0_chain(e0_config, e0_result, root))
    if binding_path:
        path = Path(binding_path).resolve()
        cfg = read_json(path)
        def bound(name):
            require(cfg.get(name) is not None, "missing bound " + name)
            return checked_file(cfg[name], path.parent)
        from .parent_qualification import validate_parent
        from .catalog_qualification import validate_catalog_review
        check("common_sft_parent", lambda: validate_parent(bound("parent"))[2])
        check("verified_catalog", lambda: validate_catalog_review(bound("catalog"), bound("exclusions"), bound("catalog_review")))
        check("complete_input_binding", lambda: inputs(path, plan, root)[3])
    else:
        checks["common_sft_parent"] = {"status": "UNQUALIFIED", "reason": "no real reviewed parent binding supplied; template false/0 is ineligible"}
        checks["verified_catalog"] = {"status": "UNQUALIFIED", "reason": "no source-reviewed catalog/exclusion authority binding supplied"}
    checks["native_arm_mechanics"] = {"status": "UNQUALIFIED", "reason": "no per-arm native receipts acquired by this offline packet"}
    parent_evidence = checks["common_sft_parent"].get("evidence", {})
    teachers = {"status": "UNQUALIFIED_UNTIL_REAL_PARENT_AND_NATIVE_MECHANICS",
                "parent_adapter_sha256": parent_evidence.get("adapter_sha256"),
                "frozen": "the exact common reviewed parent, unchanged across all updates",
                "evolving": "the exact current student adapter before each update; detached probabilities",
                "placement": "identical answer-privileged solution placement and student response-token prefixes",
                "unaided_student": "public problem only; no solution or retention/calibration gold",
                "gold_free_srcl": "preserved but not selected; verified naturally terminated strictly shorter rewrites only"}
    blocked = [name for name, item in checks.items() if item["status"] == "UNQUALIFIED"]
    source = {"branch": BRANCH, "head": git("rev-parse", "HEAD"),
              "local_changes": git("status", "--porcelain", "--untracked-files=all"),
              "native_sources": sources, "native_source_digest": digest(sources), "package_files": inventory}
    result = {"schema": 1, "experiment": "RSD-001", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "OFFLINE_READINESS_BLOCKED_NATIVE_NOT_AUTHORIZED", "source": source,
              "base_weight_sha256": BASE_SHA256, "checks": checks, "blocked_requirements": blocked,
              "teacher_qualification": teachers,
              "historical_receipts": receipts, "native_execution_authorized": False,
              "activity": {"pretrained_loads": 0, "native_optimizer_steps": 0, "native_evaluations": 0,
                           "sealed_confirmation_bank_reads": 0, "paid_teacher_calls": 0, "promotions": 0},
              "limitations": ["CPU fixtures are software evidence only", "human-review metadata does not prove truthful provenance",
                              "historical E0 failure is not a current host memory measurement",
                              "no native model fit, CUDA parity/resume, containment or capability measured here"]}
    proposal = tiny_design(plan)
    proposal["source_binding"] = {"head": source["head"], "native_source_digest": source["native_source_digest"],
                                  "plan_path": str(Path(plan_path).resolve()), "plan_sha256": file_hash(plan_path)}
    proposal["unresolved_prerequisites"] = blocked
    proposal["freeze_rule"] = "Bind actual E0 companion receipts, catalog/provenance/exclusions and reviewed parent hashes, commit exact source, and preregister before native probe/qualification or treatment outcomes. This draft is not completed enrollment."
    return result, proposal, source


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(root / PACKAGE / "configs/tiny_guidance_v1.json"))
    parser.add_argument("--e0-config", default=str(root / CENTRAL / "configs/E0_TITAN_BF16.json"))
    parser.add_argument("--e0-result", default=str(root / CENTRAL / "results/e0_titan_v2/E0_RESULT.json"))
    parser.add_argument("--binding")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = external_output(args.output, root)
    result, proposal, source = prepare(root, args.plan, args.e0_config, args.e0_result, args.binding)
    proposal["next_native_proposal"] = {
        "status": "REVIEW_CANDIDATE_ONLY_NOT_AUTHORIZED", "operation": "one unchanged E0 invocation",
        "command": [str(root.parent / ".venv-mq001/Scripts/python.exe"), "-B", "-m",
                    "experiments.post_training_mimo_centralized_v1.scripts.supervise_native",
                    "--config", str(Path(args.e0_config).resolve()), "--output", str(out / "future-native-e0")],
        "maximum_disposable_optimizer_calls": 5, "wall_seconds": 1800,
        "conditions": ["CE8/pruning/matched_mlp priority permits a slot", "same-owner explicit E0 authorization",
                       "review and commit source; exact registered runtime", "live lock/process inventory and unchanged resource admission"],
        "e1_authorized": False, "parent_or_treatment_training_authorized": False}
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "READINESS.json", result)
    write_new(out / "MINIMAL_COMPARISON_PROPOSAL.json", proposal)
    write_new(out / "SOURCE_MANIFEST.json", source)
    print(str(out / "READINESS.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
