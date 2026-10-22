"""Runnable CPU mechanism demo and tiny-policy pilot, plus fail-closed native intake."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
import platform
import random
import sys
from typing import Any

import torch
from torch import Tensor, nn

from .archive import Archive
from .domain import ACTIONS, Task, State, audit_splits, digest, root, step, success
from .metrics import paired, summarize
from .objectives import path_loss
from .runtime import Ledger, Sample, ScriptedPolicy, replay, rollout


class TinyActor(nn.Module):
    """230-parameter shared CPU actor, NOT a language model or Rineya checkpoint."""
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, len(ACTIONS)))

    def distribution(self, task: Task, state: State) -> Tensor:
        features = torch.tensor([state.x / 8, state.y / 8, task.goal[0] / 8,
                                 task.goal[1] / 8, (task.goal[0] - state.x) / 8,
                                 (task.goal[1] - state.y) / 8,
                                 (task.horizon - state.depth) / task.horizon])
        return self.net(features).log_softmax(-1)

    def sample(self, task: Task, state: State, rng: random.Random,
               max_new_tokens: int) -> Sample:
        with torch.no_grad():
            probabilities = self.distribution(task, state).exp().tolist()
        # Symbolic categorical actions are not reported as language-model tokens.
        return Sample(rng.choices(ACTIONS, weights=probabilities, k=1)[0])

    def score(self, task: Task, state: State, action: str,
              ledger: Ledger | None = None) -> float:
        with torch.no_grad():
            return float(self.distribution(task, state)[ACTIONS.index(action)])

    def paths(self, task: Task, paths: list[tuple[str, ...]], ledger: Ledger
              ) -> tuple[Tensor, Tensor]:
        if task.role not in ("train", "replay_train") or not paths or len(set(paths)) != len(paths):
            raise ValueError("Invalid training candidate set")
        count = sum(map(len, paths))
        ledger.require(trained_actions=count)
        ledger.spend(trained_actions=count)
        values = []
        for actions in paths:
            state, terms = root(task), []
            for action in actions:
                terms.append(self.distribution(task, state)[ACTIONS.index(action)])
                state = step(task, state, action)
            if not terms or not success(task, state):
                raise ValueError("Unverified training path")
            values.append(torch.stack(terms).sum())
        return torch.stack(values), torch.tensor([len(p) for p in paths])


def actor_hash(actor: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


def make_tasks(seed: int, counts: dict[str, int]) -> list[Task]:
    rng, tasks, seen = random.Random(seed), [], set()
    for role, count in counts.items():
        attempts = 0
        while sum(t.role == role for t in tasks) < count:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Cannot build distinct task worlds")
            initial = (rng.randrange(-2, 4), rng.randrange(-2, 4))
            # A construction witness makes the world feasible; it is NEVER retained
            # as an actor sample, a training target, or an archive edge.
            temporary = Task("generator", "generator", "train", initial, (31, 30), 4)
            state = root(temporary)
            for _ in range(rng.choice((2, 3))):
                state = step(temporary, state, rng.choice(ACTIONS[:-1]))
            goal = (state.x, state.y)
            if goal == initial:
                continue
            i = len(tasks)
            task = Task(f"{role}-{i}", f"parent-{i}", role, initial, goal, 4)
            if task.semantic_key in seen:
                continue
            seen.add(task.semantic_key)
            tasks.append(task)
    audit_splits(tasks)
    return tasks


def evaluate(actor: Any, tasks: list[Task], seed: int, attempts: int = 4,
             decoding_contract: dict | None = None) -> tuple[list[dict], dict]:
    if not tasks or attempts < 1:
        raise ValueError("Empty evaluation")
    if decoding_contract is None:
        if not isinstance(actor, TinyActor):
            raise ValueError("Native evaluation needs an explicit decoding contract")
        decoding_contract = {"sampler": "categorical-temperature-1", "tokenizer": "none-symbolic"}
    if not decoding_contract:
        raise ValueError("Empty decoding contract")
    contract_id = digest([decoding_contract, seed, attempts, "register-pair-v1"])
    ledger, rows = Ledger(), []
    actor.eval()
    for task in tasks:
        outcomes, actions = [], []
        for index in range(attempts):
            # Independent streams prevent early stopping in one row shifting others.
            draw_seed = int(digest([seed, task.key, index])[:16], 16)
            episode = rollout(task, actor, random.Random(draw_seed), ledger,
                              f"eval-{task.task_id}-{index}")
            outcomes.append(episode.accepted)
            actions.append(list(episode.actions) + ([episode.invalid_action]
                            if episode.invalid_action is not None else []))
        rows.append({"task_id": task.task_id, "task_key": task.key,
                     "parent_id": task.parent_id, "contract_id": contract_id,
                     "successes": outcomes, "actions": actions})
    return rows, ledger.report()


def bank_diagnostics(archives: list[Archive], minimum_multi_fraction: float = 0.5) -> dict:
    """A discriminability screen, never a native execution or promotion permit."""
    if not archives or not math.isfinite(minimum_multi_fraction) or not 0 <= minimum_multi_fraction <= 1:
        raise ValueError("Invalid bank diagnostic inputs")
    audit_splits([a.task for a in archives])
    counts = [len(a.candidates) for a in archives]
    multiple = sum(n >= 2 for n in counts)
    duplicate_worlds = len(archives) - len({a.task.semantic_key for a in archives})
    identity = [[a.task.key, sorted(a.candidates), sorted(a.root_observed)]
                for a in sorted(archives, key=lambda a: a.task.key)]
    ready = multiple / len(archives) >= minimum_multi_fraction and duplicate_worlds == 0
    return {"source_bank_sha256": digest(identity), "enrolled_problems": len(archives),
            "eligible_problems": sum(n > 0 for n in counts), "verified_paths": sum(counts),
            "multiple_path_problems": multiple, "multiple_path_fraction": multiple / len(archives),
            "duplicate_worlds": duplicate_worlds,
            "status": "OBJECTIVE_DISCRIMINATION_READY" if ready else "INSUFFICIENT_CANDIDATE_DIVERSITY",
            "native_execution_authorized": False}


def train_fixed_bank(actor: Any, archives: list[Archive], arm: str, updates: int,
                     learning_rate: float = 0.01) -> dict:
    """Same engine accepts TinyActor or an already-qualified object-bound CausalActor.

    This runner is an unprotected acquisition arm. Replay/KL losses are separate
    primitives in objectives.py; it MUST NOT be promoted as a retention treatment.
    Selection is recomputed before each update, with dropout disabled throughout.
    """
    if arm not in ("root_uniform", "uniform", "hard_max", "coverage", "soft_025", "soft_1", "paper_token_mean"):
        raise ValueError("Unknown treatment")
    if type(updates) is not int or updates < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid optimization budget")
    ledger = Ledger({"optimizer_steps": updates})
    parameters = [p for p in actor.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No explicitly trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    curves, eligible = [], 0
    actor.eval()  # eval disables dropout; gradients remain enabled.
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        losses, trained = [], []
        for archive in archives:
            candidates = sorted(archive.candidates.values(), key=lambda c: c.key)
            if arm == "root_uniform":
                candidates = [c for c in candidates if c.key in archive.root_observed]
            if not candidates:
                continue
            revision = f"fixed-bank-step-{update}"
            if arm in ("coverage", "paper_token_mean"):
                archive.rescore(revision, lambda t, s, a: actor.score(t, s, a, ledger=ledger), ledger)
                chosen = archive.coverage_selection(revision)
                indices = tuple(candidates.index(c) for c in chosen)
            else:
                indices, chosen = None, candidates
            logps, lengths = actor.paths(archive.task, [c.actions for c in candidates], ledger)
            objective = "uniform" if arm == "root_uniform" else "soft" if arm.startswith("soft_") else arm
            losses.append(path_loss(logps, lengths, objective,
                                    temperature=0.25 if arm == "soft_025" else 1.0,
                                    selected=indices))
            if arm == "hard_max":
                chosen = [candidates[int(logps.detach().argmax())]]
            trained.append((archive, chosen))
        eligible = len(losses)
        if not losses:
            return {"status": "NO_VERIFIED_TRAINING_PATHS", "eligible_tasks": 0,
                    "curve": [], "ledger": ledger.report()}
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        ledger.require(optimizer_steps=1)
        optimizer.step()
        ledger.spend(optimizer_steps=1)
        for archive, selected in trained:
            archive.mark_trained(selected)
        curves.append({"loss": float(loss.detach()), "gradient_norm": float(norm)})
    return {"status": "CPU_TOY_OR_CALLER_QUALIFIED_ACTOR_ONLY", "eligible_tasks": eligible,
            "curve": curves, "ledger": ledger.report(), "protection": "NONE"}


def mechanism_demo() -> dict:
    task = Task("fixture", "fixture-parent", "train", (0, 0), (1, 1), 3)
    archive, ledger, rng = Archive(task), Ledger(), random.Random(7)
    failed = archive.collect(ScriptedPolicy(("inc_x", "INVALID")), rng, ledger, "failed-root")
    archive.rescore("fixture-v1", lambda *_: -math.log(6), ledger)
    before = archive.compose("fixture-v1", ledger)
    prefix = archive.frontier("fixture-v1")
    suffix = archive.collect(ScriptedPolicy(("inc_y",)), rng, ledger, "successful-frontier", prefix)
    archive.rescore("fixture-v1", lambda *_: -math.log(6), ledger)
    archive.compose("fixture-v1", ledger)
    # Root-only invariant: no observed accepting terminal => no certified solution.
    assert not failed.accepted and not before and suffix.accepted
    assert len(archive.candidates) == 1 and not archive.root_observed
    candidate = next(iter(archive.candidates.values()))
    assert candidate.origin == "restart_completion"
    # Numerical counterexample is illustrative, not an observed neural update.
    return {"status": "SCRIPTED_MECHANISM_PASS_NOT_MODEL_EFFICACY",
            "failed_only_candidates": len(before), "root_sampled_successes": 0,
            "certified_solutions": len(archive.candidates), "first_origin": candidate.origin,
            "max_vs_pass1_counterexample": {"before": [0.30, 0.30], "after": [0.50, 0.05]},
            "archive": archive.export(), "ledger": ledger.report()}


def toy_pilot(seeds: tuple[int, ...] = (7, 19, 31), updates: int = 8) -> dict:
    import copy
    torch.set_num_threads(1)
    tasks = make_tasks(2601007, {"train": 8, "dev": 8, "guard": 8})
    train_tasks = [t for t in tasks if t.role == "train"]
    evaluation = [t for t in tasks if t.role != "train"]
    reports = []
    for seed in seeds:
        base = TinyActor(seed)
        collector, archives = Ledger(), []
        for task in train_tasks:
            archive = Archive(task)
            for i in range(8):
                archive.collect(base, random.Random(seed * 1000 + i), collector, f"root-{i}")
            archive.rescore(f"seed-{seed}", base.score, collector)
            for i in range(8):
                prefix = archive.frontier(f"seed-{seed}")
                archive.collect(base, random.Random(seed * 2000 + i), collector,
                                f"restart-{i}", prefix)
                archive.rescore(f"seed-{seed}", base.score, collector)
            archive.compose(f"seed-{seed}", collector, candidate_cap=32)
            archives.append(archive)
        before, base_cost = evaluate(base, evaluation, seed + 500)
        arms = []
        for arm in ("root_uniform", "uniform", "hard_max", "coverage", "soft_025", "soft_1", "paper_token_mean"):
            actor = copy.deepcopy(base)
            training = train_fixed_bank(actor, copy.deepcopy(archives), arm, updates)
            after, cost = evaluate(actor, evaluation, seed + 500)
            dev_ids = {t.task_id for t in evaluation if t.role == "dev"}
            a_dev, b_dev = [r for r in before if r["task_id"] in dev_ids], [r for r in after if r["task_id"] in dev_ids]
            a_guard, b_guard = [r for r in before if r["task_id"] not in dev_ids], [r for r in after if r["task_id"] not in dev_ids]
            arms.append({"arm": arm, "actor_sha256": actor_hash(actor), "training": training,
                         "dev": summarize(b_dev), "paired_dev": paired(a_dev, b_dev),
                         "guard": paired(a_guard, b_guard), "raw_evaluation": after,
                         "evaluation_cost": cost})
        reports.append({"seed": seed, "base_sha256": actor_hash(base),
                        "collector_cost": collector.report(), "baseline_cost": base_cost,
                        "bank_diagnostics": bank_diagnostics(archives),
                        "baseline": before,
                        "bank": [{"task_id": a.task.task_id,
                                  "paths": [{"actions": list(c.actions), "origin": c.origin,
                                             "sampled_root": c.key in a.root_observed}
                                            for c in a.candidates.values()]} for a in archives],
                        "arms": arms})
    return {"status": "EXPOSED_CPU_TOY_PILOT_NOT_RINEYA_EFFICACY",
            "parameter_count": sum(p.numel() for p in TinyActor(0).parameters()),
            "tasks": [asdict(t) for t in tasks], "split_audit": audit_splits(tasks),
            "seeds": list(seeds), "updates": updates, "reports": reports,
            "limitations": ["No pretrained model, tokenizer, GPU or native adapter loaded",
                            "Same fixed bank for objective arms; root_uniform intentionally has less data",
                            "Matched update cap, not matched wall-clock or supervised-action exposure",
                            "Tiny guard is not a qualified legacy-skill instrument",
                            "Exposed disjoint worlds, not sealed compositional-OOD confirmation"]}


def preflight(manifest: dict) -> dict:
    required = ("checkpoint_sha256", "tokenizer_sha256", "verifier_sha256",
                "train_manifest_sha256", "replay_manifest_sha256", "evaluation_manifest_sha256",
                "runtime_receipt_sha256", "parameter_scope_sha256", "code_commit")
    reasons = []
    for key in required:
        value = manifest.get(key)
        length = 40 if key == "code_commit" else 64
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value) or len(set(value)) == 1:
            reasons.append(f"Missing or invalid immutable identity: {key}")
    # Static strings are never accepted as evidence of a real native qualification.
    reasons.append("Native callback parity/backward/resume and resource evidence has not been qualified by this package")
    return {"status": "BLOCKED_NATIVE_UNQUALIFIED", "reasons": reasons,
            "model_loads": 0, "gpu_jobs": 0}


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create-without-overwrite; never replaces an existing receipt.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "toy", "preflight"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; use a new path")
    if args.command == "demo":
        result = mechanism_demo()
    elif args.command == "toy":
        result = toy_pilot(updates=args.updates)
    else:
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        result = preflight(manifest)
    result["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                         "platform": platform.platform(), "device": "cpu"}
    write_new(args.out, result)
    print(json.dumps({"status": result["status"], "output": str(args.out)}))
    return 2 if args.command == "preflight" else 0


if __name__ == "__main__":
    sys.exit(main())
