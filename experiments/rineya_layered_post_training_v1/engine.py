"""An explicit stage graph: no silent teacher/critic/soup substitutions."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import digest, require
from .data import balanced_order, fingerprint, partition, score, validate
from .evaluation import choose, evaluate, paired_confirmation, value_metrics
from .soup import state_hash, weighted_soup
from .storage import load_snapshot, resource_report, save_snapshot, write_json


def _seed(config, *parts):
    return int(digest([config.seed, *parts])[:12], 16)


def critic_audit(runtime, rows):
    values, rewards = [], []
    with torch.no_grad():
        for row in rows:
            for ep in runtime.collect(row.prompt, runtime.config.group_size, _seed(runtime.config, "critic_audit", row.id)):
                if ep.terminated:
                    _, h = runtime.logps(ep.prompt, ep.response, features=True)
                    # Pre-EOS state has seen the candidate answer, NOT the EOS action.
                    values.append(float(runtime.critic(h[-1:].float()).squeeze()))
                    rewards.append(score(row, ep.text))
    return value_metrics(values, rewards)


def run(runtime, rows, output: Path):
    cfg = runtime.config
    validate(rows)
    train, selection, confirmation = (partition(rows, s) for s in ("train", "select", "confirm"))
    require(train and selection and confirmation, "three disjoint splits required")
    require(set(cfg.required_suites) <= {r.suite for r in train}, "training data lacks protected suites")
    require(set(cfg.required_suites) <= {r.suite for r in confirmation}, "confirmation suite registry incomplete")
    acquire = [r for r in train if r.suite == "acquisition"]
    replay = [r for r in train if r.suite != "acquisition"]
    preferences = [r for r in train if r.rejected is not None]
    acquire, replay, preferences = (balanced_order(pool, cfg.seed) for pool in (acquire, replay, preferences))
    require(acquire and replay and preferences, "acquisition, retention replay and preference data required")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "data_manifest.json", {s: {"hash": fingerprint(partition(rows, s)), "rows": len(partition(rows, s))} for s in ("train", "select", "confirm")})
    events, reports, checkpoints = [], {}, {}
    saved, last_tail, successes = {}, [], {}
    reference_ready, critic_ready = False, False

    def snapshot(name, parents=()):
        state = runtime.state()
        meta = save_snapshot(output / "checkpoints" / name, runtime, state, parents)
        checkpoints[name] = meta["state_hash"]
        saved[name] = output / "checkpoints" / name
        return state

    def assess(name):
        runtime.budget.phase = "selection_evaluation"
        reports[name] = evaluate(runtime, selection, "select")
        write_json(output / "evaluations" / f"{name}.json", reports[name])
        return reports[name]

    try:
        runtime.budget.phase = "qualification"
        probe = acquire[0]
        qualification_episodes = runtime.collect(probe.prompt, 2, _seed(cfg, "qualification"))
        with torch.no_grad():
            require(bool(torch.isfinite(runtime.ce(probe))), "nonfinite supervised qualification")
        write_json(output / "qualification.json", {"status": "NATIVE_LOAD_AND_LIKELIHOOD_ONLY" if runtime.native else "SYNTHETIC_LIKELIHOOD_PASS", "parity": dict(runtime.parity), "all_probe_responses_terminated": all(e.terminated for e in qualification_episodes), "tolerance": cfg.logprob_tolerance, "identity_hash": runtime.identity_hash})
        snapshot("base")
        baseline = assess("base")
        choose({"base": baseline}, baseline, cfg)  # Validate mandatory sample counts before any training.
        current, latest_rl = "base", None
        for stage, steps in cfg.schedule():
            if runtime.native:
                require(not resource_report(cfg, output, worker=True)["blockers"], "runtime resource guard failed")
            runtime.budget.phase = stage
            runtime.budget.charge()
            parent = current
            if "soup" in stage:
                candidates = {}
                states = {}
                recipes = {}
                if stage in {"distill_soup", "tail_soup"}:
                    require(len(last_tail) >= 2, "checkpoint tail unavailable")
                    n = min(cfg.tail_checkpoints, len(last_tail))
                    keys = last_tail[-n:]
                    tensors = [load_snapshot(saved[k], runtime) for k in keys]
                    states["mean"] = weighted_soup(tensors, [1 / n] * n, [runtime.identity_hash] * n)
                    states["endpoint"] = load_snapshot(saved[parent], runtime)
                    ancestry = keys
                    recipes["mean"] = {"parents": keys, "weights": [1/n] * n}
                    recipes["endpoint"] = {"parents": [parent], "weights": [1.]}
                else:
                    anchor = latest_rl if stage == "soup1" else ("soup1" if "soup1" in saved else latest_rl)
                    require(anchor in saved, "missing RL/consolidation anchor")
                    left, right = load_snapshot(saved[anchor], runtime), load_snapshot(saved[parent], runtime)
                    for alpha in cfg.soup_alphas:
                        states[f"alpha_{alpha:g}"] = weighted_soup([left, right], [1-alpha, alpha], [runtime.identity_hash] * 2)
                    if stage == "soup2":
                        base = load_snapshot(saved["base"], runtime)
                        states["three_parent"] = weighted_soup([base, left, right], [.1, .65, .25], [runtime.identity_hash] * 3)
                    ancestry = [anchor, parent]
                    for alpha in cfg.soup_alphas:
                        recipes[f"alpha_{alpha:g}"] = {"parents": [anchor, parent], "weights": [1-alpha, alpha]}
                    if "three_parent" in states:
                        recipes["three_parent"] = {"parents": ["base", anchor, parent], "weights": [.1, .65, .25]}
                for candidate, state in states.items():
                    runtime.restore(state)
                    name = f"{stage}.{candidate}"
                    snapshot(name, [checkpoints[k] for k in recipes[candidate]["parents"]])
                    candidates[name] = assess(name)
                # Explicit baseline fallback; never silently declare a damaging soup promoted.
                candidates["base"] = baseline
                winner = choose(candidates, baseline, cfg)
                runtime.restore(load_snapshot(saved[winner], runtime))
                snapshot(stage, [checkpoints[winner]])
                reports[stage] = reports[winner]
                write_json(output / "evaluations" / f"{stage}.json", reports[stage])
                events.append({"stage": stage, "chosen": winner, "parents": ancestry, "candidate_count": len(candidates), "recipes": recipes, "optimizer_reset": True, "baseline_fallback": winner == "base"})
                current = stage
                continue
            if stage == "critic_fit":
                for i in range(steps):
                    row = acquire[i % len(acquire)]
                    eps = runtime.collect(row.prompt, cfg.group_size, _seed(cfg, stage, i))
                    complete = [e for e in eps if e.terminated]
                    require(complete, "no complete rollouts for critic fitting")
                    runtime.critic_fit(complete, [score(row, e.text) for e in complete])
                runtime.budget.phase = "critic_selection_audit"
                audit = critic_audit(runtime, [r for r in selection if r.suite == "acquisition"])
                critic_ready = audit["auroc"] is not None and audit["auroc"] >= cfg.critic_min_auc and audit["mse"] <= audit["constant_mse"]
                save_file({k: v.detach().cpu() for k, v in runtime.critic.state_dict().items()}, str(output / "critic.safetensors"))
                events.append({"stage": stage, "audit": audit, "qualified_for_native_rl": critic_ready, "fixture_gate_bypass": not runtime.native})
                require(critic_ready or not runtime.native, "critic failed held-out ranking/MSE gate; no critic RL executed")
                continue
            runtime.reset_optimizer()
            if stage == "dpo":
                runtime.anchor_reference()
            if stage.startswith("rl") and not reference_ready:
                runtime.anchor_reference()
                reference_ready = True
            last_tail = []
            if stage.startswith("sft"):
                require(bool(successes), "no verifier-approved RL traces; consolidation cannot be substituted by teacher SFT")
                consolidation = list(successes.values())
            losses, terminated, sampled, mixed = [], 0, 0, 0
            for i in range(steps):
                runtime.budget.phase = stage
                old_row = replay[i % len(replay)]
                if stage == "distill":
                    loss = runtime.sft(acquire[i % len(acquire)], old_row)
                elif stage == "dpo":
                    loss = runtime.dpo(preferences[i % len(preferences)], old_row)
                elif stage.startswith("sft"):
                    loss = runtime.sft(consolidation[i % len(consolidation)], old_row)
                else:
                    row = acquire[i % len(acquire)]
                    eps = runtime.collect(row.prompt, cfg.group_size, _seed(cfg, stage, i))
                    sampled += len(eps)
                    complete = [e for e in eps if e.terminated]
                    terminated += len(complete)
                    # All-zero/all-one groups remain in GRPO; they have zero reward advantage.
                    require(len(complete) == len(eps), "truncated rollout group rejected; increase qualified response cap before rerunning")
                    rewards = [score(row, e.text) for e in complete]
                    mixed += int(min(rewards) != max(rewards))
                    for e, reward in zip(complete, rewards):
                        if reward == 1. and len(successes) < 1000:
                            successes[digest([row.id, e.text])] = replace(row, target=e.text)
                    use_critic = stage == "rl3" and cfg.arm not in {"full_no_critic"}
                    require(not use_critic or critic_ready or not runtime.native, "unqualified critic")
                    loss = runtime.rl(complete, rewards, old_row, use_critic=use_critic)
                losses.append(loss)
                name = f"{stage}.step_{i+1:04d}"
                snapshot(name, [checkpoints[parent]] if not last_tail else [checkpoints[last_tail[-1]]])
                last_tail.append(name)
            snapshot(stage, [checkpoints[last_tail[-1]]])
            assess(stage)
            events.append({"stage": stage, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "reference_hash": runtime.reference_hash, "sampled": sampled, "terminated": terminated, "mixed_groups": mixed, "verified_consolidation_traces": len(successes)})
            current = stage
            if stage.startswith("rl"):
                latest_rl = stage
        selected = choose(reports, baseline, cfg)
        lock = {"schema": 1, "checkpoint": selected, "state_hash": checkpoints[selected], "identity_hash": runtime.identity_hash, "config_hash": cfg.fingerprint, "select_data_hash": fingerprint(selection), "confirm_data_hash": fingerprint(confirmation), "baseline_hash": checkpoints["base"]}
        lock["lock_hash"] = digest(lock)
        write_json(output / "selection.lock.json", lock)
        report = {"experiment": "RLP-001", "status": "NATIVE_RUN_COMPLETE_NOT_EFFICACY_CERTIFIED" if runtime.native else "SYNTHETIC_TORCH_MECHANICS_PASS", "arm": cfg.arm, "seed": cfg.seed, "selected_checkpoint": selected, "terminal_checkpoint": current, "selected_scores": reports[selected]["scores"], "baseline_scores": baseline["scores"], "terminal_scores": reports[current]["scores"], "selection_candidates": len(reports), "confirmation_used": False, "events": events, "budget": runtime.budget.report(), "torch": torch.__version__, "parity": runtime.parity, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if runtime.native else 0}
        write_json(output / "run_report.json", report)
        return report
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_OR_BLOCKED_NO_PROMOTION", "error_type": type(error).__name__, "error": str(error), "events": events, "budget": runtime.budget.report()})
        raise


def confirm(runtime, rows, output):
    """One confirmation artifact per immutable lock; selection is never rerun here."""
    require(not (output / "confirmation.json").exists(), "confirmation already opened for this run")
    lock = json.loads((output / "selection.lock.json").read_text())
    checksum = lock.pop("lock_hash")
    require(digest(lock) == checksum and lock["config_hash"] == runtime.config.fingerprint and lock["identity_hash"] == runtime.identity_hash, "selection lock tampered/mismatched")
    require("/" not in lock["checkpoint"] and "\\" not in lock["checkpoint"] and lock["checkpoint"] not in {".", ".."}, "invalid checkpoint path")
    evaluation = partition(validate(rows), "confirm")
    require(fingerprint(evaluation) == lock["confirm_data_hash"], "confirmation data changed after selection")
    runtime.restore(load_snapshot(output / "checkpoints" / "base", runtime))
    require(state_hash(runtime.state()) == lock["baseline_hash"], "baseline lock mismatch")
    runtime.budget.phase = "confirmation_baseline"
    baseline = evaluate(runtime, evaluation, "confirm")
    runtime.restore(load_snapshot(output / "checkpoints" / lock["checkpoint"], runtime))
    require(state_hash(runtime.state()) == lock["state_hash"], "selected checkpoint lock mismatch")
    runtime.budget.phase = "confirmation_candidate"
    candidate = evaluate(runtime, evaluation, "confirm")
    report = {"status": "NATIVE_SINGLE_SEED_CONFIRMATION" if runtime.native else "SYNTHETIC_CONFIRMATION_ONLY", "selection_lock_hash": checksum, "candidate": candidate, "baseline": baseline, "analysis": paired_confirmation(candidate, baseline, runtime.config), "budget": runtime.budget.report()}
    write_json(output / "confirmation.json", report)
    return report
