"""Complete-matrix DEV selection across seeds; confirmation is single-use."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .contracts import Plan, digest, file_hash, matrix, read_json, require, sealed, unseal, write_new
from .metrics import decision_features


def select(plan: Plan, paths: list[Path], output: Path, *, stage="full", identity: str):
    expected = {a.name: a for a in matrix(plan, stage)}
    observed, initial, frozen, kinds = {}, {}, set(), set()
    for path in paths:
        path = Path(path).resolve()
        r = unseal(read_json(path))
        require(r.get("status") == "COMPLETE" and r["plan"] == plan.fingerprint and r["identity"] == identity,
                "incomplete or differently registered run")
        name = r["arm"]["name"]
        require(name in expected and name not in observed, "unexpected/duplicate arm")
        a = expected[name]
        require(r["arm"] == vars(a), "arm metadata drift")
        require(a.seed not in initial or initial[a.seed] == r["initial_adapter"], "unpaired initial adapters")
        initial[a.seed] = r["initial_adapter"]
        frozen.add(digest(r["frozen"]))
        kinds.add(r["evidence_kind"])
        observed[name] = (r, path)
    require(set(observed) == set(expected), "complete registered stage required; missing arms/seeds cannot be dropped")
    require(len(frozen) == 1 and len(kinds) == 1, "mixed teachers/bases/evidence kinds")
    expected_steps = set(range(plan.checkpoint_every, plan.steps+1, plan.checkpoint_every)) | {plan.steps}
    candidates = defaultdict(dict)
    for name, (r, path) in observed.items():
        require({c["step"] for c in r["checkpoints"]} == expected_steps
                and len(r["checkpoints"]) == len(expected_steps), "checkpoint coverage mismatch")
        for c in r["checkpoints"]:
            # Verify persisted model files, not just copied score metadata.
            checkpoint = path.parent / "checkpoints" / f'{c["step"]:06d}'
            receipt = unseal(read_json(checkpoint / "receipt.json"))
            require(receipt == c["receipt"] and receipt["adapter_sha256"] == file_hash(checkpoint / "adapter.safetensors"),
                    "selected checkpoint artifact drift")
            f = decision_features(c["summary"])
            group = (name.rsplit("-s", 1)[0], c["step"])
            candidates[group][r["arm"]["seed"]] = {"features": f, "checkpoint": str(checkpoint),
                "result_path": str(path), "result_sha256": file_hash(path), "receipt": receipt,
                "arm": r["arm"], "meter": c["meter"]}
    eligible = []
    for key, seeds in candidates.items():
        require(set(seeds) == set(plan.seeds), "seed-complete configuration required")
        if all(v["features"]["acquisition_gain"] >= plan.min_acquisition_gain
               and v["features"]["worst_retention_drop"] <= plan.max_retention_drop for v in seeds.values()):
            n = len(seeds)
            order = (sum(v["features"]["damaged"] for v in seeds.values())/n,
                     -min(v["features"]["acquisition_gain"] for v in seeds.values()),
                     -sum(v["features"]["hard_accuracy"] for v in seeds.values())/n,
                     key[1], key[0])
            eligible.append((order, key, seeds))
    chosen = min(eligible, default=None, key=lambda row: row[0])
    result = {"schema": 1, "identity": identity, "plan": plan.fingerprint, "stage": stage,
              "evidence_kind": next(iter(kinds)), "status": "DEVELOPMENT_SELECTED" if chosen else "NO_ELIGIBLE_CONFIGURATION",
              "selection_rule": "all-seed gates; mean damage, worst-seed acquisition, mean hard accuracy, earlier step, name",
              "selected_configuration": list(chosen[1]) if chosen else None,
              "selected_seeds": {str(k): v for k, v in chosen[2].items()} if chosen else {},
              "candidates_examined": len(candidates), "eligible_configurations": len(eligible),
              "promotion_authorized": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "SELECTION.json", sealed(result))
    return result


def spend_confirmation(selection_path, identity: str, confirmation_sha256: str):
    path = Path(selection_path)
    selection = unseal(read_json(path))
    require(selection.get("status") == "DEVELOPMENT_SELECTED" and selection["identity"] == identity,
            "locked eligible selection required")
    marker = {"selection_sha256": file_hash(path), "identity": identity,
              "confirmation_sha256": confirmation_sha256, "interruption_remains_spent": True}
    # Create BEFORE opening/evaluating the confirmation bank. Never auto-delete.
    write_new(path.parent / "CONFIRMATION_STARTED.json", sealed(marker))
    return selection
