"""Fresh ledger fixtures and analysis-only oracle, never a deployable expert.

This independent generator does not copy or amend registered RCWM banks. It is
not an exact RCWM reproduction. Labels are used for SFT/evaluation, not routing.
"""
import json
import numpy as np
from .control import digest
from .artifacts import validate_rows

RULE = ("Return only the next state as JSON. Apply the event only when status is exactly 'posted'. "
        "Every other status leaves the state unchanged. add increases the source balance by amount; "
        "scale multiplies the source balance by amount; transfer subtracts amount from source "
        "and adds it to destination. All arithmetic is integer arithmetic.\n")


def transition(state, event):
    if not state or any(type(v) is not int for v in state.values()):
        raise ValueError("state must contain integer balances")
    result = dict(state)
    if event["status"] != "posted":
        return result
    source, amount = event["source"], event["amount"]
    if source not in result or type(amount) is not int:
        raise ValueError("invalid event source/amount")
    if event["op"] == "add":
        result[source] += amount
    elif event["op"] == "scale":
        result[source] *= amount
    elif event["op"] == "transfer":
        destination = event["destination"]
        if destination not in result or destination == source:
            raise ValueError("invalid transfer destination")
        result[source] -= amount
        result[destination] += amount
    else:
        raise ValueError("unknown operation")
    return result


def input_features(state, event):
    """Cheap, goal-blind input features. No expected state or success label."""
    return [float(len(state)), float(max(abs(v) for v in state.values())) / 100000,
            float(event["amount"]) / 500, float(event["status"] == "posted"),
            float(len(event["status"])) / 32,
            *[float(event["op"] == op) for op in ("add", "scale", "transfer")]]


def ledger_rows(split, parents=40, seed=0):
    if split not in {"train", "calibration", "dev", "confirm"} or parents < 2:
        raise ValueError("valid split and at least two parents required")
    split_number = ["train", "calibration", "dev", "confirm"].index(split)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_number]))
    seen_statuses = ["posted", "void", "cancelled", "pending"]
    unseen = ["unposted", "not_posted", "neutral", "never_applied"] if split == "dev" else [
              "not posted", "not_executed", "deferred", "nonce_unseen_status"]
    rows = []
    for parent in range(parents):
        normal_state = dict(zip(["a", "b", "c"], [int(v) for v in rng.integers(0, 100, 3)]))
        op = ["add", "scale", "transfer"][parent % 3]
        normal_amount = int(rng.integers(1, 10))
        variants = ["in_distribution"] if split in {"train", "calibration"} else [
                    "in_distribution", "unseen_status", "magnitude", "renamed", "key_order", "combined"]
        for variant in variants:
            state = dict(normal_state)
            status = seen_statuses[(parent // 3) % len(seen_statuses)] if split in {"train", "calibration"} else "posted"
            amount = normal_amount
            if variant in {"unseen_status", "combined"}:
                status = unseen[parent % len(unseen)]
            if variant in {"magnitude", "combined"}:
                state = {k: int(rng.integers(1000, 100000)) for k in state}
                amount = int(rng.integers(13, 491))
            if variant in {"renamed", "combined"}:
                state = dict(zip([f"account_{split}_{parent}_{i}" for i in range(3)], state.values()))
            keys = list(state)
            event = {"op": op, "status": status, "source": keys[0], "destination": keys[1], "amount": amount}
            if variant in {"key_order", "combined"}:
                state = dict(reversed(list(state.items())))
                event = dict(reversed(list(event.items())))
            expected = transition(state, event)
            prompt = RULE + "State: " + json.dumps(state) + "\nEvent: " + json.dumps(event) + "\nNext state:"
            rows.append({"id": f"ledger-{split}-{parent}-{variant}",
                         "parent_id": f"ledger-{split}-{parent}", "family": "ledger_transition",
                         "split": split, "stratum": variant, "payload": {"prompt": prompt},
                         "world_input": {"state": state, "event": event},
                         "features": input_features(state, event),
                         "feature_definition": "ledger_input_v1_8d",
                         "expected_state": expected, "target": json.dumps(expected, sort_keys=True)})
    validate_rows(rows)
    return rows


def self_rollout(initial_state, events, predict):
    """Predict chains from previous PREDICTED state, never teacher-force oracle states.

    predict(state, event) returns a state mapping. This helper is goal-blind; a
    separate analysis may score the returned final state against held-out goals.
    """
    state, trace = dict(initial_state), []
    for event in events:
        before = dict(state)
        candidate = predict(dict(before), dict(event))
        if (not isinstance(candidate, dict) or set(candidate) != set(initial_state)
                or any(type(v) is not int for v in candidate.values())):
            return {"valid": False, "final_state": None, "trace": trace}
        state = dict(candidate)
        trace.append({"input_state": before, "event": dict(event), "predicted_state": dict(state)})
    return {"valid": True, "final_state": state, "trace": trace}
