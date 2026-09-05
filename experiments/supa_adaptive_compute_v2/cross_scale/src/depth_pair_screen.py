"""Complete calibration-only transformation matrix and fixed selection rule."""
from __future__ import annotations

import itertools
import math
import time

import torch

from shared.hashing import canonical_sha256
from shared.legacy import legacy_graft
from .bridge_fit import fit_initial, fit_delta_bridge
from .delta_alignment import deadline, donor_delta, frozen, stratified_metrics, token_masks

OBJECTIVES = ("B0", "B1", "B2", "B3", "BS", "BR")


def depth_pairs(config):
    if (not config["supa_layers"] or not config["donor_layers"]
            or len(set(config["supa_layers"])) != len(config["supa_layers"])
            or len(set(config["donor_layers"])) != len(config["donor_layers"])):
        raise ValueError("matrix axes must be nonempty unique depths")
    return list(itertools.product(config["supa_layers"], config["donor_layers"]))


def validate_bundle(bundle, config):
    if bundle.get("track") != "cross_scale" or bundle.get("role") != "calibration":
        raise ValueError("invalid activation role or track")
    rows, masks = bundle["rows"], bundle["train_prompt_mask"]
    if (not rows or len(rows) != len(masks) or any(type(m) is not bool for m in masks)
            or not any(masks) or all(masks)):
        raise ValueError("training and scenario-held-out prompts are required")
    train = {r["scenario_id"] for r, keep in zip(rows, masks) if keep}
    holdout = {r["scenario_id"] for r, keep in zip(rows, masks) if not keep}
    if train & holdout or holdout != set(bundle["heldout_scenarios"]):
        raise ValueError("scenario leakage or mismatched holdout")
    if any(set(r) != {"example_id", "scenario_id", "family", "prompt"} for r in rows):
        raise ValueError("activation metadata must contain prompt-only identity fields")
    if (len({r["example_id"] for r in rows}) != len(rows)
            or any(not isinstance(value, str) or not value for r in rows for value in r.values())):
        raise ValueError("duplicate or invalid activation identities")
    if (len(bundle["input_ids"]) != len(rows)
            or any(ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long
                   or not 0 < ids.numel() <= config["calibration_max_tokens"] for ids in bundle["input_ids"])):
        raise ValueError("invalid captured token sequences")
    if (len(set(bundle["special_ids"])) != len(bundle["special_ids"])
            or any(type(value) is not int or value < 0 for value in bundle["special_ids"])):
        raise ValueError("invalid special token IDs")
    if set(bundle["small"]) != set(config["supa_layers"]):
        raise ValueError("Supa depth matrix incomplete")
    required = {k for j in config["donor_layers"] for k in (j-1, j)}
    if set(bundle["donor"]) != required:
        raise ValueError("donor depth matrix incomplete")
    for group in (bundle["small"], bundle["donor"]):
        widths = set()
        for states in group.values():
            if len(states) != len(rows):
                raise ValueError("activation/prompt count mismatch")
            for state, ids in zip(states, bundle["input_ids"]):
                if (state.ndim != 2 or len(state) != ids.numel() or state.shape[1] < 1
                        or state.dtype != torch.float32 or state.requires_grad or state.grad is not None
                        or state.device.type != "cpu" or not torch.isfinite(state).all()):
                    raise ValueError("malformed, trainable or nonfinite activation tensor")
                widths.add(state.shape[1])
        if len(widths) != 1:
            raise ValueError("inconsistent backbone residual widths")


def evaluate_bridge(bridge, layer, rope, prompts, special_ids, config, *, check_budget=None):
    results = {"up": [], "down": [], "delta": [], "native_delta": [], "small": [], "big": []}
    with torch.inference_mode():
        for prompt in prompts:
            if check_budget:
                check_budget()
            started = time.perf_counter()
            legacy_graft().memory_guard(config["max_rss_gib"])
            # Actual execution also holds the programme-wide machine lease.
            small, big = prompt["small"], prompt["big"]
            z = bridge.project(small.unsqueeze(0))
            results["up"].append(z[0])
            results["down"].append(big@bridge.down+bridge.down_bias)
            results["delta"].append(donor_delta(layer, rope, z, config=config)[0])
            results["native_delta"].append(prompt["delta"])
            results["small"].append(small)
            results["big"].append(big)
            deadline(started, config["max_prompt_seconds"])
            legacy_graft().memory_guard(config["max_rss_gib"])
            if check_budget:
                check_budget()
    values = {key: torch.cat(value) for key, value in results.items()}
    masks = token_masks([p["ids"] for p in prompts], special_ids)
    report = {name: stratified_metrics(values[pred], values[target], masks) for name, pred, target in
              (("up", "up", "big"), ("down", "down", "small"), ("delta", "delta", "native_delta"))}
    mapped = values["delta"].double()@bridge.down.double()
    report["mapped_delta_to_supa_rms"] = float(mapped.square().mean().sqrt()/values["small"].double().square().mean().sqrt().clamp_min(1e-12))
    if not torch.isfinite(mapped).all():
        raise FloatingPointError("nonfinite mapped donor delta")
    return report


def run_pair(bundle, config, supa_layer, donor_layer, layer, rope):
    from armr.provenance import state_hash
    validate_bundle(bundle, config)
    if (supa_layer, donor_layer) not in depth_pairs(config):
        raise ValueError("unregistered depth pair")
    started = time.perf_counter()
    frozen(layer)
    frozen(rope)
    if getattr(getattr(layer, "self_attn", None), "layer_idx", None) != donor_layer:
        raise ValueError("loaded donor layer does not match registered depth")
    before = state_hash(layer)
    rope_before = state_hash(rope)
    train, held = [], []
    for n, (row, is_train) in enumerate(zip(bundle["rows"], bundle["train_prompt_mask"])):
        prompt = {"small": bundle["small"][supa_layer][n], "big": bundle["donor"][donor_layer-1][n],
                  "delta": bundle["donor"][donor_layer][n]-bundle["donor"][donor_layer-1][n],
                  "ids": bundle["input_ids"][n], "scenario_id": row["scenario_id"]}
        (train if is_train else held).append(prompt)
    with torch.inference_mode():
        # Verify actual layer identity/positions against independently captured states.
        for prompt in (train[0], held[0]):
            native = donor_delta(layer, rope, prompt["big"].unsqueeze(0), config=config)[0]
            if not torch.equal(native, prompt["delta"]):
                raise ValueError("native donor replay differs from captured transformation")
    small, big = torch.cat([p["small"] for p in train]), torch.cat([p["big"] for p in train])
    scenarios = [p["scenario_id"] for p in train for _ in range(len(p["small"]))]
    outcomes, artifacts = {}, {}
    def pair_budget():
        deadline(started, config["max_pair_seconds"])
    for objective in OBJECTIVES:
        deadline(started, config["max_pair_seconds"])
        bridge = fit_initial(small, big, objective, config, scenarios)
        history = []
        if objective in ("B2", "B3"):
            bridge, history = fit_delta_bridge(bridge, layer, rope, train, config, token_normalized=objective=="B3", check_budget=pair_budget)
        outcomes[objective] = {"heldout": evaluate_bridge(bridge, layer, rope, held, bundle["special_ids"], config, check_budget=pair_budget),
                               "optimizer_steps": len(history), "training_trace": history,
                               "bridge_coefficients": sum(t.numel() for t in bridge.buffers())}
        artifacts[objective] = bridge.state_dict()
        deadline(started, config["max_pair_seconds"])
    after = state_hash(layer)
    if before != after or rope_before != state_hash(rope):
        raise RuntimeError("donor backbone mutation or gradient")
    frozen(layer)
    frozen(rope)
    return {"supa_layer": supa_layer, "donor_layer": donor_layer, "objectives": outcomes,
            "donor_state_before": before, "donor_state_after": after,
            "native_replay_bit_identical": True,
            "trainable_backbone_parameters": 0, "backbone_gradient_tensors": 0,
            "config_sha256": canonical_sha256(config)}, artifacts


def select_matrix(records, config):
    expected = set(depth_pairs(config))
    keys = [(r["supa_layer"], r["donor_layer"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("matrix missing, duplicate, or unregistered depth pair")
    eligible, decisions = [], []
    gates = config["mechanism_gate"]
    for row in records:
        if set(row["objectives"]) != set(OBJECTIVES) or row["config_sha256"] != canonical_sha256(config):
            raise ValueError("incomplete controls or configuration mismatch")
        if (row.get("donor_state_before") != row.get("donor_state_after")
                or not row.get("donor_state_before") or row.get("native_replay_bit_identical") is not True
                or row.get("trainable_backbone_parameters") != 0 or row.get("backbone_gradient_tensors") != 0):
            raise ValueError("matrix lacks frozen, qualified donor evidence")
        for objective, result in row["objectives"].items():
            metrics = result["heldout"]["delta"]
            if not all(metrics[name]["tokens"] for name in ("all_tokens", "ordinary_tokens", "final_prompt_token")):
                raise ValueError("held-out strata incomplete")
            for name in ("up", "down", "delta"):
                strata = result["heldout"][name]
                if set(strata) != {"all_tokens", "ordinary_tokens", "special_tokens", "final_prompt_token"}:
                    raise ValueError("matrix omitted a registered token stratum")
                counts = {key: value["tokens"] for key, value in strata.items()}
                if (counts["ordinary_tokens"]+counts["special_tokens"] != counts["all_tokens"]
                        or not 0 < counts["final_prompt_token"] <= counts["all_tokens"]):
                    raise ValueError("inconsistent held-out token strata")
                baseline_counts = {key: value["tokens"] for key, value in row["objectives"]["B0"]["heldout"]["delta"].items()}
                if counts != baseline_counts:
                    raise ValueError("bridge metrics used different held-out tokens")
                for stratum, values in result["heldout"][name].items():
                    if values["tokens"] < 0 or type(values["tokens"]) is not int:
                        raise ValueError("invalid held-out token count")
                    if not values["tokens"]:
                        if stratum != "special_tokens":
                            raise ValueError("required held-out stratum is empty")
                        continue
                    required = ("mean_token_cosine", "mean_token_relative_mse", "global_relative_mse",
                                "predicted_rms", "target_rms", "residual_magnitude_ratio")
                    if any(not math.isfinite(values[key]) for key in required):
                        raise ValueError("nonfinite matrix metric")
                    if (not -1.000001 <= values["mean_token_cosine"] <= 1.000001
                            or any(values[key] < 0 for key in required[1:])):
                        raise ValueError("invalid matrix metric range")
        base = row["objectives"]["B0"]["heldout"]["delta"]["ordinary_tokens"]
        null_cos = max(row["objectives"][name]["heldout"]["delta"]["ordinary_tokens"]["mean_token_cosine"] for name in ("BS", "BR"))
        for objective in ("B1", "B2", "B3"):
            metrics = row["objectives"][objective]["heldout"]["delta"]
            ordinary = metrics["ordinary_tokens"]
            checks = {
                "absolute_delta_cosine": ordinary["mean_token_cosine"] >= gates["minimum_delta_cosine"],
                "improvement_over_B0": ordinary["mean_token_cosine"] >= base["mean_token_cosine"]+gates["minimum_cosine_improvement"],
                "relative_error_improvement": ordinary["mean_token_relative_mse"] <= base["mean_token_relative_mse"]*gates["maximum_mse_fraction_of_B0"],
                "above_null_controls": ordinary["mean_token_cosine"] >= null_cos+gates["minimum_null_margin"],
                "all_token_floor": metrics["all_tokens"]["mean_token_cosine"] >= gates["minimum_all_token_cosine"],
                "scale_stability": gates["residual_ratio_bounds"][0] <= ordinary["residual_magnitude_ratio"] <= gates["residual_ratio_bounds"][1],
            }
            candidate = {"supa_layer": row["supa_layer"], "donor_layer": row["donor_layer"], "objective": objective,
                         "checks": checks, "go": all(checks.values()),
                         "ordinary_delta_cosine": ordinary["mean_token_cosine"],
                         "ordinary_delta_relative_mse": ordinary["mean_token_relative_mse"]}
            decisions.append(candidate)
            if candidate["go"]:
                eligible.append(candidate)
    eligible.sort(key=lambda r: (-r["ordinary_delta_cosine"], r["ordinary_delta_relative_mse"],
                                r["supa_layer"], r["donor_layer"], r["objective"]))
    return {"status": "MECHANISM_GO" if eligible else "MECHANISM_NO_GO", "selected": eligible[:1],
            "selection_data": "cross_scale/calibration scenario-held-out activations only",
            "comparisons": decisions, "large_generation_licensed": False,
            "small_generation_eligible_for_separate_freeze": bool(eligible)}
