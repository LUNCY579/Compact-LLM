"""Internal E1 accumulation component, with no model import or launch CLI.

Injected-backend controls are synthetic evidence. A native backend, current
source/runtime/reader binding, resource supervisor, native accumulation/resume
qualification and scientific launch validators are still required. This module
does not implement or grant those validators. Historical plans are never edited.
"""
from __future__ import annotations

from copy import deepcopy
import math

from experiments.post_training_mimo_centralized_v1.scripts import e1_data, e1_schedule
from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting


CHECKPOINT_KIND = "E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT"
BACKEND_METHODS = ("forward_loss", "loss_value", "backward", "reset_gradients", "clip_gradients",
                   "optimizer_step", "save_checkpoint", "restore_checkpoint")
BACKEND_SCOPES = ("SYNTHETIC_COMPONENT_CONTROL", "UNQUALIFIED_NATIVE_COMPONENT")
FORBIDDEN_ARRAY_FIELDS = {"packing", "packed", "cu_seqlens", "position_ids", "past_key_values",
                          "cache_position", "inputs_embeds"}
STATE_FIELDS = {"parameters", "optimizer", "training_rng"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid " + label)
    return float(value)


def validate_row(row, planned):
    require(type(row) is dict and not FORBIDDEN_ARRAY_FIELDS.intersection(row), "unregistered sequence metadata")
    observed = e1_data.validate_array(row, planned["component"])
    require(observed == planned, "scheduled row identity or dose differs")
    require(row["nonpadding_tokens"] <= 512, "CE512 envelope exceeded; no truncation allowed")
    return row


def validate_backend_state(state):
    # Structural completeness is not numerical/native state qualification.
    require(type(state) is dict and set(state) == STATE_FIELDS, "parameters, optimizer and training RNG required")
    require(all(type(value) is dict and value for value in state.values()), "empty or invalid backend state component")
    return state


def validate_checkpoint(checkpoint, plan):
    require(type(checkpoint) is dict and set(checkpoint) == {
        "schema_version", "kind", "plan_sha256", "cursor", "cost_snapshot", "backend_state",
        "save_call_id", "native_resume_qualified", "E1_authorized"}, "checkpoint envelope fields")
    require(type(checkpoint["schema_version"]) is int and checkpoint["schema_version"] == 1
            and checkpoint["kind"] == CHECKPOINT_KIND and checkpoint["plan_sha256"] == plan["plan_sha256"],
            "checkpoint plan contract drift")
    require(checkpoint["native_resume_qualified"] is False and checkpoint["E1_authorized"] is False,
            "component checkpoint is not native qualification")
    cursor = checkpoint["cursor"]
    e1_schedule.validate_resume_cursor(cursor, plan)
    update = cursor["completed_optimizer_steps"]
    require(update in plan["checkpoint_steps"], "resume requires a registered saved boundary")
    validate_backend_state(checkpoint["backend_state"])
    costs = checkpoint["cost_snapshot"]
    E1CallAccounting.from_snapshot(lambda *args, **kwargs: None, costs)
    require(type(checkpoint["save_call_id"]) is int and checkpoint["save_call_id"] == costs["last_call_id"],
            "checkpoint must bind its returned save call")
    expected_saves = sum(step <= update for step in plan["checkpoint_steps"])
    require(costs["operations"]["checkpoint_save"]["returned"] == expected_saves
            and costs["operations"]["checkpoint_save"]["failed"] == 0, "saved-boundary checkpoint count mismatch")
    for operation, count in (("forward", 20 * update), ("backward", 20 * update),
                             ("optimizer_step", update), ("gradient_reset", update), ("gradient_clip", update)):
        observed = costs["operations"][operation]
        require(observed == {"attempted": count, "returned": count, "failed": 0, "pending": 0},
                "checkpoint only supports a complete controlled boundary:" + operation)
    positions = costs["positions"]
    require(positions == {
        "input_positions_attempted": cursor["consumed_nonpadding_tokens"],
        "input_positions_returned": cursor["consumed_nonpadding_tokens"],
        "supervised_positions_attempted": cursor["consumed_completion_tokens"],
        "supervised_positions_returned": cursor["consumed_completion_tokens"]}, "checkpoint sequence-dose cursor mismatch")
    return update


def run(plan, row_get, backend, accounting, *, resume=None, latest_terminal_costs=None, stop_after_update=None):
    """Execute the fixed accumulation contract through an injected backend.

    Backend forward_loss returns one sequence's mean supervised-token loss.
    backward(loss, scale) must differentiate that loss times scale. clip_gradients
    must check finite confined gradients before clipping once at norm1. save
    returns a detached state mapping, containing adapters/optimizer/training RNG;
    the engine freezes it and binds the returned save call. A native implementation
    must additionally persist/recheck adapter bytes, optimizer schema, RNG devices,
    source/base hashes and resource guards; none is proven by this component.

    Resume currently supports a controlled stop exactly after a saved boundary,
    with the latest terminal cost snapshot identical to that boundary. Arbitrary
    rollback after later work is refused rather than erasing its costs. The one
    reload ceiling is retained; this is not cross-process native resume evidence.
    """
    e1_schedule.validate_plan(plan)
    require(callable(row_get) and type(accounting) is E1CallAccounting, "bound reader and E1 journal required")
    require(all(callable(getattr(backend, name, None)) for name in BACKEND_METHODS), "backend methods missing")
    scope = getattr(backend, "execution_scope", None)
    require(scope in BACKEND_SCOPES, "backend evidence scope required")
    target = 256 if stop_after_update is None else stop_after_update
    require(type(target) is int and target in plan["checkpoint_steps"], "stop only at a registered saved boundary")
    completed = 0
    if resume is None:
        require(latest_terminal_costs is None, "unexpected continuation costs")
        snapshot = accounting.snapshot()
        require(all(snapshot["operations"][name]["attempted"] == 0 for name in (
            "forward", "backward", "optimizer_step", "checkpoint_save", "checkpoint_reload", "gradient_reset", "gradient_clip")),
            "new engine run cannot reset consumed training budgets")
    else:
        completed = validate_checkpoint(resume, plan)
        require(type(latest_terminal_costs) is dict and latest_terminal_costs == resume["cost_snapshot"],
                "later or missing terminal costs: uncontrolled rollback unsupported")
        require(accounting.snapshot() == latest_terminal_costs, "restore cumulative accounting before resume")
    require(completed <= target, "target precedes restored boundary")
    initial_costs = accounting.snapshot()
    rows = {row["key"]: row for row in plan["ordered_rows"]}
    result = {"status": "RUNNING_ENGINE_COMPONENT", "backend_scope": scope, "plan_sha256": plan["plan_sha256"],
              "start_optimizer_step": completed + 1, "completed_optimizer_steps": completed,
              "completed_sequence_rows_this_invocation": 0, "sequence_losses_this_step": [],
              "update_summaries": [], "checkpoint_envelopes": [], "active_optimizer_step": None,
              "native_launch_authorized": False, "native_resume_qualified": False,
              "E1_authorized": False, "CHECK_FINAL_access": False}

    def save_boundary(update):
        cursor = e1_schedule.make_resume_cursor(plan, update)
        pre_save = accounting.snapshot()
        state = accounting.call("checkpoint_save", lambda: backend.save_checkpoint(deepcopy(cursor), deepcopy(pre_save)),
                                stage=f"checkpoint_{update}")
        validate_backend_state(state)
        envelope = {"schema_version": 1, "kind": CHECKPOINT_KIND, "plan_sha256": plan["plan_sha256"],
                    "cursor": cursor, "cost_snapshot": accounting.snapshot(), "backend_state": deepcopy(state),
                    "save_call_id": accounting.snapshot()["last_call_id"],
                    "native_resume_qualified": False, "E1_authorized": False}
        validate_checkpoint(envelope, plan)
        result["checkpoint_envelopes"].append(envelope)

    try:
        if resume is not None:
            accounting.call("checkpoint_reload", lambda: backend.restore_checkpoint(deepcopy(resume["backend_state"])),
                            stage=f"restore_{completed}")
        else:
            save_boundary(0)
        for step in plan["steps"][completed:target]:
            update = step["optimizer_step"]
            result["active_optimizer_step"] = update
            result["sequence_losses_this_step"] = []
            accounting.call("gradient_reset", backend.reset_gradients, stage=f"update_{update}_reset")
            for slot, key in enumerate(step["row_keys"], 1):
                row = validate_row(row_get(key), rows[key])
                stage = f"update_{update}_sequence_{slot}"
                loss = accounting.call("forward", lambda: backend.forward_loss(row), stage=stage,
                    input_positions=row["nonpadding_tokens"], supervised_positions=row["completion_tokens"])
                value = finite_nonnegative(backend.loss_value(loss), "sequence mean completion loss")
                result["sequence_losses_this_step"].append(value)
                accounting.call("backward", lambda: backend.backward(loss, scale=1 / 20), stage=stage)
                result["completed_sequence_rows_this_invocation"] += 1
                del loss, row
            norm = accounting.call("gradient_clip", lambda: backend.clip_gradients(max_norm=1.0), stage=f"update_{update}_clip")
            norm = finite_nonnegative(norm, "gradient norm")
            accounting.call("optimizer_step", lambda: backend.optimizer_step(lr=step["learning_rate"]), stage=f"update_{update}")
            completed = update
            result["completed_optimizer_steps"] = update
            result["update_summaries"].append({"optimizer_step": update, "learning_rate": step["learning_rate"],
                "mean_sequence_loss": sum(result["sequence_losses_this_step"]) / 20,
                "gradient_norm_before_clip": norm, "sequence_rows": 20})
            result["active_optimizer_step"] = None
            if step["checkpoint_after"]:
                save_boundary(update)
        require(completed == target, "engine ended before requested boundary")
        result["status"] = "COMPLETE_ENGINE_COMPONENT_UNQUALIFIED" if target == 256 else "STOPPED_AT_SAVED_COMPONENT_BOUNDARY"
    except Exception as error:
        result["status"] = "FAILED_ENGINE_COMPONENT_UNQUALIFIED"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)[:1024]
        result["automatic_resume_authorized"] = False
    finally:
        result["call_accounting"] = accounting.snapshot()
        result["returned_sequence_backwards_this_invocation"] = (
            result["call_accounting"]["operations"]["backward"]["returned"] -
            initial_costs["operations"]["backward"]["returned"])
        result["returned_optimizer_calls_this_invocation"] = (
            result["call_accounting"]["operations"]["optimizer_step"]["returned"] -
            initial_costs["operations"]["optimizer_step"]["returned"])
        result["completed_step_semantics"] = "Engine-acknowledged boundaries; actual returned calls remain authoritative if journal publication fails."
    return result
