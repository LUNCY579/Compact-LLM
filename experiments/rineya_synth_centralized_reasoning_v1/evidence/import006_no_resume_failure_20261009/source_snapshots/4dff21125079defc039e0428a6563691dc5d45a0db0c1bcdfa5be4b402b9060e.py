"""Injected CPU component engine; no model import, native backend or launch CLI.

Concepts adapted from post_training_mimo_centralized_v1/scripts/
e1_training_engine_v2.py: complete accumulation boundaries, posted-save cost
binding, publication before promotion, and refusal of cost-erasing rollback.
The central implementation is neither imported nor modified. Its byte hash is
bound by recipe(). Component controls never authorize native study training.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import sys

import study_schedule as schedule

SCOPE = "CPU_COMPONENT_CONTROL"
KIND = "SYNTH_STUDY_CPU_COMPONENT_BOUNDARY_V1"
OPERATIONS = {"gradient_reset": 27, "forward": 540, "completion_loss": 540, "backward": 540,
              "finite_gradient_clip": 27, "optimizer_step": 27, "checkpoint_save": 3,
              "checkpoint_publish": 3, "checkpoint_promote": 3, "checkpoint_reload": 2}
BACKEND_METHODS = ("reset_gradients", "forward", "completion_loss", "backward", "finite_clip_gradients",
                   "optimizer_step", "save_checkpoint", "publish_checkpoint", "restore_checkpoint",
                   "validate_state", "verify_publication")


class JsonlSink:
    """Exclusive CPU component journal with flush+fsync on every event."""
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")

    def __call__(self, event):
        self.stream.write(schedule.canonical(event) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def require(value, message):
    schedule.require(value, message)


def framework_free():
    require("torch" not in sys.modules and "transformers" not in sys.modules, "CPU component forbids model framework")


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative " + name)
    return float(value)


def recipe(registration, arm):
    require(arm in ("M1", "M2"), "M0 is unchanged and cannot enter adaptation engine")
    origin = schedule.ROOT / "experiments" / "post_training_mimo_centralized_v1" / "scripts" / "e1_training_engine_v2.py"
    body = {"registration_sha256": registration["sha256"], "arm": arm, "seed": 101,
            "cohort_sha256": registration["arms"][arm]["occurrence_order_sha256"],
            "source_pins": registration["source_and_decision_pins"], "engine_source_sha256": schedule.filehash(__file__),
            "adapted_concept_source": str(origin.relative_to(schedule.ROOT)).replace("\\", "/"),
            "adapted_concept_source_sha256": schedule.filehash(origin), "execution_scope": SCOPE}
    body["sha256"] = schedule.digest(body)
    return body


class CallLedger:
    """Attempts survive failed sinks; actual returns are not rewritten as failures."""
    def __init__(self, emit, snapshot=None):
        require(callable(emit), "durable event sink required")
        self.emit = emit
        self.last_call_id = 0
        self.operations = {name: {key: 0 for key in ("attempted", "returned", "failed", "pending")} for name in OPERATIONS}
        self.positions = {key: 0 for key in ("input_attempted", "input_returned", "supervised_attempted", "supervised_returned")}
        self.pending = {}
        if snapshot is not None:
            validate_costs(snapshot)
            self.last_call_id = snapshot["last_call_id"]
            self.operations, self.positions, self.pending = deepcopy(snapshot["operations"]), deepcopy(snapshot["positions"]), deepcopy(snapshot["pending"])

    def snapshot(self):
        return deepcopy({"schema_version": 1, "scope": SCOPE, "last_call_id": self.last_call_id,
                         "operations": self.operations, "positions": self.positions, "pending": self.pending})

    def call(self, operation, function, *, stage, row=None):
        framework_free()
        require(operation in OPERATIONS and self.operations[operation]["attempted"] < OPERATIONS[operation], "operation budget " + operation)
        self.last_call_id += 1
        identity = self.last_call_id
        meta = {"call_id": identity, "operation": operation, "stage": stage}
        if row is not None:
            require(operation == "forward", "dose counted only once at forward")
            meta.update({"row_id": row["row_id"], "input_positions": len(row["input_ids"]), "supervised_positions": len(row["completion_ids"])})
            self.positions["input_attempted"] += meta["input_positions"]
            self.positions["supervised_attempted"] += meta["supervised_positions"]
        counts = self.operations[operation]
        counts["attempted"] += 1
        counts["pending"] += 1
        self.pending[str(identity)] = meta
        self.emit({"event": "attempt", **meta})
        try:
            value = function()
        except BaseException as error:
            counts["pending"] -= 1
            counts["failed"] += 1
            self.pending.pop(str(identity))
            self.emit({"event": "failed", **meta, "error_type": type(error).__name__, "error": str(error)})
            raise
        counts["pending"] -= 1
        counts["returned"] += 1
        self.pending.pop(str(identity))
        if row is not None:
            self.positions["input_returned"] += meta["input_positions"]
            self.positions["supervised_returned"] += meta["supervised_positions"]
        self.emit({"event": "returned", **meta})
        framework_free()
        return value


def validate_costs(value):
    require(type(value) is dict and value.get("scope") == SCOPE and value.get("schema_version") == 1, "cost scope/schema")
    require(set(value["operations"]) == set(OPERATIONS), "cost operation fields")
    for name, ceiling in OPERATIONS.items():
        counts = value["operations"][name]
        require(set(counts) == {"attempted", "returned", "failed", "pending"}
                and all(type(v) is int and v >= 0 for v in counts.values()), "typed costs")
        require(counts["attempted"] == counts["returned"] + counts["failed"] + counts["pending"]
                and counts["attempted"] <= ceiling, "cost conservation/budget")
    require(value["last_call_id"] == sum(v["attempted"] for v in value["operations"].values()), "cost call cursor")
    require(len(value["pending"]) == sum(v["pending"] for v in value["operations"].values()), "pending cost conservation")
    require(set(value["positions"]) == {"input_attempted", "input_returned", "supervised_attempted", "supervised_returned"}
            and all(type(v) is int and v >= 0 for v in value["positions"].values()), "position cost fields")
    require(value["positions"]["input_returned"] <= value["positions"]["input_attempted"] <= 131072
            and value["positions"]["supervised_returned"] <= value["positions"]["supervised_attempted"], "position budget/conservation")


def validate_state(state, completed, backend):
    require(type(state) is dict and set(state) == {"parameters", "optimizer", "rng", "gradients"}, "complete parameter/moment/RNG/gradient state")
    require(all(type(state[name]) is dict and state[name] for name in state), "empty state component")
    require(state["gradients"].get("empty") is True, "partial accumulation is not a checkpoint boundary")
    require(state["optimizer"].get("step") == completed and type(state["optimizer"].get("step")) is int, "optimizer/global cursor mismatch")
    require(backend.validate_state(deepcopy(state), completed) is True, "backend state completeness/finite/schema")
    framework_free()


def cursor(registration, arm, completed):
    ids = [row for update in registration["arms"][arm]["updates"][:completed] for row in update["occurrence_row_ids"]]
    updates = registration["arms"][arm]["updates"][:completed]
    return {"completed_updates": completed, "consumed_occurrences": completed * 20,
            "input_tokens": sum(v["input_tokens"] for v in updates), "supervised_tokens": sum(v["supervised_tokens"] for v in updates),
            "consumed_order_sha256": schedule.digest(ids), "next_update_ordinal": completed + 1,
            "gradient_accumulation_sequences": 0}


def validate_checkpoint(checkpoint, registration, arm, backend, current_recipe):
    require(type(checkpoint) is dict and checkpoint.get("kind") == KIND, "study component checkpoint kind; qualification STATE1 forbidden")
    envelope = checkpoint["envelope"]
    require(envelope["sha256"] == schedule.digest({k: v for k, v in envelope.items() if k != "sha256"}), "checkpoint envelope checksum")
    require(envelope["recipe"] == current_recipe and envelope["arm"] == arm and envelope["seed"] == 101, "checkpoint source/recipe/cohort/arm/seed binding")
    require(envelope["native_resume_qualified"] is False and envelope["training_authorized"] is False, "component cannot grant native permission")
    completed = envelope["cursor"]["completed_updates"]
    require(type(completed) is int and completed in (9, 18, 27), "registered full-group chunk boundary")
    require(envelope["cursor"] == cursor(registration, arm, completed), "checkpoint global occurrence/token/LR cursor")
    validate_state(envelope["backend_state"], completed, backend)
    costs = envelope["posted_save_costs"]
    validate_costs(costs)
    require(envelope["save_call_id"] == costs["last_call_id"], "posted-save call binding")
    require(not costs["pending"] and all(v["failed"] == 0 and v["pending"] == 0 for v in costs["operations"].values()), "failed/pending costs are not a resumable checkpoint")
    previous_publications = completed // 9 - 1
    require(costs["operations"]["checkpoint_publish"] == {"attempted": previous_publications, "returned": previous_publications,
                                                         "failed": 0, "pending": 0}, "preceding publication boundary count")
    require(costs["operations"]["checkpoint_promote"] == {"attempted": previous_publications, "returned": previous_publications,
                                                         "failed": 0, "pending": 0}, "preceding promotion boundary count")
    for operation, count in (("forward", completed * 20), ("completion_loss", completed * 20), ("backward", completed * 20),
                             ("gradient_reset", completed), ("finite_gradient_clip", completed), ("optimizer_step", completed),
                             ("checkpoint_save", completed // 9)):
        require(costs["operations"][operation] == {"attempted": count, "returned": count, "failed": 0, "pending": 0}, "checkpoint complete returned boundary " + operation)
    require(costs["positions"] == {"input_attempted": envelope["cursor"]["input_tokens"], "input_returned": envelope["cursor"]["input_tokens"],
                                  "supervised_attempted": envelope["cursor"]["supervised_tokens"], "supervised_returned": envelope["cursor"]["supervised_tokens"]}, "checkpoint dose/cost binding")
    terminal = checkpoint["terminal_costs"]
    validate_costs(terminal)
    expected = deepcopy(costs)
    expected["last_call_id"] += 2
    for operation in ("checkpoint_publish", "checkpoint_promote"):
        expected["operations"][operation]["attempted"] += 1
        expected["operations"][operation]["returned"] += 1
    require(terminal == expected, "checkpoint publication ledger binding")
    require(checkpoint["publication"]["envelope_sha256"] == envelope["sha256"]
            and checkpoint["publication"].get("durable") is True, "durable envelope publication")
    require(backend.verify_publication(deepcopy(checkpoint["publication"]), envelope_sha256=envelope["sha256"]) is True,
            "publication bytes/hash/envelope not verified")
    framework_free()
    return completed


def run(registration, arm, row_get, backend, ledger, *, resume=None, latest_terminal_costs=None, stop_after_update=27):
    """Run component controls only. Stop/resume exclusively after9/18/27 updates."""
    framework_free()
    schedule.validate_registration(registration)
    require(getattr(backend, "execution_scope", None) == SCOPE, "mandatory CPU_COMPONENT_CONTROL backend; native permit absent")
    require(all(callable(getattr(backend, name, None)) for name in BACKEND_METHODS), "backend protocol")
    require(type(ledger) is CallLedger and callable(row_get), "bound reader/ledger")
    require(type(stop_after_update) is int and stop_after_update in (9, 18, 27), "registered stop boundary")
    binding = recipe(registration, arm)
    completed = 0
    if resume is None:
        require(latest_terminal_costs is None and ledger.last_call_id == 0, "fresh run cannot reset consumed costs")
    else:
        completed = validate_checkpoint(resume, registration, arm, backend, binding)
        require(latest_terminal_costs == resume["terminal_costs"] == ledger.snapshot(), "later costs/rollback refused; restore latest cumulative ledger")
        require(not latest_terminal_costs["pending"], "pending operations forbid resume")
    require(completed < stop_after_update, "target must follow saved boundary")
    # Prevalidate every scheduled row before any restore/model-like operation.
    _, _, pools, records = schedule._inputs()
    read_rows = {}
    for update in registration["arms"][arm]["updates"][completed:stop_after_update]:
        for row_id in update["occurrence_row_ids"]:
            if row_id not in read_rows:
                row = deepcopy(row_get(row_id))
                schedule.validate_token_row(row)
                require(row == pools[arm][row_id], "reader row differs from frozen arrays/source")
                read_rows[row_id] = row
    result = {"status": "RUNNING_CPU_COMPONENT_CONTROL", "scope": SCOPE, "arm": arm, "seed": 101,
              "recipe": binding, "completed_updates": completed, "checkpoints": [], "update_summaries": [],
              "native_launch_authorized": False, "native_resume_qualified": False, "training_authorized": False}
    try:
        if resume is not None:
            ledger.call("checkpoint_reload", lambda: backend.restore_checkpoint(deepcopy(resume["envelope"]["backend_state"])), stage="restore_" + str(completed))
        for update in registration["arms"][arm]["updates"][completed:stop_after_update]:
            ordinal = update["update_ordinal"]
            ledger.call("gradient_reset", backend.reset_gradients, stage="reset_" + str(ordinal))
            losses = []
            for slot, row_id in enumerate(update["occurrence_row_ids"], 1):
                row = deepcopy(read_rows[row_id])
                stage = "u" + str(ordinal) + "_row" + str(slot)
                output = ledger.call("forward", lambda: backend.forward(row), stage=stage, row=row)
                loss = ledger.call("completion_loss", lambda: backend.completion_loss(output, row), stage=stage)
                losses.append(finite(loss, "per-sequence mean loss"))
                ledger.call("backward", lambda: backend.backward(loss, scale=1 / 20), stage=stage)
            norm = finite(ledger.call("finite_gradient_clip", lambda: backend.finite_clip_gradients(max_norm=1.0), stage="clip_" + str(ordinal)), "gradient norm")
            ledger.call("optimizer_step", lambda: backend.optimizer_step(lr=update["learning_rate"]), stage="step_" + str(ordinal))
            completed = ordinal
            result["completed_updates"] = ordinal
            result["update_summaries"].append({"update_ordinal": ordinal, "learning_rate": update["learning_rate"],
                                               "mean_per_sequence_loss": sum(losses) / 20, "gradient_norm": norm})
            if ordinal % 9 == 0:
                state = ledger.call("checkpoint_save", backend.save_checkpoint, stage="save_" + str(ordinal))
                validate_state(state, ordinal, backend)
                envelope = {"schema_version": 1, "kind": KIND, "arm": arm, "seed": 101, "recipe": binding,
                    "cursor": cursor(registration, arm, ordinal), "backend_state": deepcopy(state),
                    "posted_save_costs": ledger.snapshot(), "save_call_id": ledger.last_call_id,
                    "native_resume_qualified": False, "training_authorized": False}
                envelope["sha256"] = schedule.digest(envelope)
                publication = ledger.call("checkpoint_publish", lambda: backend.publish_checkpoint(deepcopy(envelope)), stage="publish_" + str(ordinal))
                require(type(publication) is dict and publication.get("durable") is True
                        and publication.get("envelope_sha256") == envelope["sha256"], "durable envelope publication")
                require(backend.verify_publication(deepcopy(publication), envelope_sha256=envelope["sha256"]) is True,
                        "publication bytes/hash/envelope not verified")
                # A failed promotion has its own preserved failed/pending cost;
                # it cannot masquerade as the older successfully saved boundary.
                ledger.call("checkpoint_promote", lambda: ledger.emit({"event": "checkpoint_promoted",
                    "envelope_sha256": envelope["sha256"], "completed_updates": ordinal,
                    "costs_before_promotion": deepcopy(envelope["posted_save_costs"])}), stage="promote_" + str(ordinal))
                checkpoint = {"kind": KIND, "envelope": envelope, "publication": deepcopy(publication), "terminal_costs": ledger.snapshot()}
                validate_checkpoint(checkpoint, registration, arm, backend, binding)
                result["checkpoints"].append(checkpoint)
        result["status"] = "COMPLETE_CPU_COMPONENT_CONTROL" if completed == 27 else "STOPPED_CPU_COMPONENT_BOUNDARY"
    except BaseException as error:
        result["status"] = "FAILED_CPU_COMPONENT_CONTROL"
        result["error_type"], result["error"] = type(error).__name__, str(error)
        result["automatic_resume_authorized"] = False
    finally:
        result["costs"] = ledger.snapshot()
    return result
