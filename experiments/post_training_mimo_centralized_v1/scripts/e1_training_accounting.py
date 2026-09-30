"""Bounded E1 component call journal; no framework import or launch authority.

This is separate from the five-update E0 journal. It preserves every attempt and
supports restoring a complete terminal cost snapshot without resetting budgets.
Restoring a checkpoint's older ledger is insufficient when later work occurred.
The native loader/wrapper and campaign-wide rollback policy remain unqualified.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re


SCHEMA_VERSION = 1
SCOPE = "E1_TRAINING_COMPONENT_UNQUALIFIED"
CEILINGS = {
    "model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310,
    "forward": 5120, "backward": 5120, "optimizer_step": 256,
    "checkpoint_save": 5, "checkpoint_reload": 1,
    "gradient_reset": 256, "gradient_clip": 256,
}
POSITION_CEILING = 5_242_880
SEQUENCE_CEILING = 512
STATES = ("attempted", "returned", "failed", "pending")
POSITION_FIELDS = ("input_positions_attempted", "input_positions_returned",
                   "supervised_positions_attempted", "supervised_positions_returned")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(operation, stage, input_positions, supervised_positions):
    require(type(operation) is str and operation in CEILINGS, "unknown E1 operation")
    require(type(stage) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", stage), "invalid stage")
    require(type(input_positions) is int and type(supervised_positions) is int, "position counts must be integers")
    if operation == "forward":
        require(1 < input_positions <= SEQUENCE_CEILING and 0 < supervised_positions < input_positions,
                "single CE512 sequence with supervised targets required")
    else:
        require(input_positions == supervised_positions == 0, "positions apply only to forwards")
    return {"operation": operation, "stage": stage, "input_positions": input_positions,
            "supervised_positions": supervised_positions}


class E1CallAccounting:
    def __init__(self, emit):
        require(callable(emit), "durable event sink required")
        self.emit = emit
        self.last_call_id = 0
        self.operations = {name: dict.fromkeys(STATES, 0) for name in CEILINGS}
        self.positions = dict.fromkeys(POSITION_FIELDS, 0)
        self.pending = {}
        self.stack = []

    def begin(self, operation, *, stage, input_positions=0, supervised_positions=0):
        values = metadata(operation, stage, input_positions, supervised_positions)
        require(self.operations[operation]["attempted"] < CEILINGS[operation], "E1_OPERATION_BUDGET:" + operation)
        if operation == "forward":
            require(self.positions["input_positions_attempted"] + input_positions <= POSITION_CEILING,
                    "E1_INPUT_POSITION_BUDGET")
        self.last_call_id += 1
        call = {"call_id": self.last_call_id, **values}
        self.operations[operation]["attempted"] += 1
        self.operations[operation]["pending"] += 1
        self.pending[call["call_id"]] = call
        self.stack.append(call["call_id"])
        if operation == "forward":
            self.positions["input_positions_attempted"] += input_positions
            self.positions["supervised_positions_attempted"] += supervised_positions
        # Failed journal publication retains the unmatched attempted call.
        self.emit("operation_attempt", schema_version=SCHEMA_VERSION, scope=SCOPE, **call)
        return call["call_id"]

    def _finish(self, call_id, state):
        require(type(call_id) is int and self.stack and self.stack[-1] == call_id,
                "completion must match innermost pending call")
        call = self.pending.pop(call_id)
        self.stack.pop()
        self.operations[call["operation"]]["pending"] -= 1
        self.operations[call["operation"]][state] += 1
        if state == "returned" and call["operation"] == "forward":
            self.positions["input_positions_returned"] += call["input_positions"]
            self.positions["supervised_positions_returned"] += call["supervised_positions"]
        return call

    def returned(self, call_id):
        call = self._finish(call_id, "returned")
        self.emit("operation_returned", schema_version=SCHEMA_VERSION, scope=SCOPE, **call)

    def failed(self, call_id, error):
        call = self._finish(call_id, "failed")
        self.emit("operation_failed", schema_version=SCHEMA_VERSION, scope=SCOPE,
                  error_type=type(error).__name__, error_message=str(error)[:1024], **call)

    def call(self, operation, function, *, stage, input_positions=0, supervised_positions=0):
        require(callable(function), "callable operation required")
        call_id = self.begin(operation, stage=stage, input_positions=input_positions,
                             supervised_positions=supervised_positions)
        try:
            value = function()
        except BaseException as error:
            self.failed(call_id, error)
            raise
        # Publishing a returned marker can fail after actual work returned. That
        # is not a failed model/optimizer operation and must not be reclassified.
        self.returned(call_id)
        return value

    def snapshot(self):
        return deepcopy({"schema_version": SCHEMA_VERSION, "scope": SCOPE,
            "ceilings": CEILINGS, "sequence_ceiling": SEQUENCE_CEILING,
            "input_position_ceiling": POSITION_CEILING,
            "last_call_id": self.last_call_id, "operations": self.operations,
            "positions": self.positions,
            "pending_calls": [self.pending[index] for index in self.stack],
            "native_launch_authorized": False, "E1_authorized": False})

    @classmethod
    def from_snapshot(cls, emit, snapshot):
        """Restore cumulative costs; never qualify checkpoint tensors or RNG.

        The caller must supply the latest terminal ledger for the attempt, not
        silently roll accounting back to a checkpoint when later calls occurred.
        Pending calls require reconciliation outside this component and refuse
        automatic continuation. Failed attempts remain counted against ceilings.
        """
        blank = cls(emit)
        require(type(snapshot) is dict and set(snapshot) == set(blank.snapshot()), "snapshot fields")
        for name in ("schema_version", "scope", "ceilings", "sequence_ceiling", "input_position_ceiling",
                     "native_launch_authorized", "E1_authorized"):
            require(json.dumps(snapshot[name], sort_keys=True, allow_nan=False) ==
                    json.dumps(blank.snapshot()[name], sort_keys=True, allow_nan=False),
                    "snapshot contract drift:" + name)
        require(snapshot["pending_calls"] == [], "pending calls require explicit reconciliation")
        ops = snapshot["operations"]
        require(type(ops) is dict and set(ops) == set(CEILINGS), "operation inventory")
        for name, counts in ops.items():
            require(type(counts) is dict and set(counts) == set(STATES), "operation counter fields")
            require(all(type(value) is int and value >= 0 for value in counts.values()), "invalid operation counter")
            require(counts["pending"] == 0 and counts["attempted"] == counts["returned"] + counts["failed"]
                    and counts["attempted"] <= CEILINGS[name], "operation counts inconsistent or over budget")
        require(type(snapshot["last_call_id"]) is int and snapshot["last_call_id"] == sum(x["attempted"] for x in ops.values()),
                "call ID total drift")
        positions = snapshot["positions"]
        require(type(positions) is dict and set(positions) == set(POSITION_FIELDS)
                and all(type(value) is int and value >= 0 for value in positions.values()), "position fields")
        for suffix, state in (("attempted", "attempted"), ("returned", "returned")):
            count = ops["forward"][state]
            inputs, supervised = positions["input_positions_" + suffix], positions["supervised_positions_" + suffix]
            require(2 * count <= inputs <= SEQUENCE_CEILING * count and count <= supervised <= inputs - count,
                    "positions incompatible with forward counts")
        require(positions["input_positions_returned"] <= positions["input_positions_attempted"] <= POSITION_CEILING
                and positions["supervised_positions_returned"] <= positions["supervised_positions_attempted"], "position budget drift")
        failed_count = ops["forward"]["failed"]
        failed_inputs = positions["input_positions_attempted"] - positions["input_positions_returned"]
        failed_supervised = positions["supervised_positions_attempted"] - positions["supervised_positions_returned"]
        require(2 * failed_count <= failed_inputs <= SEQUENCE_CEILING * failed_count and
                failed_count <= failed_supervised <= failed_inputs - failed_count,
                "failed forward positions inconsistent")
        blank.last_call_id = snapshot["last_call_id"]
        blank.operations = deepcopy(ops)
        blank.positions = deepcopy(positions)
        return blank
