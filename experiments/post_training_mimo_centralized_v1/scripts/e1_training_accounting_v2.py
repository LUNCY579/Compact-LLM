"""Versioned E1 journal correction; stdlib only and no launch authority.

An inner attempt-publication failure leaves its span pending. Preserve the
original error and the enclosing pending spans rather than inventing a failed
completion for an outer call while an inner call is unresolved.

The frozen V1 accounting source and native qualifier remain unchanged. This
component is not wired into an existing engine or worker. Future consumers need
a new reviewed source registration. Snapshot schema, scope, and budgets remain
compatible with V1; producer source version must be bound separately.
"""
from __future__ import annotations

from experiments.post_training_mimo_centralized_v1.scripts import e1_training_accounting as legacy


SOURCE_VERSION = 2
SOURCE_SCOPE = "E1_TRAINING_ACCOUNTING_V2_COMPONENT_UNQUALIFIED"


class E1CallAccountingV2(legacy.E1CallAccounting):
    source_version = SOURCE_VERSION
    source_scope = SOURCE_SCOPE

    def call(self, operation, function, *, stage, input_positions=0, supervised_positions=0):
        legacy.require(callable(function), "callable operation required")
        call_id = self.begin(operation, stage=stage, input_positions=input_positions,
                             supervised_positions=supervised_positions)
        try:
            value = function()
        except BaseException as error:
            # A nested attempt may remain pending when its journal publication
            # fails. Its outer span cannot be completed out of order.
            if self.stack and self.stack[-1] == call_id:
                self.failed(call_id, error)
            raise
        # A returned marker failure cannot reclassify work that returned.
        self.returned(call_id)
        return value
