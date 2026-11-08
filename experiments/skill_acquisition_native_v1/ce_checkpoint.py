"""Synthetic CPU CE8 adapter to the reviewed bounded checkpoint mechanics.

This adapter supplies no native admission, production CE ledger, runner or cost
qualification. Its exact bindings distinguish it from the default B2 adapter.
The caller installs the returned restore fence and recomputes actual policy
identity before another CE boundary; checkpoint snapshots are symbolic only.
"""
from __future__ import annotations

from . import checkpoint as base

MODE = 'CE8_SYNTHETIC_CPU_1'
ROWS = 8


class CECheckpoints(base.NativeCheckpoints):
    def __init__(self, directory, bindings, numel, gate, *, metadata_reserve, synthetic_cpu):
        # Refuse native mode before any path conversion, framework import or IO.
        base.require(synthetic_cpu is True, 'native CE checkpoint scope is not admitted')
        base.require(type(bindings) is dict and bindings.get('synthetic_cpu') is True
                     and bindings.get('checkpoint_mode') == MODE,
                     'exact synthetic CE8 checkpoint-mode binding required')
        base.require(type(bindings.get('data')) is dict
                     and base._hex(bindings['data'].get('fixture_tokens_sha256')),
                     'exact prepared fixture token/target digest required')
        recipe = bindings.get('recipe')
        base.require(type(recipe) is dict and recipe.get('schema') == 'SKILL-ACQUISITION-CE-UPDATE-RECIPE-1'
                     and recipe.get('synthetic_cpu') is True
                     and not {'boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'} & set(recipe),
                     'static synthetic CE recipe binding required')
        from . import ce_update
        ce_update.validate_recipe(dict(recipe, boundary_id=0, generation=0, U_before=0,
                                       symbolic_snapshot='0' * 64, start_policy_id='0' * 64))
        base.require(base._hex(bindings.get('ordered_plan_sha256'))
                     and bindings['ordered_plan_sha256'] == base.digest(bindings.get('ordered_plan')),
                     'exact CE8 ordered plan digest required')
        super().__init__(directory, bindings, numel, gate, metadata_reserve=metadata_reserve)

    def _extract_plan_rows(self, bindings):
        plan = bindings.get('ordered_plan')
        base.require(type(plan) is list and plan, 'exact nonempty ordered CE8 plan required')
        result, seen = [], set()
        for index, boundary in enumerate(plan):
            base.require(type(boundary) is dict, 'CE8 ordered boundary schema')
            ids = boundary.get('row_ids')
            base.require(type(ids) is list and len(ids) == ROWS
                         and all(type(row) is str and row for row in ids)
                         and len(set(ids)) == ROWS and not seen.intersection(ids),
                         'eight distinct globally unique ordered CE row identities required')
            base.require(type(boundary.get('boundary')) is int and boundary['boundary'] == index
                         and type(boundary.get('logical_exposure')) is int
                         and boundary['logical_exposure'] == 0, 'CE8 boundary order/exposure drift')
            result.append(list(ids))
            seen.update(ids)
        return result

    def _validate_progress(self, progress):
        base.validate_progress(progress, self.plan_rows)
        base.require(progress['C'] == progress['U']
                     and all(entry['updated'] is True for entry in progress['history']),
                     'successful CE8 history requires C=U and every boundary updated')
        return progress

    @staticmethod
    def _require_cpu(model):
        values = list(model.parameters()) + list(model.buffers())
        base.require(values and all(value.device.type == 'cpu' for value in values),
                     'synthetic CE8 checkpoints require CPU parameters and buffers')
        import torch
        base.require(not torch.cuda.is_initialized(),
                     'synthetic CE8 checkpoint proof requires uninitialized CUDA and zero CUDA RNG streams')

    def save(self, model, optimizer, progress, ledger):
        self._validate_progress(progress)
        self._require_cpu(model)
        return super().save(model, optimizer, progress, ledger)

    def restore(self, model, optimizer):
        self._require_cpu(model)
        return super().restore(model, optimizer)
