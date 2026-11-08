"""CE8 synthetic checkpoint controls; no production CE grant or cost proof.

Actual cases require root's COMP_RS_RUN_NATIVE_CPU_TESTS=1 clearance. The resume
comparison uses fresh objects in one OS process, not a fresh-process/native
qualification. Collecting this file imports no Torch or NumPy.
"""
import ast
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import checkpoint as cp, ce_checkpoint as ce_cp, ce_update
from experiments.skill_acquisition_native_v1.tests import test_ce_update as helpers


def plan(count=2):
    return [{'boundary': i, 'row_ids': ['b%d-r%d' % (i, j) for j in range(8)], 'logical_exposure': 0}
            for i in range(count)]


def static_recipe(value=None):
    dynamic = {'boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'}
    return {key: copy.deepcopy(item) for key, item in (value or helpers.recipe()).items() if key not in dynamic}


def bindings():
    return {'checkpoint_mode': ce_cp.MODE, 'synthetic_cpu': True, 'ordered_plan': plan(),
            'ordered_plan_sha256': cp.digest(plan()),
            'inventory': [{'name': 'weight', 'shape': [2], 'dtype': 'F32'}],
            'code': 'synthetic-source-only', 'runtime': 'synthetic-cpu-only',
            'data': {'fixture_tokens_sha256': cp.digest({'fixture': 'prepared-tokens-and-targets'})},
            'recipe': static_recipe(), 'own_start_sha256': 'a' * 64, 'common_anchor_sha256': 'b' * 64}


def progress(rows, C=0, generation=0):
    history = [{'boundary': i, 'row_ids': list(rows[i]), 'updated': True} for i in range(C)]
    value = {'C': C, 'U': C, 'history': history, 'generation': generation}
    return dict(value, snapshot=cp.digest(value))


class Budget:
    def __init__(self): self.events = []
    def __call__(self, kind, bytes_upper, *, write_id, path):
        cp.require(type(bytes_upper) is int and bytes_upper > 0, 'exact positive reservation')
        self.events.append(dict(kind=kind, bytes_upper=bytes_upper, write_id=write_id, path=path))


class PureTests(unittest.TestCase):
    def test_native_and_wrong_mode_denied_before_path_or_io_even_forged_flags(self):
        class UnreadablePath:
            def __fspath__(self): raise AssertionError('path accessed before scope refusal')
        observed = []
        for flag in (False, 1, None):
            value = bindings(); value.update(approved=True, training_authorized=True, native_qualified=True)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'not admitted'):
                ce_cp.CECheckpoints(UnreadablePath(), value, 2, observed.append,
                                    metadata_reserve=observed.append, synthetic_cpu=flag)
        for mode in (None, 'B2', 'CE8_NATIVE'):
            value = bindings(); value['checkpoint_mode'] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'checkpoint-mode'):
                ce_cp.CECheckpoints(UnreadablePath(), value, 2, observed.append,
                                    metadata_reserve=observed.append, synthetic_cpu=True)
        self.assertEqual(observed, [])

    def test_invalid_eight_rows_global_uniqueness_and_order_leave_no_files(self):
        for change in ('B2', 'within', 'global', 'order', 'exposure'):
            value = bindings()
            if change == 'B2': value['ordered_plan'][0]['row_ids'] = ['a', 'b']
            if change == 'within': value['ordered_plan'][0]['row_ids'][7] = value['ordered_plan'][0]['row_ids'][0]
            if change == 'global': value['ordered_plan'][1]['row_ids'][0] = value['ordered_plan'][0]['row_ids'][0]
            if change == 'order': value['ordered_plan'][1]['boundary'] = 0
            if change == 'exposure': value['ordered_plan'][0]['logical_exposure'] = True
            value['ordered_plan_sha256'] = cp.digest(value['ordered_plan'])
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'denied'; events = []; budget = Budget()
                with self.subTest(change=change), self.assertRaises(ValueError):
                    ce_cp.CECheckpoints(target, value, 2, events.append, metadata_reserve=budget, synthetic_cpu=True)
                self.assertFalse(target.exists()); self.assertFalse(events); self.assertFalse(budget.events)

    def test_parent_B2_scope_preserved_and_instance_hooks_accept_CE8_only(self):
        with self.assertRaisesRegex(ValueError, 'B2'): cp._plan_rows(bindings())
        with tempfile.TemporaryDirectory() as directory, patch.object(cp.NativeCheckpoints, '_check_gate', return_value={}):
            adapter = ce_cp.CECheckpoints(directory, bindings(), 2, lambda stage: None,
                                         metadata_reserve=Budget(), synthetic_cpu=True)
            self.assertEqual(adapter.plan_rows, [item['row_ids'] for item in plan()])
            value = progress(adapter.plan_rows, 1)
            self.assertIs(adapter._validate_progress(value), value)
            for change in ('C!=U', 'uniform', 'row-order'):
                bad = copy.deepcopy(value)
                if change == 'C!=U': bad['U'] = 0
                if change == 'uniform': bad['U'] = 0; bad['history'][0]['updated'] = False
                if change == 'row-order': bad['history'][0]['row_ids'].reverse()
                with self.subTest(change=change), self.assertRaises(ValueError): adapter._validate_progress(bad)
            # Default behavior still admits skipped B2 boundaries and C>U.
            generic = cp.NativeCheckpoints.__new__(cp.NativeCheckpoints)
            generic.plan_rows = [['r', 's']]
            generic._validate_progress({'C': 1, 'U': 0, 'generation': 0, 'snapshot': 'a' * 64,
                                       'history': [{'boundary': 0, 'row_ids': ['r', 's'], 'updated': False}]})

    def test_tokens_and_static_recipe_bindings_cannot_silently_cross_restore(self):
        for field in ('data', 'recipe'):
            value = bindings(); value[field] = 'unbound-fixture-name'
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'denied'
                with self.assertRaises(ValueError):
                    ce_cp.CECheckpoints(target, value, 2, lambda stage: None, metadata_reserve=Budget(), synthetic_cpu=True)
                self.assertFalse(target.exists())
        with tempfile.TemporaryDirectory() as directory, patch.object(cp.NativeCheckpoints, '_check_gate', return_value={}):
            ce_cp.CECheckpoints(directory, bindings(), 2, lambda stage: None, metadata_reserve=Budget(), synthetic_cpu=True)
            for field in ('tokens', 'recipe'):
                changed = bindings()
                if field == 'tokens': changed['data']['fixture_tokens_sha256'] = 'f' * 64
                else: changed['recipe']['optimizer']['lr'] *= 2
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'bindings changed'):
                    ce_cp.CECheckpoints(directory, changed, 2, lambda stage: None,
                                        metadata_reserve=Budget(), synthetic_cpu=True)

    def test_nonCPU_buffer_and_initializedCUDA_denied_before_parent_operation(self):
        cpu = SimpleNamespace(device=SimpleNamespace(type='cpu'))
        gpu = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        adapter = ce_cp.CECheckpoints.__new__(ce_cp.CECheckpoints); adapter.plan_rows = [plan()[0]['row_ids']]
        model = SimpleNamespace(parameters=lambda: iter([cpu]), buffers=lambda: iter([gpu]))
        with patch.object(cp.NativeCheckpoints, 'save') as save, patch.object(cp.NativeCheckpoints, 'restore') as restore:
            with self.assertRaisesRegex(ValueError, 'parameters and buffers'): adapter.save(model, None, progress(adapter.plan_rows), None)
            with self.assertRaisesRegex(ValueError, 'parameters and buffers'): adapter.restore(model, None)
            save.assert_not_called(); restore.assert_not_called()
        model.buffers = lambda: iter([cpu])
        with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))}), \
             patch.object(cp.NativeCheckpoints, 'restore') as restore:
            with self.assertRaisesRegex(ValueError, 'uninitialized CUDA'): adapter.restore(model, None)
            restore.assert_not_called()

    def test_collection_has_no_framework_imports(self):
        for module in (ce_cp,):
            tree = ast.parse(Path(module.__file__).read_text())
            imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
            self.assertFalse(set(imports) & {'torch', 'numpy', 'transformers'})

    def test_restore_consumer_rejects_forged_nested_fence_counts_and_ancestry_without_mutation(self):
        rows = [item['row_ids'] for item in plan()]
        caller = FixtureCaller(rows); caller.progress = progress(rows, 2)
        nested = progress(rows, 1, 1); nested['symbolic_snapshot'] = nested['snapshot']
        record = {'C': 1, 'U': 1, 'generation': 1, 'snapshot': nested['snapshot'],
                  'symbolic_snapshot': nested['snapshot'], 'history': copy.deepcopy(nested['history']),
                  'progress': nested, 'manifest_sha256': 'f' * 64, 'install_fence_required': True}
        caller.operation('checkpoint_restore', 'started', operation_id='restore-proof')
        caller.operation('checkpoint_restore', 'returned', operation_id='restore-proof', C=1, U=1,
                         generation=1, symbolic_snapshot=nested['snapshot'], manifest_sha256='f' * 64)
        before = copy.deepcopy((caller.progress, caller.used_restores, caller.events))
        for change in ('old-fence', 'wrong-counts', 'wrong-U', 'row-order', 'nonancestor-history', 'alias'):
            bad = copy.deepcopy(record)
            if change == 'old-fence':
                old = progress(rows, 1); old['symbolic_snapshot'] = old['snapshot']; bad['progress'] = old
            if change == 'wrong-counts':
                zero = progress(rows, 0, 1); zero['symbolic_snapshot'] = zero['snapshot']; bad['progress'] = zero
            if change == 'wrong-U': bad['progress']['U'] = 0
            if change == 'row-order': bad['progress']['history'][0]['row_ids'].reverse()
            if change == 'nonancestor-history': bad['progress']['history'][0]['unobserved_proof'] = 'forged'
            if change == 'alias': bad['progress']['symbolic_snapshot'] = 'a' * 64
            with self.subTest(change=change), self.assertRaises(ValueError): caller.install_restore(bad)
            self.assertEqual((caller.progress, caller.used_restores, caller.events), before)
        caller.install_restore(record)
        self.assertEqual(caller.progress, nested)
        self.assertEqual(caller.used_restores, {'restore-proof'})
        self.assertEqual(caller.events, before[2])


class FixtureCaller:
    """In-memory test proof consumer, NOT a production/durable CE ledger."""
    def __init__(self, rows):
        self.rows, self.events, self.used_restores = rows, [], set()
        self.progress = progress(rows)
    @property
    def generation(self): return self.progress['generation']
    def operation(self, kind, phase, **fields): self.events.append(dict(fields, kind=kind, phase=phase))
    def snapshot(self): return {'synthetic_fixture_only': True, 'events': len(self.events), 'head': cp.digest(self.events)}
    def commit(self, result):
        p = self.progress; boundary = p['C']
        cp.require(result['generation'] == p['generation'] and result['symbolic_snapshot'] == p['snapshot'], 'stale CE result fence')
        cp.require(result['boundary_id'] == boundary and result['U_before'] == p['U']
                   and result['U_after'] == p['U'] + 1 and result['updated'] is True
                   and result['requires_restore'] is False and result['synthetic_cpu'] is True
                   and [row['id'] for row in result['rows']] == self.rows[boundary], 'complete CE result prefix proof')
        def paired(record, kind):
            events = [e for e in self.events if e.get('operation_id') == record['operation_id']]
            cp.require(len(events) == 2 and [e['phase'] for e in events] == ['started', 'returned']
                       and all(e['kind'] == kind and e['generation'] == p['generation']
                               and e['symbolic_snapshot'] == p['snapshot']
                               and e['boundary_id'] == boundary and e['start_policy_id'] == result['start_policy_id']
                               for e in events), 'paired CE physical proof')
            cp.require(all(events[-1].get(key) == value for key, value in record.items() if key != 'kind'), 'returned CE proof binding')
            return events
        pre = paired(result['identity_operation'], 'fingerprint')[-1]
        post = paired(result['final_identity_operation'], 'fingerprint')[-1]
        cp.require(pre['policy_sha256'] == result['start_policy_id'] and pre['U'] == p['U']
                   and post['policy_sha256'] == result['final_policy_id'] and post['U'] == p['U'] + 1
                   and pre['frozen_state_sha256'] == post['frozen_state_sha256'] == result['frozen_state_sha256'], 'actual CE policy/U/frozen proof')
        for row in result['rows']:
            for key, kind in (('forward_operation', 'loss_forward'), ('backward_operation', 'backward')):
                cp.require(paired(row[key], kind)[-1]['prompt_id'] == row['id'], 'row-specific proof')
        paired(result['gradient_validation_operation'], 'gradient_validation')
        paired(result['clipping_operation'], 'gradient_clipping')
        optimizer = paired(result['optimizer_operation'], 'optimizer')
        cp.require(optimizer[0]['U_before'] == p['U'] and optimizer[0]['expected_U_after'] == p['U'] + 1, 'Adam start proof')
        scoped = [e for e in self.events if e.get('boundary_id') == boundary and e.get('generation') == p['generation']
                  and e.get('symbolic_snapshot') == p['snapshot'] and e.get('start_policy_id') == result['start_policy_id']]
        for kind in ('loss_forward', 'backward'):
            cp.require(sum(e['kind'] == kind and e['phase'] == 'returned' for e in scoped) == 8, 'exact eight returned rows')
        cleanup = [e for e in scoped if e['kind'] == 'gradient_cleanup']
        cp.require(len(cleanup) == 2 and [e['phase'] for e in cleanup] == ['started', 'returned']
                   and cleanup[0]['operation_id'] == cleanup[1]['operation_id'] and cleanup[1]['optimizer_attempted'] is True
                   and self.events.index(cleanup[0]) > self.events.index(post), 'successful CE cleanup required before commit')
        self.progress = progress(self.rows, boundary + 1, p['generation'])
    def install_restore(self, record):
        events = [e for e in self.events if e['kind'] == 'checkpoint_restore' and e['phase'] == 'returned'
                  and e.get('manifest_sha256') == record['manifest_sha256']
                  and all(e.get(key) == record[key] for key in ('C', 'U', 'generation', 'symbolic_snapshot'))]
        cp.require(len(events) == 1 and events[0]['operation_id'] not in self.used_restores
                   and record['install_fence_required'] is True and record['generation'] > self.generation,
                   'one-use returned restore fence proof')
        pair = [e for e in self.events if e.get('operation_id') == events[0]['operation_id']]
        cp.require(len(pair) == 2 and [e['phase'] for e in pair] == ['started', 'returned']
                   and all(e['kind'] == 'checkpoint_restore' for e in pair), 'paired returned restore proof')
        nested = record['progress']
        cp.validate_progress(nested, self.rows)
        cp.require(nested['C'] == nested['U'] and all(entry['updated'] is True for entry in nested['history']),
                   'restored nested progress must be successful CE8 history')
        cp.require(all(type(record[key]) is int and nested[key] == record[key]
                       for key in ('C', 'U', 'generation'))
                   and nested['snapshot'] == record['snapshot'] == record['symbolic_snapshot']
                   and nested.get('symbolic_snapshot') == record['symbolic_snapshot'] == events[0]['symbolic_snapshot'],
                   'nested restore fence/counts differ from returned proof')
        cp.require(nested['C'] <= self.progress['C']
                   and nested['history'] == self.progress['history'][:nested['C']]
                   and record['history'] == nested['history'], 'restored history is not the exact current ancestor prefix')
        self.used_restores.add(events[0]['operation_id'])
        self.progress = copy.deepcopy(nested)


@unittest.skipUnless(os.environ.get('COMP_RS_RUN_NATIVE_CPU_TESTS') == '1', 'actual CE8 checkpoint graphs await root clearance')
class TinyCPUIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        helpers.TinyCPU.setUpClass()
        cls.torch, cls.np = torch, np
        if torch.cuda.is_initialized(): raise AssertionError('CE8 CPU proof must not initialize CUDA')

    def make(self):
        helper = helpers.TinyCPU(methodName='runTest'); model = helper.model(); torch = self.torch
        class ObservedDropout(torch.nn.Dropout):
            def __init__(self): super().__init__(.25); self.changed = False; self.calls = 0
            def forward(self, value):
                output = super().forward(value)
                if self.training:
                    self.calls += 1; self.changed |= bool((value.detach() != output.detach()).any())
                return output
        layer = model.model.layers[0]; layer.linear = torch.nn.Sequential(layer.linear, ObservedDropout())
        _, recipe = helper.inputs(model)
        return helper, model, helper.optimizer(model, recipe), recipe

    def prepared(self, py_rng, np_rng):
        all_rows = []
        for boundary in range(2):
            data = copy.deepcopy(helpers.rows())
            for i, row in enumerate(data):
                row['id'] = plan()[boundary]['row_ids'][i]
                # Both streams select prompt tokens consumed by the actual loss.
                row['prompt_ids'] = [13 + py_rng.randrange(4), 17 + int(np_rng.randint(4))] + row['prompt_ids']
            all_rows.append(data)
        return all_rows

    def seed(self):
        random.seed(1701); self.np.random.seed(1702); self.torch.manual_seed(1703)

    def fixture_bindings(self, model, recipe, prepared):
        value = bindings()
        value['inventory'] = [{'name': n, 'shape': list(p.shape), 'dtype': 'float32'}
                              for n, p in model.named_parameters() if p.requires_grad]
        value['recipe'] = static_recipe(recipe)
        value['data'] = {'fixture_tokens_sha256': cp.digest(prepared), 'source': 'synthetic-prepared-two-boundary-fixture'}
        value['own_start_sha256'] = value['common_anchor_sha256'] = helpers.TinyCPU.runtime.anchor_digest(model)
        return value

    def boundary(self, helper, model, optimizer, recipe, caller, expected):
        boundary = caller.progress['C']; data = copy.deepcopy(helpers.rows())
        for i, row in enumerate(data):
            row['id'] = caller.rows[boundary][i]
            row['prompt_ids'] = [13 + random.randrange(4), 17 + int(self.np.random.randint(4))] + row['prompt_ids']
        self.assertEqual(data, expected)
        recipe = dict(recipe, boundary_id=boundary, U_before=caller.progress['U'], generation=caller.generation,
                      symbolic_snapshot=caller.progress['snapshot'], start_policy_id=helper.runtime.anchor_digest(model))
        result = ce_update.boundary_update(model, optimizer, data, recipe, caller.operation)
        # Logical C/U remain unchanged until the full result plus cleanup is consumed.
        self.assertEqual((caller.progress['C'], caller.progress['U']), (boundary, boundary))
        caller.commit(result)
        self.assertEqual((caller.progress['C'], caller.progress['U']), (boundary + 1, boundary + 1))
        return result

    def snapshot(self, model, optimizer):
        rng, tensors = cp.streams.rng_state()
        return ({n: p.detach().clone() for n, p in model.named_parameters()},
                {n: p.detach().clone() for n, p in model.named_buffers()},
                [{key: value.detach().clone() for key, value in optimizer.state.get(p, {}).items()} for p in model.parameters()],
                json.loads(cp.canonical(rng)), {key: value.clone() for key, value in tensors.items()},
                {n: module.training for n, module in model.named_modules()})

    def assert_state_equal(self, expected, actual):
        for index in (0, 1, 4):
            self.assertEqual(set(expected[index]), set(actual[index]))
            self.assertTrue(all(self.torch.equal(expected[index][key], actual[index][key]) for key in expected[index]))
        self.assertEqual(expected[3], actual[3]); self.assertEqual(expected[5], actual[5])
        for left, right in zip(expected[2], actual[2]):
            self.assertEqual(set(left), set(right))
            self.assertTrue(all(self.torch.equal(left[key], right[key]) for key in left))

    def adapter(self, directory, model, value, caller, budget):
        return ce_cp.CECheckpoints(directory, value, sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  lambda stage: None, metadata_reserve=budget, synthetic_cpu=True).attach_ledger(caller)

    def test_stochastic_two_boundaries_bitwise_fresh_objects_restore_and_fenced_commit(self):
        prepared = self.prepared(random.Random(1701), self.np.random.RandomState(1702))
        helper, model, optimizer, recipe = self.make(); expected_caller = FixtureCaller([p['row_ids'] for p in plan()])
        self.seed()
        self.boundary(helper, model, optimizer, recipe, expected_caller, prepared[0]); model.lm_head.train(True)
        self.boundary(helper, model, optimizer, recipe, expected_caller, prepared[1])
        expected = self.snapshot(model, optimizer)
        self.assertTrue(model.model.layers[0].linear[1].changed)
        self.assertGreater(model.model.layers[0].linear[1].calls, 0)
        helper, model, optimizer, recipe = self.make(); caller = FixtureCaller([p['row_ids'] for p in plan()]); budget = Budget()
        value = self.fixture_bindings(model, recipe, prepared); self.seed()
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, model, value, caller, budget)
            self.boundary(helper, model, optimizer, recipe, caller, prepared[0]); model.lm_head.train(True)
            saved_state = self.snapshot(model, optimizer)
            saved_policy = helper.runtime.anchor_digest(model)
            manifest = adapter.save(model, optimizer, caller.progress, caller)
            self.assertEqual(manifest['rng']['cuda_count'], 0)
            old = self.boundary(helper, model, optimizer, recipe, caller, prepared[1])
            spent_events, spent_bytes = len(caller.events), sum(e['bytes_upper'] for e in budget.events)
            helper2, fresh, fresh_optimizer, recipe2 = self.make()
            reopened = self.adapter(directory, fresh, value, caller, budget)
            restored = reopened.restore(fresh, fresh_optimizer)
            self.assert_state_equal(saved_state, self.snapshot(fresh, fresh_optimizer))
            self.assertEqual(restored['manifest_sha256'], manifest['manifest_sha256'])
            self.assertEqual((restored['C'], restored['U']), (1, 1))
            self.assertNotEqual(restored['snapshot'], restored['restored_snapshot'])
            self.assertGreater(len(caller.events), spent_events)
            self.assertGreater(sum(e['bytes_upper'] for e in budget.events), spent_bytes)
            caller.install_restore(restored)
            with self.assertRaisesRegex(ValueError, 'one-use'): caller.install_restore(restored)
            with self.assertRaisesRegex(ValueError, 'stale CE result'): caller.commit(old)
            self.assertEqual(helper2.runtime.anchor_digest(fresh), saved_policy)
            result = self.boundary(helper2, fresh, fresh_optimizer, recipe2, caller, prepared[1])
            self.assertEqual(result['generation'], restored['generation'])
            self.assertEqual(result['symbolic_snapshot'], restored['snapshot'])
            self.assert_state_equal(expected, self.snapshot(fresh, fresh_optimizer))
            # Cleanup is absent from result IDs: the callback is an essential proof.
            scoped = FixtureCaller(caller.rows); scoped.progress = progress(caller.rows, 1, restored['generation'])
            scoped.progress['snapshot'] = restored['snapshot']
            scoped.events = [e for e in caller.events if e['kind'] != 'gradient_cleanup']
            with self.assertRaisesRegex(ValueError, 'cleanup'): scoped.commit(result)
            self.assertEqual((scoped.progress['C'], scoped.progress['U']), (1, 1))
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_latest_corruption_refuses_without_live_state_mutation_or_older_fallback(self):
        prepared = self.prepared(random.Random(1701), self.np.random.RandomState(1702))
        helper, model, optimizer, recipe = self.make(); caller = FixtureCaller([p['row_ids'] for p in plan()])
        value = self.fixture_bindings(model, recipe, prepared); self.seed()
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, model, value, caller, Budget())
            adapter.save(model, optimizer, caller.progress, caller)
            self.boundary(helper, model, optimizer, recipe, caller, prepared[0])
            adapter.save(model, optimizer, caller.progress, caller)
            paths = adapter._latest(); self.assertEqual(len(paths), 2)
            latest = paths[-1].with_suffix('.safetensors')
            with latest.open('r+b') as stream:
                stream.seek(-1, 2); byte = stream.read(1); stream.seek(-1, 2); stream.write(bytes([byte[0] ^ 1]))
            before = self.snapshot(model, optimizer); before_progress = copy.deepcopy(caller.progress); reads = []
            read = adapter._read_json
            def observed(path): reads.append(path); return read(path)
            with patch.object(adapter, '_read_json', side_effect=observed), self.assertRaisesRegex(ValueError, 'payload bytes corrupt'):
                adapter.restore(model, optimizer)
            self.assert_state_equal(before, self.snapshot(model, optimizer)); self.assertEqual(caller.progress, before_progress)
            self.assertIn(paths[-1], reads); self.assertNotIn(paths[0], reads)
            self.assertTrue(any(e['kind'] == 'checkpoint_restore' and e['phase'] == 'failed' for e in caller.events))
        self.assertFalse(self.torch.cuda.is_initialized())
