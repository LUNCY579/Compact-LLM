"""Resource-gated actual CPU CE/checkpoint/durable-journal integration.

Reopened ledgers and replacement model/Adam objects stay in one OS process.
The injected boot identity describes that fixture epoch, not a Windows boot or
fresh-process qualification. Declared test profiles are not native measurements.
Collection imports no framework; root alone clears/runs the actual CPU cases.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import uuid

from experiments.skill_acquisition_native_v1 import ce_plan, ce_update, ce_checkpoint, checkpoint


class PureSafety(unittest.TestCase):
    def test_native_refusal_precedes_path_access_even_with_forged_grants(self):
        from experiments.skill_acquisition_native_v1 import ce_journal
        class UnreadablePath:
            def __fspath__(self): raise AssertionError('native scope touched path')
        for flag in (False, 1, None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'scope denied'):
                ce_journal.CeJournal(UnreadablePath(), {'mode': 'synthetic_fixture', 'approved': True},
                    {'approved': True, 'training_authorized': True}, {}, synthetic_cpu=flag)


@unittest.skipUnless(os.environ.get('COMP_RS_RUN_NATIVE_CPU_TESTS') == '1',
                     'actual CPU journal integration awaits root resource clearance')
class TinyCPUIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        from experiments.skill_acquisition_native_v1 import ce_journal
        from experiments.skill_acquisition_native_v1.tests import test_ce_update
        test_ce_update.TinyCPU.setUpClass()
        cls.torch, cls.np, cls.cj, cls.fixtures = torch, np, ce_journal, test_ce_update
        if torch.cuda.is_initialized(): raise AssertionError('CPU integration must not initialize CUDA')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ce8-durable-cpu-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def tearDown(self):
        self.assertFalse(self.torch.cuda.is_initialized())

    def seed(self):
        random.seed(1701); self.np.random.seed(1702); self.torch.manual_seed(1703)

    def model(self):
        helper = self.fixtures.TinyCPU(methodName='runTest')
        model = helper.model()
        layer = model.model.layers[0]
        # Real train-mode stochastic work is replayed under nonreentrant GC.
        layer.linear = self.torch.nn.Sequential(layer.linear, self.torch.nn.Dropout(.25))
        model.eval()  # Newly attached modules default to training=True.
        _, recipe = helper.inputs(model)
        return helper, model, helper.optimizer(model, recipe), recipe

    @staticmethod
    def static_recipe(recipe):
        dynamic = {'boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'}
        return {key: copy.deepcopy(value) for key, value in recipe.items() if key not in dynamic}

    def prepared(self):
        records = [{'id': 'journal-row-%02d' % index, 'target_sha256': ce_plan.digest(['temporary', index])}
                   for index in range(16)]
        skeleton = ce_plan.build_synthetic(records)
        ids = [row_id for boundary in skeleton['ordered_plan'] for row_id in boundary['row_ids']]
        py_rng, np_rng = random.Random(1701), self.np.random.RandomState(1702)
        flat = []
        for index, row_id in enumerate(ids):
            row = copy.deepcopy(self.fixtures.rows()[index % 8])
            row['id'] = row_id
            if len(row['completion_ids']) == 1: row['completion_ids'].insert(0, 3)
            row['prompt_ids'] = [13 + py_rng.randrange(4), 17 + int(np_rng.randint(4))] + row['prompt_ids']
            row['target_sha256'] = ce_plan.digest(row['completion_ids'])
            flat.append(row)
        plan = ce_plan.build_synthetic([{'id': row['id'], 'target_sha256': row['target_sha256']} for row in flat])
        self.assertEqual([i for b in plan['ordered_plan'] for i in b['row_ids']], ids)
        return plan, flat

    def context(self, directory, *, limits=None, resume=False, shared=None):
        helper, model, optimizer, recipe = self.model()
        if shared is None:
            plan, prepared = self.prepared()
            profile = {'read_bytes_upper': 10 ** 9, 'write_bytes_upper': 10 ** 9,
                       'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 10 ** 9,
                       'wall_s_upper': 10.0}
            setup = dict(profile, identity_hash_attempts_upper=10000)
            caps = {'schema': self.cj.CAP_SCHEMA, 'synthetic_cpu': True,
                    'operation_starts': {kind: 10000 for kind in self.cj.KINDS},
                    **{key: 10 ** 12 for key in self.cj.COUNTERS if key != 'wall_seconds'}}
            caps.update(metadata_reserved_bytes=10 ** 9, invocations=10, resumes=10, wall_seconds=10 ** 6)
            for key, value in (limits or {}).items():
                if key == 'operation_starts': caps[key].update(value)
                else: caps[key] = value
            anchor = helper.runtime.anchor_digest(model)
            sources = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for module in (ce_update, ce_checkpoint, checkpoint, ce_plan, self.cj)}
            bindings = {'scope': self.cj.SCOPE, 'recipe': self.static_recipe(recipe),
                'anchors': {'own_start_sha256': anchor, 'common_anchor_sha256': anchor}, 'source': sources,
                'runtime': {'device': 'cpu', 'dtype': 'float32', 'cuda_initialized': False},
                'prepared_rows': prepared, 'operation_bounds': {kind: dict(profile) for kind in self.cj.KINDS},
                'setup_bounds': setup}
            cp_bindings = {**ce_plan.checkpoint_export(plan), 'inventory': [
                {'name': name, 'shape': list(parameter.shape), 'dtype': 'float32'}
                for name, parameter in model.named_parameters() if parameter.requires_grad],
                'recipe': self.static_recipe(recipe), 'code': sources, 'runtime': bindings['runtime'],
                'data': {'fixture_tokens_sha256': ce_plan.digest(prepared)},
                'own_start_sha256': anchor, 'common_anchor_sha256': anchor}
            shared = (plan, prepared, bindings, caps, cp_bindings)
        plan, prepared, bindings, caps, cp_bindings = shared
        ledger = self.cj.CeJournal(directory, plan, bindings, caps, synthetic_cpu=True, resume=resume)
        ledger.start_invocation(resume=resume, boot_id='SAME-PROCESS-SYNTHETIC-EPOCH',
                                reason='documented same-process object replacement')
        return {'helper': helper, 'model': model, 'optimizer': optimizer, 'recipe': recipe,
                'ledger': ledger, 'shared': shared, 'prepared': prepared, 'directory': Path(directory)}

    def physical(self, ledger, kind, callback, **fields):
        identifier = uuid.uuid4().hex
        ledger.operation(kind, 'started', operation_id=identifier, **fields)
        try: value = callback()
        except BaseException as error:
            ledger.operation(kind, 'failed', operation_id=identifier, error=str(error)[:1000])
            raise
        ledger.operation(kind, 'returned', operation_id=identifier)
        return value

    def adapter(self, context):
        ledger = context['ledger']; bindings = context['shared'][4]
        model = context['model']; count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return self.physical(ledger, 'checkpoint_construct', lambda:
            ce_checkpoint.CECheckpoints(context['directory'] / 'checkpoints', bindings, count,
                lambda stage: None, metadata_reserve=ledger.reserve_metadata, synthetic_cpu=True).attach_ledger(ledger),
            **checkpoint.constructor_hash_lower_bound(bindings))

    def save(self, context, adapter):
        ledger = context['ledger']
        return self.physical(ledger, 'checkpoint_adapter_io', lambda:
            adapter.save(context['model'], context['optimizer'], ledger.progress_record(), ledger))

    def restore(self, context, adapter):
        ledger = context['ledger']
        record = self.physical(ledger, 'checkpoint_adapter_io', lambda:
            adapter.restore(context['model'], context['optimizer']))
        ledger.install_restore(record)
        return record

    def rows_for_boundary(self, context):
        C = context['ledger'].progress_record()['C']
        # Python/NumPy draws are consumed by actual prepared prompt construction.
        rows = []
        for expected in context['prepared'][8 * C:8 * C + 8]:
            row = copy.deepcopy(expected)
            row['prompt_ids'][:2] = [13 + random.randrange(4), 17 + int(self.np.random.randint(4))]
            self.assertEqual(row, expected)
            rows.append(row)
        return rows

    def boundary(self, context, *, commit=True, rows=None):
        ledger = context['ledger']; p = ledger.progress_record()
        rows = self.rows_for_boundary(context) if rows is None else rows
        recipe = dict(context['recipe'], boundary_id=p['C'], U_before=p['U'], generation=p['generation'],
            symbolic_snapshot=p['snapshot'], start_policy_id=context['helper'].runtime.anchor_digest(context['model']))
        result = ce_update.boundary_update(context['model'], context['optimizer'], rows, recipe, ledger.operation)
        self.assertEqual(ledger.progress_record(), p)
        if commit: ledger.commit_boundary(result)
        return result

    def snapshot(self, context):
        model, optimizer = context['model'], context['optimizer']
        rng, tensors = checkpoint.streams.rng_state()
        return ({n: p.detach().clone() for n, p in model.named_parameters()},
                {n: p.detach().clone() for n, p in model.named_buffers()},
                [{key: value.detach().clone() for key, value in optimizer.state.get(p, {}).items()}
                 for p in model.parameters()], json.loads(checkpoint.canonical(rng)),
                {key: value.clone() for key, value in tensors.items()},
                {name: module.training for name, module in model.named_modules()})

    def assert_state_equal(self, expected, actual, *, rng=True):
        for index in (0, 1, 4) if rng else (0, 1):
            self.assertEqual(set(expected[index]), set(actual[index]))
            self.assertTrue(all(self.torch.equal(expected[index][key], actual[index][key]) for key in expected[index]))
        for left, right in zip(expected[2], actual[2]):
            self.assertEqual(set(left), set(right))
            self.assertTrue(all(self.torch.equal(left[key], right[key]) for key in left))
        if rng: self.assertEqual(expected[3], actual[3])
        self.assertEqual(expected[5], actual[5])

    def assert_cost_retained(self, before, after):
        for key, value in before['cumulative'].items(): self.assertGreaterEqual(after['cumulative'][key], value, key)
        for key, value in before['operation_starts'].items(): self.assertGreaterEqual(after['operation_starts'][key], value, key)

    def test_real_partial_Adam_reopen_restore_replay_retains_cost_and_clean_state(self):
        clean = self.context(self.root / 'clean'); self.seed(); self.boundary(clean)
        expected = self.snapshot(clean)
        split = self.context(self.root / 'split'); self.seed(); adapter = self.adapter(split); self.save(split, adapter)
        original = split['optimizer'].step
        def partial(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('real Adam returned internally then fixture interruption')
        with patch.object(split['optimizer'], 'step', partial), self.assertRaisesRegex(RuntimeError, 'fixture interruption') as error:
            self.boundary(split)
        self.assertTrue(error.exception.requires_restore)
        self.assertTrue(split['ledger'].requires_restore)
        self.assertEqual((split['ledger'].progress_record()['C'], split['ledger'].progress_record()['U']), (0, 0))
        self.assertEqual({int(s['step']) for s in split['optimizer'].state.values()}, {1})
        split['ledger'].end_invocation('FAILED_PARTIAL_ADAM')
        before = split['ledger'].snapshot()
        resumed = self.context(split['directory'], resume=True, shared=split['shared'])
        self.assertIsNot(resumed['ledger'], split['ledger']); self.assertIsNot(resumed['model'], split['model'])
        self.assert_cost_retained(before, resumed['ledger'].snapshot())
        record = self.restore(resumed, self.adapter(resumed))
        self.assertEqual((record['C'], record['U']), (0, 0)); self.assertGreater(record['generation'], 0)
        self.assertFalse(resumed['optimizer'].state)
        self.boundary(resumed)
        self.assert_state_equal(expected, self.snapshot(resumed))
        self.assertEqual(resumed['ledger'].snapshot()['operation_starts']['optimizer'], 2)
        self.assertEqual(resumed['ledger'].progress_record()['U'], 1)
        self.assert_cost_retained(before, resumed['ledger'].snapshot())

    def test_CE2_bitwise_actual_weights_moments_and_all_CPU_RNG_after_fresh_object_restore(self):
        clean = self.context(self.root / 'clean'); self.seed(); self.boundary(clean)
        clean['model'].lm_head.train(True); self.boundary(clean); expected = self.snapshot(clean)
        split = self.context(self.root / 'split'); self.seed(); adapter = self.adapter(split); self.save(split, adapter)
        self.boundary(split); split['model'].lm_head.train(True); self.save(split, adapter)
        saved = self.snapshot(split)
        self.boundary(split, commit=False)
        self.assertTrue(split['ledger'].requires_restore)
        split['ledger'].end_invocation('RETURNED_UNCOMMITTED_SECOND_BOUNDARY'); before = split['ledger'].snapshot()
        resumed = self.context(split['directory'], resume=True, shared=split['shared'])
        self.assert_cost_retained(before, resumed['ledger'].snapshot())
        restored = self.restore(resumed, self.adapter(resumed))
        self.assertEqual((restored['C'], restored['U']), (1, 1))
        self.assert_state_equal(saved, self.snapshot(resumed))
        self.boundary(resumed)
        self.assert_state_equal(expected, self.snapshot(resumed))
        self.assertEqual((resumed['ledger'].progress_record()['C'], resumed['ledger'].progress_record()['U']), (2, 2))
        self.assertEqual(resumed['ledger'].snapshot()['operation_starts']['optimizer'], 3)

    def test_real_ancestor_restore_proof_is_one_use_and_old_CE_result_cannot_commit(self):
        context = self.context(self.root / 'ancestor'); self.seed(); adapter = self.adapter(context)
        self.boundary(context); self.save(context, adapter); ancestor = self.snapshot(context)
        old_result = self.boundary(context); before = context['ledger'].snapshot()
        record = self.restore(context, adapter)
        self.assert_state_equal(ancestor, self.snapshot(context)); self.assert_cost_retained(before, context['ledger'].snapshot())
        current = context['ledger'].progress_record()
        with self.assertRaises(ValueError): context['ledger'].install_restore(record)
        self.assertEqual(context['ledger'].progress_record(), current)
        with self.assertRaises(ValueError): context['ledger'].commit_boundary(old_result)
        self.assertEqual(context['ledger'].progress_record(), current)
        self.assert_state_equal(ancestor, self.snapshot(context))

    def test_same_length_prompt_or_target_substitution_rejected_before_physical_forward(self):
        for change in ('prompt', 'target'):
            with self.subTest(change=change):
                context = self.context(self.root / change); self.seed(); before = self.snapshot(context)
                rows = self.rows_for_boundary(context)
                if change == 'prompt': rows[0]['prompt_ids'][0] = 12
                else: rows[0]['completion_ids'][0] = 10
                original = context['helper'].generate.teacher_forced_logprobs
                with patch.object(context['helper'].generate, 'teacher_forced_logprobs', wraps=original) as forward, \
                     patch.object(context['optimizer'], 'step', wraps=context['optimizer'].step) as step, self.assertRaisesRegex(ValueError, 'census drift'):
                    self.boundary(context, rows=rows)
                forward.assert_not_called(); step.assert_not_called()
                self.assert_state_equal(before, self.snapshot(context), rng=False)
                self.assertTrue(all(not module.training for module in context['model'].modules()))
                self.assertFalse(context['optimizer'].state); self.assertTrue(context['ledger'].requires_restore)
                self.assertEqual(context['ledger'].progress_record()['C'], 0)

    def test_exhausted_forward_and_Adam_caps_admit_no_next_physical_primitive(self):
        context = self.context(self.root / 'forward', limits={'operation_starts': {'loss_forward': 1}})
        self.seed(); before = self.snapshot(context)
        original = context['helper'].generate.teacher_forced_logprobs
        with patch.object(context['helper'].generate, 'teacher_forced_logprobs', wraps=original) as forward, \
             patch.object(context['optimizer'], 'step', wraps=context['optimizer'].step) as step, self.assertRaisesRegex(ValueError, 'ceiling'):
            self.boundary(context)
        self.assertEqual(forward.call_count, 1); step.assert_not_called()
        self.assertEqual(context['ledger'].snapshot()['operation_starts']['loss_forward'], 1)
        self.assert_state_equal(before, self.snapshot(context), rng=False)
        self.assertTrue(all(not module.training for module in context['model'].modules()))
        self.assertFalse(context['optimizer'].state); self.assertTrue(context['ledger'].requires_restore)
        context = self.context(self.root / 'Adam', limits={'operation_starts': {'optimizer': 1}})
        self.seed(); self.boundary(context); before = self.snapshot(context)
        with patch.object(context['optimizer'], 'step', wraps=context['optimizer'].step) as step, self.assertRaisesRegex(ValueError, 'ceiling'):
            self.boundary(context)
        step.assert_not_called(); self.assert_state_equal(before, self.snapshot(context), rng=False)
        self.assertTrue(all(not module.training for module in context['model'].modules()))
        self.assertEqual(context['ledger'].progress_record()['U'], 1)
        self.assertEqual(context['ledger'].snapshot()['operation_starts']['optimizer'], 1)
        self.assertTrue(context['ledger'].requires_restore)

    def test_permanent_unknown_metadata_reservations_block_CP_write_before_file_open(self):
        context = self.context(self.root / 'metadata', limits={'metadata_reserved_bytes': 64 * 1024 ** 2})
        self.seed(); adapter = self.adapter(context); self.save(context, adapter)
        ledger = context['ledger']; cap = context['shared'][3]['metadata_reserved_bytes']; ordinal = 0
        while cap - ledger.snapshot()['cumulative']['metadata_reserved_bytes'] >= 2 * 1024 ** 2:
            ledger.reserve_metadata('checkpoint_json', 1024 ** 2, write_id='unwritten-%d' % ordinal,
                path=adapter.directory / ('.unknown-%d.json.partial' % ordinal))
            ordinal += 1
        spent = ledger.snapshot()['cumulative']['metadata_reserved_bytes']; self.assertGreater(ordinal, 0)
        child_opens = []; original_open = Path.open; original_reserve = adapter.metadata_reserve; admissions = []
        def observed_open(path, mode='r', *args, **kwargs):
            if Path(path).resolve().parent == adapter.directory and any(value in mode for value in ('w', 'a', 'x')):
                child_opens.append(str(path))
            return original_open(path, mode, *args, **kwargs)
        def reserve(kind, amount, **fields):
            admissions.append((kind, amount)); return original_reserve(kind, amount, **fields)
        target = adapter.directory / 'DENIED.json'
        with patch.object(adapter, 'metadata_reserve', side_effect=reserve), patch.object(Path, 'open', observed_open), \
             self.assertRaisesRegex(ValueError, 'metadata_reserved_bytes'):
            adapter._publish_json(target, {'fixture': 'x' * (3 * 1024 ** 2)})
        self.assertEqual(len(admissions), 1); self.assertFalse(child_opens); self.assertFalse(target.exists())
        self.assertFalse(list(adapter.directory.glob('.DENIED.json.*.partial')))
        self.assertGreaterEqual(ledger.snapshot()['cumulative']['metadata_reserved_bytes'], spent)


if __name__ == '__main__': unittest.main()
