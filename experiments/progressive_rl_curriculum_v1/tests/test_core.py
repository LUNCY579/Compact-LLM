from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from experiments.progressive_rl_curriculum_v1.backends import DemoBackend
from experiments.progressive_rl_curriculum_v1.cli import demo_inputs, main
from experiments.progressive_rl_curriculum_v1.curriculum import TaskStats, batch, gate, paired_interval, unlocked
from experiments.progressive_rl_curriculum_v1.data import Catalog, import_mimo, load_catalog, verified_demonstrations, write_catalog
from experiments.progressive_rl_curriculum_v1.engine import Engine
from experiments.progressive_rl_curriculum_v1.evidence import BudgetExhausted, Ledger, Store
from experiments.progressive_rl_curriculum_v1.objectives import advantages
from experiments.progressive_rl_curriculum_v1.schema import Episode, Gates, Limits, Plan, Task, digest


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.catalog = demo_inputs()

    def test_roundtrip_plan(self):
        self.assertEqual(self.plan, Plan.from_dict(json.loads(json.dumps(asdict(self.plan)))))

    def test_unknown_field_rejected(self):
        value = asdict(self.plan)
        value['unvalidated_override'] = True
        with self.assertRaises(TypeError):
            Plan.from_dict(value)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.plan, learning_rate=float('nan'))

    def test_context_overflow_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.plan, limits=replace(self.plan.limits, max_context_tokens=1))

    def test_fraction_cannot_decrease(self):
        with self.assertRaises(ValueError):
            replace(self.plan, stages=tuple(reversed(self.plan.stages)))

    def test_mixture_must_sum_one(self):
        with self.assertRaises(ValueError):
            replace(self.plan.stages[0], source_weights={'arithmetic': .9})

    def test_exploration_cannot_be_removed(self):
        with self.assertRaises(ValueError):
            replace(self.plan.stages[0], band_weights={'frontier': 1.0})

    def test_group_size_one_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.plan.stages[0], group_size=1)

    def test_sealed_and_test_splits_rejected(self):
        for split in ('test', 'sealed', 'challenge', 'dev', 'TRAIN'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                replace(self.catalog.tasks[0], split=split)

    def test_gold_not_in_policy_view(self):
        self.assertEqual(set(asdict(self.catalog.tasks[0].public())), {'key', 'text'})

    def test_empty_gold_missing_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.catalog.tasks[0], answer=None)

    def test_invalid_episode_rejected(self):
        with self.assertRaises(ValueError):
            Episode('x', '', (1,), (2,), (float('nan'),), 0)
        with self.assertRaises(ValueError):
            Episode('x', '', (1,), (2,), (), 0)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.catalog = demo_inputs()
        self.task = self.catalog.tasks[0]

    def test_duplicate_id(self):
        with self.assertRaises(ValueError):
            Catalog([self.task, self.task])

    def test_parent_split_leak(self):
        with self.assertRaises(ValueError):
            Catalog([self.task, replace(self.task, id='other', prompt='other', split='calibration')])

    def test_normalized_cross_source_leak(self):
        other = replace(self.task, source='other', id='other', group='other',
                        prompt='  ' + self.task.prompt.replace(' ', '  '), split='calibration')
        with self.assertRaises(ValueError):
            Catalog([self.task, other])

    def test_multiple_revisions_rejected(self):
        with self.assertRaises(ValueError):
            Catalog([self.task, replace(self.task, id='other', revision='changed')])

    def test_order_invariant_fingerprint(self):
        self.assertEqual(self.catalog.fingerprint, Catalog(reversed(self.catalog.tasks)).fingerprint)

    def test_catalog_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tasks.jsonl'
            write_catalog(self.catalog, path)
            self.assertEqual(self.catalog.fingerprint, load_catalog(path).fingerprint)

    def test_external_never_matches_text_runtime(self):
        task = replace(self.task, verifier='external', requirements=('text', 'sandbox'))
        self.assertEqual(Catalog([task]).subset('train', frozenset({'text'})), [])

    def test_truncated_correct_text_not_success(self):
        self.assertEqual(self.catalog.verify(self.task.key, self.task.answer, terminated=False), 0)

    def test_mimo_requires_pinned_revision(self):
        with self.assertRaises(ValueError):
            import_mimo([], revision='main', split='train', domain='code', group_field='parent')

    def test_mimo_preserves_environment_without_executing(self):
        rows = [{'id': 'fixture', 'parent': 'repo1', 'prompt': [{'role': 'user', 'content': 'Fix this.'}],
                 'reward_config': {'do_not_execute': 'arbitrary command'}}]
        result = import_mimo(rows, revision='a'*40, split='train', domain='code', group_field='parent')
        self.assertEqual(result.tasks[0].metadata['environment_spec'], rows[0])
        self.assertEqual(result.subset('train', frozenset({'text'})), [])
        self.assertIsNone(result.tasks[0].answer)

    def test_mimo_missing_parent_rejected(self):
        with self.assertRaises(ValueError):
            import_mimo([{'prompt': 'hi'}], revision='a'*40, split='train', domain='code', group_field='parent')

    def test_mimo_multimodal_rejected(self):
        with self.assertRaises(ValueError):
            import_mimo([{'parent': 'x', 'prompt': [{'role': 'user', 'content': [{'image': 'x'}]}]}],
                        revision='a'*40, split='train', domain='web', group_field='parent')

    def test_teacher_verified_and_deduped(self):
        row = {'task_key': self.task.key, 'response': self.task.answer, 'teacher': 'teacher@sha', 'provenance': 'receipt'}
        self.assertEqual(len(verified_demonstrations(self.catalog, [row, row])), 1)
        with self.assertRaises(ValueError):
            verified_demonstrations(self.catalog, [{**row, 'response': 'wrong'}])

    def test_teacher_holdout_rejected(self):
        task = self.catalog.subset('calibration', frozenset({'text'}))[0]
        with self.assertRaises(ValueError):
            verified_demonstrations(self.catalog, [{'task_key': task.key, 'response': task.answer,
                                                  'teacher': 'teacher', 'provenance': 'receipt'}])


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.catalog = demo_inputs()
        self.train = self.catalog.subset('train', frozenset({'text'}))
        self.replay = self.catalog.subset('retention_train', frozenset({'text'}))

    def test_unlocks_nested_and_ceiling_exact(self):
        previous = set()
        for stage in self.plan.stages:
            pool = unlocked(self.train, stage, self.plan.seed)
            current = {t.group for t in pool}
            self.assertTrue(previous <= current)
            self.assertLessEqual(len(current), int(len(self.train) * stage.fraction))
            previous = current

    def test_unlock_answer_and_order_unaware(self):
        a = [t.key for t in unlocked(self.train, self.plan.stages[0], 1)]
        changed = [replace(t, answer='changed') for t in reversed(self.train)]
        self.assertEqual(a, [t.key for t in unlocked(changed, self.plan.stages[0], 1)])

    def test_four_failures_remain_unknown(self):
        self.assertEqual(TaskStats(4, 0, 0).band(0, self.plan.gates), 'unknown')

    def test_stale_stats_reprobe(self):
        self.assertEqual(TaskStats(100, 100, 0).band(100, self.plan.gates), 'unknown')

    def test_difficulty_bands(self):
        for wins, expected in ((0, 'hard'), (40, 'frontier'), (70, 'easy'), (99, 'mastered')):
            with self.subTest(expected=expected):
                self.assertEqual(TaskStats(100, wins, 1).band(1, self.plan.gates), expected)

    def test_deterministic_mixture_and_replay(self):
        args = (self.train, self.replay, {}, self.plan.stages[0], self.plan.gates, 0, 42)
        a, report = batch(*args)
        self.assertEqual(a, batch(*args)[0])
        self.assertEqual(len(a), self.plan.stages[0].batch_tasks)
        self.assertEqual(sum(t.split == 'retention_train' for t in a), report['replay_tasks'])
        self.assertTrue(all(t.split in {'train', 'retention_train'} for t in a))

    def test_empty_pool_is_not_filled_from_holdout(self):
        with self.assertRaises(ValueError):
            batch([], self.replay, {}, self.plan.stages[0], self.plan.gates, 0, 42)

    def test_constant_groups_zero_advantage(self):
        self.assertEqual(advantages([0, 0, 0, 0]), [0]*4)
        self.assertEqual(advantages([1, 1, 1, 1]), [0]*4)

    def test_mixed_advantages_centered(self):
        a = advantages([0, 0, 1, 0])
        self.assertAlmostEqual(sum(a), 0)
        self.assertGreater(a[2], 0)

    def test_invalid_rewards_rejected(self):
        for values in ([1], [0, float('nan')], [0, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                advantages(values)

    def test_parent_cluster_interval(self):
        before = {str(i): 0. for i in range(20)}
        groups = {str(i): str(i // 2) for i in range(20)}
        gates = replace(self.plan.gates, min_eval_groups=10)
        interval = paired_interval(before, {k: 1. for k in before}, groups, gates, 1)
        self.assertEqual(interval, {'mean': 1., 'lcb': 1., 'ucb': 1., 'groups': 10})

    def test_pseudo_replication_rejected(self):
        before = {str(i): 0. for i in range(20)}
        with self.assertRaises(ValueError):
            paired_interval(before, before, {k: 'same-parent' for k in before}, self.plan.gates, 1)

    def test_retention_banks_not_averaged_away(self):
        signal = {'groups': 20, 'mixed_fraction': .5, 'valid_action_rate': 1., 'truncation_rate': 0.}
        passed, reasons = gate({'lcb': .1}, {'a': {'lcb': .5}, 'b': {'lcb': -.2}}, signal, self.plan.gates)
        self.assertFalse(passed)
        self.assertIn('retention_lcb:b', reasons)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan, _ = demo_inputs()

    def test_reserve_and_settle(self):
        state = {}
        ledger = Ledger(self.plan.limits, state)
        ledger.reserve(generated_tokens=100)
        ledger.settle({'generated_tokens': 25})
        self.assertEqual(state['used']['generated_tokens'], 25)

    def test_interrupted_reservation_not_refunded(self):
        state = {}
        ledger = Ledger(self.plan.limits, state)
        ledger.reserve(generated_tokens=100)
        ledger.settle()
        self.assertEqual(state['used']['generated_tokens'], 100)

    def test_budget_overflow_rejected_without_mutation(self):
        state = {}
        ledger = Ledger(replace(self.plan.limits, generated_tokens=1), state)
        with self.assertRaises(BudgetExhausted):
            ledger.reserve(generated_tokens=2)
        self.assertEqual(state['used']['generated_tokens'], 0)

    def test_duplicate_settlement_rejected(self):
        ledger = Ledger(self.plan.limits, {})
        ledger.reserve(optimizer_steps=1)
        ledger.settle()
        with self.assertRaises(ValueError):
            ledger.settle()

    def test_actual_cost_above_reservation_rejected(self):
        ledger = Ledger(self.plan.limits, {})
        ledger.reserve(generated_tokens=1)
        with self.assertRaises(ValueError):
            ledger.settle({'generated_tokens': 2})

    def test_store_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp), {'binding': 'one'})
            name = store.commit({'hello': 'world'}, DemoBackend(self.plan))
            (Path(tmp) / name / 'state.json').write_text('{}')
            with self.assertRaises(ValueError):
                store.inspect()

    def test_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Store(Path(tmp), {'binding': 'one'}).commit({}, DemoBackend(self.plan))
            with self.assertRaises(ValueError):
                Store(Path(tmp), {'binding': 'two'}).inspect()

    def test_lock_excludes_other_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp), {})
            with store.lock():
                with self.assertRaises(RuntimeError):
                    with store.lock():
                        pass
            self.assertFalse((Path(tmp) / 'RUNNING.lock').exists())


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.catalog = demo_inputs()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def engine(self, directory=None, plan=None, allow=True, backend=None):
        plan = plan or self.plan
        return Engine(plan, self.catalog, backend or DemoBackend(plan), directory or self.directory, allow_training=allow)

    def test_qualification_never_updates(self):
        engine = self.engine(allow=False)
        result = engine.qualify()
        self.assertEqual(result['status'], 'READY')
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['budgets_used']['optimizer_steps'], 0)
        self.assertFalse(result['native_model_run'])

    def test_training_requires_authorization(self):
        with self.assertRaises(ValueError):
            self.engine(allow=False).run()

    def test_demo_promotes_after_distinct_updates(self):
        engine = self.engine()
        result = engine.run(4)
        self.assertEqual(result['stage'], 'expansion')
        events = [e for e in engine.state['events'] if e['event'] == 'EVALUATION']
        self.assertEqual([e['version'] for e in events], [2, 4])

    def test_resume_matches_uninterrupted(self):
        a = self.engine(self.directory / 'a')
        a.run(4)
        b = self.engine(self.directory / 'b')
        b.run(2)
        resumed = self.engine(self.directory / 'b')
        resumed.resume()
        resumed.run(2)
        self.assertEqual(a.backend.logit, resumed.backend.logit)
        for key in ('stage', 'stage_updates', 'stats', 'updates', 'version', 'collection'):
            self.assertEqual(a.state[key], resumed.state[key])
        for key in Ledger.KEYS:
            self.assertEqual(a.state['used'][key], resumed.state['used'][key])

    def test_duplicate_evaluation_cannot_promote(self):
        engine = self.engine()
        engine.run(2)
        with self.assertRaises(ValueError):
            engine.evaluate()

    def test_train_statistics_do_not_contain_holdout(self):
        engine = self.engine()
        engine.run(2)
        self.assertTrue(all(self.catalog.by_key[key].split == 'train' for key in engine.state['stats']))

    def test_no_training_on_blocked_mimo_only_data(self):
        external = [replace(t, requirements=('text', 'mimo:code')) for t in self.catalog.tasks]
        with self.assertRaises(ValueError):
            Engine(self.plan, Catalog(external), DemoBackend(self.plan), self.directory)

    def test_rollback_keeps_spent_budget(self):
        engine = self.engine()
        engine.run(1)
        used = dict(engine.state['used'])
        previous_version = engine.backend.version
        engine._rollback('test')
        self.assertEqual(engine.state['status'], 'ROLLED_BACK')
        self.assertEqual(engine.state['used'], used)
        self.assertGreater(engine.backend.version, previous_version)

    def test_budget_exhaustion_is_terminal(self):
        plan = replace(self.plan, limits=replace(self.plan.limits, generated_tokens=1))
        engine = self.engine(plan=plan)
        self.assertEqual(engine.run()['status'], 'EXHAUSTED')
        self.assertEqual(engine.state['updates'], 0)

    def test_interrupted_operation_restores_preupdate_state(self):
        engine = self.engine()
        engine.qualify()
        engine.ledger.reserve(optimizer_steps=1, processed_tokens=100)
        engine.store.commit(engine.state, engine.backend)
        engine.backend.logit = 99
        resumed = self.engine()
        resumed.resume()
        self.assertNotEqual(resumed.backend.logit, 99)
        self.assertEqual(resumed.state['used']['optimizer_steps'], 1)
        self.assertIsNone(resumed.state['pending'])

    def test_all_zero_bootstrap_path(self):
        backend = DemoBackend(self.plan)
        backend.logit = -100
        engine = self.engine(backend=backend)
        self.assertEqual(engine.qualify()['status'], 'BOOTSTRAP_REQUIRED')
        self.assertEqual(engine.state['updates'], 0)

    def test_bootstrap_rejects_locked_training_rows(self):
        engine = self.engine()
        engine.state['status'] = 'BOOTSTRAP_REQUIRED'
        eligible = {t.key for t in unlocked(engine.train, engine.stage, engine.plan.seed)}
        task = next(t for t in engine.train if t.key not in eligible)
        with self.assertRaises(ValueError):
            engine.bootstrap([(task, task.answer)])

    def test_replay_ce_remains_present(self):
        engine = self.engine()
        engine.run(1)
        update = next(e for e in engine.state['events'] if e['event'] == 'RL_UPDATE')
        self.assertGreater(update['metrics']['replay_rows'], 0)

    def test_plan_command_does_not_import_model_loader(self):
        config, catalog = self.directory / 'plan.json', self.directory / 'catalog.jsonl'
        config.write_text(json.dumps(asdict(self.plan)))
        write_catalog(self.catalog, catalog)
        with patch('experiments.progressive_rl_curriculum_v1.cli.local_hf_backend', side_effect=AssertionError):
            with patch('builtins.print'):
                self.assertEqual(main(['plan', '--config', str(config), '--catalog', str(catalog)]), 0)

    def test_native_train_cannot_bypass_qualification(self):
        config, catalog = self.directory / 'plan.json', self.directory / 'catalog.jsonl'
        config.write_text(json.dumps(asdict(self.plan)))
        write_catalog(self.catalog, catalog)
        args = ['train', '--config', str(config), '--catalog', str(catalog), '--run-dir', str(self.directory),
                '--model-dir', 'missing', '--allow-model-load', '--allow-training']
        with patch('experiments.progressive_rl_curriculum_v1.cli.local_hf_backend', side_effect=AssertionError):
            with patch('builtins.print'):
                self.assertEqual(main(args), 2)

class FailureBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.catalog = demo_inputs()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def engine(self):
        return Engine(self.plan, self.catalog, DemoBackend(self.plan), self.root, allow_training=True)

    def test_committed_update_preserves_due_evaluation_on_crash(self):
        engine = self.engine()
        engine.run(1)
        # Crash after optimizer commit, before due evaluation enters its first operation.
        with patch.object(engine, 'evaluate', side_effect=RuntimeError('injected crash')):
            with self.assertRaises(RuntimeError):
                engine.step()
        resumed = self.engine()
        resumed.resume()
        self.assertEqual(resumed.state['stage_updates'], 2)
        self.assertTrue(resumed.state['evaluation_due'])
        resumed.step()
        self.assertEqual(resumed.state['updates'], 2)
        self.assertFalse(resumed.state['evaluation_due'])
        self.assertEqual(resumed.state['passes'], 1)

    def test_orphan_checkpoint_never_overwritten(self):
        store = Store(self.root, {})
        backend = DemoBackend(self.plan)
        store.commit({}, backend)
        orphan = self.root / 'checkpoint-00000001'
        orphan.mkdir()
        (orphan / 'uncommitted').write_text('keep')
        name = store.commit({}, backend)
        self.assertEqual(name, 'checkpoint-00000002')
        self.assertEqual((orphan / 'uncommitted').read_text(), 'keep')

    def test_rollback_receipt_tampering_is_detected(self):
        store = Store(self.root, {})
        backend = DemoBackend(self.plan)
        first = store.commit({}, backend)
        store.commit({}, backend)
        path = self.root / first / 'receipt.json'
        receipt = json.loads(path.read_text())
        receipt['sequence'] = 123
        path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            store.restore(backend, first)

    def test_active_plan_mutation_is_rejected(self):
        engine = self.engine()
        engine.qualify()
        engine.stage.band_weights['frontier'] = .9
        with self.assertRaises(ValueError):
            engine.step()

    def test_unknown_stats_never_negative(self):
        for stat in ((-1, 0, 0), (1, 2, 0), (1, 0, -1)):
            with self.subTest(stat=stat), self.assertRaises(ValueError):
                TaskStats(*stat)

    def test_zero_weight_does_not_revoke_data_access(self):
        stage = replace(self.plan.stages[0], source_weights={'arithmetic': 0., 'other': 1.})
        pool = self.catalog.subset('train', frozenset({'text'}))
        self.assertTrue(unlocked(pool, stage, self.plan.seed))

    def test_later_stages_cannot_remove_eligible_source(self):
        later = replace(self.plan.stages[1], source_weights={'different': 1.})
        with self.assertRaises(ValueError):
            replace(self.plan, stages=(self.plan.stages[0], later))

    def test_role_contract_rejects_id_collisions(self):
        task = self.catalog.tasks[0]
        with self.assertRaises(ValueError):
            replace(task, source='source:part')

    def test_failed_optimizer_restores_weights_but_charges_attempt(self):
        engine = self.engine()
        engine.qualify()
        original = engine.backend.logit
        def fail(*args, **kwargs):
            engine.backend.logit = 999
            raise RuntimeError('injected optimizer failure')
        with patch.object(engine.backend, 'update', side_effect=fail):
            with self.assertRaises(RuntimeError):
                engine.step()
        resumed = self.engine()
        resumed.resume()
        self.assertEqual(resumed.backend.logit, original)
        self.assertEqual(resumed.state['updates'], 0)
        self.assertEqual(resumed.state['used']['optimizer_steps'], 1)


if __name__ == '__main__':
    unittest.main()
