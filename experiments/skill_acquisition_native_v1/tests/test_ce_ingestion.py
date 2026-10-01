"""Pure full-stream ingestion controls; fake rendering is explicitly synthetic.

No framework, pretrained tokenizer, causal model or native scope is executed.
"""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_ingestion as ing, ce_plan, data
from experiments.maxrl_composable_skills_v2 import contracts, oracle


def fake_render(tokenizer, row):
    prompt = [9] * (3 + row['line'] % 3)
    completion = list(row['target'].encode('ascii')) + ([41] * 3 if row['line'] % 3 == 1 else []) + [151645]
    return {'id': row['id'], 'prompt_ids': prompt, 'completion_ids': completion,
            'render_sha256': hashlib.sha256(('fixture-render/' + row['id']).encode()).hexdigest(),
            'target_sha256': row['target_sha256']}


class PureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='ce-ingestion-pure-')
        cls.root = Path(cls.temporary.name); cls.dataset = cls.root / 'data'; cls.out = cls.root / 'teacher'
        for path in (cls.dataset / 'manifests', cls.dataset / 'canonical', cls.dataset / 'policy', cls.out): path.mkdir(parents=True)
        ids = ce_plan._order(['row-%04d' % i for i in range(3000)])
        spec = {'initial': [0, 1, 2, 3, 4, 5], 'lookup': {'links': {'a%d' % i: i for i in range(6)},
                                                           'requested': ['a%d' % i for i in range(6)]}}
        prompt = contracts.render(spec)
        cls.rows = [{'id': rid, 'parent_id': 'parent/' + rid, 'prompt': prompt, 'prompt_hash': ing.digest(prompt),
                     'role': 'train', 'tag': 'core', 'kind': 'composition', 'skills': ['R'],
                     'data_source': 'rineya_maxrl_comp_v2', 'verifier': {'version': contracts.VERSION, 'kind': 'composition', 'spec': spec}}
                    for rid in ids]
        atomic = {'name': 'atomic_R', 'rows': 3000, 'ids': ids}
        files = {}
        for name, raw in (('manifests/atomic_R.json', ing.canonical(atomic) + b'\n'),
                          (data.TRAIN_FILE, b''.join(ing.canonical(row) + b'\n' for row in cls.rows)),
                          (data.POLICY_FILE, b''.join(ing.canonical({'id': row['id'], 'messages': row['prompt']}) + b'\n' for row in cls.rows))):
            (cls.dataset / name).write_bytes(raw); files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        manifest_raw = ing.canonical({'files': files}) + b'\n'; (cls.dataset / 'manifest.json').write_bytes(manifest_raw)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        _, _, source = data.load_atomic_train(cls.dataset, ('R',), expected_manifest=manifest_sha)
        cls.preparation = {'schema': 'R-ORACLE-SUPERVISION-PREPARATION-1', 'approved': False,
            'training_authorized': False, 'rows': 3000, 'parent_ids': 3000, 'source_sha256': 'd' * 64,
            'order_rule': ce_plan.ORDER, 'permutation_rule': [ce_plan.PERMUTE, 1301, 'row_id', 0, 'index'],
            'data': source, 'atomic_R_sha256': ing.digest(atomic), 'targets': {}, 'census': {}}
        cls.target_records = {}
        for arm in ce_plan.ARMS:
            records = []
            for i, row in enumerate(cls.rows):
                target = oracle.answers(spec, limit=1)[0]
                if arm == 'permuted':
                    answer = sorted(list(range(6)), key=lambda item: (ing.digest([ce_plan.PERMUTE, 1301, row['id'], 0, item]), item))
                    target = ing.canonical({'answer': answer}).decode('ascii')
                records.append({'id': row['id'], 'parent_id': row['parent_id'], 'line': i + 1, 'logical_exposure': 0,
                    'prompt': row['prompt'], 'messages': row['prompt'] + [{'role': 'assistant', 'content': target}],
                    'target': target, 'target_sha256': hashlib.sha256(target.encode()).hexdigest()})
            cls.target_records[arm] = records
            path = cls.out / (arm + '.messages.jsonl'); raw = b''.join(ing.canonical(r) + b'\n' for r in records); path.write_bytes(raw)
            cls.preparation['targets'][arm] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'rows': 3000, 'ordered_target_digest': ing.digest([[r['id'], r['target_sha256']] for r in records])}
            rendered = [fake_render('fixture', row) for row in records]
            Ps = [len(r['prompt_ids']) for r in rendered]; Cs = [len(r['completion_ids']) for r in rendered]
            cls.preparation['census'][arm] = {'rows_rendered': 3000, 'eos_id': 151645, 'max_target_tokens': 128,
                'tokenizer_receipt_sha256': 'e' * 64, 'prompt_tokens': {'min': min(Ps), 'max': max(Ps), 'total': sum(Ps)},
                'labelled_tokens': {'min': min(Cs), 'max': max(Cs), 'total': sum(Cs)}, 'total_prompt_tokens': sum(Ps),
                'total_supervised_tokens': sum(Cs), 'total_sequence_positions': sum(Ps) + sum(Cs),
                'max_sequence_positions': max(p + c for p, c in zip(Ps, Cs)),
                'prompt_render_sha256': ing.digest([r['render_sha256'] for r in rendered]),
                'sequence_ids_sha256': ing.digest([r['prompt_ids'] + r['completion_ids'] for r in rendered])}
        cls.prep_path = cls.out / 'PREPARATION.json'; cls.prep_path.write_bytes(ing.canonical(cls.preparation) + b'\n')
        cls.prep_ref = {'path': str(cls.prep_path), 'sha256': hashlib.sha256(cls.prep_path.read_bytes()).hexdigest()}
        cls.plan_refs = {}
        for arm in ce_plan.ARMS:
            plan = ce_plan.build_proposal(cls.prep_path, cls.prep_ref['sha256'], arm)
            path = cls.out / (arm + '.plan.json'); path.write_bytes(ing.canonical(plan) + b'\n')
            cls.plan_refs[arm] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()
    def setUp(self): self.events = []; self.gates = []; self.admissions = []
    def admit(self, stage, refs):
        self.admissions.append((stage, copy.deepcopy(refs)))
        return {'schema': ing.ADMISSION_SCHEMA, 'mode': 'synthetic_fixture', 'admitted': True,
            'phase': 'qualification', 'arm': refs['arm'], 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': 'c' * 64, 'source_sha256': 'd' * 64, 'runtime_sha256': 'e' * 64}
    def operation(self, kind, phase, **fields): self.events.append((kind, phase, fields))
    def verify(self, arm='canonical'):
        return ing.verify_sources(self.prep_ref, self.plan_refs[arm], arm, self.admit,
                                  operation=self.operation, gate=self.gates.append)

    def test_unapproved_scope_denied_before_path_io_or_framework(self):
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        class PathTrap:
            def __fspath__(self): raise AssertionError('path touched before denial')
        with patch.object(Path, 'open', side_effect=AssertionError('IO before denial')):
            with self.assertRaisesRegex(ValueError, 'unapproved'):
                ing.verify_sources({'path': PathTrap()}, {}, 'canonical', lambda *a: {'admitted': False},
                                   operation=self.operation, gate=self.gates.append)
        self.assertFalse(self.events); self.assertFalse(self.gates)
        self.assertEqual(set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}, before)

    def test_forged_registered_json_receipt_is_not_an_opaque_stage_permit(self):
        def forged(stage, refs):
            value = self.admit(stage, refs); value['mode'] = 'registered_scope'
            value.update(approved=True, training_authorized=True); return value
        with patch.object(Path, 'open', side_effect=AssertionError('IO before capability gate')):
            with self.assertRaisesRegex(ValueError, 'opaque outer stage capability'):
                ing.verify_sources({'path': 'invalid'}, {}, 'canonical', forged,
                                   operation=self.operation, gate=self.gates.append)
        self.assertFalse(self.events); self.assertFalse(self.gates)

    def test_full_sources_actual_census_and_only_eight_copied_cached_rows(self):
        verified = self.verify(); self.assertFalse(verified.receipt['tokenizer_rendered'])
        prepared = ing.PreparedRows(verified, 'fixture', outer_admission=self.admit,
                                   operation=self.operation, gate=self.gates.append, renderer=fake_render)
        self.assertEqual(prepared.census['rows'], 3000); self.assertEqual(prepared.cached_rows, 0)
        self.assertEqual(prepared.census['sequence_ids_sha256'], self.preparation['census']['canonical']['sequence_ids_sha256'])
        self.assertLess(prepared.census['max_sequence_positions'],
                        prepared.census['max_prompt_tokens'] + prepared.census['max_labeled_tokens'])
        self.assertFalse(set(prepared.witnesses['max_prompt_tokens']['row_ids'])
                         & set(prepared.witnesses['max_labeled_tokens']['row_ids']))
        for witness in prepared.witnesses.values():
            self.assertTrue(witness['row_ids'])
        first = prepared.iter_boundary(0); self.assertEqual(prepared.cached_rows, 8)
        first[0]['prompt_ids'][0] = 123
        self.assertNotEqual(first, prepared.iter_boundary(0))
        second = prepared.iter_boundary(1); self.assertEqual(len(second), 8); self.assertEqual(prepared.cached_rows, 8)
        self.assertEqual([r['id'] for r in second], [r['id'] for r in self.target_records['canonical'][8:16]])
        self.assertFalse(prepared.receipt['training_authorized']); self.assertFalse(prepared.receipt['native_qualified'])

    def test_same_length_teacher_substitution_refused_by_oracle_even_if_row_rehashed(self):
        reference = self.preparation['targets']['canonical']; path = Path(reference['path']); original = path.read_bytes()
        lines = original.splitlines(keepends=True); record = ce_plan._json(lines[0])
        record['target'] = ing.canonical({'answer': [0, 1, 2, 3, 4, 4]}).decode()
        record['target_sha256'] = hashlib.sha256(record['target'].encode()).hexdigest()
        record['messages'][-1]['content'] = record['target']; lines[0] = ing.canonical(record) + b'\n'
        try:
            path.write_bytes(b''.join(lines))
            with self.assertRaisesRegex(ValueError, 'oracle policy'): self.verify()
        finally: path.write_bytes(original)

    def test_full_unselected_policy_and_canonical_stream_tampering_refused(self):
        for relative in (data.TRAIN_FILE, data.POLICY_FILE):
            path = self.dataset / relative; original = path.read_bytes()
            try:
                path.write_bytes(original + b'{}\n')
                with self.subTest(file=relative), self.assertRaises(ValueError): self.verify()
            finally: path.write_bytes(original)

    def test_tokenizer_denial_and_native_injection_refused_before_renderer(self):
        verified = self.verify()
        with self.assertRaisesRegex(ValueError, 'unapproved'):
            ing.PreparedRows(verified, 'fixture', outer_admission=lambda *a: {'admitted': False},
                             operation=self.operation, gate=self.gates.append, renderer=lambda *a: self.fail('render before denial'))
        native = copy.deepcopy(verified); native._admission['mode'] = 'registered_scope'
        def admitted(stage, refs):
            value = self.admit(stage, refs); value['mode'] = 'registered_scope'; return value
        with self.assertRaisesRegex(ValueError, 'opaque outer stage capability'):
            ing.PreparedRows(native, 'fixture', outer_admission=admitted, operation=self.operation,
                             gate=self.gates.append, renderer=lambda *a: self.fail('native injected renderer'))

    def test_permuted_policy_and_same_length_render_change_bind_different_contents(self):
        verified = self.verify('permuted')
        prepared = ing.PreparedRows(verified, 'fixture', outer_admission=self.admit,
                                   operation=self.operation, gate=self.gates.append, renderer=fake_render)
        self.assertEqual(prepared.receipt['arm'], 'permuted')
        def changed(tokenizer, row):
            value = fake_render(tokenizer, row); value['prompt_ids'][0] += 1; return value
        prepared.renderer = changed
        with self.assertRaisesRegex(ValueError, 'changed after full census'): prepared.iter_boundary(0)
        self.assertEqual(prepared.cached_rows, 0)

    def test_same_length_target_line_change_after_verification_refused_before_render(self):
        verified = self.verify(); row_id = self.target_records['canonical'][0]['id']
        path = Path(self.preparation['targets']['canonical']['path']); original = path.read_bytes()
        try:
            mutated = bytearray(original); mutated[verified._index[row_id]['offset'] + 10] ^= 1; path.write_bytes(mutated)
            with self.assertRaisesRegex(ValueError, 'changed after source verification'):
                verified._target(row_id, self.operation, self.gates.append)
        finally: path.write_bytes(original)

    def test_producer_source_budget_denial_precedes_source_read_hash_and_receipt(self):
        source = Path(ing.__file__)
        opened = []
        original_open = Path.open
        def tracked_open(path, *args, **kwargs):
            if path == source: opened.append(path)
            return original_open(path, *args, **kwargs)
        def deny_source(kind, phase, **fields):
            self.events.append((kind, phase, fields))
            if phase == 'started' and fields.get('path') == str(source):
                self.assertEqual(fields['read_bytes_upper'], source.stat().st_size)
                self.assertEqual(fields['identity_hash_bytes_upper'], source.stat().st_size)
                self.assertEqual(fields['identity_hash_attempts_upper'], 1)
                raise ValueError('producer source identity budget exhausted')
        with patch.object(Path, 'open', tracked_open):
            with self.assertRaisesRegex(ValueError, 'producer source identity budget exhausted'):
                ing.verify_sources(self.prep_ref, self.plan_refs['canonical'], 'canonical', self.admit,
                                   operation=deny_source, gate=self.gates.append)
        self.assertFalse(opened)
        source_events = [(phase, fields) for _, phase, fields in self.events if fields.get('path') == str(source)]
        self.assertEqual([phase for phase, _ in source_events], ['started'])

    def test_failed_accounting_preserves_original_work_error_with_cause(self):
        observed = []
        primary = OSError('original source read failed')
        accounting = RuntimeError('failed observer unavailable')
        def operation(kind, phase, **fields):
            observed.append((kind, phase, fields))
            if phase == 'failed': raise accounting
        def work(): raise primary
        with self.assertRaises(OSError) as caught:
            ing._io(operation, lambda stage: None, 'ingestion_sources', {}, work)
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, accounting)
        self.assertEqual(caught.exception.accounting_failure,
                         {'type': 'RuntimeError', 'message': 'failed observer unavailable'})
        self.assertEqual([(kind, phase) for kind, phase, _ in observed],
                         [('input_io', 'started'), ('input_io', 'failed')])
        self.assertEqual(observed[0][2]['operation_id'], observed[1][2]['operation_id'])
        self.assertEqual(observed[1][2]['error_type'], 'OSError')
        self.assertEqual(observed[1][2]['error'], str(primary))

    def test_started_observation_precedes_work_and_failure_retained(self):
        observed = []
        def operation(kind, phase, **fields): observed.append((kind, phase))
        def work():
            self.assertEqual(observed, [('input_io', 'started')]); raise OSError('pure read failure')
        with self.assertRaisesRegex(OSError, 'pure read'):
            ing._io(operation, lambda stage: None, 'ingestion_sources', {}, work)
        self.assertEqual(observed, [('input_io', 'started'), ('input_io', 'failed')])
