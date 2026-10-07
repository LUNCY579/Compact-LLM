"""Independent CPU review: three envelopes, causal saved evidence, closed native scope."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import hashlib, json, sys, time, unittest

HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parent / 'long_guard_collection_preparation_work'
PINNED = ('long_guard_collection.py', 'test_long_guard_collection.py')
module = ModuleType('independent_source_authored_fixtures')
module.__file__ = str(AUTHOR / PINNED[1])
exec(compile((AUTHOR / PINNED[1]).read_bytes(), module.__file__, 'exec'), module.__dict__)
Base = module.LongControls

class IndependentControls(Base):
    def test_three_envelopes_preserve_packet_indices_and_actual_dose(self):
        source = self.make_source()
        plan = self.L.freeze_plan(source, self.component.validate_row)
        self.assertEqual(sorted(plan['capture_to_source']), list(range(2304)))
        for ordinal, tokens in ((1, (151643,)), (13, (9, 151645)), (25, (9, 10, 151645))):
            with self.subTest(ordinal=ordinal):
                envelope, saved, model = self.collect(source, plan, ordinal, tokens=tokens)
                checked = self.L.audit_saved_batch(source, plan, ordinal, saved,
                    component=self.component, fixture_only=True)
                selected = plan['batches'][ordinal - 1]
                self.assertEqual(checked['attempted_forwards'], 64 * len(tokens))
                self.assertEqual(checked['input_positions'],
                    64 * sum(selected['context_cap'] - 32 + i for i in range(len(tokens))))
                self.assertEqual(len(model.calls), checked['attempted_forwards'])
                self.assertEqual([r['source_index'] for r in checked['rows']], selected['source_indices'])
                for joined in checked['rows']:
                    self.assertEqual(joined['raw']['source_packet_sha256'], selected['source_packet_sha256'])
                    self.assertFalse(joined['raw']['truncated'])
                self.assertFalse(envelope['native_PASS'])

    def test_rebound_return_before_attempt_journal_is_rejected(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        _, saved, _ = self.collect(source, plan, 1, tokens=(151645,))
        path = Path(saved['events']['path']); original = path.read_bytes()
        events = [json.loads(line) for line in original.splitlines()]
        self.assertEqual([events[i]['event'] for i in (0, 1)], ['attempt', 'returned'])
        events[0], events[1] = events[1], events[0]
        raw = b''.join((json.dumps(e, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii') for e in events)
        path.write_bytes(raw)
        altered = deepcopy(saved)
        altered['events'] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            with self.assertRaisesRegex(ValueError, 'journal|causal|event'):
                self.L.audit_saved_batch(source, plan, 1, altered, component=self.component, fixture_only=True)
        finally: path.write_bytes(original)

    def test_whole_source_and_plan_mutations_refuse_before_forward(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        altered = deepcopy(source)
        altered['rows'][0], altered['rows'][4] = altered['rows'][4], altered['rows'][0]
        altered['metadata'][0], altered['metadata'][4] = altered['metadata'][4], altered['metadata'][0]
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.L.freeze_plan(altered, self.component.validate_row)
        altered = deepcopy(source)
        altered['declared_domains'].append({'domain_id': 'CPU_EMPTY', 'class': 'long_context', 'row_ids': [], 'component_ids': []})
        with self.assertRaises(ValueError): self.L.freeze_plan(altered, self.component.validate_row)
        altered_plan = deepcopy(plan); altered_plan['batches'][0]['source_indices'][1] = altered_plan['batches'][0]['source_indices'][0]
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            self.L.check_plan(source, altered_plan, self.component.validate_row)

    def test_native_audit_refuses_before_selected_reads(self):
        with self.assertRaisesRegex(ValueError, 'native|authored|fixture|CPU'):
            self.L.audit_saved_batch({}, {}, 1, {}, component=self.component, fixture_only=False)
        with self.assertRaisesRegex(ValueError, 'native|authored|fixture|CPU'):
            self.L.audit_saved_population({}, {}, [], component=self.component, fixture_only=False)

    def test_typed_zero_logprob_field_rebound_is_rejected(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        envelope, saved, _ = self.collect(source, plan, 13, tokens=(151645,))
        raw = deepcopy(envelope['collection']['rows'][0])
        raw['selected_logprobs'][0] = 0.0; raw['trace'][0]['logprob'] = False
        with self.assertRaisesRegex(ValueError, 'logprob|lineage|typed'):
            self.L._check_raw(source['rows'][plan['batches'][12]['source_indices'][0]], raw,
                plan['batches'][12], fixture_only=True)

    def test_synchronous_token_sink_failure_stops_next_forward(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        directory = self.root / 'sink_fault'; directory.mkdir()
        torch = self.mock['FakeTorch']()
        model = self.mock['LongModel'](torch, tokens=(9, 151645), prompt_length=992)
        decoder = self.mock['Decoder'](); journal = self.Journal(directory)
        original = journal.evidence; seen = []
        def evidence(name, value):
            seen.append(name)
            if name == 'raw_row0001_token01': raise OSError('CPU authored token sink denied')
            return original(name, value)
        journal.evidence = evidence
        try:
            with self.assertRaisesRegex(OSError, 'token sink denied'):
                self.L._CPU_collect(source, plan, 1, capability=self.capability,
                    torch=torch, model=model, decoder=decoder, journal=journal,
                    guard=lambda stage: None, device='cpu', component=self.component)
        finally: journal.stream.close()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(decoder.calls, [])
        self.assertIn('inference_failure_0001', seen)
        self.assertFalse((directory / 'stage_long_guard_batch.json').exists())

    def test_rebound_partial_boolean_zero_counter_is_rejected(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        _, saved, _ = self.collect(source, plan, 1, tokens=(151645,))
        altered = deepcopy(saved); path = Path(saved['tokens'][0][0]['path'])
        original = path.read_bytes(); partial = json.loads(original)
        partial['costs']['input_positions']['failed'] = False
        path.write_bytes((json.dumps(partial, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii'))
        altered['tokens'][0][0] = self.ref(path)
        try:
            with self.assertRaisesRegex(ValueError, 'partial|cost|typed'):
                self.L.audit_saved_batch(source, plan, 1, altered, component=self.component, fixture_only=True)
        finally: path.write_bytes(original)

    def test_rebound_duplicate_journal_key_is_rejected(self):
        source = self.make_source(); plan = self.L.freeze_plan(source, self.component.validate_row)
        _, saved, _ = self.collect(source, plan, 1, tokens=(151645,))
        path = Path(saved['events']['path']); original = path.read_bytes()
        first, remainder = original.split(b'\n', 1)
        raw = b'{"event":"ignored",' + first[1:] + b'\n' + remainder
        path.write_bytes(raw); altered = deepcopy(saved)
        altered['events'] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            with self.assertRaisesRegex(ValueError, 'duplicate|journal|JSON'):
                self.L.audit_saved_batch(source, plan, 1, altered, component=self.component, fixture_only=True)
        finally: path.write_bytes(original)
        raw = original[:-1]
        self.assertTrue(original.endswith(b'\n')); path.write_bytes(raw)
        altered['events'] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            with self.assertRaisesRegex(ValueError, 'newline|journal|record'):
                self.L.audit_saved_batch(source, plan, 1, altered, component=self.component, fixture_only=True)
        finally: path.write_bytes(original)

if __name__ == '__main__':
    before = {name: hashlib.sha256((AUTHOR / name).read_bytes()).hexdigest() for name in PINNED}
    started = time.monotonic()
    names = [name for name in IndependentControls.__dict__ if name.startswith('test_')]
    suite = unittest.TestSuite(IndependentControls(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {name: hashlib.sha256((AUTHOR / name).read_bytes()).hexdigest() for name in PINNED}
    receipt = {'kind': 'INDEPENDENT_LONG_GUARD_FOCUSED_CPU_REVIEW', 'tests': result.testsRun,
        'success': result.wasSuccessful(), 'seconds': time.monotonic() - started,
        'source_before': before, 'source_after': after, 'source_unchanged': before == after,
        'read_paths_and_counts': dict(Counter(IndependentControls.reads)),
        'forbidden_import_attempts': IndependentControls.imports,
        'intentional_real_path_denials': len(IndependentControls.denials),
        'fixture_only': True, 'native_PASS': False, 'scoring_authorized': False,
        'limits': ['Authored fixture producers reused with independent adversarial expectations.',
            'Three representative cap-homogeneous batches replayed, not an independent all36 replay.',
            'No native model, data, tokenizer or API consumed; native qualification and decoder binding remain absent.']}
    receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
    destination = HERE / 'INDEPENDENT_CONTROLS_001.json'
    with destination.open('x', encoding='utf-8') as stream: json.dump(receipt, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'path': str(destination), 'tests': result.testsRun, 'success': result.wasSuccessful(), 'source_unchanged': before == after, 'sha256': receipt['sha256']}))
    raise SystemExit(0 if result.wasSuccessful() and before == after else 1)
