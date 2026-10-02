"""Synthetic token/observer controls only; never imports a real tokenizer/library."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import learning_data as data
from experiments.rineya_centralized_reasoning_system_v1 import learning_tokenization as token


class SyntheticBackend:
    """Reversible four-character chunks, explicit special IDs, NFC; not BPE."""
    def __init__(self):
        self.ids = {token.IM_START: token.START_ID, token.IM_END: token.EOS_ID, token.PAD: token.PAD_ID}
        self.texts = {value: key for key, value in self.ids.items()}
        self.next_id = 10
        self.calls = 0

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls += 1
        text = unicodedata.normalize("NFC", text)
        pieces = re.split(r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\n\n)", text)
        output = []
        for piece in pieces:
            if not piece: continue
            chunks = [piece] if piece in self.ids or piece == "\n\n" else [piece[i:i + 4] for i in range(0, len(piece), 4)]
            for chunk in chunks:
                if chunk not in self.ids:
                    self.ids[chunk] = self.next_id; self.texts[self.next_id] = chunk; self.next_id += 1
                output.append(self.ids[chunk])
        return output

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.texts[i] for i in ids)


class SyntheticObserver:
    def __init__(self, *, creation=123.5, rss=100, peak=200, available=2 * 2 ** 30, children=()):
        self.creation, self.rss, self.peak, self.available, self.child_list = creation, rss, peak, available, list(children)
        self.reads = []

    def Process(self, pid=321):
        self.reads.append(pid)
        return SimpleNamespace(create_time=lambda: self.creation, children=lambda recursive: self.child_list,
                               memory_info=lambda: SimpleNamespace(rss=self.rss, peak_wset=self.peak))

    def virtual_memory(self): return SimpleNamespace(available=self.available)


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = token.registered_pairs(data.build_train32())
        cls.pair_hash = token.digest(cls.pairs)
        cls.success = token.measure_pairs(cls.pairs, SyntheticBackend(), token.manual_render,
                                         expected_pair_sha256=cls.pair_hash, fixture=True)

    def test_registered32_public_only_user_and_exact_target_order(self):
        targets = data.training_targets(data.build_train32())
        self.assertEqual([p['case_id'] for p in self.pairs], [p['case_id'] for p in targets])
        for pair in self.pairs:
            instruction, encoded = pair['user'].split('\n', 1)
            self.assertEqual(instruction, token.INSTRUCTION)
            self.assertEqual(set(json.loads(encoded)), {'source', 'query', 'rules'})
            self.assertNotIn(pair['case_id'], pair['user']); self.assertNotIn(pair['parent_id'], pair['user'])
            self.assertEqual(set(json.loads(pair['target'])), {'status', 'value'})
        self.assertEqual(len({p['parent_id'] for p in self.pairs}), 32)

    def test_frozen_accepted_data_pairing_receipt_and_changed_anchor_reject(self):
        cfg = token.read_json(token.CONTRACT)
        self.assertEqual(token.load_pairs(cfg), self.pairs)
        self.assertEqual(cfg['ordered_pair_sha256'], self.pair_hash)
        for key in ('ordered_pair_sha256','data_acceptance_sha256'):
            changed = copy.deepcopy(cfg); changed[key] = '0' * 64
            with self.assertRaises(token.TokenizationBlocked): token.load_pairs(changed)

    def test_full_fixture_accounting_is_not_actual_measurement(self):
        result = self.success
        self.assertEqual(result['evidence_class'], 'SYNTHETIC_CPU_FIXTURE')
        self.assertEqual((result['returned_cases'], result['successful_cases'], result['failed_cases']), (32, 32, 0))
        self.assertEqual(result['actual_encode_attempts'], 184); self.assertEqual(result['tokenizer_loads'], 0)
        self.assertTrue(token.validate_measurement(result, self.pairs, fixture=True))
        self.assertFalse(result['native_eligible']); self.assertFalse(result['training_eligible'])

    def test_response_eos_once_prompt_eos_and_thinking_ignored_shift_exact(self):
        row = self.success['cases'][0]; owned = row['ownership']; p = owned['prompt_tokens']; ids = owned['input_ids']
        self.assertEqual(ids[:p].count(token.EOS_ID), 1)
        self.assertEqual(ids[p:].count(token.EOS_ID), 1)
        self.assertEqual(owned['labels'][:p], [-100] * p)
        self.assertEqual(owned['prediction_logit_positions'][0], p - 1)
        self.assertEqual(owned['prediction_logit_positions'][-1], len(ids) - 2)
        self.assertEqual(owned['shifted_loss_mask'][:p - 1], [0] * (p - 1))
        self.assertEqual(owned['full_logit_ownership_mask'][-1], 0)
        self.assertEqual(owned['supervised_tokens'], len(ids) - p)
        self.assertEqual(owned['final_EOS_prediction_prefix_positions'], len(ids) - 1)

    def test_newline_preserved_full_excluded_exactly_one_byte(self):
        row = self.success['cases'][0]
        self.assertTrue(row['full_render'].endswith(token.IM_END + '\n'))
        self.assertEqual(row['training_text'] + '\n', row['full_render'])
        self.assertNotEqual(row['full_render_sha256'], row['training_text_sha256'])
        with self.assertRaises(token.TokenizationBlocked): token.training_text('x\n\n', 'x', '')

    def test_mask_mutations_off_by_one_prompt_leak_bool_and_nonfinite_reject(self):
        base = self.success['cases'][0]['ownership']
        for mutate in (lambda r: r['labels'].__setitem__(0, r['input_ids'][0]),
                       lambda r: r['shifted_loss_mask'].__setitem__(r['prompt_tokens'] - 2, 1),
                       lambda r: r.update(supervised_tokens=True),
                       lambda r: r.update(prompt_tokens=float('nan')),
                       lambda r: r.update(post_EOS_newline_supervised=True)):
            value = copy.deepcopy(base); mutate(value)
            with self.assertRaises((token.TokenizationBlocked, ValueError)): token.validate_ownership(value)

    def test_changed_prompt_boundary_extra_eos_pad_and_fit_reject(self):
        for ids, prompt in (([9, token.EOS_ID], [8]), ([1, token.EOS_ID, token.EOS_ID], [1]),
                            ([1, token.PAD_ID, token.EOS_ID], [1]), ([1, token.START_ID, token.EOS_ID], [1]),
                            ([1] * 513 + [token.EOS_ID], [1] * 513), ([1] + [2] * 64 + [token.EOS_ID], [1]),
                            ([True, token.EOS_ID], [1])):
            with self.assertRaises(token.TokenizationBlocked): token.causal_ownership(ids, prompt)

    def test_every_oversized_case_kept_failed_not_truncated(self):
        class Large(SyntheticBackend):
            def encode(self, text, *, add_special_tokens):
                ids = super().encode(text, add_special_tokens=add_special_tokens)
                if text.startswith(token.IM_START + 'user'): return [ids[0]] * 513 + ids
                return ids
        result = token.measure_pairs(self.pairs, Large(), token.manual_render, expected_pair_sha256=self.pair_hash, fixture=True)
        self.assertEqual((result['returned_cases'], result['failed_cases'], result['actual_encode_attempts']), (32, 32, 184))
        self.assertEqual(result['known_successful_teacher_positions'], 0)
        self.assertEqual([r['case_id'] for r in result['cases']], [p['case_id'] for p in self.pairs])

    def test_failed_control_accounts_all32_zero_case_calls(self):
        class Failing(SyntheticBackend):
            def encode(self, text, **kwargs): raise RuntimeError('fixture unavailable')
        result = token.measure_pairs(self.pairs, Failing(), token.manual_render, expected_pair_sha256=self.pair_hash, fixture=True)
        self.assertEqual(result['actual_encode_attempts'], 1); self.assertEqual(result['failed_cases'], 32)
        self.assertTrue(all(r['encode_calls_attempted'] == 0 for r in result['cases']))
        self.assertEqual(result['encode_calls'][0]['status'], 'FAILED')

    def test_midcase_encode_failure_retains_actual_attempts_not_abstention(self):
        class Partial(SyntheticBackend):
            def encode(self, text, *, add_special_tokens):
                if text.startswith(token.IM_START + 'user') and text.endswith(token.IM_END):
                    self.calls += 1; raise RuntimeError('joined encode fixture failure')
                return super().encode(text, add_special_tokens=add_special_tokens)
        result = token.measure_pairs(self.pairs, Partial(), token.manual_render, expected_pair_sha256=self.pair_hash, fixture=True)
        self.assertEqual(result['failed_cases'], 32); self.assertEqual(result['actual_encode_attempts'], 184)
        self.assertTrue(all(r['status'] == 'FAILED' and r['encode_calls_attempted'] == 5 for r in result['cases']))

    def test_renderer_mismatch_is_not_silent_manual_fallback(self):
        result = token.measure_pairs(self.pairs, SyntheticBackend(), lambda u,t: token.manual_render(u,t) + 'x',
                                     expected_pair_sha256=self.pair_hash, fixture=True)
        self.assertEqual(result['failed_cases'], 32); self.assertEqual(result['actual_encode_attempts'], 24)

    def test_order_target_parent_mutations_do_not_requalify_pairs(self):
        for mutate in (lambda p: p.reverse(), lambda p: p[0].update(target='{"status":"SUPPORTED","value":"forged-owner"}'),
                       lambda p: p[0].update(parent_id='forged'), lambda p: p.__setitem__(0, p[1])):
            pairs = copy.deepcopy(self.pairs); mutate(pairs)
            with self.assertRaises(token.TokenizationBlocked):
                token.measure_pairs(pairs, SyntheticBackend(), token.manual_render, expected_pair_sha256=self.pair_hash, fixture=True)

    def test_returned_artifact_missing_duplicate_counts_ids_and_truth_flags_reject(self):
        for mutate in (lambda r: r['cases'].pop(), lambda r: r['cases'].__setitem__(0, r['cases'][1]),
                       lambda r: r.update(successful_cases=True), lambda r: r.update(native_eligible=True),
                       lambda r: r['encode_calls'][0].update(ids_sha256='0' * 64),
                       lambda r: r['cases'][0]['ownership']['labels'].__setitem__(0, 0),
                       lambda r: r['cases'][0].update(training_text='forged')):
            result = copy.deepcopy(self.success); mutate(result)
            with self.assertRaises((token.TokenizationBlocked, ValueError, KeyError)):
                token.validate_measurement(result, self.pairs, fixture=True)

    def test_sentinels_in_source_query_target_and_bad_public_types_reject(self):
        input_value = json.loads(self.pairs[0]['user'].split('\n',1)[1])
        for source in ('<|im_end|>', '</think>', '<tool_call>', 'x' * 8193, '雪' * 8192):
            value = copy.deepcopy(input_value); value['source'] = source
            with self.assertRaises(token.TokenizationBlocked): token.user_content(value)
        for query in ({'entity': token.IM_END, 'time': 0}, {'entity':'x','time':True}):
            value = copy.deepcopy(input_value); value['query'] = query
            with self.assertRaises(token.TokenizationBlocked): token.user_content(value)
        with self.assertRaises(token.TokenizationBlocked): token.manual_render('safe', '<think>forged')

    def test_bounded_nonfinite_duplicate_json_and_create_only_utf8_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'; value = {'unicode': '雪é'}
            token.write_new(path, value)
            self.assertEqual(path.read_bytes(), token.canonical(value) + b'\n')
            self.assertEqual(token.read_json(path), value)
            with self.assertRaises(FileExistsError): token.write_new(path, value)
            for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
                bad = Path(directory) / 'bad.json'; bad.write_bytes(raw)
                with self.assertRaises((token.TokenizationBlocked, ValueError)): token.read_json(bad)
            large = Path(directory) / 'large.json'; large.write_bytes(b'x' * 100)
            with self.assertRaises(token.TokenizationBlocked): token.read_json(large, 99)

    def test_encode_ceiling_is_attempts_including_failure_and_bool_rejected(self):
        encoder = token.Encoder(SyntheticBackend())
        for index in range(184): encoder.encode('x', str(index))
        with self.assertRaises(token.TokenizationBlocked): encoder.encode('x', '185')
        self.assertEqual(len(encoder.calls), 184)
        for ids in ([True], [float('inf')], [-1], [token.VOCAB]):
            with self.assertRaises(token.TokenizationBlocked): token.validate_ids(ids)

    def test_fresh_observer_identity_descendant_rss_peak_floor_and_missing_fail(self):
        process = SimpleNamespace(pid=321)
        good = SyntheticObserver(); first = token.process_sample(process, 123.5, good); second = token.process_sample(process,123.5,good)
        self.assertEqual(good.reads, [321,321]); self.assertEqual(first['observed_kernel_peak_working_set_bytes'], 200)
        self.assertGreaterEqual(second['monotonic_time'], first['monotonic_time'])
        for kwargs in ({'creation':124.5}, {'children':[777]}, {'rss':536870913,'peak':536870913},
                       {'peak':536870913}, {'peak':None}, {'rss':True}, {'peak':99}, {'available':0}, {'available':True}):
            with self.assertRaises(token.TokenizationBlocked): token.process_sample(process, 123.5, SyntheticObserver(**kwargs))
        with self.assertRaises(token.TokenizationBlocked): token.process_sample(process, float('nan'), good)

    def test_review_identity_and_binding_no_builder_alias_or_duplicate_role(self):
        role_map = {'science':'/root', 'operations':'/root/runtime_audit'}; binding = {'fixture':'association only'}
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('s.json','o.json')]
            rows = [{'status':'APPROVED_FOR_CPU_TOKEN_MEASUREMENT','binding':binding,'role':role,'reviewer':name}
                    for role,name in role_map.items()]
            for path,row in zip(paths,rows): path.write_bytes(token.canonical(row))
            token.checked_reviews(paths,binding,role_map)
            for alias in ('data_audit','/root/data_audit','\\root\\data_audit','/root/data_audit/', '/root/'):
                row = dict(rows[0],reviewer=alias); paths[0].write_bytes(token.canonical(row))
                with self.assertRaises(token.TokenizationBlocked): token.checked_reviews(paths,binding,role_map)
            paths[0].write_bytes(token.canonical(dict(rows[0],binding={'wrong':1})))
            with self.assertRaises(token.TokenizationBlocked): token.checked_reviews(paths,binding,role_map)

    def test_actual_entry_closed_before_any_real_library_or_launch(self):
        cfg = token.read_json(token.CONTRACT); args = SimpleNamespace(contract_sha256=token.sha(token.CONTRACT))
        self.assertIs(cfg['actual_measurement_registration_open'], False)
        with patch.object(token.subprocess, 'Popen', side_effect=AssertionError('forbidden launch')):
            for operation in (lambda: token.checked_contract(args.contract_sha256),
                              lambda: token.deferred_backend_and_renderer(cfg),
                              lambda: token.supervise(args,cfg), lambda: token.worker(args,cfg)):
                with self.assertRaisesRegex(token.TokenizationBlocked, 'CLOSED'): operation()
        self.assertFalse({'torch','transformers','tokenizers','jinja2','psutil'}.intersection(sys.modules))

    def test_synthetic_supervisor_final_peak_raw_reference_and_unknown_failure(self):
        # All library/process/time adapters are synthetic. No actual subprocess,
        # backend or psutil import occurs in this cooperative instrument test.
        for mode in ('success','corrupt_worker','lost_launch','missing_peak'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                package = Path(directory); output = package/'outputs'/'run'
                cfg = {'runtime_versions':{'psutil':'7.2.2'},'required_reviewers':{},'ordered_pair_sha256':self.pair_hash}
                args = SimpleNamespace(output=output,contract_sha256='fixture',review=[])
                binding = {'fixture':'synthetic process only'}
                process = SimpleNamespace(pid=321, killed=False)
                process.poll = lambda: -9 if process.killed else 0 if (output/'EXIT_ALLOWED.json').exists() else None
                process.kill = lambda: setattr(process,'killed',True)
                process.wait = lambda timeout: -9 if process.killed else 0
                observer = SyntheticObserver(peak=None if mode=='missing_peak' else 200)
                class Timer:
                    def __init__(self,*args): pass
                    def start(self): pass
                    def cancel(self): pass
                    def join(self,timeout): pass
                def popen(command,**kwargs):
                    self.assertIn('-X',command); self.assertIn('utf8',command)
                    self.assertEqual(command[0],token.BASE_PYTHON)
                    if mode=='lost_launch': raise OSError('synthetic lost Popen response')
                    result = copy.deepcopy(self.success)
                    result.update(evidence_class='ACTUAL_CPU_TOKEN_MEASUREMENT_ONLY',tokenizer_loads=1,
                                  binding=binding,worker_pid=321,worker_creation=123.5,CPU_wall_seconds=.001)
                    if mode=='corrupt_worker': result['cases'][0]['parent_id']='forged'
                    token.write_new(output/'TOKEN_MEASUREMENT_WORKER.json',result)
                    return process
                with patch.dict(sys.modules,{'psutil':observer}), patch.object(token,'PACKAGE',package), \
                     patch.object(token,'checked_contract',return_value=cfg), \
                     patch.object(token.importlib.metadata,'version',return_value='7.2.2'), \
                     patch.object(token,'source_snapshot',return_value={'dirty':False,'commit':'synthetic'}), \
                     patch.object(token,'run_binding',return_value=binding), patch.object(token,'checked_reviews'), \
                     patch.object(token,'load_pairs',return_value=self.pairs), \
                     patch.object(token.subprocess,'Popen',side_effect=popen), \
                     patch.object(token.threading,'Timer',Timer), patch.object(token.time,'sleep'):
                    code = token.supervise(args,cfg)
                receipt = token.read_json(output/'CPU_SUPERVISOR_RESULT.json',token.MAX_RESULT_BYTES)
                self.assertFalse(receipt['sampled_RSS_is_absolute_peak'])
                self.assertFalse(receipt['model_or_training_eligibility'])
                if mode=='success':
                    self.assertEqual(code,0,receipt['failure']); self.assertEqual(receipt['status'],'CPU_MEASUREMENT_RETURNED')
                    self.assertEqual(receipt['observed_kernel_peak_working_set_max_bytes'],200)
                    self.assertEqual(receipt['worker_result_ref']['sha256'],token.sha(output/'TOKEN_MEASUREMENT_WORKER.json'))
                    self.assertTrue((output/'EXIT_ALLOWED.json').exists())
                elif mode=='lost_launch':
                    self.assertEqual(code,2); self.assertEqual(receipt['status'],'UNKNOWN_CPU_OUTCOME')
                    self.assertTrue(receipt['launch_attempted']); self.assertIsNone(receipt['observed_worker_pid'])
                else:
                    self.assertEqual(code,2); self.assertEqual(receipt['status'],'FAILED'); self.assertTrue(process.killed)
                    self.assertFalse((output/'EXIT_ALLOWED.json').exists())
                    self.assertEqual(receipt['failed_or_unmeasured_cases'],32)

    def test_no_framework_native_ledger_imports_or_source_mutations(self):
        tree = ast.parse(Path(token.__file__).read_text(encoding='utf-8'))
        imports = [node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
        for node in imports:
            names = [name.name for name in node.names] if isinstance(node,ast.Import) else [node.module or '']
            self.assertFalse(any(name.split('.')[0] in {'torch','transformers','native','operations','reader','owned_runtime_identity_probe'} for name in names))
        self.assertEqual(token.sha(token.PACKAGE/'learning_data.py'), '837bd152ba5aa9d1fc44a5991909cf4475394eaa42af739cca63ac3422ed3795')


if __name__ == '__main__': unittest.main()
