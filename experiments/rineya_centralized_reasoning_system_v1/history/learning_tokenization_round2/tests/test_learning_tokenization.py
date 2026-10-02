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

    def fixture_registration(self, package, cfg):
        """Temporary synthetic coordinator record; never an actual activation."""
        args = SimpleNamespace(output=package/token.OUTPUT_RELATIVE, contract_sha256='fixture-contract',
                               activation=package/token.ACTIVATION_RELATIVE, activation_sha256=None,
                               review=[package/'science.json',package/'operations.json'])
        args.activation.parent.mkdir(parents=True,exist_ok=True)
        basic = token.run_binding(cfg,args.contract_sha256,{'commit':'synthetic','dirty':False})
        token.write_new(args.activation,token.activation_payload(cfg,basic))
        args.activation_sha256=token.sha(args.activation)
        binding=dict(basic,activation_sha256=args.activation_sha256,attempt_key=token.ATTEMPT_KEY,
                     output_relative=token.OUTPUT_RELATIVE,attempt_relative=token.ATTEMPT_RELATIVE)
        for path,(role,name) in zip(args.review,cfg['required_reviewers'].items()):
            token.write_new(path,{'status':'APPROVED_FOR_CPU_TOKEN_MEASUREMENT','binding':binding,
                                 'role':role,'reviewer':name})
        return args

    def run_supervisor_fixture(self, mode):
        """Production supervisor with synthetic activation, OS/library/timer."""
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory); cfg=token.read_json(token.CONTRACT)
            process=SimpleNamespace(pid=321,killed=False,post_exit_polls=0)
            state={'popen_calls':0,'timer_started':False,'fresh_samples':0}
            test=self
            class Observer(SyntheticObserver):
                def Process(self,pid=321):
                    base=super().Process(pid)
                    if mode in ('normal_exit','live_missing','live_error','pid_reused') and pid==321 and (output/'EXIT_ALLOWED.json').exists():
                        if mode in ('normal_exit','live_missing'): raise ProcessLookupError('synthetic exact root vanished')
                        if mode=='live_error': raise PermissionError('synthetic still-live telemetry error')
                        return SimpleNamespace(create_time=lambda:124.5,children=lambda recursive:[],
                                               memory_info=lambda:SimpleNamespace(rss=100,peak_wset=200))
                    if mode=='child_vanished' and pid==777: raise ProcessLookupError('synthetic child identity unavailable')
                    if mode=='child_reused' and pid==777:
                        return SimpleNamespace(create_time=lambda:999.5)
                    return base
            observer=Observer(peak=None if mode=='missing_peak' else 200,
                              children=[777] if mode in ('child_survives','child_vanished','child_reused') else [])
            if mode=='child_reused': observer.child_list=[SimpleNamespace(pid=777,create_time=lambda:123.5)]
            class Timer:
                def __init__(self,*args): pass
                def start(self): state['timer_started']=True
                def cancel(self): pass
                def join(self,timeout): pass
            def poll():
                if process.killed: return -9
                if (output/'GO.json').exists() and not (output/'TOKEN_MEASUREMENT_WORKER.json').exists():
                    go=token.read_json(output/'GO.json')
                    result=copy.deepcopy(test.success)
                    result.update(evidence_class='ACTUAL_CPU_TOKEN_MEASUREMENT_ONLY',tokenizer_loads=1,
                                  binding=go['context']['admission']['binding'],context=go['context'],
                                  attempt_ref=go['attempt_ref'],intent_ref=go['intent_ref'],
                                  go_ref=token.artifact_ref(output/'GO.json'),worker_pid=321,
                                  worker_creation=123.5,CPU_wall_seconds=.001)
                    if mode=='corrupt_worker': result['cases'][0]['parent_id']='forged'
                    token.write_new(output/'TOKEN_MEASUREMENT_WORKER.json',result)
                if (output/'EXIT_ALLOWED.json').exists():
                    process.post_exit_polls+=1
                    if mode=='live_missing': return None
                    if mode in ('normal_exit','live_error','pid_reused') and process.post_exit_polls==1: return None
                    return 0
                return None
            process.poll=poll
            process.kill=lambda:setattr(process,'killed',True)
            def wait(timeout):
                if mode=='cleanup_error': raise TimeoutError('synthetic root reap unavailable')
                return -9 if process.killed else 0
            process.wait=wait
            def popen(command,**kwargs):
                state['popen_calls']+=1
                test.assertTrue(state['timer_started'])
                test.assertIn('-X',command);test.assertIn('utf8',command)
                test.assertEqual(command[0],token.BASE_PYTHON)
                if mode=='lost_launch': raise OSError('synthetic lost Popen response')
                return process
            actual_validator=token.validate_worker_result
            def validate(*args):
                returned=actual_validator(*args)
                if mode=='late_peak': observer.peak=536870913
                return returned
            actual_write=token.write_new
            def write(path,value):
                if mode=='publication_error' and Path(path).name=='CPU_SUPERVISOR_RESULT.json':
                    raise OSError('synthetic final publication failure')
                return actual_write(path,value)
            with patch.object(token,'PACKAGE',package), patch.object(token,'source_contract',return_value=cfg),                  patch.object(token,'source_snapshot',return_value={'dirty':False,'commit':'synthetic'}):
                args=self.fixture_registration(package,cfg);output=args.output
                with patch.dict(sys.modules,{'psutil':observer}),                      patch.object(token.importlib.metadata,'version',return_value='7.2.2'),                      patch.object(token,'load_pairs',return_value=self.pairs),                      patch.object(token.subprocess,'Popen',side_effect=popen),                      patch.object(token.threading,'Timer',Timer),patch.object(token.time,'sleep'),                      patch.object(token,'validate_worker_result',side_effect=validate),                      patch.object(token,'write_new',side_effect=write):
                    if mode=='publication_error':
                        with self.assertRaises(OSError): token.supervise(args,cfg)
                        with self.assertRaisesRegex(token.TokenizationBlocked,'unresolved'): token.supervise(args,cfg)
                        self.assertEqual(state['popen_calls'],1);return
                    code=token.supervise(args,cfg)
                    receipt=token.read_json(output/'CPU_SUPERVISOR_RESULT.json',token.MAX_RESULT_BYTES)
                    original=(output/'CPU_SUPERVISOR_RESULT.json').read_bytes()
                    if mode=='terminal_corruption':
                        corrupt=copy.deepcopy(receipt);corrupt['exit_code']=True
                        (output/'CPU_SUPERVISOR_RESULT.json').write_bytes(token.canonical(corrupt))
                        with self.assertRaises(token.TokenizationBlocked):token.supervise(args,cfg)
                        self.assertEqual(state['popen_calls'],1)
                        (output/'CPU_SUPERVISOR_RESULT.json').write_bytes(original)
                    repeat=token.supervise(args,cfg)
                    self.assertEqual(repeat,code)
                    self.assertEqual((output/'CPU_SUPERVISOR_RESULT.json').read_bytes(),original)
                    self.assertEqual(state['popen_calls'],1)
                    changed=copy.copy(args);changed.output=package/'outputs'/'another'
                    with self.assertRaises(token.TokenizationBlocked): token.supervise(changed,cfg)
                    self.assertEqual(state['popen_calls'],1)
                self.assertFalse(receipt['sampled_RSS_is_absolute_peak'])
                self.assertFalse(receipt['model_or_training_eligibility'])
                if mode in ('success','normal_exit','terminal_corruption'):
                    self.assertEqual(code,0,receipt['failure']);self.assertEqual(receipt['status'],'CPU_MEASUREMENT_RETURNED')
                    marker=token.read_json(output/'EXIT_ALLOWED.json')
                    self.assertEqual(marker['final_live_sample'],receipt['final_live_sample'])
                    self.assertEqual(receipt['samples'][-1],receipt['final_live_sample'])
                    self.assertGreaterEqual(len(receipt['samples']),3)
                    self.assertEqual(receipt['worker_result_ref']['sha256'],token.sha(output/'TOKEN_MEASUREMENT_WORKER.json'))
                    self.assertFalse(process.killed)
                else:
                    self.assertEqual(code,2);self.assertIsNotNone(receipt['failure'])
                    if mode!='corrupt_worker':
                        self.assertTrue(receipt['CPU_outcome_unknown'])
                        self.assertEqual(receipt['status'],'UNKNOWN_CPU_OUTCOME')
                    if mode in ('late_peak','corrupt_worker','missing_peak'):
                        self.assertFalse((output/'EXIT_ALLOWED.json').exists())
                    if mode.startswith('child_'):
                        children=receipt['unresolved_observations'][0]['descendants']
                        self.assertEqual(children[0]['pid'],777)
                        self.assertIn(children[0]['state'],('OBSERVED_UNRESOLVED','IDENTITY_UNAVAILABLE','PID_REUSE_OR_LIST_RACE_UNRESOLVED'))
                        if mode=='child_reused':
                            self.assertEqual(children[0]['listed_creation_time'],123.5)
                            self.assertEqual(children[0]['creation_time'],999.5)
                    if mode=='lost_launch':
                        self.assertTrue(receipt['launch_attempted']);self.assertIsNone(receipt['observed_worker_pid'])

    def test_postvalidation_peak_breach_denies_exit_and_blocks_repeat(self):
        self.run_supervisor_fixture('late_peak')

    def test_unresolved_descendant_surviving_vanished_reused_capture_and_block_repeat(self):
        for mode in ('child_survives','child_vanished','child_reused'): self.run_supervisor_fixture(mode)

    def test_durable_unknown_lost_launch_one_attempt_for_all_output_choices(self):
        self.run_supervisor_fixture('lost_launch')

    def test_final_receipt_publication_or_cleanup_failure_never_relaunch(self):
        self.run_supervisor_fixture('publication_error');self.run_supervisor_fixture('cleanup_error')

    def test_normal_permitted_exit_race_valid_and_opposing_live_errors_reject(self):
        for mode in ('success','normal_exit','live_missing','live_error','pid_reused'): self.run_supervisor_fixture(mode)

    def test_corrupt_worker_missing_peak_fail_before_exit_authorization(self):
        self.run_supervisor_fixture('corrupt_worker');self.run_supervisor_fixture('missing_peak')

    def test_corrupt_prior_terminal_receipt_never_promotes_or_relaunches(self):
        self.run_supervisor_fixture('terminal_corruption')

    def test_source_record_flag_cannot_be_flipped_into_activation(self):
        cfg=token.read_json(token.CONTRACT);changed=copy.deepcopy(cfg)
        changed['actual_measurement_registration_open']=True
        with patch.object(token,'sha',return_value='fixture'),patch.object(token,'read_json',return_value=changed):
            with self.assertRaisesRegex(token.TokenizationBlocked,'must remain false'):token.source_contract('fixture')

    def test_immutable_activation_requires_exact_coordinator_bindings_not_open_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory);cfg=token.read_json(token.CONTRACT)
            with patch.object(token,'PACKAGE',package),patch.object(token,'source_contract',return_value=cfg),                  patch.object(token,'source_snapshot',return_value={'dirty':False,'commit':'synthetic'}):
                args=self.fixture_registration(package,cfg)
                actual_cfg,admitted=token.admission(args)
                self.assertEqual(actual_cfg,cfg);self.assertEqual(admitted['activation_ref']['sha256'],args.activation_sha256)
                self.assertIs(cfg['actual_measurement_registration_open'],False)
                original=args.activation.read_bytes()
                for mutate in (lambda r:r.update(coordinator='/root/data_audit'),
                               lambda r:r.update(output_relative='outputs/another'),
                               lambda r:r['source_binding'].update(pair_sha256='0'*64),
                               lambda r:r['limits'].update(wall_seconds=61)):
                    value=json.loads(original);mutate(value);args.activation.write_bytes(token.canonical(value))
                    changed=copy.copy(args);changed.activation_sha256=token.sha(args.activation)
                    with self.assertRaises(token.TokenizationBlocked): token.admission(changed)
                args.activation.write_bytes(original)
                changed=copy.copy(args);changed.activation_sha256='0'*64
                with self.assertRaises(token.TokenizationBlocked):token.admission(changed)
                changed=copy.copy(args);changed.activation=package/'foreign.json'
                with self.assertRaises(token.TokenizationBlocked):token.admission(changed)

    def test_worker_intent_attempt_output_command_parent_and_exit_bound(self):
        import os
        for mode in ('valid','missing_intent','wrong_attempt','wrong_command','wrong_parent','prewritten_exit','bad_sample'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                package=Path(directory);cfg=token.read_json(token.CONTRACT)
                with patch.object(token,'PACKAGE',package),patch.object(token,'source_contract',return_value=cfg),                      patch.object(token,'source_snapshot',return_value={'dirty':False,'commit':'synthetic'}):
                    args=self.fixture_registration(package,cfg);_,admitted=token.admission(args)
                    args.output.mkdir(parents=True)
                    context=token.context_record(admitted,os.getppid(),123.5)
                    guard=package/token.ATTEMPT_RELATIVE
                    token.write_new(guard,{'admission':admitted,'context':context})
                    intent=args.output/'CPU_INTENT.json'
                    token.write_new(intent,{'context':context,'attempt_ref':token.artifact_ref(guard)})
                    sample={'pid':os.getpid(),'creation_time':123.5,'rss_bytes':100,
                            'observed_kernel_peak_working_set_bytes':200,'host_available_bytes':2*2**30,
                            'descendants':[],'wall_time':token.time.time(),'monotonic_time':token.time.monotonic()}
                    go={'context':context,'attempt_ref':token.artifact_ref(guard),'intent_ref':token.artifact_ref(intent),
                        'worker_pid':os.getpid(),'worker_creation':123.5,'initial_live_sample':sample}
                    token.write_new(args.output/'GO.json',go)
                    class Observer(SyntheticObserver):
                        def Process(self,pid):
                            return SimpleNamespace(create_time=lambda:124.5 if mode=='wrong_parent' and pid==os.getppid() else 123.5,
                                                   exe=lambda:token.BASE_PYTHON,
                                                   cmdline=lambda:['forged'] if mode=='wrong_command' else admitted['command'])
                    observer=Observer()
                    if mode=='missing_intent':intent.unlink()
                    if mode=='wrong_attempt':guard.write_bytes(b'{}')
                    if mode=='prewritten_exit':token.write_new(args.output/'EXIT_ALLOWED.json',{'forged':True})
                    if mode=='bad_sample':
                        go['initial_live_sample']['rss_bytes']=536870913
                        (args.output/'GO.json').write_bytes(token.canonical(go))
                    if mode!='valid':
                        with self.assertRaises((token.TokenizationBlocked,FileNotFoundError,KeyError)):
                            token.validate_worker_admission(args,admitted,observer)
                        continue
                    checked_context,checked_go,go_ref=token.validate_worker_admission(args,admitted,observer)
                    self.assertEqual(checked_context,context)
                    worker_path=args.output/'TOKEN_MEASUREMENT_WORKER.json'
                    token.write_new(worker_path,{'failed_cases':0})
                    worker_ref=token.artifact_ref(worker_path)
                    marker=token.exit_payload(admitted,context,go['attempt_ref'],go['intent_ref'],go_ref,worker_ref,sample,0)
                    token.write_new(args.output/'EXIT_ALLOWED.json',marker)
                    token.validate_exit_authorization(args,admitted,context,go,go_ref,worker_ref)
                    for mutate in (lambda r:r.update(worker_pid=999),lambda r:r.update(expected_exit_code=True),
                                   lambda r:r['final_live_sample'].update(monotonic_time=0),
                                   lambda r:r['worker_result_ref'].update(sha256='0'*64)):
                        bad=copy.deepcopy(marker);mutate(bad)
                        (args.output/'EXIT_ALLOWED.json').write_bytes(token.canonical(bad))
                        with self.assertRaises(token.TokenizationBlocked):
                            token.validate_exit_authorization(args,admitted,context,go,go_ref,worker_ref)

    def test_full_render_returned_vectors_count_hash_parity_and_exclusion_replayed(self):
        for mutate in (lambda r:r['encode_calls'][26].update(token_count=1),
                       lambda r:[r['encode_calls'][i].update(ids_sha256='0'*64) for i in (26,27)],
                       lambda r:r['encode_calls'][26].pop('returned_ids'),
                       lambda r:r['encode_calls'][26].update(returned_ids=[True]),
                       lambda r:r['encode_calls'][26].update(returned_ids=[1]*8193)):
            bad=copy.deepcopy(self.success);mutate(bad)
            with self.assertRaises((token.TokenizationBlocked,KeyError)):
                token.validate_measurement(bad,self.pairs,fixture=True)
        bad=copy.deepcopy(self.success)
        for index in (26,27):
            ids=list(bad['encode_calls'][index]['returned_ids']);ids[0]=1
            bad['encode_calls'][index].update(returned_ids=ids,ids_sha256=token.digest(ids),token_count=len(ids))
        with self.assertRaises(token.TokenizationBlocked):token.validate_measurement(bad,self.pairs,fixture=True)

    def test_no_framework_native_ledger_imports_or_source_mutations(self):
        tree = ast.parse(Path(token.__file__).read_text(encoding='utf-8'))
        imports = [node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
        for node in imports:
            names = [name.name for name in node.names] if isinstance(node,ast.Import) else [node.module or '']
            self.assertFalse(any(name.split('.')[0] in {'torch','transformers','native','operations','reader','owned_runtime_identity_probe'} for name in names))
        self.assertEqual(token.sha(token.PACKAGE/'learning_data.py'), '837bd152ba5aa9d1fc44a5991909cf4475394eaa42af739cca63ac3422ed3795')


if __name__ == '__main__': unittest.main()
