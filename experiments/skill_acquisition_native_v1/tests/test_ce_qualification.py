"""Receipt-shaped request/orchestration fixtures only; no native qualification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification as q, ce_plan


class RequestControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.refs, self.values, self.events = {}, {}, []
        self.code = {'synthetic-test-source': 'a' * 64}
    def bound(self, name, value):
        path = self.root / (name + '.json'); raw = q.canonical(value)
        path.write_bytes(raw); ref = {'path': str(path), 'sha256': __import__('hashlib').sha256(raw).hexdigest()}
        self.refs[name], self.values[name] = ref, value; return ref
    def request(self):
        census = {'rows_rendered': 3000, 'total_prompt_tokens': 2560912, 'total_supervised_tokens': 64712,
            'total_sequence_positions': 2625624, 'max_sequence_positions': 1287,
            'prompt_render_sha256': 'b' * 64, 'sequence_ids_sha256': 'c' * 64,
            'prompt_tokens': {'max': 1259}, 'labelled_tokens': {'max': 29}}
        preparation = self.bound('preparation', {'approved': False, 'training_authorized': False, 'census': {'canonical': census}})
        records = [{'id': 'row-%04d' % i, 'target_sha256': q.digest(['fixture-label', i])} for i in range(3000)]
        records = sorted(records, key=lambda r: (ce_plan.__dict__['_order']([r['id']])[0]))
        index = {r['id']: r for r in records}; records = [index[rid] for rid in ce_plan._order(index)]
        plan = ce_plan._finish('unapproved_proposal', 'canonical', records,
            {'target_policy': {'kind': 'oracle_first_compact_ascending'}, 'file': {'fixture': True}, 'census': census},
            preparation, None)
        self.bound('plan', plan)
        recipe = {'accumulation_rows': 8, 'microbatch_rows': 1,
            'loss_reduction': 'mean_labeled_tokens_per_row_then_equal_rows',
            'gradient_checkpointing': {'use_reentrant': False},
            'gradient_clipping': {'max_norm': 1.0, 'norm_type': 2.0, 'foreach': False, 'error_if_nonfinite': True},
            'optimizer_origin': 'FRESH_PER_ARM', 'optimizer': {'class': 'torch.optim.AdamW', 'lr': 1e-5,
                'eps': 1e-8, 'weight_decay': 0, 'betas': [.9, .999], 'foreach': False, 'capturable': True},
            'target_eos_id': 151645, 'eos_ids': [151645, 151643], 'pad_id': 151643, 'max_target_tokens': 128}
        self.bound('recipe', {'recipe': recipe})
        self.bound('source', {'code_sha256': self.code, 'stage_workers': {}})
        runtime = {'device': 'cuda:0', 'dtype': 'float32', 'packages': dict(q.PINS),
            'deterministic_algorithms': True, 'tf32_cuda_matmul': False, 'tf32_cudnn': False}
        self.bound('runtime', {'decoding_preimage': runtime, 'decoding_sha256': q.digest(runtime)})
        self.bound('anchors', dict(own_start_sha256='d' * 64, common_anchor_sha256='d' * 64,
            base_weights_sha256='e' * 64, warm_delta_sha256='f' * 64, tokenizer_receipt_sha256='1' * 64))
        self.bound('selectedoperatorchoice', {'schema': 'CE8-OPERATOR-CHOICE-1', 'approved': True,
            'choice': 'canonical_only', 'new_experiment': True, 'preserved_stop_receipt_sha256': q.WARM_STOP})
        self.bound('evaluation', {'registered_bank_specification': 'SYNTHETIC METADATA ONLY'})
        self.bound('resources', {'declared': 'SYNTHETIC; NO NATIVE MEASUREMENT'})
        cost = {k: 100000000 for k in q.COST_FIELDS}; cost['wall_seconds_upper'] = 1000.0
        self.bound('attempt_caps', {'schema': 'CE8-QUALIFICATION-CAPS-1',
            'stage_caps': {stage: dict(cost, max_attempts=2) for stage in q.STAGES},
            'global_caps': {k: v * 16 for k, v in cost.items()}})
        self.bound('ingestion', {'schema': 'CE8-INGESTION-SPEC-1', 'arm': 'canonical',
            'plan_sha256': plan['plan_sha256'], 'preparation_ref': preparation,
            'tokenizer_receipt_sha256': '1' * 64, 'expected_census': q.declared_census(self.values['preparation'], 'canonical')})
        self.bound('scope', {'schema': q.SCOPE, 'role': 'qualification_only', 'approved': True,
            'qualification_authorized': True, 'production_training_authorized': False, 'arm': 'canonical',
            'choice_sha256': self.refs['selectedoperatorchoice']['sha256'], 'static_recipe_sha256': q.digest(recipe),
            'attempt_caps_sha256': self.refs['attempt_caps']['sha256'], 'resources_sha256': self.refs['resources']['sha256'],
            'likelihood_tolerance_nats': 1e-4, 'allowed_data_roles': ['train'],
            'gradient_reference': {'absolute_tolerance': 1e-6, 'relative_tolerance': 1e-6, 'norm': 'l2'}})
        request = {'schema': q.REQUEST, 'phase': 'qualification', 'approved': True,
            'qualification_authorized': True, 'training_authorized': False,
            'status': 'AUTHORIZED_QUALIFICATION_ONLY', 'arm': 'canonical', 'bootstrap_bounds': cost}
        for name, ref in self.refs.items(): request[name + '_ref'] = ref
        request['scope_authorization_ref'] = self.bound('scope_authorization', {'approved': True,
            'qualification_authorized': True, 'training_authorized': False,
            'request_preimage_sha256': q.digest(q.request_preimage(request))})
        return request
    def admit(self, request):
        def outer(stage, refs):
            self.events.append(('outer', stage))
            return {'schema': 'CE8-OUTER-ADMISSION-1', 'mode': 'synthetic_fixture', 'admitted': True,
                'phase': 'qualification', 'arm': request['arm'], 'request_sha256': q.digest(q.request_preimage(request)),
                'scope_sha256': request['scope_ref']['sha256'], 'source_sha256': request['source_ref']['sha256'],
                'runtime_sha256': request['runtime_ref']['sha256'],
                'static_recipe_sha256': q.digest(self.values['recipe']['recipe'])}
        return q.admit_qualification(request, code_sha256=self.code, outer_admission=outer,
            io_observer=lambda kind, phase, **fields: self.events.append((kind, phase, fields)))
    def test_unapproved_missingchoice_refuses_before_IO_or_backend(self):
        before = {n for n in ('torch', 'numpy', 'transformers', 'safetensors') if n in sys.modules}
        for value in ({'schema': q.REQUEST, 'phase': 'qualification', 'approved': False},
                      {'schema': q.REQUEST, 'phase': 'qualification', 'approved': True,
                       'qualification_authorized': True, 'training_authorized': False,
                       'status': 'AUTHORIZED_QUALIFICATION_ONLY', 'arm': 'canonical'}):
            with self.assertRaises(q.RequestRejected): self.admit(value)
        self.assertEqual(self.events, [])
        self.assertEqual(before, {n for n in ('torch', 'numpy', 'transformers', 'safetensors') if n in sys.modules})
    def test_qualification_no_completed_qualification_dependency_or_training_grant(self):
        request = self.request(); context = self.admit(request)
        self.assertEqual(context['phase'], 'qualification'); self.assertFalse(context['production_training_authorized'])
        self.assertEqual(context['stage_plan'][0]['stage'], 'ingestion_tokenizer')
        self.assertEqual(self.events[0], ('outer', 'qualification_metadata'))
        self.assertNotIn('qualifiedreceipt_ref', request)
        request['qualifiedreceipt_ref'] = {'path': 'missing', 'sha256': '0' * 64}
        with self.assertRaisesRegex(q.RequestRejected, 'own completed output'): self.admit(request)
    def test_exact_authorization_and_source_bindings(self):
        request = self.request(); request['bootstrap_bounds']['wall_seconds_upper'] += 1
        with self.assertRaisesRegex(q.RequestRejected, 'authorization'): self.admit(request)
        request = self.request(); self.code = {'changed-source': 'b' * 64}
        with self.assertRaisesRegex(q.RequestRejected, 'source bundle'): self.admit(request)
    def test_concrete_outer_adapter_shared_request_preimage_and_null_output(self):
        from experiments.skill_acquisition_native_v1 import ce_controller as controller
        request = self.request()
        request['qualifiedreceipt_ref'] = None
        request['scope_authorization_ref'] = self.bound('scope_authorization', {'approved': True,
            'qualification_authorized': True, 'training_authorized': False,
            'request_preimage_sha256': q.digest(q.request_preimage(request))})
        expected = q.digest(q.request_preimage(request))
        authority = object(); reservations = []
        def verify(capability, request_hash, scope_hash, phase):
            return (capability is authority and request_hash == expected
                    and scope_hash == request['scope_ref']['sha256'] and phase == 'qualification')
        outer = controller.OuterAdmission(request,
            {'mode': 'synthetic_fixture', 'static_recipe_sha256': q.digest(self.values['recipe']['recipe'])},
            authority, lambda capability, stage, refs: reservations.append((capability, stage, refs)),
            verify, bootstrap=True)
        context = q.admit_qualification(request, code_sha256=self.code, outer_admission=outer,
            io_observer=lambda kind, phase, **fields: self.events.append((kind, phase, fields)))
        self.assertEqual(context['request_sha256'], expected)
        self.assertEqual(context['mode'], 'synthetic_fixture')
        self.assertEqual(context['static_recipe_sha256'], q.digest(self.values['recipe']['recipe']))
        self.assertFalse(context['production_training_authorized'])
        self.assertEqual(reservations[0][1], 'qualification_metadata')
        reused = controller.OuterAdmission(request, context, authority, outer.reserve, verify, bootstrap=True)
        self.assertEqual(reused('qualification_metadata', context['refs'])['request_sha256'], expected)
    def test_gradient_reference_uses_l2_absolute_relative_bound_at_zero_and_nonzero(self):
        context = self.admit(self.request())
        ingestion = {'prepared_row_bindings': [{'id': 'witness', 'labeled_tokens': 2}],
                     'witnesses': {'max_sequence_positions': {'row_ids': ['witness']}}}
        def observed(reference_norm, difference_norm, max_abs_error):
            return {'per_token_comparisons': [{'id': 'witness', 'positions': 2, 'max_abs_nats': 0}],
                'gradient_reference': {'norm': 'l2', 'fixed_loss_denominator': 8,
                    'reference_norm': reference_norm, 'difference_norm': difference_norm,
                    'max_abs_error': max_abs_error}}
        q.validate_stage_observations(context, 'likelihood_gradient_parity', observed(0, 1e-6, 5e-7), ingestion)
        with self.assertRaisesRegex(q.RequestRejected, 'gradient reference'):
            q.validate_stage_observations(context, 'likelihood_gradient_parity', observed(0, 1e-3, 5e-7), ingestion)
        q.validate_stage_observations(context, 'likelihood_gradient_parity', observed(2, 3e-6, 5e-7), ingestion)
        with self.assertRaisesRegex(q.RequestRejected, 'gradient reference'):
            q.validate_stage_observations(context, 'likelihood_gradient_parity', observed(2, 4e-6, 5e-7), ingestion)
        with self.assertRaisesRegex(q.RequestRejected, 'gradient reference'):
            q.validate_stage_observations(context, 'likelihood_gradient_parity', observed(2, 1e-8, 5e-7), ingestion)
    def actual_ingestion_fixture(self, context):
        plan = context['bindings']['plan']
        ids = [rid for boundary in plan['ordered_plan'] for rid in boundary['row_ids']]
        targets = dict(pair for boundary in plan['label_binding']['boundaries'] for pair in boundary['row_target_pairs'])
        expected = context['bindings']['ingestion']['expected_census']
        census = {key: value for key, value in expected.items() if key != 'rows_rendered'}
        census['rows'] = len(ids)
        # Max P and max S share one witness; max C is a different real row.
        # Do not synthesize maxP+maxC (=1288) when actual max S is 1287.
        Ps, Cs = [1259, 1000], [28, 29]
        for values, key in ((Ps, 'total_prompt_tokens'), (Cs, 'total_supervised_tokens')):
            base, remainder = divmod(expected[key] - sum(values), len(ids) - 2)
            values.extend(base + (i < remainder) for i in range(len(ids) - 2))
        rows = [{'id': rid, 'prompt_tokens': P, 'labeled_tokens': C, 'full_sequence_tokens': P + C,
                 'prompt_ids_sha256': q.digest(['fixture-prompt', rid]),
                 'completion_ids_sha256': q.digest(['fixture-completion', rid]),
                 'prepared_row_sha256': q.digest(['fixture-prepared', rid]),
                 'render_sha256': q.digest(['fixture-render', rid]), 'target_sha256': targets[rid]}
                for rid, P, C in zip(ids, Ps, Cs)]
        witnesses = {field: {'value': census[field], 'row_ids': [row['id'] for row in rows if row[key] == census[field]]}
            for field, key in (('max_prompt_tokens', 'prompt_tokens'), ('max_labeled_tokens', 'labeled_tokens'),
                               ('max_sequence_positions', 'full_sequence_tokens'))}
        body = {'schema': 'CE8-INGESTION-1',
            'tokenizer_rendered': True, 'model_loads': 0, 'optimizer_steps': 0,
            'native_qualified': False, 'training_authorized': False, 'arm': context['arm'],
            'plan_sha256': plan['plan_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
            'tokenizer_receipt_sha256': context['bindings']['anchors']['tokenizer_receipt_sha256'],
            'preparation_ref': context['refs']['preparation_ref'], 'plan_ref': context['refs']['plan_ref'],
            'census': census, 'prepared_row_bindings': rows, 'witnesses': witnesses}
        return dict(body, receipt_sha256=q.digest(body))
    def test_actual_ingestion_checks_summed_counts_and_plan_targets_after_self_rehash(self):
        context = self.admit(self.request()); receipt = self.actual_ingestion_fixture(context)
        q.validate_actual_ingestion(context, receipt)
        self.assertEqual(receipt['witnesses']['max_prompt_tokens']['value'], 1259)
        self.assertEqual(receipt['witnesses']['max_labeled_tokens']['value'], 29)
        self.assertEqual(receipt['witnesses']['max_sequence_positions']['value'], 1287)
        self.assertNotEqual(receipt['witnesses']['max_prompt_tokens']['row_ids'],
                            receipt['witnesses']['max_labeled_tokens']['row_ids'])
        changed = copy.deepcopy(receipt)
        changed['prepared_row_bindings'][2]['prompt_tokens'] += 1
        changed['prepared_row_bindings'][2]['full_sequence_tokens'] += 1
        changed['receipt_sha256'] = q.digest({k: v for k, v in changed.items() if k != 'receipt_sha256'})
        with self.assertRaisesRegex(q.RequestRejected, 'summed census'):
            q.validate_actual_ingestion(context, changed)
        changed = copy.deepcopy(receipt); changed['prepared_row_bindings'][2]['target_sha256'] = '0' * 64
        changed['receipt_sha256'] = q.digest({k: v for k, v in changed.items() if k != 'receipt_sha256'})
        with self.assertRaisesRegex(q.RequestRejected, 'bound label plan'):
            q.validate_actual_ingestion(context, changed)
    def test_no_default_registered_worker_or_synthetic_native_receipt(self):
        context = self.admit(self.request())
        with self.assertRaisesRegex(q.RequestRejected, 'default backend'):
            q.build_stage_command(context, q.STAGES[0], request_path=self.root/'request.json', output_dir=self.root/'output')
        with self.assertRaisesRegex(q.RequestRejected, 'not native qualification'):
            q.producer_receipt(context, [], {}, backend_mode='synthetic_fixture')
        draft = q.producer_receipt(context, [{'passed': True, 'native_actual': True}], {}, backend_mode='registered_scope')
        self.assertEqual(draft['status'], 'DRAFT_UNQUALIFIED'); self.assertFalse(draft['passed'])
    def test_flipped_activation_flag_and_invented_verifier_cannot_issue_native_pass(self):
        context = self.admit(self.request()); invoked = []
        def invented_verifier(*args, **kwargs):
            invoked.append((args, kwargs)); return True
        self.assertFalse(q._REGISTERED_NATIVE_VERIFIERS)
        with patch.object(q, 'NATIVE_BACKEND_IMPLEMENTED', True):
            for verifier in (None, object(), invented_verifier):
                draft = q.producer_receipt(context, [{'passed': True, 'native_actual': True}],
                    {'completed': True}, backend_mode='registered_scope', native_verifier=verifier)
                self.assertEqual(draft['status'], 'DRAFT_UNQUALIFIED')
                self.assertFalse(draft['passed']); self.assertFalse(draft['native_qualified'])
                self.assertEqual(draft['actual_activation_gaps'], q.ACTUAL_ACTIVATION_GAPS)
        self.assertEqual(invoked, [])
    def test_missing_actual_stage_observations_cannot_pass_validators(self):
        context = self.admit(self.request())
        for stage in q.STAGES[1:]:
            with self.subTest(stage=stage), self.assertRaises(q.RequestRejected):
                q.validate_stage_observations(context, stage, {})


if __name__ == '__main__': unittest.main()
