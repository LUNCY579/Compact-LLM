"""Synthetic receipt-shaped metadata and loader spies; no native qualification performed."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from experiments.skill_acquisition_native_v1 import protocol as p, plan as planning


def update_recipe(synthetic=False):
    return {'schema': p.UPDATE_SCHEMA, 'synthetic_cpu': synthetic, 'approved': not synthetic,
        'training_authorized': not synthetic, 'B': 2, 'G': 8, 'objective': {'method': 'maxrl', 'epsilon': 1e-6},
        'optimizer': {'class': 'torch.optim.AdamW', 'lr': 1e-6, 'betas': [.9, .999], 'eps': 1e-8,
                      'weight_decay': 0, 'foreach': False, 'capturable': not synthetic},
        'ratio_clipping': None, 'gradient_clipping': None, 'anchor_KL': {'enabled': False, 'coefficient': 0},
        'gradient_checkpointing': {'use_reentrant': False}, 'max_new_tokens': 128,
        'pad_id': 151643, 'eos_ids': [151645, 151643]}


def proposal():
    return {'schema': 'COMP-RS-001-S1-RECIPE-PROPOSAL-2', 'version': 2, 'approved': False,
        'training_authorized': False, 'recommended_recipe': {'choice_id': 'S1_MAXRL_FIXED_EXPOSURE_B2_G8_V1',
            'prompt_batch': {'prompts_per_optimizer_boundary': 2, 'rollouts_per_prompt': 8, 'rollouts_per_boundary': 16},
            'sampler': {'max_new_tokens': 128}, 'objective': {'name': 'Practical centered MaxRL', 'epsilon': 1e-6},
            'optimizer': {'learning_rate': 1e-6, 'betas': [.9, .999], 'epsilon': 1e-8, 'schedule': 'constant', 'warmup_steps': 0},
            'ratio_clipping': {'enabled': False}, 'anchor_KL': {'enabled': False}, 'other_interventions': {'replay': False}},
        'proposed_unit_and_panel': {'prompts_per_skill_per_unit': 3000, 'nominal_boundaries_per_unit': 1500,
            'nominal_rollouts_per_unit': 24000, 'arms': {'MT': {'units': 2, 'boundaries': 3000, 'prompts_R': 3000, 'prompts_S': 3000}}},
        'proposed_checkpoint_and_rollback_policy': {'proposed_resume_count_cap': 8,
            'cadence_completed_boundaries': 10, 'proposed_rollback_work_allowance_fraction': .25},
        'proposed_seed_and_order_policy': {'seed_assignment': {'R': 1301, 'S': 1302, 'P': 1303, 'MT': 1304,
            'R_to_S': 1305, 'S_to_R': 1306, 'R_prime': 2301, 'S_prime': 2302}}}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.root = Path(temp.name)
        self.addCleanup(temp.cleanup)
        self.edits, self.values, self.refs = [], {}, {}

    def edit(self, label, dotted, value):
        self.edits.append((label, dotted.split('.'), value))

    def adjusted(self, label, value):
        value = copy.deepcopy(value)
        for name, keys, change in self.edits:
            if name == label:
                target = value
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = change
        self.values[label] = value
        return value

    def bound(self, label, value, path=None):
        value = self.adjusted(label, value)
        path = path or self.root / (label + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        self.refs[label] = p.file_record(path)
        return self.refs[label]

    def qualification_evidence(self, qualification, planned):
        """Fictional operation observations exercise validation only; no CUDA work."""
        bindings = ('code_sha256', 'anchor', 'start', 'inventory', 'update_recipe_sha256',
            'sampler_sha256', 'plan_sha256', 'manifest_sha256', 'tokenizer_receipt_sha256',
            'runtime_preimage', 'decoding_sha256', 'scorer_bundle_sha256')
        scope = {key: qualification[key] for key in bindings}
        scope.update(schema='S1_QUALIFICATION_SCOPE-1', strict_scorers={'native': '1' * 64, 'independent': '2' * 64}, gradient_reference={
            'norm': 'l2', 'error_metric': 'max_abs_and_relative_l2', 'zero_reference_rule': 'absolute_only',
            'absolute_tolerance': 1e-6, 'relative_tolerance': 1e-5, 'implementation_sha256': '0' * 64})
        scope_ref = self.bound('qualification_scope', scope)
        authorization = self.bound('qualification_authorization', {'approved': True,
            'qualification_authorized': True, 'scope_sha256': scope_ref['sha256']})
        operations, samples = [], []
        def operation(kind, inp, output, origin='declared_mechanics_fixture', phase='returned'):
            identifier = 'SYNTHETIC-OP-%d' % len(operations)
            operations.append({'id': identifier, 'scope_sha256': scope_ref['sha256'], 'kind': kind,
                'origin': origin, 'phase': phase, 'device': 'cuda:0', 'input': inp, 'output': output})
            return identifier
        for prompt in range(2):
            for k in range(8):
                trajectory = planned.trajectory(0, prompt, k)
                policy = qualification['anchor']['arm_sha256']
                generated = {'completion_ids': [17, 151645], 'stop_reason': 'eos',
                    'completion_text': 'SYNTHETIC TRACE ONLY', 'raw_step_logprobs': [-1.0, -2.0]}
                generation = operation('generation', {'trajectory': trajectory, 'policy_sha256': policy,
                    'sampler_sha256': qualification['sampler_sha256'], 'token_source': 'native_model_generate',
                    'reward_override': False}, generated, 'actual_native_sampling')
                expected = {'generation_operation_id': generation, 'completion_sha256': p.digest(generated),
                            'policy_sha256': policy}
                teacher = operation('teacher_forced', dict(expected, differentiable=True,
                    token_source='returned_native_generation'), {'selected_logprobs': [-1.0, -2.0]}, 'actual_native_sampling')
                scoring = dict(expected, trajectory=trajectory, reward_source='strict_task_scoring',
                    reward_override=False, scorer_bundle_sha256=qualification['scorer_bundle_sha256'])
                native = operation('strict_task_scorer', dict(scoring, implementation_sha256='1' * 64),
                                   {'reward': 0}, 'actual_task_scoring')
                independent = operation('independent_strict_task_scorer', dict(scoring, implementation_sha256='2' * 64),
                                        {'reward': 0}, 'actual_task_scoring')
                samples.append({'id': 'SYNTHETIC-NATIVE-%d-%d' % (prompt, k), 'origin': 'actual_native_sample',
                    'trajectory': trajectory, 'policy_sha256': policy, 'generation_operation_id': generation,
                    'teacher_operation_id': teacher, 'native_scorer_operation_id': native,
                    'independent_scorer_operation_id': independent})
        tokens = []
        for identifier, completion, reason in (('SYNTHETIC-EOS', [17, 151645], 'eos'),
                                               ('SYNTHETIC-CAP', [17] * 128, 'length_cap')):
            fixture = {'id': identifier, 'origin': 'declared_mechanics_token_fixture',
                'token_source': 'supplied_fixture_tokens', 'counts_as_native_sample': False,
                'prompt_length': 3, 'padding_length': 2, 'completion_ids': completion, 'stop_reason': reason}
            fixture['mask_operation_id'] = operation('loss_forward', {'token_fixture_id': identifier, 'fixture_ids': [identifier]},
                {'mask': {'selected_predictor_positions': list(range(2, 2 + len(completion))),
                          'prompt_positions_selected': 0, 'pad_positions_selected': 0,
                          'eos_positions_selected': int(reason == 'eos')}})
            tokens.append(fixture)
        rewards = [{'id': name, 'origin': 'declared_mechanics_reward_fixture',
            'reward_source': 'supplied_fixture_rewards', 'counts_as_task_reward': False, 'rewards': value}
            for name, value in (('SYNTHETIC-UNIFORM', [[0] * 8, [1] * 8]),
                                ('SYNTHETIC-MIXED', [[0] * 8, [0, 1] * 4]))]
        for _ in range(14): operation('loss_forward', {'fixture_ids': ['SYNTHETIC-MIXED']}, {})
        for _ in range(16): operation('backward', {'fixture_ids': ['SYNTHETIC-MIXED']}, {})
        operation('reference_backward', {'fixture_ids': ['SYNTHETIC-MIXED']}, {'gradient_reference': {
            'norm': 'l2', 'error_metric': 'max_abs_and_relative_l2', 'fixed_loss_denominator': 16,
            'reference_norm': 1.0, 'difference_norm': 0.0, 'max_abs_error': 0.0, 'implementation_sha256': '0' * 64}})
        adam = []
        for u in range(2):
            adam.append(operation('optimizer_step', {'fixture_ids': ['SYNTHETIC-MIXED']},
                {'U_before': u, 'U_after': u + 1, 'step_proof_sha256': '3' * 64}))
            operation('checkpoint_save', {'fixture_ids': ['SYNTHETIC-MIXED']}, {'manifest_sha256': str(6 + u) * 64})
        for initialized in (False, True):
            before = {'C': 2, 'U': int(initialized), 'adam_initialized': initialized,
                      'parameters_sha256': '4' * 64, 'adam_sha256': '5' * 64}
            operation('uniform_skip', {'fixture_ids': ['SYNTHETIC-UNIFORM']}, {'before': before, 'after': dict(before, C=3)})
        operation('checkpoint_restore', {'fixture_ids': ['SYNTHETIC-MIXED']}, {'manifest_sha256': '6' * 64,
            'state_sha256': {key: ['8' * 64] * 2 for key in ('parameters', 'adam', 'python_rng',
                'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng')}})
        operation('rollback_reuse_check', {'fixture_ids': ['SYNTHETIC-MIXED']}, {
            'returned_optimizer_operation_id': adam[1], 'restored_manifest_sha256': '6' * 64,
            'generation_before': 0, 'generation_after': 1, 'reuse_outcome': 'STALE_GENERATION_REJECTED',
            'charged_before': 99, 'charged_after': 100})
        operation('backward', {'fixture_ids': ['SYNTHETIC-MIXED'], 'reserved_work_units': 1},
                  {'charged_work_units': 1}, phase='failed')
        evidence_ref = self.bound('qualification_evidence', {'schema': 'S1_QUALIFICATION_EVIDENCE-1',
            'scope_sha256': scope_ref['sha256'], 'native_samples': samples, 'token_fixtures': tokens,
            'reward_fixtures': rewards, 'operations': operations})
        evidence = self.values['qualification_evidence']
        counts = dict(p.QUAL_COUNTS, native_parity_positions=32)
        ledger_ref, terminal = self.qualification_ledger(evidence_ref, evidence, counts)
        seal_ref = self.bound('qualification_seal', {'schema': 'S1_QUALIFICATION_COST_SEAL-1',
            'kind': 'INVOCATION_END', 'completed': True, 'scope_sha256': scope_ref['sha256'],
            'evidence_sha256': evidence_ref['sha256'], 'operations_sha256': p.digest(evidence['operations']),
            'counts': counts, 'cost_ledger': ledger_ref, 'terminal_row_sha256': terminal})
        return {'qualification_scope': scope_ref, 'scope_authorization': authorization, 'evidence': evidence_ref,
            'cost_seal': seal_ref, 'counts': counts, 'mechanics_cases': {name: True for name in p.QUAL_CASES},
            'native_score_parity': {'tolerance_nats': 1e-4, 'positions': 32, 'max_abs_nats': 0.0}}

    def qualification_ledger(self, evidence_ref, evidence, counts):
        entries, previous = [], '0' * 64
        def append(value):
            nonlocal previous
            value.update(seq=len(entries), previous_sha256=previous, scope_sha256=evidence['scope_sha256'])
            value['row_sha256'] = p.digest(value)
            previous = value['row_sha256']
            entries.append(value)
        attempts, failed, unknown = {}, 0, 0
        for operation in evidence['operations']:
            kind = operation['kind']
            attempts[kind] = attempts.get(kind, 0) + 1
            append({'event': 'started', 'operation_id': operation['id'], 'operation_kind': kind,
                    'origin': operation['origin'], 'input_sha256': p.digest(operation['input']), 'reserved_work_units': 1})
            if operation['phase'] == 'unknown':
                unknown += 1
            else:
                failed += int(operation['phase'] == 'failed')
                append({'event': operation['phase'], 'operation_id': operation['id'],
                        'output_sha256': p.digest(operation['output']), 'known_work_units': 1})
        append({'event': 'qualification_complete', 'evidence_sha256': evidence_ref['sha256'],
                'operations_sha256': p.digest(evidence['operations']),
                'operation_ids': [operation['id'] for operation in evidence['operations']], 'counts': counts,
                'charged_physical_totals': {'attempts_by_kind': attempts, 'reserved_work_units': len(evidence['operations']),
                                           'failed_attempts': failed, 'unknown_attempts': unknown}})
        ref = self.bound('qualification_ledger', {'schema': 'S1_QUALIFICATION_COST_LEDGER-1',
            'scope_sha256': evidence['scope_sha256'], 'entries': entries})
        return ref, previous

    def packet(self, arm='R'):
        native = {'test_mode': False, 'registered_run': True}
        data_root = self.root / 'data'
        data_root.mkdir(exist_ok=True)
        files, manifests, manifest_refs = {}, {}, {}
        for number, skill in enumerate(planning.ARMS[arm]):
            manifests[skill] = {'name': 'atomic_' + skill, 'schema_version': '2.0.0', 'rows': 3000,
                'ids': ['MC2-%064x' % (number * 3000 + row) for row in range(3000)]}
            relative = 'manifests/atomic_' + skill + '.json'
            manifest_refs[skill] = self.bound('manifest_' + skill, manifests[skill], data_root / relative)
            files[relative] = {'sha256': manifest_refs[skill]['sha256']}
        for relative in ('canonical/train.jsonl', 'policy/train.jsonl'):
            path = data_root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'SYNTHETIC TRAINING FILE ONLY\n')
            files[relative] = {'sha256': p.file_record(path)['sha256']}
        data_ref = self.bound('data_manifest', {'files': files}, data_root / 'manifest.json')
        manifest_patch = patch.object(p, 'MANIFEST', data_ref['sha256'])
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)
        asset_root = self.root / 'assets'
        asset_root.mkdir(exist_ok=True)
        assets = {}
        for name in ('model.safetensors', 'config.json', 'generation_config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'merges.txt', 'vocab.json'):
            path = asset_root / name
            path.write_bytes(('SYNTHETIC ' + name).encode())
            assets[name] = p.file_record(path)
        asset_ref = self.bound('assets', {'stage': 'Q1_ASSET_CANDIDATE', 'model_path': str(asset_root), 'assets': assets})
        delta_path = self.root / 'warm.delta'
        header = json.dumps({'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        delta_path.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', .01))
        a = {'base_weights_sha256': assets['model.safetensors']['sha256'], 'delta_path': str(delta_path),
             'delta_sha256': p._delta_digest(delta_path), 'arm_sha256': 'c' * 64}
        runtime = {'schema': 'COMP-RS-001-S0-DECODING-1', 'device': 'cuda:0', 'dtype': 'float32',
            'num_threads': 4, 'cublas_workspace_config': ':4096:8', 'deterministic_algorithms': True,
            'deterministic_warn_only': False, 'tf32_cuda_matmul': False, 'tf32_cudnn': False,
            'float32_matmul_precision': 'highest', 'packages': dict(p.PACKAGES), 'python': '3.11.15',
            'cuda': {'device_name': 'SYNTHETIC METADATA CUDA', 'driver_version': 'SYNTHETIC'},
            'decoding': {'greedy': True, 'do_sample': False, 'batch_size': 1, 'max_new_tokens': 128,
                'enable_thinking': False, 'pad_token_id': 151643, 'eos_token_id': [151645, 151643]}}
        self.runtime = runtime
        q1_ref = self.bound('q1', {**native, 'schema': 'COMP-RS-001-S0-Q1-FP32-1', 'stage': 'S0_Q1_FP32',
            'passed': True, 'failed_checks': [], 'decoding_preimage': runtime, 'decoding_sha256': p.digest(runtime),
            'host_floor': {'peak_worker_rss_gib': 2.5}})
        warm_ref = self.bound('warm', {**native, 'schema': 'COMP-RS-001-S0-WARMSTART-1', 'stage': 'S0_WARMSTART',
            'completed': True, 'outcome': 'WARMSTART_COMPLETE', 'anchor': a,
            'plan': {'rows': 600, 'steps': 75}, 'training': {'steps': 75}, 'steps': [{'index': n} for n in range(75)],
            'fingerprints': {'frozen_unchanged': True, 'trainable_changed': True, 'final_equals_theta0': True},
            'inputs': {'dataset_manifest_sha256': p.MANIFEST, 'asset_receipt': asset_ref['path'],
                'asset_receipt_sha256': asset_ref['sha256'], 'tokenizer_receipt_sha256': 'f' * 64},
            'delta': {'path': str(delta_path), 'data_section_sha256': a['delta_sha256'],
                'file_sha256': p.file_record(delta_path)['sha256']}})
        fa = dict(a, warmstart_receipt_sha256=warm_ref['sha256'])
        initial_ref = self.bound('initial', {**native, 'stage': 'S0_RUNTIME_FREEZE', 'schema': 'COMP-RS-001-S0-FREEZE-WARM-1',
            'anchor': fa, 'decoding_preimage': runtime, 'decoding_sha256': p.digest(runtime),
            'scoring_code_bundle_sha256': 'd' * 64, 'host_floor_gib': 3.5, 'inventory': {'name': 'PENDING'},
            'inputs': {'warmstart': warm_ref, 'q1': q1_ref}})
        initial = self.values['initial']
        bound_initial = dict(initial, sha256=initial_ref['sha256'])
        competence_ref = self.bound('competence', {**native, 'schema': 'COMP-RS-001-S0-COMPETENCE-2',
            'stage': 'S0_COMPETENCE', 'completed': True, 'outcome': 'SIGNAL', 'anchor': a,
            'runtime_freeze': bound_initial, 'decoding_sha256': p.digest(runtime), 'scorer_bundle': {'expected': 'd' * 64},
            'group_size': 8, 'm_min': .1, 'delta_acq_rows': 13,
            'inputs': {'dataset_manifest_sha256': p.MANIFEST, 'full_plan': True, 'planned': 450,
                      'plan_by_skill': {'R': 150, 'S': 150, 'P': 150}},
            'summary': {skill: {'prompts': 150, 'samples': 1200, 'G': 8,
                'groups': {'mixed': 15, 'all_fail': 135, 'all_success': 0}} for skill in ('R', 'S', 'P')}})
        a_ref = self.bound('pilotA', {**native, 'stage': 'S0_PILOT_A', 'anchor': fa,
            'runtime_freeze': bound_initial, 'decoding_sha256': p.digest(runtime),
            'decision': {'outcome': 'RERUN_WITH_B', 'peak_worker_rss_gib': 2.0}})
        b_ref = self.bound('pilotB', {**native, 'stage': 'S0_PILOT_B', 'anchor': fa,
            'runtime_freeze': bound_initial, 'decoding_sha256': p.digest(runtime), 'completed': True,
            'error': None, 'oom': None, 'abort': None, 'updates': 1,
            'decision': {'inventory': 'B', 'outcome': 'ADOPT_B', 'peak_worker_rss_gib': 2.2},
            'after_a_pilot': {'sha256': a_ref['sha256'], 'anchor': fa, 'outcome': 'RERUN_WITH_B', 'registered_run': True}})
        adopted_ref = self.bound('adopted', {**initial, 'schema': 'COMP-RS-001-S0-FREEZE-WARM-PILOT-1',
            'inventory': {'name': 'B', 'blocks': [14, 27], 'registered_numel': 220233216,
                'decision': {'inventory': 'B', 'outcome': 'ADOPT_B'}},
            'inputs': {'initial_warm_freeze': initial_ref, 'competence': competence_ref, 'pilot': b_ref, 'a_pilot': a_ref}})
        adopted = self.values['adopted']
        bound_adopted = dict(adopted, sha256=adopted_ref['sha256'])
        runs = {}
        for run in (1, 2):
            path = self.root / ('enroll%d.jsonl' % run)
            ids = ['synthetic-row-%d' % index for index in range(1650)]
            bind = {'stage': 'S0_ENROLL_RUN%d' % run, 'run': run, 'anchor': a,
                    'decoding_sha256': p.digest(runtime), 'plan_sha256': p.digest(ids)}
            with path.open('wb') as stream:
                for row_id in ids:
                    body = dict(bind, id=row_id, process={'pid': run, 'process_created': 1.0, 'session': 'session%d' % run})
                    row = dict(body, row_sha256=p.digest(body))
                    stream.write(p.canonical(row) + b'\n')
            journal = dict(p.file_record(path), rows=1650, bind=bind, bind_sha256=p.digest(bind))
            ref = self.bound('enrollment%d' % run, {**native, 'schema': 'COMP-RS-001-S0-ENROLL-1',
                'stage': 'S0_ENROLL_RUN%d' % run, 'run': run, 'completed': True, 'native_outputs_scored': True,
                'fingerprint_unchanged': True, 'runtime_freeze': bound_adopted, 'anchor': a,
                'decoding_sha256': p.digest(runtime), 'scorer_bundle': {'expected': 'd' * 64},
                'inputs': {'full_guard_dev': True, 'full_calibration': True, 'dataset_manifest_sha256': p.MANIFEST,
                    'planned': 1650, 'plan_by_bank': {'guard_dev': 1200, 'calibration': 450}},
                'summary': {'guard_dev': {'rows': 1200}, 'calibration': {'rows': 450}}, 'journal': journal})
            runs[str(run)] = {'receipt': ref['path'], 'receipt_sha256': ref['sha256'],
                'journal': journal['path'], 'journal_sha256': journal['sha256'],
                'processes': [[run, 1.0]], 'sessions': ['session%d' % run]}
        churn_ref = self.bound('churn', {**native, 'schema': 'COMP-RS-001-S0-CHURN-1', 'stage': 'S0_CHURN',
            'outcome': 'CHURN_ZERO', 'churn': 0, 'rows_compared': 1650, 'runtime_freeze': bound_adopted,
            'separate_processes': True, 'separate_processes_required': True, 'shared_processes': [],
            'scores_equal': {'guard_dev_scoring': True, 'calibration_scoring': True}, 'runs': runs})
        proposed = proposal()
        proposal_ref = self.bound('proposal', proposed)
        planned = planning.build_plan(proposed, manifests, arm)
        plan_ref = self.bound('plan', planned.export())
        census = {'schema': 'S1_PROMPT_LENGTH_CENSUS-1', 'measured': True, 'plan_sha256': planned.sha256,
            'tokenizer_receipt_sha256': 'f' * 64,
            'prompt_lengths': {item.row_id: 80 for boundary in planned.boundaries for item in boundary.prompts}}
        census_ref = self.bound('census', census)
        recipe, code = update_recipe(), {'protocol.py': 'e' * 64}
        start = {'kind': 'warm_anchor', 'arm_sha256': a['arm_sha256']}
        prices = {name: 1.0 for name in ('t_gen95', 't_update95', 't_ckpt_upper', 't_init_upper',
                                       't_restore_upper', 't_cleanup_upper', 't_finalize_upper')}
        qualification = {**native, 'schema': p.QUALIFICATION_SCHEMA,
            'passed': True, 'completed': True, 'error': None, 'device': 'cuda:0', 'code_sha256': code,
            'anchor': a, 'start': start, 'inventory': {'name': 'B', 'blocks': [14, 27], 'numel': 220233216},
            'update_recipe_sha256': p.digest(recipe), 'sampler_sha256': p.digest(p.SAMPLER),
            'plan_sha256': planned.sha256, 'manifest_sha256': p.MANIFEST, 'tokenizer_receipt_sha256': 'f' * 64,
            'runtime_preimage': runtime, 'decoding_sha256': p.digest(runtime), 'scorer_bundle_sha256': 'd' * 64,
            'score_parity_passed': True, 'resume_parameters_adam_rng_bitwise': True,
            'gradient_reference_passed': True, 'finite_confined_gradients': True, 'frozen_state_unchanged': True,
            'timing_prices': prices,
            'resources': {'peak_worker_rss_gib': 2, 'checkpoint_delta_peak_worker_rss_gib': 2, 'gpu_total_gib': 8,
                'peak_reserved_gib': 5, 'min_gpu_free_gib': 1, 'min_disk_free_bytes': 40 * 2**30,
                'memory_fraction': .78, 'reserved_fraction_limit': .95}}
        qualification.update(self.qualification_evidence(qualification, planned))
        qualification_ref = self.bound('qualification', qualification)
        timing_ref = self.bound('timings', {'schema': 'S1_TIMING_PRICES-1', 'measured': True,
            'inventory': {'name': 'B', 'blocks': [14, 27], 'numel': 220233216},
            'update_recipe_sha256': p.digest(recipe), 'qualification_sha256': qualification_ref['sha256'], 'prices': prices})
        io_caps = {name: 100000000 for name in p.IO_LIMITS}
        io_value = {'schema': 'S1_IO_BUDGET-1', 'measured': True,
            'plan_sha256': planned.sha256, 'inventory': {'name': 'B', 'blocks': [14, 27], 'numel': 220233216},
            'qualification_sha256': qualification_ref['sha256'], 'caps': io_caps}
        io_ref = self.bound('io_budget', io_value)
        caps = planning.derive_caps(planned, census['prompt_lengths'], prices, 220233216,
                                   unit_io_budget=io_value)
        mapping = {'generation_attempts': 'rollouts_started', 'generated_token_charge': 'generated_tokens_charged',
            'prefill_token_charge': 'prefill_tokens_charged', 'forward_attempts': 'loss_forward_started',
            'backward_attempts': 'backward_started', 'optimizer_attempts': 'optimizer_started',
            'checkpoint_attempts': 'checkpoint_started', 'invocations': 'invocations_started', 'wall_seconds': 'wall_charged_s'}
        limits = {name: caps[key] for name, key in mapping.items()}
        limits.update(host_floor_gib=3.5, resumes=8,
            active_metadata_reserve_bytes=caps['active_metadata_reserve_bytes'],
            active_storage_reserve_bytes=caps['active_storage_reserve_bytes'], disk_floor_bytes=14 * 2**30)
        limits.update(io_caps)
        packet = {'schema': p.SCHEMA, 'approved': True, 'training_authorized': True, 'status': 'APPROVED_FOR_NATIVE_TRAINING',
            'mode': 'production', 'lease_timeout_s': 900, 'checkpoint_cadence': 10,
            'output_dir': str(self.root / 'output'), 'source_checkout': str(self.root / 'checkout'),
            'arm': arm, 'manifest_sha256': p.MANIFEST, 'anchor': a, 'start': start,
            'inventory': {'name': 'B', 'blocks': [14, 27], 'numel': 220233216}, 'code_sha256': code,
            'update_recipe': recipe, 'sampler': dict(p.SAMPLER), 'proposal': proposal_ref, 'qualification': qualification_ref,
            'data_manifest': data_ref, 'atomic_manifests': manifest_refs, 'plan': plan_ref, 'census': census_ref, 'timings': timing_ref,
            'io_budget': io_ref,
            'prerequisites': {'initial_warm_freeze': initial_ref, 'competence': competence_ref,
                              'adopted_warm_freeze': adopted_ref, 'churn': churn_ref}, 'limits': limits}
        return self.authorize(self.adjusted('packet', packet))

    def authorize(self, packet):
        ref = self.bound('authorization', {'approved': True, 'training_authorized': True,
            'packet_preimage_sha256': p.digest({k: v for k, v in packet.items() if k != 'authorization'})})
        packet['authorization'] = ref
        return packet

    def admit(self, packet, called=None):
        return p.admit_packet(packet, code_sha256={'protocol.py': 'e' * 64}, current_runtime_preimage=self.runtime,
            framework_loader=(lambda: called.append(True)) if called is not None else None)

    def reject(self, packet, message):
        called = []
        with self.assertRaisesRegex(p.AdmissionRejected, message): self.admit(packet, called)
        self.assertEqual(called, [])

    def mutations(self, cases):
        for label, path, value, message in cases:
            with self.subTest(label=label, path=path):
                self.edits = []
                self.edit(label, path, value)
                self.reject(self.packet(), message)

    def test_realistic_synthetic_metadata_chain_reaches_spy_only(self):
        called = []
        self.admit(self.packet(), called)
        self.assertEqual(called, [True])

    def test_MT_uses_two_unit_C_and_caps(self):
        packet = self.packet('MT')
        self.admit(packet)
        self.assertEqual(packet['limits']['generation_attempts'], 60000)
        self.assertEqual(packet['limits']['optimizer_attempts'], 3750)

    def test_unapproved_never_reads_receipts(self):
        with self.assertRaisesRegex(p.AdmissionRejected, 'unapproved'):
            p.admit_packet({'schema': p.SCHEMA, 'approved': False}, code_sha256={})

    def test_change_after_authorization_and_file_tampering(self):
        packet = self.packet()
        packet['arm'] = 'S'
        self.reject(packet, 'authorization')
        packet['arm'] = 'R'
        Path(packet['qualification']['path']).write_text('{}')
        self.reject(packet, 'qualification changed')

    def test_native_grpo_and_sequential_are_not_implemented(self):
        self.mutations([('packet', 'arm', arm, 'sequential') for arm in ('R_to_S', 'S_to_R')])
        self.edits = []
        self.edit('packet', 'update_recipe.objective.method', 'grpo')
        self.edit('packet', 'update_recipe.objective.ddof', 1)
        self.reject(self.packet(), 'GRPO')

    def test_source_and_optimizer_drift(self):
        self.mutations([('packet', 'code_sha256', {'protocol.py': '0' * 64}, 'source bundle'),
                        ('packet', 'update_recipe.optimizer.lr', 2e-6, 'optimizer drift')])

    def test_warm_completion_and_state(self):
        self.mutations([('warm', path, value, 'warmstart') for path, value in
            (('plan.rows', 599), ('training.steps', 74), ('completed', False), ('fingerprints.frozen_unchanged', False))])

    def test_competence_counts_signal_and_scorer(self):
        self.mutations([('competence', path, value, 'competence') for path, value in
            (('group_size', 7), ('inputs.full_plan', False), ('inputs.planned', 449),
             ('summary.R.samples', 1199), ('summary.R.groups.mixed', 14), ('m_min', .01), ('scorer_bundle.expected', '0' * 64))])
        self.edits = []
        self.edit('competence', 'summary.R.groups.mixed', 14)
        self.edit('competence', 'summary.R.groups.all_fail', 136)
        self.reject(self.packet(), 'strict mixed-group gate')

    def test_pilot_update_and_exact_A_ancestry(self):
        self.mutations([('pilotB', path, value, 'pilot') for path, value in
            (('updates', 0), ('after_a_pilot.sha256', '0' * 64), ('completed', False))])

    def test_initial_freeze_precision_digest_floor(self):
        self.mutations([('initial', path, value, 'freeze|runtime|host floor') for path, value in
            (('decoding_sha256', '0' * 64), ('decoding_preimage.dtype', 'bfloat16'), ('host_floor_gib', 2), ('schema', 'wrong'))])

    def test_full_separate_enrollment(self):
        self.mutations([('churn', 'rows_compared', 1200, 'churn'), ('churn', 'churn', False, 'churn'),
            ('churn', 'separate_processes', False, 'churn'), ('enrollment1', 'inputs.full_guard_dev', False, 'enrollment'),
            ('enrollment2', 'completed', False, 'enrollment')])

    def test_rehashed_journal_cannot_fake_full_count_or_row_integrity(self):
        for damage, message in (('truncate', 'population'), ('bad_row', 'row hash')):
            with self.subTest(damage=damage):
                packet = self.packet()
                churn_ref = packet['prerequisites']['churn']
                churn, _ = p.read_json(churn_ref['path'])
                reference = churn['runs']['1']
                enroll, _ = p.read_json(reference['receipt'])
                journal_path = Path(reference['journal'])
                lines = journal_path.read_bytes().splitlines(keepends=True)
                if damage == 'truncate':
                    lines = lines[:1]
                else:
                    row = json.loads(lines[0])
                    row['id'] = 'changed-without-row-hash'
                    lines[0] = p.canonical(row) + b'\n'
                journal_path.write_bytes(b''.join(lines))
                new_hash = p.file_record(journal_path)['sha256']
                enroll['journal']['sha256'] = new_hash
                updated = self.bound('rehashed_enroll', enroll, Path(reference['receipt']))
                reference.update(receipt_sha256=updated['sha256'], journal_sha256=new_hash)
                packet['prerequisites']['churn'] = self.bound('rehashed_churn', churn, Path(churn_ref['path']))
                self.authorize(packet)
                self.reject(packet, message)

    def test_positive_but_wrong_compute_caps(self):
        for key in ('optimizer_attempts', 'generation_attempts', 'invocations', 'prefill_token_charge', 'wall_seconds'):
            with self.subTest(key=key):
                packet = self.packet()
                packet['limits'][key] += 1
                self.authorize(packet)
                self.reject(packet, 'exact plan')

    def test_unit_metadata_reserve_cannot_keep_the_old_fixed_allowance(self):
        packet = self.packet()
        n, writes = packet['inventory']['numel'], packet['limits']['chunk_io_write_bytes']
        self.assertEqual(packet['limits']['active_metadata_reserve_bytes'], writes)
        self.assertEqual(packet['limits']['active_storage_reserve_bytes'], 28 * n + writes)
        packet['limits'].update(active_metadata_reserve_bytes=64 * 1024 ** 2,
                               active_storage_reserve_bytes=28 * n + 64 * 1024 ** 2)
        self.authorize(packet)
        self.reject(packet, 'reservation drift')
        packet['limits'].update(active_metadata_reserve_bytes=None,
                               active_storage_reserve_bytes=28 * n + writes)
        self.authorize(packet)
        self.reject(packet, 'reservation drift')

    def metadata_census_packet(self):
        """Synthetic size envelopes exercise admission; no measured native census."""
        packet = self.packet()
        io = copy.deepcopy(self.values['io_budget'])
        metadata = {'schema': 'S1_METADATA_CENSUS-1', 'code_sha256': packet['code_sha256'],
            'update_recipe_sha256': p.digest(packet['update_recipe']), 'plan_sha256': self.values['plan']['plan_sha256'],
            'io_preimage_sha256': p.digest(io), 'fixed_bytes_upper': 64 * 1024 ** 2,
            'classes': {name: {'writes_upper': 1024, 'serialized_bytes_upper': 4096}
                        for name in ('physical_ledger', 'checkpoint_json', 'checkpoint_journal', 'runner_receipt')}}
        io['metadata_census'] = metadata
        packet['io_budget'] = self.bound('io_budget', io)
        reserve = metadata['fixed_bytes_upper'] + sum(item['writes_upper'] * item['serialized_bytes_upper']
                                                     for item in metadata['classes'].values())
        packet['limits'].update(active_metadata_reserve_bytes=reserve,
                                active_storage_reserve_bytes=28 * packet['inventory']['numel'] + reserve)
        return self.authorize(packet)

    def test_bound_metadata_census_can_admit_smaller_enforced_storage_budget(self):
        packet = self.metadata_census_packet()
        self.admit(packet)
        self.assertEqual(packet['limits']['active_metadata_reserve_bytes'], 80 * 1024 ** 2)
        self.assertLess(packet['limits']['active_metadata_reserve_bytes'], packet['limits']['chunk_io_write_bytes'])
        context = p.admitted_context(packet)
        self.assertEqual(context['caps']['active_metadata_reserve_bytes'], 80 * 1024 ** 2)

    def test_metadata_census_must_bind_source_recipe_plan_and_IO_without_self_hash(self):
        for field in ('code_sha256', 'update_recipe_sha256', 'plan_sha256', 'io_preimage_sha256'):
            with self.subTest(field=field):
                packet = self.metadata_census_packet()
                io = copy.deepcopy(self.values['io_budget'])
                io['metadata_census'][field] = {'protocol.py': '0' * 64} if field == 'code_sha256' else '0' * 64
                packet['io_budget'] = self.bound('io_budget', io)
                self.authorize(packet)
                self.reject(packet, 'metadata census .*binding')

    def test_qualification_counts_cases_and_score_error(self):
        self.mutations([('qualification', path, value, 'qualification|parity|acceptance|fixture case') for path, value in
            (('counts.backwards', 0), ('counts.optimizer_steps', True), ('mechanics_cases.C_greater_than_U', False),
             ('native_score_parity.max_abs_nats', .0002), ('native_score_parity.positions', 16), ('gradient_reference_passed', False))])

    def audit_fixture(self):
        packet = self.packet()
        planned = planning.build_plan(self.values['proposal'],
            {skill: self.values['manifest_' + skill] for skill in planning.ARMS[packet['arm']]}, packet['arm'])
        return copy.deepcopy(self.values['qualification']), packet, planned

    def rebind_evidence(self, qualification, evidence):
        """Rehash downstream fixtures so semantic controls do not stop at stale hashes."""
        ref = self.bound('qualification_evidence', evidence)
        ledger_ref, terminal = self.qualification_ledger(ref, evidence, qualification['counts'])
        seal = copy.deepcopy(self.values['qualification_seal'])
        seal.update(evidence_sha256=ref['sha256'], operations_sha256=p.digest(evidence['operations']),
                    cost_ledger=ledger_ref, terminal_row_sha256=terminal)
        qualification.update(evidence=ref, cost_seal=self.bound('qualification_seal', seal))

    def rebind_ledger(self, qualification, ledger):
        previous = '0' * 64
        for index, entry in enumerate(ledger['entries']):
            entry.update(seq=index, previous_sha256=previous)
            entry.pop('row_sha256', None)
            entry['row_sha256'] = p.digest(entry)
            previous = entry['row_sha256']
        seal = copy.deepcopy(self.values['qualification_seal'])
        seal.update(cost_ledger=self.bound('qualification_ledger', ledger), terminal_row_sha256=previous)
        qualification['cost_seal'] = self.bound('qualification_seal', seal)

    def test_native_samples_need_no_natural_cap_and_uniform_rewards_still_have_parity(self):
        qualification, packet, planned = self.audit_fixture()
        evidence = self.values['qualification_evidence']
        native_generations = [op for op in evidence['operations'] if op['kind'] == 'generation']
        self.assertTrue(all(len(op['output']['completion_ids']) == 2 for op in native_generations))
        self.assertTrue(all(op['output']['reward'] == 0 for op in evidence['operations']
                            if op['kind'] in ('strict_task_scorer', 'independent_strict_task_scorer')))
        audited = p.validate_qualification_evidence(qualification, packet, planned)
        self.assertEqual(audited['counts']['native_sampled_trajectories'], 16)
        self.assertEqual(audited['counts']['native_parity_positions'], 32)
        self.assertEqual(audited['counts']['mechanics_token_fixtures'], 2)
        self.assertTrue(audited['mechanics_cases']['length_cap_without_eos'])

    def test_fixture_relabeling_cannot_be_native_sampling_or_task_rewards(self):
        for change in ('reuse_fixture_forward', 'supplied_tokens', 'override_reward'):
            with self.subTest(change=change):
                qualification, packet, planned = self.audit_fixture()
                evidence = copy.deepcopy(self.values['qualification_evidence'])
                sample = evidence['native_samples'][0]
                by_id = {operation['id']: operation for operation in evidence['operations']}
                if change == 'reuse_fixture_forward':
                    sample['generation_operation_id'] = evidence['token_fixtures'][0]['mask_operation_id']
                elif change == 'supplied_tokens':
                    by_id[sample['generation_operation_id']]['input']['token_source'] = 'supplied_fixture_tokens'
                else:
                    by_id[sample['native_scorer_operation_id']]['input']['reward_override'] = True
                self.rebind_evidence(qualification, evidence)
                with self.assertRaisesRegex(p.AdmissionRejected, 'provenance|fixtures|fixture rewards'):
                    p.validate_qualification_evidence(qualification, packet, planned)

    def test_every_native_token_including_eos_needs_independent_score_parity(self):
        for change in ('missing_eos', 'wrong_eos_likelihood', 'same_scorer', 'different_reward', 'wrong_seed'):
            with self.subTest(change=change):
                qualification, packet, planned = self.audit_fixture()
                evidence = copy.deepcopy(self.values['qualification_evidence'])
                sample = evidence['native_samples'][0]
                by_id = {operation['id']: operation for operation in evidence['operations']}
                if change == 'missing_eos':
                    by_id[sample['teacher_operation_id']]['output']['selected_logprobs'].pop()
                elif change == 'wrong_eos_likelihood':
                    by_id[sample['teacher_operation_id']]['output']['selected_logprobs'][-1] += .001
                elif change == 'same_scorer':
                    by_id[sample['independent_scorer_operation_id']]['input']['implementation_sha256'] = '1' * 64
                elif change == 'different_reward':
                    by_id[sample['independent_scorer_operation_id']]['output']['reward'] = 1
                else:
                    sample['trajectory']['sample_seed'] += 1
                self.rebind_evidence(qualification, evidence)
                with self.assertRaisesRegex(p.AdmissionRejected, 'likelihood|scorer|trajectory'):
                    p.validate_qualification_evidence(qualification, packet, planned)

    def test_mechanics_coverage_derives_from_masks_and_returned_operations(self):
        for change in ('mask', 'missing_adam', 'skip_changed_state', 'gradient_error'):
            with self.subTest(change=change):
                qualification, packet, planned = self.audit_fixture()
                evidence = copy.deepcopy(self.values['qualification_evidence'])
                if change == 'mask':
                    op = next(op for op in evidence['operations'] if op['kind'] == 'loss_forward' and 'mask' in op['output'])
                    op['output']['mask']['eos_positions_selected'] = 0
                elif change == 'missing_adam':
                    op = next(op for op in evidence['operations'] if op['kind'] == 'optimizer_step')
                    op.update(phase='unknown', output={})
                elif change == 'skip_changed_state':
                    op = next(op for op in evidence['operations'] if op['kind'] == 'uniform_skip')
                    op['output']['after']['parameters_sha256'] = '0' * 64
                else:
                    op = next(op for op in evidence['operations'] if op['kind'] == 'reference_backward')
                    op['output']['gradient_reference'].update(max_abs_error=.01, difference_norm=.02)
                self.rebind_evidence(qualification, evidence)
                with self.assertRaisesRegex(p.AdmissionRejected, 'mask|counts|skip|gradient-reference'):
                    p.validate_qualification_evidence(qualification, packet, planned)

    def test_gradient_tolerances_and_scope_are_separately_authorized(self):
        self.mutations([('qualification_scope', 'gradient_reference.absolute_tolerance', None, 'gradient reference'),
            ('qualification_scope', 'gradient_reference.relative_tolerance', 0, 'gradient reference'),
            ('qualification_scope', 'update_recipe_sha256', '0' * 64, 'scope source'),
            ('qualification_authorization', 'qualification_authorized', False, 'scope authorization')])

    def test_bound_cost_ledger_retains_failed_and_unknown_starts(self):
        qualification, packet, planned = self.audit_fixture()
        evidence = copy.deepcopy(self.values['qualification_evidence'])
        evidence['operations'].append({'id': 'SYNTHETIC-UNKNOWN', 'kind': 'generation',
            'scope_sha256': evidence['scope_sha256'], 'origin': 'actual_native_sampling', 'phase': 'unknown',
            'device': 'cuda:0', 'input': {'aborted_before_return': True}, 'output': {}})
        self.rebind_evidence(qualification, evidence)
        result = p.validate_qualification_evidence(qualification, packet, planned)
        costs = result['charged_physical_totals']
        self.assertEqual(costs['failed_attempts'], 1)
        self.assertEqual(costs['unknown_attempts'], 1)
        self.assertEqual(costs['attempts_by_kind']['generation'], 17)
        self.assertEqual(costs['attempts_by_kind']['backward'], 17)
        self.assertEqual(costs['reserved_work_units'], len(evidence['operations']))
        ledger = copy.deepcopy(self.values['qualification_ledger'])
        ledger['entries'] = [entry for entry in ledger['entries'] if entry.get('operation_id') != 'SYNTHETIC-UNKNOWN']
        self.rebind_ledger(qualification, ledger)
        with self.assertRaisesRegex(p.AdmissionRejected, 'all attempts retained'):
            p.validate_qualification_evidence(qualification, packet, planned)

    def test_failed_operation_charge_cannot_exceed_the_retained_reservation(self):
        qualification, packet, planned = self.audit_fixture()
        evidence = copy.deepcopy(self.values['qualification_evidence'])
        failed = next(op for op in evidence['operations'] if op['phase'] == 'failed')
        failed['output']['charged_work_units'] = 2
        self.rebind_evidence(qualification, evidence)
        with self.assertRaisesRegex(p.AdmissionRejected, 'charged start reservation'):
            p.validate_qualification_evidence(qualification, packet, planned)

    def test_ledger_hash_chain_and_rehashed_terminal_charge_cannot_be_substituted(self):
        for change in ('broken_chain', 'lower_charge', 'hex_only'):
            with self.subTest(change=change):
                qualification, packet, planned = self.audit_fixture()
                ledger = copy.deepcopy(self.values['qualification_ledger'])
                if change == 'broken_chain':
                    ledger['entries'][0]['reserved_work_units'] += 1
                    seal = copy.deepcopy(self.values['qualification_seal'])
                    seal['cost_ledger'] = self.bound('qualification_ledger', ledger)
                    qualification['cost_seal'] = self.bound('qualification_seal', seal)
                    message = 'hash chain'
                elif change == 'lower_charge':
                    ledger['entries'][-1]['charged_physical_totals']['failed_attempts'] = 0
                    self.rebind_ledger(qualification, ledger)
                    message = 'charged physical totals'
                else:
                    seal = copy.deepcopy(self.values['qualification_seal'])
                    seal.pop('cost_ledger')
                    seal['cost_ledger_sha256'] = 'a' * 64
                    qualification['cost_seal'] = self.bound('qualification_seal', seal)
                    message = 'cost ledger receipt binding'
                with self.assertRaisesRegex(p.AdmissionRejected, message):
                    p.validate_qualification_evidence(qualification, packet, planned)

    def test_start_order_cannot_substitute_for_returned_dependency_order(self):
        for change in ('generation', 'checkpoint_save', 'optimizer', 'restore'):
            with self.subTest(change=change):
                qualification, packet, planned = self.audit_fixture()
                evidence = self.values['qualification_evidence']
                operations = evidence['operations']
                ledger = copy.deepcopy(self.values['qualification_ledger'])
                if change == 'generation':
                    sample = evidence['native_samples'][0]
                    source, dependent = sample['generation_operation_id'], sample['teacher_operation_id']
                elif change == 'checkpoint_save':
                    source = next(op['id'] for op in operations if op['kind'] == 'checkpoint_save')
                    dependent = next(op['id'] for op in operations if op['kind'] == 'checkpoint_restore')
                elif change == 'optimizer':
                    source = next(op for op in operations if op['kind'] == 'rollback_reuse_check')['output']['returned_optimizer_operation_id']
                    dependent = next(op['id'] for op in operations if op['kind'] == 'checkpoint_restore')
                else:
                    source = next(op['id'] for op in operations if op['kind'] == 'checkpoint_restore')
                    dependent = next(op['id'] for op in operations if op['kind'] == 'rollback_reuse_check')
                returned = next(entry for entry in ledger['entries'] if entry.get('event') == 'returned'
                                and entry.get('operation_id') == source)
                ledger['entries'].remove(returned)
                after = next(index for index, entry in enumerate(ledger['entries'])
                             if entry.get('event') == 'started' and entry.get('operation_id') == dependent)
                ledger['entries'].insert(after + 1, returned)
                self.rebind_ledger(qualification, ledger)
                with self.assertRaisesRegex(p.AdmissionRejected, 'return.*before|returned-work dependency order'):
                    p.validate_qualification_evidence(qualification, packet, planned)

    def test_qualification_runtime_resource_fit(self):
        self.mutations([('qualification', path, value, 'runtime|resource|VRAM|disk') for path, value in
            (('runtime_preimage.dtype', 'bfloat16'), ('resources.peak_reserved_gib', 7), ('resources.min_gpu_free_gib', .1),
             ('resources.peak_worker_rss_gib', None), ('resources.min_disk_free_bytes', 15 * 2**30))])

    def test_plan_census_timing_binding(self):
        self.mutations([('census', 'measured', False, 'census'), ('census', 'plan_sha256', '0' * 64, 'census'),
            ('plan', 'seed', 17, 'plan'), ('timings', 'measured', False, 'timing'), ('timings', 'prices.t_init_upper', 2, 'timing|exact plan')])

    def test_actual_runtime_is_a_separate_postframework_premodel_gate(self):
        packet = self.packet()
        called = []
        p.admit_packet(packet, code_sha256=packet['code_sha256'], framework_loader=lambda: called.append('framework_only'))
        self.assertEqual(called, ['framework_only'])
        bad = copy.deepcopy(self.runtime)
        bad['dtype'] = 'bfloat16'
        with self.assertRaisesRegex(p.AdmissionRejected, 'runtime'):
            p.validate_current_runtime(packet, bad)
        self.assertEqual(p.validate_current_runtime(packet, self.runtime), self.runtime)

    def test_output_paths_and_IO_measurement_scope(self):
        self.mutations([('packet', 'output_dir', str(self.root / 'data' / 'out'), 'output overlaps'),
            ('packet', 'mode', 'qualification', 'production mode'),
            ('io_budget', 'measured', False, 'IO budget'),
            ('io_budget', 'plan_sha256', '0' * 64, 'IO budget')])

    def test_admission_IO_observer_has_exact_bytes_and_bound_operation_ids(self):
        packet = self.packet()
        events = []
        observer = lambda kind, phase, **fields: events.append(dict(kind=kind, phase=phase, **fields))
        p.admit_packet(packet, code_sha256=packet['code_sha256'], io_observer=observer)
        starts = {event['operation_id']: event for event in events if event['phase'] == 'started'}
        ends = {event['operation_id']: event for event in events if event['phase'] == 'returned'}
        self.assertEqual(set(starts), set(ends))
        self.assertGreater(sum(event['bytes_read'] for event in ends.values()), 0)
        self.assertTrue(any(event['identity_hash_attempts_upper'] == 1 for event in starts.values()))
        for identity, event in ends.items():
            self.assertLessEqual(event['bytes_read'], starts[identity]['read_bytes_upper'])
            self.assertGreaterEqual(event['wall_s'], 0)
        denied = []
        with self.assertRaises(p.AdmissionRejected):
            p.admit_packet({'schema': p.SCHEMA, 'approved': False}, code_sha256={},
                           io_observer=lambda *args, **kwargs: denied.append((args, kwargs)))
        self.assertEqual(denied, [])

    def test_public_context_rereads_are_observed_and_grant_nothing(self):
        packet = self.packet()
        self.admit(packet)
        events = []
        context = p.admitted_context(packet, io_observer=lambda kind, phase, **fields: events.append((kind, phase, fields)))
        self.assertFalse(context['execution_authorized_by_context'])
        self.assertEqual(context['planned'].export(), self.values['plan'])
        self.assertEqual(context['caps']['final_C'], 1500)
        self.assertTrue(any(phase == 'started' for _, phase, _ in events))
        self.assertTrue(any(phase == 'returned' and fields['bytes_read'] > 0 for _, phase, fields in events))

    def test_recipe_types_and_explicit_synthetic_CPU(self):
        for key, value in (('B', True), ('G', 1), ('max_new_tokens', 129), ('gradient_clipping', 1)):
            recipe = update_recipe()
            recipe[key] = value
            with self.assertRaises(p.AdmissionRejected): p.validate_update_recipe(recipe)
        recipe = update_recipe(True)
        self.assertEqual(p.validate_update_recipe(recipe, synthetic_cpu=True), recipe)
        with self.assertRaises(p.AdmissionRejected): p.validate_update_recipe(recipe)

    def test_duplicate_and_nonfinite_JSON(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            path = self.root / 'bad.json'
            path.write_text(text)
            with self.assertRaises(p.AdmissionRejected): p.read_json(path)


if __name__ == '__main__': unittest.main()
