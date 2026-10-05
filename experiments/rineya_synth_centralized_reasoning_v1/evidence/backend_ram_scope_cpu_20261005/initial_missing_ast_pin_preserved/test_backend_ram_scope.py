"""Focused stdlib/mock amendment003 controls; no frameworks/model/native run."""
import ast
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE))
import backend_qual_contract as contract
sys.path.insert(0,str(HERE))
import backend_qual_worker as worker
import worker_ram_probe as ram

OLD=json.loads((PACKAGE/'configs/SYNTH_BACKENDQ002_RECOMPUTE_A1.json').read_bytes())


def fixture_scope(phase='A'):
    """Use frozen actual rows/LR/dose; replace heavy admission/proof reads only."""
    reg={'sha256':OLD['qualification_recipe']['registration_sha256'],
         'source_and_decision_pins':OLD['qualification_recipe']['source_decision_pins']}
    binding={'profile':'CPU_MOCK_RAM_POLICY_ONLY','callback_sha256':'a'*64,'native_profile_fit_qualified':False}
    checkpoint=deepcopy(OLD['checkpoint_amendment'])
    if phase=='B':
        groups=40;checkpoint['counts'].update(forward=1120,recompute=1120,per_layer_forward=groups,per_layer_recompute=groups)
    with ExitStack() as stack:
        stack.enter_context(patch.object(contract,'selected_packet',return_value=(reg,deepcopy(OLD['rows']))))
        stack.enter_context(patch.object(contract,'inference_binding',return_value=deepcopy(OLD['inference_qualification'])))
        stack.enter_context(patch.object(contract,'worker_ram_binding',return_value=binding))
        stack.enter_context(patch.object(contract,'checkpoint_binding',return_value=checkpoint))
        stack.enter_context(patch.object(contract,'checkpoint_runtime_files',return_value=deepcopy(OLD['checkpoint_runtime_pins'])))
        stack.enter_context(patch.object(contract,'cursor',side_effect=lambda reg,n:deepcopy(OLD['c9_binding'] if n==9 else OLD['c10_binding'])))
        stack.enter_context(patch.object(contract,'_observed_duration',side_effect=lambda path,op:6.874 if op=='verify_frozen_base' else 45.900))
        return contract.build_scope(phase,predecessor=None if phase=='A' else{})


class Controls(unittest.TestCase):
    def test_mock_built_A_B_keep_every_native_dose_LR_model_runtime_gate(self):
        for phase in ('A','B'):
            scope=fixture_scope(phase)
            self.assertEqual('SYNTH-BACKENDQ-003-WORKER-RAM',scope['id']);self.assertEqual(contract.WORKER_PROFILE,scope['profile'])
            self.assertEqual(OLD['base_compute_profile'],scope['base_compute_profile'])
            for key in ('rows','source_packet_sha256','resources','tolerances','model_pins','precision','global_learning_rates','c9_binding','c10_binding'):
                self.assertEqual(OLD[key],scope[key],key)
            self.assertEqual(82,len(scope['runtime_pins']));self.assertEqual(OLD['runtime_pins'],scope['runtime_pins'])
            self.assertEqual((16,9),(scope['resource_feasibility']['A_total_full_base_checks'],scope['resource_feasibility']['B_total_full_base_checks']))
            self.assertFalse(scope['owned_qualification_authorized']);self.assertTrue(scope['staged_native_dispatch_forbidden'])
            self.assertEqual((202,200,10) if phase=='A' else(42,20,1),tuple(scope['dose'][k] for k in ('maximum_forward_calls','native_backwards','native_optimizer_steps')))
            self.assertEqual(48112 if phase=='A' else 10650,scope['dose']['expected_forward_input_positions'])
            self.assertEqual(5600 if phase=='A' else 1120,scope['checkpoint_amendment']['counts']['recompute'])
            self.assertEqual(scope['worker_ram_amendment'],scope['qualification_recipe']['worker_ram_amendment'])
    def test_old_A1_and_V2_config_profile_and_contract_rejected_before_modelwork(self):
        for name in ('SYNTH_BACKENDQ001_A1.json','SYNTH_BACKENDQ002_RECOMPUTE_A1.json'):
            body=json.loads((PACKAGE/'configs'/name).read_bytes())
            with self.assertRaisesRegex(ValueError,'authority/phase'):contract.validate_packet(body)
        body=fixture_scope();body['profile']=contract.CHECKPOINT_PROFILE
        with patch.object(contract,'checkpoint_binding',return_value=body['checkpoint_amendment']):
            with self.assertRaisesRegex(ValueError,'profile/proof'):contract.validate_packet(body)
    def test_exact_RAM_policy_and_recipe_closure_rejects_forged_callback(self):
        body=fixture_scope();policy=deepcopy(body['worker_ram_amendment'])
        for mutate in (lambda b:b.pop('worker_ram_amendment'),lambda b:b['worker_ram_amendment'].update(callback_sha256='0'*64),
                       lambda b:b['qualification_recipe']['worker_ram_amendment'].update(profile='other')):
            changed=deepcopy(body);mutate(changed)
            with patch.object(contract,'checkpoint_binding',return_value=body['checkpoint_amendment']),patch.object(contract,'worker_ram_binding',return_value=policy):
                with self.assertRaisesRegex(ValueError,'RAM callback'):contract.validate_packet(changed)
    def test_actual_small_CPU_primary_proof_pins_and_bootstrap_transplant_body(self):
        binding=contract.worker_ram_binding()
        self.assertFalse(binding['numeric_cache']);self.assertFalse(binding['fallback']);self.assertFalse(binding['native_profile_fit_qualified'])
        self.assertEqual(contract.RAM_PRIMARY_BODY_SHA,binding['primary_review_body_sha256'])
        self.assertEqual(contract._callback_ast(HERE/'worker_ram_probe.py'),contract._callback_ast(contract.RAM_PROOF_DIR/'worker_ram_probe.py'))
        self.assertEqual(PACKAGE,ram.PACKAGE);self.assertEqual(PACKAGE,contract.PACKAGE)
        self.assertEqual(PACKAGE/'native_process.py',Path(ram.native_process.__file__).resolve())
    def test_CPU_primary_and_preservation_proof_drift_denied(self):
        for field in ('RAM_PROOF_SHA','RAM_PRESERVATION_SHA','RAM_PRIMARY_FILE_SHA'):
            with self.subTest(field=field),patch.object(contract,field,'0'*64):
                with self.assertRaisesRegex(ValueError,'proof byte drift'):contract.worker_ram_binding()
    def test_worker_diff_changes_only_bootstrap_import_and_available_injection(self):
        original=(PACKAGE/'backend_qual_worker.py').read_text();staged=(HERE/'backend_qual_worker.py').read_text()
        expected=original.replace('"backend_qual_work","backend_recompute_scope_work"','"backend_qual_work","backend_recompute_scope_work","backend_ram_scope_work"')
        expected=expected.replace('from inference_qual_worker import make_guard\n','from inference_qual_worker import make_guard\nimport worker_ram_probe\n')
        expected=expected.replace('guard=make_guard(output,owner,start,lambda:torch)','guard=make_guard(output,owner,start,lambda:torch,available=worker_ram_probe.available_bytes)')
        self.assertEqual(expected,staged)
        calls=[n for n in ast.walk(ast.parse(staged)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_guard']
        self.assertEqual(1,len(calls));self.assertEqual(['available'],[kw.arg for kw in calls[0].keywords])
    def test_guard_fresh_RAM_sameAPIerrors_and_unchanged_floors(self):
        owner={'job_name':'CPU_FIXTURE','supervisor_pid':1};amounts=[2*1024**3,1]
        with patch.object(ram.native_process,'system_memory_status',side_effect=lambda:{'available_bytes':amounts.pop(0),'total_bytes':4*1024**3}) as actual:
            guard=worker.make_guard(HERE,owner,0.,lambda:None,available=ram.available_bytes,owned_resources=lambda name:{'tree_rss_bytes':0},
                heartbeat_reader=lambda path:{'unix_seconds':10.,'supervisor_pid':1},monotonic=lambda:0.,wall=lambda:10.)
            guard('one')
            with self.assertRaisesRegex(ValueError,'host resource'):guard('two')
            self.assertEqual(2,actual.call_count)
        error=OSError('GlobalMemoryStatusEx failure')
        with patch.object(ram.native_process,'system_memory_status',side_effect=error):
            with self.assertRaises(OSError) as caught:ram.available_bytes()
            self.assertIs(error,caught.exception)
    def test_source_policy_and_staging_denial_before_framework_or_files(self):
        pins=contract.source_pins()
        self.assertIn(str((HERE/'worker_ram_probe.py').resolve()),pins)
        self.assertIn(str((PACKAGE/'tests/test_backend_recompute_scope.py').resolve()),pins)
        with self.assertRaisesRegex(ValueError,'staged native dispatch'):contract.preflight(HERE/'missing.json')
        with self.assertRaisesRegex(ValueError,'staging cannot'):contract.write_scope(HERE/'missing.json','A',authorize_owned_qualification=True)
        with self.assertRaisesRegex(ValueError,'staged native dispatch'):worker.run(HERE/'missing.json',HERE)
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers')))


if __name__=='__main__':unittest.main()
