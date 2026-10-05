"""Actual pinned metadata and draft refusal controls; no framework/native work."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_registration as r
from experiments.skill_acquisition_native_v1 import ce_native_backend as b
from experiments.skill_acquisition_native_v1 import ce_native_services as services, ce_service_bootstrap_verifier as replay


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[3]
        cls.proposal_ref,raw=r.raw_reference(cls.root/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json',expected_sha=r.PROPOSAL_SHA)
        cls.proposal=json.loads(raw)
        cls.decision_ref,raw=r.raw_reference(cls.root/'experiments/skill_acquisition_method_lab_v1/evidence/SELECTED_R_ORACLE_SFT_20261002.json',expected_sha=r.DECISION_SHA)
        cls.decision=json.loads(raw)
        arm=cls.proposal['qualification']['witnesses']['arms']['canonical']
        cls.prior_ref,raw=r.raw_reference(arm['ingestion_receipt_ref']['path'],expected_sha=arm['ingestion_receipt_ref']['sha256'])
        cls.prior=json.loads(raw)
        cls.plan=cls.proposal['native_backend']['native_ledger_artifact_interface']['per_phase_arm']['qualification']['canonical']['plan_ref']
        cls.preview=r.metadata_source_preview(cls.proposal['input_refs']['preparation'],cls.plan,cls.proposal['static_recipe_sha256'])
        cls.source=r.source_snapshot(cls.root)
        cls.resource=json.loads((cls.root/'experiments/skill_acquisition_native_v1/evidence/CE8_RESOURCE_READONLY_PROBE_V2_20261001.json').read_bytes())
        cls.worker=json.loads((cls.root/'experiments/skill_acquisition_native_v1/evidence/CE8_WORKER_READONLY_PROBE_V3_20261001.json').read_bytes())

    def documents(self,**changes):
        values={'proposal':self.proposal,'decision':self.decision,'decision_ref':self.decision_ref,'proposal_ref':self.proposal_ref,
            'prior_receipt':self.prior,'source_preview':self.preview,'source':self.source,
            'output_root':Path(tempfile.gettempdir())/'ce8-registration-future-output',
            'artifact_directory':Path(tempfile.gettempdir())/'ce8-registration-artifacts',
            'source_head':'e43d2131c1b9dd145843fcefb359324285ea29cc','resource_probe':self.resource,'worker_probe':self.worker}
        values.update(changes); return r.build_documents(**values)

    def test_actual_source_preview_remains_non_native_and_prior_observation_unchanged(self):
        original=Path(self.prior_ref['path']).read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),self.prior_ref['sha256'])
        self.assertFalse(self.preview['native_execution_authorized']); self.assertFalse(self.preview['tokenizer_recomputed'])
        self.assertEqual(self.preview['model_loads'],0)
        doc=self.documents(); expected=doc['EXPECTED_PREPARED_RECEIPT.json']
        self.assertFalse(expected['observed_current_scope']); self.assertFalse(expected['tokenizer_recomputed'])
        new=expected['expected_receipt']
        self.assertEqual(new['static_recipe_sha256'],self.proposal['static_recipe_sha256'])
        self.assertNotEqual(new['receipt_sha256'],self.prior['receipt_sha256'])
        self.assertEqual(new['prepared_row_bindings'],self.prior['prepared_row_bindings'])
        self.assertEqual(new['census'],self.prior['census'])
        self.assertEqual(Path(self.prior_ref['path']).read_bytes(),original)
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_every_profile_has_known_loader_and_tensor_writer_lower_work(self):
        doc=self.documents(); scope=doc['SCOPE.json']; profiles=scope['qualifier_accounting']['operation_profiles']
        self.assertEqual(doc['SERVICE_SPEC.json']['resource']['host_transient_bytes']['ingestion_boundary'],64*r.MIB)
        self.assertEqual(set(profiles),r.accounting.KINDS)
        context={'bindings':{'scope':scope,'anchors':self.proposal['anchors']}}
        for kind,lower in b.loader_work_lower_bounds(context).items():
            for key,value in lower.items(): self.assertGreaterEqual(profiles[kind][key],value)
        n=self.proposal['inventory']['numel']
        self.assertGreaterEqual(profiles['qualification_tensor_write']['identity_hash_bytes_upper'],8*n+4*r.MIB)
        self.assertGreaterEqual(profiles['qualification_artifact']['reducer_work_bytes_upper'],4*r.MIB)
        self.assertEqual(profiles['cleanup']['wall_s_upper'],19200)
        self.assertEqual(scope['qualifier_accounting']['global_caps'],self.proposal['qualification']['global_caps'])

    def test_real_recipe_witness_contract_and_unapproved_backend_scope(self):
        doc=self.documents(); scope=doc['SCOPE.json']; receipt=doc['EXPECTED_PREPARED_RECEIPT.json']['expected_receipt']
        context={'phase':'qualification','static_recipe_sha256':self.proposal['static_recipe_sha256'],
            'bindings':{'scope':scope,'recipe':{'recipe':self.proposal['static_recipe']},'anchors':self.proposal['anchors'],
                        'runtime':doc['RUNTIME.json']}}
        spec=r.executor.validate_execution_spec(context,receipt)
        self.assertEqual(len(spec['all_witness_ids']),90); self.assertEqual(len(spec['gradient_batch_ids']),8)
        self.assertEqual(len(spec['adam_batches']),2)
        with self.assertRaisesRegex(ValueError,'unmeasured proposal'): b.validate_backend_spec(context)
        # Pure shape audit only: no source permit, registry or native call.
        audit=copy.deepcopy(context); audit['bindings']['scope']['native_backend']['operation_bounds_status']='AUTHORIZED_QUALIFICATION_UPPER'
        b.validate_backend_spec(audit)
        self.assertFalse(scope['approved']); self.assertFalse(scope['qualification_authorized'])

    def test_runtime_anchor_composed_reservations_fit_stage_cap(self):
        doc=self.documents(); spec=doc['SCOPE.json']['qualifier_accounting']
        profiles=spec['operation_profiles']; cap=spec['stage_caps']['runtime_anchor']
        # The native loader has four outer input spans and three observed-file
        # spans. The Q ledger reserves the complete profile for each, even
        # when the inner file is smaller. Thirty-two probes and 1024 metadata
        # events conservatively cover this fixed stage's guards/publication.
        counts={'cleanup':1,'module_import':1,'configured_runtime':1,'process_rng_seed':1,
            'input_io':7,'tokenizer_load':1,'model_load':1,'inventory_configuration':1,
            'fingerprint':1,'qualification_artifact':1,'resource_probe':32,'resource_cuda_probe':32}
        totals={key:sum(profiles[kind][key]*count for kind,count in counts.items())
            for key in ('read_bytes_upper','identity_hash_attempts_upper','identity_hash_bytes_upper')}
        totals['identity_hash_attempts_upper']+=1024
        totals['identity_hash_bytes_upper']+=cap['metadata_bytes_upper']
        self.assertGreater(totals['identity_hash_bytes_upper'],
            self.proposal['qualification']['stage_caps']['runtime_anchor']['identity_hash_bytes_upper'])
        for field,key in (('read_bytes_upper','io_read_bytes_upper'),
                ('identity_hash_attempts_upper','identity_hash_attempts_upper'),
                ('identity_hash_bytes_upper','identity_hash_bytes_upper')):
            self.assertLessEqual(totals[field],cap[key])
        self.assertEqual(cap['identity_hash_bytes_upper'],64*r.GIB)
        old=copy.deepcopy(self.proposal['qualification']['stage_caps']['runtime_anchor'])
        old['identity_hash_bytes_upper']=cap['identity_hash_bytes_upper']
        self.assertEqual(cap,old)
        self.assertEqual(spec['global_caps'],self.proposal['qualification']['global_caps'])
        self.assertEqual(spec['physical_wall_seconds_upper'],19200)
        self.assertFalse(doc['SCOPE.json']['qualification_authorized'])
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_foreign_choices_and_changed_source_inputs_cannot_build_selected_packet(self):
        bad=copy.deepcopy(self.decision); bad['choice']='paired_canonical_permuted'
        with self.assertRaises(ValueError): self.documents(decision=bad)
        bad=copy.deepcopy(self.decision_ref); bad['sha256']='0'*64
        with self.assertRaises(ValueError): self.documents(decision_ref=bad)
        with self.assertRaises(ValueError): r.raw_reference(Path(self.prior_ref['path']),expected_sha='0'*64)
        with self.assertRaises(ValueError): self.documents(output_root=self.root/'unsafe-output')

    def test_child_environment_and_launch_templates_are_explicit_without_credentials(self):
        with patch.dict(r.os.environ,{'API_KEY':'do-not-copy','TOKEN':'do-not-copy','PROGRAMFILES':r'C:\Program Files',
                                    'USERNAME':'native-fixture-user'}):
            environment=r.child_environment()
        self.assertNotIn('API_KEY',environment); self.assertNotIn('TOKEN',environment)
        self.assertEqual(environment['PROGRAMFILES'],r'C:\Program Files')
        self.assertEqual(environment['USERNAME'],'native-fixture-user')
        doc=self.documents(); scope=doc['SCOPE.json']
        fake_identity={'os_identity':{'pid':1,'creation_filetime_100ns':1,'boot_identity':'pure-fixture'}}
        context={'request_sha256':'a'*64,'scope_sha256':'b'*64,'bindings':{'scope':scope,'source':self.source}}
        for branch in r.checkpoint.BRANCHES:
            spec=r.processes.child_launch_spec(context,branch,{'path':str(Path(tempfile.gettempdir())/'REQUEST.json'),'sha256':'c'*64,'bytes':100},fake_identity)
            self.assertEqual(spec['worker_module'],'experiments.skill_acquisition_native_v1.ce_checkpoint_worker')
            self.assertEqual(spec['approved_code_sha256'],self.source['code_sha256'])
            self.assertEqual(spec['environment']['PYTHONHASHSEED'],'20261001')
        self.assertIn('worker_handle_duplicate',doc['SERVICE_SPEC.json']['bootstrap']['operation_profiles'])

    def test_create_only_ref_chain_and_all_native_routes_stay_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            refs=r.write_draft(self.documents(),Path(temp),{'decision':self.decision_ref,'preparation':self.proposal['input_refs']['preparation'],'plan':self.plan})
            for ref in refs.values():
                raw=Path(ref['path']).read_bytes(); self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256']); self.assertEqual(len(raw),ref['bytes'])
            request=json.loads(Path(refs['REQUEST.json']['path']).read_bytes()); auth=json.loads(Path(refs['SCOPE_AUTHORIZATION.json']['path']).read_bytes())
            self.assertEqual(auth['request_preimage_sha256'],r.controller.digest(r.controller.request_preimage(request)))
            self.assertFalse(request['approved']); self.assertFalse(request['qualification_authorized']); self.assertFalse(request['training_authorized'])
            class Trap:
                def __getattribute__(self,name): raise AssertionError('native work from draft')
            with self.assertRaises(ValueError): r.controller.run_controller(request,Trap())
            with self.assertRaisesRegex(ValueError,'create-only'): r.write_draft(self.documents(),Path(temp),{})
            self.assertEqual(set(request['preparation_ref']),{'path','sha256'})
        self.assertFalse(r.authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
        self.assertFalse(r.controller._REGISTERED_NATIVE_FACTORIES)

    def test_actual_full_source_file_census_fits_source_reservations_and_raw_replay(self):
        # Exercise the same real _Budget/_file calls that exhausted attempt1,
        # against source-built profiles/caps, without worker/OS/native authority.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); spec=copy.deepcopy(self.documents()['SERVICE_SPEC.json']['bootstrap'])
            spec['owned_directory']=str(root/'service')
            self.assertEqual(spec['explicit_reservation_kinds'],['input_io','worker_source_verification'])
            maximum=spec['operation_profiles']['worker_source_verification']
            self.assertEqual(maximum['write_bytes_upper'],0)
            ledger=services.BootstrapLedger(spec,'a'*64,synthetic_fixture=True,clock=lambda:100.)
            budget=services.worker._Budget({'bounds':{'operation_attempts_upper':100000,'io_read_bytes_upper':4*r.GIB,
                'identity_hash_attempts_upper':100000,'identity_hash_bytes_upper':4*r.GIB,'wall_seconds_upper':120}},ledger.operation)
            source=r.source_snapshot(self.root);references=list(source['source_files'].values())
            references.extend((self.worker['spec']['python_ref'],self.worker['spec']['process_image_ref'],self.decision_ref))
            self.assertGreater(len(references),65) # the old4GiB write cap failed at61source starts
            before=dict(ledger.counters)
            for reference in references:services.worker._file(reference,budget)
            self.assertEqual(ledger.counters['read_bytes']-before['read_bytes'],sum(ref['bytes']+1 for ref in references))
            self.assertLess(ledger.counters['write_bytes']-before['write_bytes'],16*r.MIB)
            starts=[entry for entry in ledger.events if entry['kind']=='operation' and entry['payload']['phase']=='started']
            for entry,reference in zip(starts,references):
                profile=entry['payload']['reservation_profile']
                self.assertEqual(profile['read_bytes_upper'],reference['bytes']+1)
                self.assertEqual(profile['identity_hash_attempts_upper'],1)
                self.assertEqual(profile['identity_hash_bytes_upper'],reference['bytes']+1)
                self.assertEqual(profile['write_bytes_upper'],0)
            # Default-zero metadata wrappers may perform implicit identity
            # arithmetic; preserve their full hash ceiling rather than zero.
            budget.run('worker_source_verification',lambda:None)
            self.assertEqual(ledger.events[-2]['payload']['reservation_profile']['identity_hash_attempts_upper'],maximum['identity_hash_attempts_upper'])
            self.assertEqual(ledger.events[-2]['payload']['reservation_profile']['identity_hash_bytes_upper'],maximum['identity_hash_bytes_upper'])
            legacy=copy.deepcopy(spec);legacy.pop('explicit_reservation_kinds')
            self.assertEqual(services.reservation_profile(legacy,'worker_source_verification',
                {'io_read_bytes_upper':17,'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17}),maximum)
            opaque=services.reservation_profile(spec,'worker_source_verification',
                {'io_read_bytes_upper':17,'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':17,'opaque_source_verification':True})
            self.assertEqual(opaque['identity_hash_bytes_upper'],maximum['identity_hash_bytes_upper'])
            ledger.end();evidence=ledger.inspect_evidence()
            binding={'stage':'service_bootstrap_verification','request_sha256':'a'*64,'scope_sha256':'b'*64,
                'static_recipe_sha256':'c'*64,'code_sha256':source['code_sha256'],'worker_identity_sha256':'d'*64,'attempt_id':'source-census-replay'}
            reader=replay.raw.RawEvidenceReader(lambda *args,**kwargs:None,
                {key:10**12 for key in replay.raw.BUDGET_FIELDS},root)
            result=replay.verify_service_bootstrap({'mode':'synthetic_fixture','request_sha256':'a'*64,'verification_binding':binding},
                spec,evidence['header_ref'],evidence['journal_ref'],evidence['seal_ref'],reader=reader,outer_preflight=lambda *args:None)
            self.assertEqual(result['derived_totals'],ledger.counters)
            self.assertFalse(result['native_qualified']);self.assertFalse(result['execution_verified'])
            self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)


if __name__=='__main__': unittest.main()
