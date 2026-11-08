"""Tiny mocked provenance and tensor fixtures; no actual matrix export or Git."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]/'src/upstream.py'
spec=importlib.util.spec_from_file_location('stage_upstream_under_test',SOURCE)
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
ORIGINAL_CONFIG=json.loads((u.TRACK/'configs/alignment_v2.json').read_text())

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')

class FixtureBase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='upstream_fixture_',dir=Path(__file__).parent)
        self.root=Path(self.tmp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(__file__).parent.resolve()))
        self.repo=self.root/'repo';self.programme=self.repo/'experiments/programme';self.track=self.programme/'cross_scale';self.stage=self.track/'generation_stage/v1'
        self.audit_root=self.track/'evidence/07_independent_audit';self.audit=self.audit_root/'pass-audit';self.exportroot=self.stage/'evidence/00_upstream_exports'
        self.patches=[patch.object(u,key,value) for key,value in {'REPO':self.repo,'PROGRAMME':self.programme,'TRACK':self.track,'STAGE':self.stage,'AUDIT_ROOT':self.audit_root,'EXPORT_ROOT':self.exportroot}.items()]
        for value in self.patches:value.start()
        self.run='mock-run';self.capture_dir=self.track/'evidence/01_alignment_matrix'/self.run;self.local=self.track/'local'/self.run
        self.aggregate_dir=self.capture_dir/'aggregate/attempt_000';self.pairroot=self.track/'evidence/02_bridge_objectives'/self.run
        self.committed=[];self.metadata_only=[]
        configpath=self.track/'configs/alignment_v2.json';write(configpath,ORIGINAL_CONFIG)
        src=self.track/'src/fixture_source.py';src.parent.mkdir(parents=True);src.write_text('# frozen fixture\n')
        stage_src=self.stage/'src/fixture_stage.py';stage_src.parent.mkdir(parents=True);stage_src.write_text('# stage fixture\n')
        self.dataset=self.programme/'data/generated/v2/cross_scale/manifest.json';cal=self.dataset.parent/'calibration.jsonl';cal.parent.mkdir(parents=True);cal.write_text('{"prompt":"fixture"}\n')
        write(self.dataset,{'datasets':{'calibration':{'path':'calibration.jsonl','sha256':u._sha(cal)}}})
        contract={'track':'cross_scale','config':copy.deepcopy(ORIGINAL_CONFIG),'config_path':configpath.relative_to(self.repo).as_posix(),'config_sha256':u._sha(configpath),'sources':{src.relative_to(self.repo).as_posix():u._sha(src)},'dataset_manifests':{self.dataset.relative_to(self.repo).as_posix():u._sha(self.dataset)}}
        regid=u._canonical(contract);regpath=self.track/'evidence/00_preregistration'/regid/'registration.json'
        disk={**contract,'registration_sha256':regid,'status':'FROZEN'};write(regpath,disk)
        self.registration={**disk,'registration_path':str(regpath)}
        self.local.mkdir(parents=True);(self.local/'activations.pt').write_bytes(b'opaque fixture activations')
        self.capture={'status':'COMPLETE','dataset_manifest':str(self.dataset),'dataset_manifest_sha256':u._sha(self.dataset),'activation_sha256':u._sha(self.local/'activations.pt'),'donor_identity':{'revision':'fixture'},'supa_identity':{'name':'fixture'},'supa_files':{'model.safetensors':ORIGINAL_CONFIG['checkpoint_sha256']}}
        for name in ('capture.json','run_manifest.json','started.json'):write(self.capture_dir/name,self.capture)
        self.capture_hash=u._sha(self.capture_dir/'capture.json')
        self.reports=[];self.sources=[]
        for i,j in u.PAIRS:
            directory=self.pairroot/f'S{i}_D{j}/attempt_000';bridge=self.local/f'pairs/S{i}_D{j}/attempt_000/bridges.pt';bridge.parent.mkdir(parents=True);bridge.write_bytes(b'opaque fixture bridge')
            report={'supa_layer':i,'donor_layer':j,'objectives':{name:{} for name in u.OBJECTIVES}}
            self.reports.append(report);write(directory/'metrics.json',report)
            terminal={'status':'COMPLETE','capture_sha256':self.capture_hash,'activation_sha256':self.capture['activation_sha256'],'dataset_manifest_sha256':self.capture['dataset_manifest_sha256'],'donor_identity':self.capture['donor_identity'],'supa_identity':self.capture['supa_identity'],'bridge_path':str(bridge),'bridge_sha256':u._sha(bridge),'metrics_sha256':u._sha(directory/'metrics.json')}
            write(directory/'started.json',{'status':'STARTED'});write(directory/'run_manifest.json',terminal)
            self.sources.append({'path':str(directory),'status':'COMPLETE','manifest_sha256':u._sha(directory/'run_manifest.json')})
        self.winner={'supa_layer':8,'donor_layer':12,'objective':'B3','go':True,'checks':{'fixture':True},'ordinary_delta_cosine':.7,'ordinary_delta_relative_mse':.5}
        self.selection={'status':'MECHANISM_GO','selected':[self.winner],'large_generation_licensed':False,'small_generation_eligible_for_separate_freeze':True}
        write(self.aggregate_dir/'matrix.json',{'pairs':self.reports,'sources':self.sources});write(self.aggregate_dir/'selection.json',self.selection)
        aggregate={'status':'COMPLETE','arm':'aggregate','selection':self.selection,'matrix_sha256':u._sha(self.aggregate_dir/'matrix.json'),'dataset_manifest_sha256':self.capture['dataset_manifest_sha256']}
        write(self.aggregate_dir/'started.json',{'status':'STARTED'});write(self.aggregate_dir/'run_manifest.json',aggregate)
        script=self.audit_root/'tools/audit_alignment_matrix_v2.py';script.parent.mkdir(parents=True);script.write_text('# independent fixture auditor\n')
        inventory={'pairs':self.sources,'capture':{'path':str(self.capture_dir),'capture_sha256':self.capture_hash},'aggregates':[{'path':str(self.aggregate_dir),'status':'COMPLETE','manifest_sha256':u._sha(self.aggregate_dir/'run_manifest.json')}]}
        write(self.audit/'attempt_inventory.json',inventory)
        inputs={str(path.resolve()):u._sha(path) for path in self.repo.rglob('*') if path.is_file() and not path.is_relative_to(self.audit_root) and not path.is_relative_to(self.stage)}
        write(self.audit/'input_hashes.json',inputs)
        self.audit_receipt={'schema':'independent_cross_scale_alignment_v2_audit_v1','status':'PASS','run_id':self.run,'matrix_pairs':16,'objective_cells':96,'candidate_gate_records':48,'selection_recomputed':True,'models_loaded':0,'torch_imported':False,'programme_helpers_imported':False,'checkpoint_weight_files_opened':False,'bridge_activation_deserialization':False,'generation_performed':False,'git_commands_run':False,'selection':self.selection,'aggregate_directory':str(self.aggregate_dir),'capture_sha256':self.capture_hash,'registration_contract_sha256':regid,'embedded_registration_sha256':u._canonical(self.registration),'config_canonical_sha256':u._canonical(ORIGINAL_CONFIG),'script_sha256':u._sha(script),'attempt_inventory_sha256':u._sha(self.audit/'attempt_inventory.json'),'input_hashes_sha256':u._sha(self.audit/'input_hashes.json')}
        write(self.audit/'audit_receipt.json',self.audit_receipt)
        def read_capture(run,config,registration,*,metadata_only):
            self.metadata_only.append(metadata_only);self.assertTrue(metadata_only);return self.capture,None
        self.helpers=SimpleNamespace(configure=lambda path:copy.deepcopy(ORIGINAL_CONFIG),verify_registration=lambda *args:self.registration,read_capture=read_capture,collect_completed_pairs=lambda *args:(self.reports,self.sources),select_matrix=lambda *args:self.selection,validate_receipt=lambda *args:None,require_committed=lambda path:self.committed.append(Path(path)))
        self.helperpatch=patch.object(u,'_legacy_helpers',return_value=self.helpers);self.helperpatch.start()

    def tearDown(self):
        self.helperpatch.stop()
        for value in reversed(self.patches):value.stop()
        # Verify the resolved recursive-cleanup target is our test workspace.
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(Path(__file__).parent.resolve()))
        self.tmp.cleanup()

    def verify(self):return u.verify_upstream(self.run,audit_directory=self.audit)

class VerificationTests(FixtureBase):
    def test_complete_mocked_handshake_keeps_generation_closed(self):
        value=self.verify()
        self.assertEqual(value['selected']['objective'],'B3')
        self.assertEqual(value['export_source_objectives'],{'XG':'B3','XS':'BS','XR':'BR'})
        self.assertFalse(value['generation_licensed'])
        self.assertEqual(self.metadata_only,[True])
        self.assertIn(self.aggregate_dir/'selection.json',self.committed)
        self.assertIn(self.audit/'audit_receipt.json',self.committed)
        self.assertEqual(value['context']['supa_hidden_size'],1024)
        self.assertEqual(value['context']['donor_hidden_size'],2560)

    def test_incomplete_audit_rejected_before_helper_import(self):
        self.audit_receipt['objective_cells']=95;write(self.audit/'audit_receipt.json',self.audit_receipt)
        with self.assertRaises(u.UpstreamError):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_post_audit_unfinished_aggregate_rejected(self):
        write(self.aggregate_dir.parent/'attempt_001/started.json',{'status':'STARTED'})
        with self.assertRaisesRegex(u.UpstreamError,'unfinished attempt'):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_post_audit_duplicate_complete_aggregate_rejected(self):
        duplicate=self.aggregate_dir.parent/'attempt_001'
        for name in ('started.json','run_manifest.json'):
            write(duplicate/name,json.loads((self.aggregate_dir/name).read_text()))
        with self.assertRaisesRegex(u.UpstreamError,'exactly one matching complete aggregate'):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_post_audit_closed_failed_aggregate_requires_new_audit(self):
        directory=self.aggregate_dir.parent/'attempt_001'
        write(directory/'started.json',{'status':'STARTED'})
        write(directory/'run_manifest.json',{'status':'FAILED','arm':'aggregate'})
        write(directory/'failure.json',{'type':'FixtureFailure'})
        with self.assertRaisesRegex(u.UpstreamError,'audit omitted/changed current aggregate attempts'):self.verify()

    def test_rehashed_audit_aggregate_status_or_digest_mismatch_rejected(self):
        original=json.loads((self.audit/'attempt_inventory.json').read_text())
        for key,value in (('status','FAILED'),('manifest_sha256','0'*64)):
            with self.subTest(field=key):
                inventory=copy.deepcopy(original);inventory['aggregates'][0][key]=value
                write(self.audit/'attempt_inventory.json',inventory)
                self.audit_receipt['attempt_inventory_sha256']=u._sha(self.audit/'attempt_inventory.json')
                write(self.audit/'audit_receipt.json',self.audit_receipt)
                with self.assertRaisesRegex(u.UpstreamError,'audit omitted/changed current aggregate attempts'):self.verify()

    def test_ambiguous_aggregate_terminal_receipts_rejected(self):
        write(self.aggregate_dir/'abandonment.json',{'status':'ABANDONED'})
        with self.assertRaisesRegex(u.UpstreamError,'ambiguous terminal receipts'):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_ambiguous_pair_terminal_receipts_rejected(self):
        write(self.pairroot/'S8_D12/attempt_000/abandonment.json',{'status':'ABANDONED'})
        with self.assertRaisesRegex(u.UpstreamError,'ambiguous terminal receipts'):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_unknown_pair_root_child_rejected(self):
        (self.pairroot/'S9_D12').mkdir()
        with self.assertRaisesRegex(u.UpstreamError,'unexpected pair-root child'):self.verify()
        self.assertEqual(self.metadata_only,[])

    def test_swapped_registration_digests_rejected(self):
        contract=self.audit_receipt['registration_contract_sha256']
        embedded=self.audit_receipt['embedded_registration_sha256']
        self.assertNotEqual(contract,embedded)
        self.audit_receipt.update(registration_contract_sha256=embedded,embedded_registration_sha256=contract)
        write(self.audit/'audit_receipt.json',self.audit_receipt)
        with self.assertRaisesRegex(u.UpstreamError,'audit registration/config binding differs'):self.verify()

    def test_missing_pair_rejected_before_selection(self):
        self.reports.pop()
        self.helpers.select_matrix=lambda *args:self.fail('selection from partial matrix')
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_missing_null_control_rejected(self):
        del self.reports[0]['objectives']['BS']
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_edited_selection_rejected(self):
        changed=copy.deepcopy(self.selection);changed['selected'][0]['objective']='B1';write(self.aggregate_dir/'selection.json',changed)
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_missing_audit_input_cannot_be_rehashed_into_acceptance(self):
        inputs=json.loads((self.audit/'input_hashes.json').read_text());del inputs[str(self.local/'pairs/S8_D12/attempt_000/bridges.pt')]
        write(self.audit/'input_hashes.json',inputs);self.audit_receipt['input_hashes_sha256']=u._sha(self.audit/'input_hashes.json');write(self.audit/'audit_receipt.json',self.audit_receipt)
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_changed_opaque_bridge_after_audit_rejected(self):
        (self.local/'pairs/S8_D12/attempt_000/bridges.pt').write_bytes(b'changed')
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_uncommitted_upstream_rejected(self):
        def check(path):
            if Path(path)==self.aggregate_dir/'selection.json':raise u.UpstreamError('uncommitted selection')
        self.helpers.require_committed=check
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_audit_cannot_request_checkpoint_file_read(self):
        unsafe=self.repo/'fake-model.safetensors';unsafe.write_bytes(b'do not load weights')
        inputs=json.loads((self.audit/'input_hashes.json').read_text());inputs[str(unsafe)]='a'*64
        write(self.audit/'input_hashes.json',inputs);self.audit_receipt['input_hashes_sha256']=u._sha(self.audit/'input_hashes.json');write(self.audit/'audit_receipt.json',self.audit_receipt)
        original=u._sha
        def protected(path):
            self.assertNotEqual(Path(path),unsafe,'verification tried to read checkpoint weights')
            return original(path)
        with patch.object(u,'_sha',side_effect=protected),self.assertRaises(u.UpstreamError):self.verify()

    def test_no_go_rejected(self):
        self.selection['status']='MECHANISM_NO_GO';self.selection['selected']=[];self.selection['small_generation_eligible_for_separate_freeze']=False
        write(self.aggregate_dir/'selection.json',self.selection)
        aggregate=json.loads((self.aggregate_dir/'run_manifest.json').read_text());aggregate['selection']=self.selection;write(self.aggregate_dir/'run_manifest.json',aggregate)
        with self.assertRaises(u.UpstreamError):self.verify()

    def test_run_and_output_path_escape_rejected(self):
        with self.assertRaises(u.UpstreamError):u.verify_upstream('../bad',audit_directory=self.audit)
        with self.assertRaises(u.UpstreamError):u.export_upstream(self.run,audit_directory=self.audit,output_directory=self.repo/'outside')

class ExportFixture(FixtureBase):
    def tiny_export(self, *, drop_null=False):
        import torch
        torch.set_num_threads(1)
        shapes={'up':(2,3),'up_bias':(3,),'down':(3,2),'down_bias':(2,)}
        original=self.local/'pairs/S8_D12/attempt_000/tiny-original.pt'
        bridges={name:{key:torch.arange(__import__('math').prod(shape),dtype=torch.float32).reshape(shape)+index for key,shape in shapes.items()} for index,name in enumerate(u.OBJECTIVES)}
        if drop_null:del bridges['BS']
        torch.save(bridges,original)
        admitted={'original_bridges':{'path':str(original),'sha256':u._sha(original)},'export_source_objectives':{'XG':'B3','XS':'BS','XR':'BR'},'context':{'generation_stage_source_files':u._stage_sources(),'generation_licensed':False},'generation_licensed':False}
        return torch,shapes,original,bridges,admitted

    def test_tiny_subset_round_trip_exact_and_exclusive(self):
        torch,shapes,original,bridges,admitted=self.tiny_export();before=u._sha(original);output=self.exportroot/'export-fixture'
        with patch.object(u,'verify_upstream',return_value=admitted),patch.object(u,'SHAPES',shapes):
            result=u.export_upstream(self.run,audit_directory=self.audit,output_directory=output)
            with self.assertRaises(u.UpstreamError):u.export_upstream(self.run,audit_directory=self.audit,output_directory=output)
        exported=torch.load(result['tensor_path'],weights_only=True,map_location='cpu')
        for arm,objective in admitted['export_source_objectives'].items():
            for key,value in exported[arm].items():self.assertEqual(u._tensor_hash(value),u._tensor_hash(bridges[objective][key]))
        self.assertEqual(u._sha(original),before);self.assertFalse(result['generation_licensed'])
        manifest=json.loads(Path(result['manifest_path']).read_text());self.assertTrue(manifest['subset_byte_identical_on_reload']);self.assertFalse(manifest['refit_performed'])

    def test_missing_null_tensor_blocks_output_creation(self):
        _,shapes,_,_,admitted=self.tiny_export(drop_null=True);output=self.exportroot/'missing-null'
        with patch.object(u,'verify_upstream',return_value=admitted),patch.object(u,'SHAPES',shapes),self.assertRaises(u.UpstreamError):
            u.export_upstream(self.run,audit_directory=self.audit,output_directory=output)
        self.assertFalse(output.exists())

    def test_wrong_dtype_nan_trainable_and_shape_rejected(self):
        torch,shapes,_,bridges,_=self.tiny_export()
        for mutation in ('dtype','nan','gradient','shape'):
            bridge={k:v.clone() for k,v in bridges['B3'].items()}
            if mutation=='dtype':bridge['up']=bridge['up'].double()
            elif mutation=='nan':bridge['up'][0,0]=float('nan')
            elif mutation=='gradient':bridge['up'].requires_grad_(True)
            else:bridge['up']=bridge['up'].T
            with self.subTest(mutation=mutation),patch.object(u,'SHAPES',shapes),self.assertRaises(u.UpstreamError):u._validate_bridge(bridge,torch)
        with self.assertRaises(u.UpstreamError):u._validate_bridge(bridges['B3'],torch)  # Production dimensions remain 1024/2560.

if __name__=='__main__':unittest.main(verbosity=2)
