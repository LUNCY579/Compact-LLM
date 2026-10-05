"""Independent parser and exact dependency controls; CPU fake evidence only."""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(PACKAGE))
import phase_replay as replay
import audit_adapter as adapter
import fixture_support as fixtures
spec=importlib.util.spec_from_file_location('auditor_fixture_supplier',PACKAGE/'tests/test_short_phase_collection.py')
supplier=importlib.util.module_from_spec(spec);spec.loader.exec_module(supplier)

def evidence():
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        f,cuda,markers,final,worker,journal,identity=supplier.complete_collection(directory)
        return markers,final,worker,journal

class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.good=evidence()
    def run_good(self,values=None):return replay.replay(*(values or deepcopy(self.good)),fixture=True)
    def test_fixture_never_native_PASS(self):
        proof=self.run_good();self.assertEqual('PASS_CPU_FIXTURE_PHASE_REPLAY_ONLY',proof['status']);self.assertFalse(proof['native_raw_audit_complete'])
        with self.assertRaisesRegex(ValueError,'CPU fixture'):replay.replay(*deepcopy(self.good),fixture=False)
    def test_exact_original_dependency_and14_site_inventory(self):
        source,changes=adapter.prepared_source();self.assertEqual(14,len(changes));self.assertIn('capture_n+79',source)
        self.assertIn('contract.base._tokenizer_runtime()',source)
        with self.assertRaisesRegex(ValueError,'dependency'):adapter.prepared_source(adapter.DEPENDENCY.read_bytes()+b'\n')
    def test_native_signal_closes_before_integrated_or_inventory_paths(self):
        with self.assertRaisesRegex(ValueError,'terminal\+preserved'):
            adapter.prerequisites(terminal_signal=None,inventory_path='UNOWNED',inventory_sha='a'*64,profile_sha='b'*64,
                config_sha='c'*64,preservation_sha='d'*64,source_commit='e'*40)
    def test_strict_missing_end_extra_keys_and_duplicate_start(self):
        m,s,w,j=deepcopy(self.good)
        for bad in (m[:-1],m+[deepcopy(m[0])]):
            with self.assertRaises(ValueError):replay.tree(bad)
        m[0]['phase']['extra']='UNREGISTERED'
        with self.assertRaisesRegex(ValueError,'strict start/end'):replay.tree(m)
    def test_json_duplicate_nonfinite_and_line_boundaries(self):
        for raw in (b'{"name":"x","name":"y"}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n',b'{}',b'x'*65537+b'\n'):
            with self.assertRaises((ValueError,json.JSONDecodeError)):replay.parse(raw)
    def test_guard_query_floor_type_and_false_branch_mutations(self):
        for field,value in (('guard_cuda_is_initialized',1),('guard_cuda_mem_get_info',[1,8*1024**3])):
            m,s,w,j=deepcopy(self.good)
            next(v for v in m if v['phase']['operation']==field and v['phase']['event']=='returned')['phase']['result']=value
            with self.assertRaises(ValueError):replay.replay(m,s,w,j,fixture=True)
    def test_guard_or_query_erasure_and_reordered_return_denied(self):
        m,s,w,j=deepcopy(self.good)
        index=next(i for i,v in enumerate(m) if v['phase']['operation']=='guard_before_inference_forward')
        for removed in (index,index+1):
            with self.assertRaises(ValueError):replay.replay(m[:removed]+m[removed+1:],s,w,j,fixture=True)
        m[0],m[1]=m[1],m[0]
        with self.assertRaises(ValueError):replay.tree(m)
    def test_main_operation_publication_and_actual_Journal_mutations(self):
        for name in ('main_load_titan','publication_loader'):
            m,s,w,j=deepcopy(self.good)
            for v in m:
                if v['phase']['operation']==name:v['phase']['operation']='UNREGISTERED_OPERATION'
            with self.assertRaises(ValueError):replay.replay(m,s,w,j,fixture=True)
        m,s,w,j=deepcopy(self.good);next(v for v in j if v['event']=='attempt')['operation']='HIDDEN_PRIMITIVE'
        with self.assertRaises(ValueError):replay.replay(m,s,w,j,fixture=True)
    def test_final_IO_closed_status_and_projected_write_denials(self):
        for change in (lambda b:b['phase_IO'].update(closed=False),lambda b:b.update(status='FAILED_PHASE_OBSERVATION'),
            lambda b:b['phase_IO']['operations']['phase_final_summary_write'].update(returned=0),
            lambda b:b['phase_IO'].update(marker_bytes_returned=0)):
            m,s,w,j=deepcopy(self.good);change(s);s['sha256']=replay.digest({k:v for k,v in s.items() if k!='sha256'})
            with self.assertRaises(ValueError):replay.replay(m,s,w,j,fixture=True)
    def test_swallowed_terminal_query_failure_is_not_complete(self):
        m,s,w,j=deepcopy(self.good)
        next(v for v in m if v['phase']['operation']=='terminal_cuda_mem_get_info' and v['phase']['event']=='returned')['phase']['error']={'type':'OSError','message':'caught by helper'}
        with self.assertRaisesRegex(ValueError,'complete nested'):replay.replay(m,s,w,j,fixture=True)

class WholePathControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.good=fixtures.native_shaped(evidence())
    def test_whole_filesystem_terminal_loader_and_composition_is_CPU_only(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            package,config,leaf,saved,registration,config_sha,preservation_sha,profile_sha,profile,header,assets=fixtures.filesystem(directory,self.good)
            source,changes=adapter.prepared_source()
            with patch.object(adapter,'PACKAGE',package),patch.object(adapter,'current_head',return_value='a'*40):
                context=adapter._terminal_context(config,leaf,saved,registration,('atomic_RS_DEV',4),config_sha,preservation_sha,profile_sha,'a'*40,fixture=True)
                result=adapter._finish_composition(context,{'status':'CPU_INJECTED_INHERITED_HEAVY_CLOSURE','scoring_authorized':False},
                    source,changes,profile,header,assets,registration,'a'*40,fixture=True)
                self.assertEqual('PASS_CPU_FIXTURE_TERMINAL_COMPOSITION_ONLY',result['status']);self.assertFalse(result['native_raw_audit_complete'])
                self.assertEqual(682,result['phase3_independent_loader_replay']['tensor_chunks'])
                with self.assertRaisesRegex(ValueError,'CPU fixture'):
                    adapter._finish_composition(context,{},source,changes,profile,header,assets,registration,'a'*40,fixture=True,output_path='DENIED')
                with self.assertRaisesRegex(ValueError,'CPU fixture'):
                    adapter._terminal_context(config,leaf,saved,registration,('atomic_RS_DEV',4),config_sha,preservation_sha,profile_sha,'a'*40)
            self.assertFalse(any(n in sys.modules for n in ('torch','transformers','tokenizers')))
    def test_canonical_receipt_does_not_define_asset_execution_order(self):
        m,s,w,j,profile,header,assets=deepcopy(self.good)
        self.assertNotEqual(list(assets),profile['asset_order']);replay.loader_tree(m,w['loader'],profile,header,assets)
        # Swap source asset guard names as a complete balanced tree and reseal.
        first,second=profile['asset_order'][:2]
        for marker in m:
            op=marker['phase']['operation']
            if op.endswith(':'+first):marker['phase']['operation']=op[:-len(first)]+second
            elif op.endswith(':'+second):marker['phase']['operation']=op[:-len(second)]+first
        with self.assertRaisesRegex(ValueError,'source literal guard'):replay.loader_tree(m,w['loader'],profile,header,assets)
    def test_first_copy_order_and_final_native_inventory_order_are_distinct(self):
        m,s,w,j,profile,header,assets=deepcopy(self.good)
        copy_order=replay.validate_header(header,profile['shapes'],header['file_bytes'])
        self.assertNotEqual(copy_order,profile['final_digest_order'])
        for prefix in ('guard_tensor_copy_before:','guard_device_digest_before:'):
            changed=deepcopy(m);found=next(v for v in changed if v['phase']['operation'].startswith(prefix))
            before=found['phase']['operation'];replacement=prefix+profile['final_digest_order'][0]
            for v in changed:
                if v['phase']['ordinal']==found['phase']['ordinal']:v['phase']['operation']=replacement
            with self.assertRaises(ValueError):replay.loader_tree(changed,w['loader'],profile,header,assets)
    def test_source_shape_order_and_header_omission_overlap_dtype_negatives(self):
        for mutation in (lambda h:h['entries'].pop(next(iter(h['entries']))),
            lambda h:next(iter(h['entries'].values())).update(dtype='FP32'),
            lambda h:next(iter(h['entries'].values()))['data_offsets'].__setitem__(0,2),
            lambda h:h.update(header_bytes=1024**2+1)):
            m,s,w,j,p,h,a=deepcopy(self.good);mutation(h)
            with self.assertRaises(ValueError):replay.loader_tree(m,w['loader'],p,h,a)
        with self.assertRaisesRegex(ValueError,'source bytes'):replay.loader_source((PACKAGE/'native_loader.py').read_bytes()+b'\n')
    def test_asset_size_hash_denominators_and_missing_guard_reject(self):
        for mutate in (lambda w,a:a['config.json'].update(bytes=0),lambda w,a:a['config.json'].update(sha256='0'*64),
            lambda w,a:w['loader'].update(payload_chunk_count=681)):
            m,s,w,j,p,h,a=deepcopy(self.good);mutate(w,a)
            with self.assertRaises(ValueError):replay.loader_tree(m,w['loader'],p,h,a)
        m,s,w,j,p,h,a=deepcopy(self.good);m.pop(next(i for i,v in enumerate(m) if v['phase']['operation'].startswith('guard_asset_hash_after:')))
        with self.assertRaises(ValueError):replay.loader_tree(m,w['loader'],p,h,a)
    def test_full_fixture_missing_altered_preserved_and_external_anchors_reject(self):
        for mode in ('missing','altered','external_config','external_profile','external_preservation','external_HEAD'):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                package,c,l,s,r,ch,ph,pf,p,h,a=fixtures.filesystem(directory,self.good)
                if mode=='missing':(l/'PHASE_ACCOUNTING_FINAL.json').unlink()
                if mode=='altered':(s/'worker_result.json').write_bytes(b'{}\n')
                if mode=='external_config':ch='0'*64
                if mode=='external_profile':pf='0'*64
                if mode=='external_preservation':ph='0'*64
                commit='b'*40 if mode=='external_HEAD' else 'a'*40
                with patch.object(adapter,'PACKAGE',package),self.assertRaises((ValueError,FileNotFoundError)):
                    adapter._terminal_context(c,l,s,r,('atomic_RS_DEV',4),ch,ph,pf,commit,fixture=True)
    def test_bounded_strict_file_and_JSONL_overflow(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path=Path(directory)/'CPU_INERT.json'
            for raw in (b'{"value":1e999}',b'{"value":1,"value":2}',b'{"value":NaN}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):replay.read_json(path)
            path.write_bytes(b'{}\n{}\n')
            with self.assertRaisesRegex(ValueError,'line/count'):replay.read_lines(path,maximum_lines=1)
            path.write_bytes(b'x'*32)
            with self.assertRaisesRegex(ValueError,'bounded'):replay.read_bytes(path,16)
            path.write_bytes(b'{ }\n')
            with self.assertRaisesRegex(ValueError,'canonical'):replay.read_lines(path,canonical_lines=True)
    def test_inherited14_site_body_and_explicit_IO_facades(self):
        source,changes=adapter.prepared_source();module=adapter._inherited_module(source)
        self.assertIs(module._phase3_admission,adapter._phase3_admission);self.assertIs(module._phase3_early,adapter._phase3_early)
        original=adapter.DEPENDENCY.read_bytes().decode('utf-8');start=" events=jsonl(leaf/'events.jsonl')";end=' def compact(values):'
        actual=original[original.index(start):original.index(end)].replace('contract.INFERENCE_AUDIT','contract.base.INFERENCE_AUDIT')
        self.assertEqual(actual,source[source.index(start):source.index(end)])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path=Path(directory)/'CPU_INERT.json';path.write_bytes(b'{"value":7}\n')
            proxy=adapter._NativeReads(object())
            self.assertEqual({'value':7},proxy.read_json(path));self.assertEqual(adapter.sha(path.read_bytes()),proxy.filehash(path))
            path.write_bytes(b'{"value":1e999}')
            with self.assertRaisesRegex(ValueError,'finite'):proxy.read_json(path)
    def test_native_public_entry_no_signal_is_inert_even_with_fixture_hooks(self):
        with patch.object(adapter,'_terminal_context',side_effect=AssertionError('terminal read')),patch.object(adapter,'_inherited_module',side_effect=AssertionError('heavy native closure')):
            with self.assertRaisesRegex(ValueError,'terminal\+preserved'):
                adapter.audit_terminal('UNUSED','UNUSED','UNUSED',expected_selector=('atomic_RS_DEV',4),expected_config_file_sha256='a'*64,
                    expected_preservation_body_sha256='b'*64,inventory_registration_path='UNUSED',expected_inventory_registration_file_sha256='c'*64,
                    expected_profile_body_sha256='d'*64,expected_source_commit='e'*40)
    def test_runtime_AST_and_core64_inventory_replacements_reject(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            package,c,l,s,r,ch,ph,pf,p,h,a=fixtures.filesystem(directory,self.good)
            scope=replay.read_json(c);sup=replay.read_json(l/'SUPERVISOR_RECEIPT.json')
            adapter._phase3_admission(scope,sup,r,pf,scope['short_phase_observation_profile'],{},replay.require)
            for change in (lambda b:b.update(qualified_runtime64_sha256='0'*64),
                lambda b:b['counts'].update(runtime=True),lambda b:b['inventories']['runtime'].pop(next(iter(b['inventories']['runtime'])))):
                bad=deepcopy(r);change(bad);bad['sha256']=replay.digest({k:v for k,v in bad.items() if k!='sha256'})
                with self.assertRaises(ValueError):adapter._phase3_admission(scope,sup,bad,pf,scope['short_phase_observation_profile'],{},replay.require)
            bad=deepcopy(scope);bad['short_RAM_monitor_profile']['runtime_extension_pins']={next(iter(scope['runtime_pins'])):'0'*64}
            with self.assertRaisesRegex(ValueError,'AST extension'):adapter._phase3_admission(bad,sup,r,pf,scope['short_phase_observation_profile'],{},replay.require)
    def test_exact_actual_source_dependency_admission_and_same_count_substitution_reject(self):
        package=adapter.PACKAGE;paths=set(package.glob('*.py'))|set((package/'tests').glob('*.py'))
        paths|={package/n for n in ('PILOT_DECISIONS.json','pilot_trainable_inventory.json','NATIVE_QUALIFICATION_PROPOSAL.json','AGENTS.md')}
        paths|={package.parents[1]/n for n in ('AGENTS.md','RINEYA_SYNTH_CENTRALIZED_REASONING.md')}
        pins={str(p.resolve()):replay.file_info(p)['sha256'] for p in paths}
        deps={str(p.resolve()):replay.file_info(p)['sha256'] for p in (HERE/'audit_adapter.py',HERE/'phase_replay.py',HERE/'test_phase_audit.py',
            HERE/'fixture_support.py',adapter.DEPENDENCY,HERE/'pinned_dependencies/PRESERVATION.json')}
        registration={'inventories':{'source':pins},'auditor_dependency_pins':deps}
        self.assertEqual(pins,adapter._source_admission(registration))
        changed=deepcopy(registration);key=next(iter(changed['inventories']['source']));changed['inventories']['source'].pop(key);changed['inventories']['source']['CPU_SUBSTITUTE_SOURCE']='0'*64
        with self.assertRaisesRegex(ValueError,'exact executing source roster'):adapter._source_admission(changed)
        changed=deepcopy(registration);changed['inventories']['source'][str((package/'native_loader.py').resolve())]='0'*64
        with self.assertRaisesRegex(ValueError,'externally pinned source'):adapter._source_admission(changed)
    def test_actual_source_GATE_precedes_import_and_wrong_HEAD_rejects(self):
        # Bad external pins and HEAD strings reject without contract/module import.
        with patch.object(adapter,'_inherited_module',side_effect=AssertionError('unadmitted import')) as loader:
            with self.assertRaisesRegex(ValueError,'exact launch HEAD'):
                adapter.prerequisites(terminal_signal=adapter.SIGNAL,inventory_path='UNUSED',inventory_sha='a'*64,profile_sha='b'*64,
                    config_sha='c'*64,preservation_sha='d'*64,source_commit='z'*40)
            loader.assert_not_called()
    def test_recorded_cleanup_and_resource_reordering_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            package,c,l,s,r,ch,ph,pf,p,h,a=fixtures.filesystem(directory,self.good)
            scope=replay.read_json(c);worker=replay.read_json(l/'worker_result.json');sup=replay.read_json(l/'SUPERVISOR_RECEIPT.json')
            lease=Path(sup['lease_path'])
            adapter._recorded_cleanup(scope,worker,sup,lease,l,ch)
            for key,value in (('owned_worker_identity_currently_alive',True),('gpu_telemetry_cleanup_verified',False),('unjoined_sampler_threads',['CPU_INERT_PENDING'])):
                bad=deepcopy(sup);bad[key]=value
                with self.assertRaises(ValueError):adapter._recorded_cleanup(scope,worker,bad,lease,l,ch)
            resources=replay.read_json(l/'RESOURCE_OBSERVATIONS.json');hb=replay.read_lines(l/'supervisor_heartbeat.jsonl')
            with self.assertRaisesRegex(ValueError,'chronological'):adapter._resource_types(worker,sup,list(reversed(resources)),hb)

if __name__=='__main__':unittest.main()
