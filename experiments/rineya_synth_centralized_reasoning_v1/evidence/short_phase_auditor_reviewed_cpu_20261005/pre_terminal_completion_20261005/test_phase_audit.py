"""Independent parser and exact dependency controls; CPU fake evidence only."""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(PACKAGE))
import phase_replay as replay
import audit_adapter as adapter
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

if __name__=='__main__':unittest.main()
