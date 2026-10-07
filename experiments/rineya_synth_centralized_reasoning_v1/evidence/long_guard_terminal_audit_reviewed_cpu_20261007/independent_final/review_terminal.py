"""Independent finite CPU replay of saved runtime-to-terminal consumer."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import builtins,hashlib,io,json,sys,time,unittest
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
TARGET=PACKAGE/'artifacts/long_guard_terminal_audit_preparation_work'
PRODUCER=PACKAGE/'evidence/long_guard_owned_runtime_reviewed_cpu_20261007/current001'
CORE=PACKAGE/'evidence/long_guard_collection_reviewed_cpu_20261007/current001'
reads=[];denials=[];imports=[]
real_io,real_open,real_import=io.open,builtins.open,builtins.__import__
def reader(path,mode='r',*args,**kwargs):
    if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
        p=Path(path).resolve();reads.append(str(p))
        if not (p.is_relative_to(TARGET) or p.is_relative_to(HERE) or p.is_relative_to(PRODUCER) and p.suffix=='.py' or p.is_relative_to(CORE) and p.suffix=='.py' or p.parent==PACKAGE and p.suffix=='.py' or p.parent==PACKAGE/'tests' and p.suffix=='.py'):
            denials.append(str(p));raise AssertionError('independent actual input forbidden')
    return real_io(path,mode,*args,**kwargs)
def importing(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','transformers','tokenizers','ctypes','psutil','safetensors','numpy'}:
        imports.append(name);raise AssertionError('independent native/framework import forbidden')
    return real_import(name,*args,**kwargs)
traps=[patch('io.open',side_effect=reader),patch.object(builtins,'open',side_effect=lambda p,*a,**k:reader(p,*a,**k) if isinstance(p,(str,bytes,Path)) else real_open(p,*a,**k)),patch.object(builtins,'__import__',side_effect=importing)]
for trap in traps:trap.start()
T=ModuleType('independent_original_terminal_controls');T.__file__=str(TARGET/'test_long_guard_terminal_audit.py')
exec(compile(Path(T.__file__).read_bytes(),T.__file__,'exec'),T.__dict__)
class Independent(T.TerminalControls):
    def rebind_worker(self,packet,value):
        path=Path(packet['worker']['path']);path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8');packet['worker']=self.ref(path)
        sup_path=Path(packet['supervisor']['path']);sup=self.L._json(sup_path.read_bytes())
        info={k:v for k,v in self.ref(path,json_value=False).items() if k!='path'}
        for row in sup['partial_outputs']:
            if row['path']=='worker_result.json':row.update(info)
        sup_path.write_text(json.dumps(sup,sort_keys=True)+'\n',encoding='utf-8');packet['supervisor']=self.ref(sup_path)
        packet['raw_files']['worker_result.json']=info
        packet['raw_files']['SUPERVISOR_RECEIPT.json']={k:v for k,v in self.ref(sup_path,json_value=False).items() if k!='path'}
        (Path(sup['lease_path'])/'terminal.json').write_bytes(sup_path.read_bytes())
    def test_positive_exact_runtime_and_two_births(self):
        packet,host=self.packet();observed=[]
        # Original supervisor creates both logs; successful workers may emit no
        # stdout/stderr. Keep these exact empty files in its frozen inventory.
        leaf=Path(packet['output_directory']);sup_path=Path(packet['supervisor']['path']);sup=self.L._json(sup_path.read_bytes())
        for name in ('worker.stdout.log','worker.stderr.log'):
            path=leaf/name;path.write_bytes(b'');info={'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}
            if not any(row['path']==name for row in sup['partial_outputs']):sup['partial_outputs'].append({'path':name,**info})
            packet['raw_files'][name]=info
        sup_path.write_text(json.dumps(sup,sort_keys=True)+'\n',encoding='utf-8');packet['supervisor']=self.ref(sup_path)
        packet['raw_files']['SUPERVISOR_RECEIPT.json']={k:v for k,v in self.ref(sup_path,json_value=False).items() if k!='path'}
        (Path(sup['lease_path'])/'terminal.json').write_bytes(sup_path.read_bytes())
        def death(birth):observed.append(deepcopy(birth));return False
        host.is_alive=death;result=self.consume(packet,host)
        self.assertEqual([row['pid'] for row in observed],[7101,7102]);self.assertEqual(result['physical_forward_dose']['attempted_calls'],66)
        self.assertEqual(result['physical_forward_dose']['input_positions'],838+64*2016)
        self.assertTrue(result['fixture_only']);self.assertFalse(result['native_PASS']);self.assertFalse(result['scoring_authorized'])
    def test_consistently_rebound_typed_counter_negatives(self):
        packet,host=self.packet();original=self.L.read_ref(packet['worker'])
        for label,mutate in (('physical',lambda v:v['forward_dose']['operations']['zero_parity_base_forward'].update(attempted_calls=True)),('accounting',lambda v:v['operation_accounting']['operations']['install_physical_forward_meter'].update(attempted=True))):
            changed=deepcopy(original);mutate(changed);changed=self.A.seal({k:v for k,v in changed.items() if k!='sha256'});self.rebind_worker(packet,changed)
            with self.subTest(counter=label),self.assertRaises(ValueError):self.consume(packet,host)
            self.rebind_worker(packet,deepcopy(original))
    def test_identical_outside_leaf_saved_files_reject(self):
        packet,host=self.packet();copydir=self.root/'outside_leaf';copydir.mkdir()
        for label,ref in (('batch',packet['saved']['batch']),('row',packet['saved']['rows'][0]),('token',packet['saved']['tokens'][0][0])):
            copy=copydir/Path(ref['path']).name;copy.write_bytes(Path(ref['path']).read_bytes());changed=deepcopy(packet);selected=self.ref(copy)
            if label=='batch':changed['saved']['batch']=selected
            elif label=='row':changed['saved']['rows'][0]=selected
            else:changed['saved']['tokens'][0][0]=selected
            with self.subTest(selected=label),self.assertRaises(ValueError):self.consume(changed,host)
    def test_independent_native_early_gate(self):
        self.test_native_refusal_and_real_input_traps()
if __name__=='__main__':
    files=('long_guard_terminal_audit.py','test_long_guard_terminal_audit.py');before={n:hashlib.sha256((TARGET/n).read_bytes()).hexdigest() for n in files};start=time.monotonic()
    names=['test_positive_exact_runtime_and_two_births','test_consistently_rebound_typed_counter_negatives','test_identical_outside_leaf_saved_files_reject','test_independent_native_early_gate']
    if '--positive-only' in sys.argv:names=['test_positive_exact_runtime_and_two_births']
    result=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(Independent(n) for n in names));after={n:hashlib.sha256((TARGET/n).read_bytes()).hexdigest() for n in files}
    for trap in reversed(traps):trap.stop()
    receipt=dict(kind='INDEPENDENT_DECODED_LONG_TERMINAL_SAVED_CPU_REVIEW',test_names=names,tests=result.testsRun,success=result.wasSuccessful(),seconds=time.monotonic()-start,source_before=before,source_after=after,source_unchanged=before==after,review_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_paths_and_counts=dict(Counter(reads)),inner_read_paths_and_counts=dict(Counter(Independent.reads)),forbidden_import_attempts=imports+Independent.imports,intentional_real_path_denials=len(denials)+len(Independent.denials),fixture_only=True,native_PASS=False,scoring_authorized=False,native_qualification_established=False)
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();path=HERE/('INDEPENDENT_FOCUSED_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(path=str(path),tests=result.testsRun,success=result.wasSuccessful(),source_unchanged=before==after)));raise SystemExit(0 if result.wasSuccessful() and before==after and not imports else 1)
