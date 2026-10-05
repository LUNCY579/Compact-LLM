"""CPU synthetic metadata/byte controls; never validates native execution."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os,shutil,subprocess,sys,tempfile,types,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='encoder_runtime_v2_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
import encoder_registration_v2 as registration
import admitted_runtime_observer as observer
import long_native_contract as contract
import long_native_worker as worker
from rs_contract import digest,canonical

SHA='1'*64


def fixture_parts():
    """Fake executable maps; helpers/input are real source bytes, no binary reads."""
    versions={'tokenizers':'0.22.2','jinja2':'3.1.6','markupsafe':'3.0.3'}
    enc={'versions':versions,'packages':{},'paths':{},'files':{},'runtime_pins':{}}
    for name,count in (('tokenizers',17),('jinja2',25),('markupsafe',3)):
        root=PACKAGE/'artifacts/CPU_FIXTURE_NOT_NATIVE'/name
        files=[str(root/f'synthetic_{i}.py') for i in range(count)]
        enc['packages'][name]={'root':str(root),'version':versions[name],'files':files}
        for path in files:
            enc['paths'][path]=name;enc['files'][path]={'path':path,'bytes':1,'sha256':SHA};enc['runtime_pins'][path]=SHA
    source={str((PACKAGE/name).resolve()):{'path':str((PACKAGE/name).resolve()),'bytes':(PACKAGE/name).stat().st_size,
        'sha256':registration.filehash(PACKAGE/name)} for name in ('encoder_inventory.py','long_native_contract.py')}
    packet=registration.encoder_prep.contract.PACKET
    input_info={'path':str(packet),'bytes':packet.stat().st_size,'sha256':registration.filehash(packet)}
    source[str(packet)]=input_info
    preview={'runtime_pins':enc['runtime_pins'],'dependency_versions':versions,'executing_package_roster':enc['packages'],
        'source_pins':source,'input_packet':input_info,
        'chat_template_sha256':registration.encoder_prep.contract.native.read_json(packet)['tokenizer_binding']['chat_template_sha256']}
    preview['sha256']=digest(preview)
    package={'kind':registration.encoder_prep.KIND,'native_authority_issued':False,'native_registration_written':False,
        'native_config_written':False,'complete_runtime_execution_closure_established':False,
        'runtime_inventory':enc,'native_registration_preview':preview}
    package['sha256']=digest(package)
    base=registration.stdlib_prep.PHYSICAL_BASE.absolute();logical=registration.stdlib_prep.BASE.absolute()
    alias={'logical_base_prefix':str(logical),'physical_base_prefix':str(base),'file_attributes':0x410,
        'reparse_tag':0xa0000003,'is_symlink':False}
    std={'fixture':False,'policy':registration.stdlib_prep.VERSION,'python_version':'3.11.15',
        'excluded_directory_names':['site-packages','__pycache__'],
        'Windows_system_driver_scope':'inherited separately; no machine closure claim',
        'approved_alias_binding':alias,'base_prefix':str(logical),'physical_base_prefix':str(base),
        'optional_stdlib_zip':{'path':str(base/'python311.zip'),'present':False},'paths':{},'files':{},'runtime_pins':{}}
    for category,count,suffix in (('stdlib_python_source',738,'.py'),('stdlib_extension_runtime',34,'.pyd'),
        ('interpreter_root_executable_runtime',6,'.dll'),('executing_Hermes_launcher',1,'.exe')):
        for i in range(count):
            path=str(base/'CPU_FIXTURE_NOT_NATIVE'/f'{category}_{i}{suffix}')
            std['paths'][path]=category;std['files'][path]={'path':path,'bytes':1,'sha256':SHA};std['runtime_pins'][path]=SHA
    stdlib={'kind':registration.stdlib_prep.KIND,'fixture':False,'before_after_byte_equal':True,
        'native_authority_issued':False,'complete_machine_trust_claimed':False,'inventory':std,
        'helper_source_sha256':registration.filehash(PACKAGE/'stdlib_inventory.py')}
    stdlib['sha256']=digest(stdlib)
    anchors={name:{'path':str(PACKAGE/f'CPU_FIXTURE_{name}.json'),'file_sha256':SHA,'body_sha256':value['sha256']}
        for name,value in (('package',package),('stdlib',stdlib))}
    pr={'status':'PASS_EXACT_ENCODER_PACKAGE_BYTE_REVIEW','candidate_file_sha256':SHA,'candidate_body_sha256':package['sha256'],
        'helper_source_sha256':registration.filehash(PACKAGE/'encoder_inventory.py'),'runtime_pins_sha256':digest(enc['runtime_pins']),
        'observed_versions':versions,'input_packet_sha256':input_info['sha256'],'chat_template_sha256':preview['chat_template_sha256'],
        'consumption_source_pins':{path:entry['sha256'] for path,entry in source.items()},'source_unchanged':True,
        'encoder_framework_model_or_GPU_execution':False,'native_registration_or_config_written':False}
    sr={'status':'PASS_CURRENT779_RUNTIME_BYTE_SUPERSET','candidate_body_sha256':stdlib['sha256'],'candidate_file_sha256':SHA,
        'helper_source_before':stdlib['helper_source_sha256'],'helper_source_after':stdlib['helper_source_sha256'],
        'runtime_files':779,'runtime_bytes':779,'alias_binding':alias,'optional_stdlib_zip':std['optional_stdlib_zip'],
        'forbidden_modules_loaded':[],'native_authority_issued':False,'encoder_framework_model_GPU_execution':False}
    return package,stdlib,pr,sr,contract.base_runtime_pins(),anchors


class RegistrationControls(unittest.TestCase):
    def setUp(self):self.parts=fixture_parts()
    def validate(self,parts=None):return registration.validate_parts(*(parts or self.parts))
    def test_exact_synthetic_merge_preserves_inherited_and_has_no_authority(self):
        value=self.validate();self.assertEqual(len(value['additional_runtime_pins']),824)
        self.assertTrue(all(value['runtime_pins'][path]==sha for path,sha in self.parts[4].items()))
        self.assertFalse(self.parts[0]['native_authority_issued'])
    def test_conflicting_hash_rejected_not_overwritten(self):
        with self.assertRaises(ValueError):registration.merge_pins({str(PACKAGE/'x'):SHA},{str(PACKAGE/'x'):'2'*64})
    def test_package_and_stdlib_omissions_or_extra_are_denied(self):
        for index in (0,1):
            for extra in (False,True):
                parts=deepcopy(self.parts);inv=parts[index]['runtime_inventory' if index==0 else 'inventory']
                path=next(iter(inv['runtime_pins']))
                if extra:
                    new=str(PACKAGE/'CPU_FIXTURE_EXTRA.py');inv['runtime_pins'][new]=SHA;inv['files'][new]={'path':new,'bytes':1,'sha256':SHA};inv['paths'][new]='extra'
                else:
                    for name in ('runtime_pins','files','paths'):inv[name].pop(path)
                with self.subTest(index=index,extra=extra),self.assertRaises(ValueError):self.validate(parts)
    def test_review_status_alone_or_other_candidate_cannot_substitute(self):
        for index,key in ((2,'candidate_file_sha256'),(2,'candidate_body_sha256'),(2,'helper_source_sha256'),
            (2,'input_packet_sha256'),(2,'chat_template_sha256'),(2,'runtime_pins_sha256'),(3,'candidate_file_sha256'),(3,'candidate_body_sha256')):
            parts=deepcopy(self.parts);parts[index][key]='2'*64
            with self.subTest(key=key,index=index),self.assertRaises(ValueError):self.validate(parts)
    def test_review_source_map_and_stale_current_source_denied(self):
        parts=deepcopy(self.parts);parts[2]['consumption_source_pins'].clear()
        with self.assertRaises(ValueError):self.validate(parts)
        parts=deepcopy(self.parts);key=next(iter(parts[0]['native_registration_preview']['source_pins']))
        parts[0]['native_registration_preview']['source_pins'][key]['sha256']='2'*64
        parts[0]['native_registration_preview']['sha256']=digest({k:v for k,v in parts[0]['native_registration_preview'].items() if k!='sha256'})
        parts[2]['consumption_source_pins'][key]='2'*64
        with self.assertRaises(ValueError):self.validate(parts)
    def test_versions_alias_zip_fixture_and_inherited_drift_denied(self):
        mutations=[lambda p:p[0]['runtime_inventory']['versions'].update(tokenizers='0.22.1'),
            lambda p:p[1]['inventory']['approved_alias_binding'].update(reparse_tag=0),
            lambda p:p[1]['inventory']['optional_stdlib_zip'].update(present=True),
            lambda p:p[1].update(fixture=True),lambda p:p[4].pop(next(iter(p[4])))]
        for i,mutate in enumerate(mutations):
            parts=deepcopy(self.parts);mutate(parts)
            with self.subTest(i=i),self.assertRaises(ValueError):self.validate(parts)
    def test_strict_parser_and_actual_anchor_byte_binding(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e309}'):
            with self.assertRaises(ValueError):registration.strict_json(raw)
        with tempfile.TemporaryDirectory() as leaf:
            path=Path(leaf)/'evidence.json';body={'CPU_FIXTURE':True};body['sha256']=digest(body);path.write_text(canonical(body))
            selected={'path':str(path),'file_sha256':registration.filehash(path),'body_sha256':body['sha256']}
            self.assertEqual(registration.anchor(selected),body);path.write_text(canonical({**body,'CPU_FIXTURE':False}))
            with self.assertRaises(ValueError):registration.anchor(selected)
    def test_stage_entrypoints_deny_before_any_paths(self):
        # Explicitly load a new stage-shaped copy even from package/tests.
        with tempfile.TemporaryDirectory() as leaf:
            fake=Path(leaf)/'artifacts/encoder_runtime_v2_work';fake.mkdir(parents=True)
            for name in ('encoder_registration_v2.py','admitted_runtime_observer.py','long_native_contract.py'):
                shutil.copyfile(Path(sys.modules[name[:-3]].__file__),fake/name)
            script="import sys;sys.path[:0]=[sys.argv[1],sys.argv[2]];import encoder_registration_v2 as r,admitted_runtime_observer as o,long_native_contract as c;\nfor f in (lambda:r.registration(None,None),lambda:o.observe(None,None,None,None,phase='before_import',recipe_sha256='1'*64),lambda:c.preflight(None),lambda:c.write_scope(None,None,None)):\n try:f();raise AssertionError('stage admitted')\n except ValueError as e:assert 'staged' in str(e)\nprint('stage denied before paths')"
            run=subprocess.run([sys.executable,'-I','-B','-c',script,str(fake),str(PACKAGE)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)


class ObserverControls(unittest.TestCase):
    def setUp(self):
        self.admission={'owned_supervision_verified':True,'model_load_and_constructor_in_scope':True,
            'binding':{'purpose':observer.PURPOSE,'recipe_sha256':SHA,'role':'trace1','seed':101}}
    def test_synthetic_admitted_byte_phases_keep_preparation_gate_intact(self):
        with tempfile.TemporaryDirectory() as leaf:
            path=Path(leaf)/'tiny.py';path.write_bytes(b'CPU_FIXTURE\n')
            value={'stdlib_inventory':{},'encoder_inventory':{'versions':{'tokenizers':'0.22.2'}},
                'additional_runtime_pins':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()}}
            operations=[];guards=[]
            def call(name,fn):operations.append(name);return fn()
            original=observer.preparation.no_execution
            with patch.object(observer,'STAGED',False),patch.object(observer,'stdlib_roster',return_value={'fixture_roster':True}),\
                patch.object(observer,'encoder_roster',return_value={'fixture_encoder':True}),patch.dict(sys.modules,{'torch':types.ModuleType('torch')}):
                with self.assertRaises(ValueError):original()
                outputs=[observer.observe(value,call,guards.append,self.admission,phase=phase,recipe_sha256=SHA) for phase in ('before_import','after_math')]
                self.assertIs(observer.preparation.no_execution,original)
            self.assertEqual([x['phase'] for x in outputs],['before_import','after_math'])
            self.assertEqual(len(operations),10);self.assertEqual(len(guards),4)
            self.assertEqual(outputs[0]['actual_additional_runtime_pins'],outputs[1]['actual_additional_runtime_pins'])
            self.assertFalse(outputs[1]['native_qualification_established'])
    def test_bad_admission_and_phase_reject_before_file_or_roster_operations(self):
        mutations=[lambda a:a.update(owned_supervision_verified=False),lambda a:a['binding'].update(recipe_sha256='2'*64),
            lambda a:a['binding'].update(purpose='old_V1'),lambda a:a['binding'].update(seed=102)]
        for mutate in mutations:
            admission=deepcopy(self.admission);mutate(admission);calls=[]
            with patch.object(observer,'STAGED',False),self.assertRaises(ValueError):
                observer.observe(None,lambda name,fn:calls.append(name),lambda x:None,admission,phase='before_import',recipe_sha256=SHA)
            self.assertEqual(calls,[])
        with patch.object(observer,'STAGED',False),self.assertRaises(ValueError):observer.validate_admission(self.admission,'wrong',SHA)
    def test_byte_drift_and_post_roster_drift_fail_with_retained_calls(self):
        with tempfile.TemporaryDirectory() as leaf:
            path=Path(leaf)/'tiny.py';path.write_bytes(b'CPU_FIXTURE')
            value={'stdlib_inventory':{},'encoder_inventory':{'versions':{}},'additional_runtime_pins':{str(path):'0'*64}}
            calls=[]
            def call(name,fn):calls.append(name);return fn()
            with patch.object(observer,'STAGED',False),patch.object(observer,'stdlib_roster',return_value={}),patch.object(observer,'encoder_roster',return_value={}):
                with self.assertRaises(ValueError):observer.observe(value,call,lambda x:None,self.admission,phase='after_math',recipe_sha256=SHA)
            self.assertEqual(len(calls),3)
            value['additional_runtime_pins'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();calls.clear()
            with patch.object(observer,'STAGED',False),patch.object(observer,'stdlib_roster',side_effect=[{'a':1},{'a':2}]),patch.object(observer,'encoder_roster',return_value={}):
                with self.assertRaises(ValueError):observer.observe(value,call,lambda x:None,self.admission,phase='after_math',recipe_sha256=SHA)
            self.assertEqual(len(calls),5)
    def test_worker_admission_rejects_before_declared_file_reads(self):
        with patch.object(observer,'STAGED',False),self.assertRaises(ValueError):
            worker.verified_files({'recipe':{'sha256':SHA}},lambda name,fn:self.fail('file read before admission'),lambda x:None,None,phase='before_import')
    def test_runtime_union_conflict_cannot_be_hidden(self):
        with self.assertRaises(ValueError):registration.merge_pins({str(PACKAGE/'same.py'):SHA},{str(PACKAGE/'same.py'):'2'*64})


if __name__=='__main__':unittest.main()
