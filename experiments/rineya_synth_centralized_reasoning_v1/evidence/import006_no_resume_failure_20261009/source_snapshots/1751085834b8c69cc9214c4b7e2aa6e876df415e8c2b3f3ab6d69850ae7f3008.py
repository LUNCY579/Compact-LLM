"""Synthetic finite-runtime roster controls; no imported encoder or native state."""
from pathlib import Path
from copy import deepcopy
import sys,tempfile,unittest,subprocess,json
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='stdlib_runtime_closure_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(CODE));sys.path.append(str(PACKAGE))
import stdlib_inventory as s


class Controls(unittest.TestCase):
    def setUp(self):
        # Fixture enumeration only. Production no_execution is separately
        # proved in an isolated child and never relaxed by the helper.
        self.execution_patch=patch.object(s,'no_execution',lambda:None);self.execution_patch.start();self.addCleanup(self.execution_patch.stop)
        self.temp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'python'
        for name in ('Lib','Lib/urllib','Lib/site-packages','Lib/__pycache__','DLLs'):(self.root/name).mkdir(parents=True,exist_ok=True)
        for name in ('Lib/core.py','Lib/urllib/parser.py','Lib/site-packages/omitted.py','Lib/__pycache__/omitted.py',
                     'DLLs/_fake.pyd','DLLs/fake.dll','python.exe','pythonw.exe','python311.dll','python3.dll','vcruntime140.dll'):
            (self.root/name).write_bytes(('SYNTHETIC_NOT_EXECUTED_'+name).encode())
        (self.root/'Lib/urllib/__init__.py').write_bytes(b'')
        self.exe=self.root/'python.exe'
    def candidate(self):return s.candidate(self.root,self.exe,'3.11.15',fixture=True)
    def verify(self,value):return s.validate(value,value['sha256'],fixture=True)
    def test_exact_finite_roster_includes_empty_source_and_excludes_site_cache(self):
        value=self.candidate();paths=value['inventory']['paths']
        self.assertEqual(10,len(paths));self.assertNotIn(str(self.root/'Lib/site-packages/omitted.py'),paths)
        self.assertNotIn(str(self.root/'Lib/__pycache__/omitted.py'),paths)
        self.assertEqual(0,value['inventory']['files'][str(self.root/'Lib/urllib/__init__.py')]['bytes'])
        self.assertEqual(s.hashlib.sha256(b'').hexdigest(),value['inventory']['files'][str(self.root/'Lib/urllib/__init__.py')]['sha256'])
        self.assertTrue(self.verify(value));self.assertFalse(value['native_authority_issued'])
        (self.root/'DLLs/fake.dll').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'empty Python'):self.candidate()
    def test_same_count_omission_extra_path_and_hash_reseal_rejected(self):
        value=self.candidate();paths=value['inventory']['runtime_pins'];key=next(iter(paths));paths[key+'x']=paths.pop(key)
        value['sha256']=s.digest({k:v for k,v in value.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'current runtime'):self.verify(value)
        value=self.candidate();key=next(iter(value['inventory']['runtime_pins']));value['inventory']['runtime_pins'][key]='0'*64
        value['sha256']=s.digest({k:v for k,v in value.items() if k!='sha256'})
        with self.assertRaises(ValueError):self.verify(value)
    def test_added_or_deleted_executable_and_source_roster_drift(self):
        value=self.candidate();(self.root/'Lib/new.py').write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError,'omitted path'):self.verify(value)
        value=self.candidate();(self.root/'DLLs/fake.dll').unlink()
        with self.assertRaisesRegex(ValueError,'omitted path'):self.verify(value)
    def test_content_change_and_changed_pointer_fail(self):
        value=self.candidate();(self.root/'Lib/core.py').write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError,'current runtime'):self.verify(value)
        value=self.candidate();value['inventory']['executable']=str(self.root/'absent.exe')
        value['sha256']=s.digest({k:v for k,v in value.items() if k!='sha256'})
        with self.assertRaises(ValueError):self.verify(value)
    def test_version_root_and_helper_source_pins_are_exact(self):
        with self.assertRaisesRegex(ValueError,'version'):s.roster(self.root,self.exe,'3.11.14',fixture=True)
        with self.assertRaisesRegex(ValueError,'actual pinned'):s.roster(self.root,self.exe,'3.11.15')
        value=self.candidate();value['helper_source_sha256']='f'*64;value['sha256']=s.digest({k:v for k,v in value.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'helper source'):self.verify(value)
    def test_optional_stdlib_zip_presence_and_absence_frozen(self):
        value=self.candidate();self.assertFalse(value['inventory']['optional_stdlib_zip']['present'])
        (self.root/'python311.zip').write_bytes(b'SYNTHETIC_ZIP_BYTES_NOT_EXECUTED')
        with self.assertRaisesRegex(ValueError,'current runtime'):self.verify(value)
        fresh=self.candidate();self.assertTrue(fresh['inventory']['optional_stdlib_zip']['present']);self.assertTrue(self.verify(fresh))
    def test_symlink_reparse_and_escape_guards_without_link_permission(self):
        file=self.root/'Lib/core.py';original=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda p:p==file or original(p)):
            with self.assertRaisesRegex(ValueError,'symlink'):self.candidate()
        with self.assertRaisesRegex(ValueError,'escaped'):s.pointer(self.exe,file=True,root=self.root/'Lib')
    def test_roster_change_during_hash_and_between_two_passes_rejects(self):
        original=s.info;changed=[False]
        def info(path,root=None):
            value=original(path,root)
            if not changed[0]:changed[0]=True;(self.root/'Lib/race.py').write_bytes(b'fixture')
            return value
        with patch.object(s,'info',info):
            with self.assertRaisesRegex(ValueError,'roster changed'):self.candidate()
    def test_external_hash_and_no_machine_or_native_authority(self):
        value=self.candidate()
        with self.assertRaisesRegex(ValueError,'externally'):s.validate(value,None,fixture=True)
        value['complete_machine_trust_claimed']=True;value['sha256']=s.digest({k:v for k,v in value.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'CPU runtime'):self.verify(value)
        with self.assertRaisesRegex(ValueError,'staging-only'):s.save_staged(self.root/'native_registration.json',value)
    def test_executed_Hermes_launcher_outside_base_is_explicitly_hashed(self):
        outside=self.root.parent/'venv_launcher.exe';outside.write_bytes(b'FAKE_HERMES_LAUNCHER_NOT_EXECUTED')
        value=s.candidate(self.root,outside,'3.11.15',fixture=True)
        self.assertEqual('executing_Hermes_launcher',value['inventory']['paths'][str(outside)])
        self.assertTrue(self.verify(value))
    def test_exact_known_alias_target_and_tag_changes_reject(self):
        expected={'logical_base_prefix':str(s.BASE.absolute()),'physical_base_prefix':str(s.PHYSICAL_BASE.absolute()),
            'file_attributes':0x410,'reparse_tag':0xa0000003,'is_symlink':False}
        with patch.object(s,'pointer',return_value=s.PHYSICAL_BASE):self.assertTrue(s.validate_alias(expected))
        for key,value in (('physical_base_prefix',str(self.root)),('reparse_tag',0),('file_attributes',0x10),('is_symlink',True)):
            wrong={**expected,key:value}
            with self.assertRaisesRegex(ValueError,'approved'):s.validate_alias(wrong)
    def test_isolated_production_import_and_forbidden_module_reject_before_roster(self):
        code="""import sys,types
sys.path.insert(0,sys.argv[1]);sys.path.append(sys.argv[2])
import stdlib_inventory as s
assert not any(n.split('.',1)[0] in s.FORBIDDEN for n in sys.modules)
s.no_execution()
sys.modules['tokenizers']=types.ModuleType('tokenizers')
try:s.roster('must_not_read','must_not_read','3.11.15')
except ValueError as error:assert 'may not execute' in str(error)
else:raise AssertionError('forbidden import accepted')
print('ISOLATED_PRODUCTION_GATE_PASS')
"""
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(CODE),str(Path(s.__file__).resolve().parents[2] if STAGED else PACKAGE)],capture_output=True,text=True,timeout=30)
        self.assertEqual(0,result.returncode,result.stderr);self.assertIn('PRODUCTION_GATE_PASS',result.stdout)


if __name__=='__main__':unittest.main()
