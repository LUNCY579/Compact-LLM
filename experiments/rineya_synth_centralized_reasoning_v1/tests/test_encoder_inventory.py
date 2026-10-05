"""CPU synthetic installed-roster controls; no encoder/framework execution."""
from copy import deepcopy
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='long_encoder_registration_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(CODE));sys.path.append(str(PACKAGE))
import encoder_inventory as inventory


class Distribution:
    def __init__(self,base,name,version):
        self.base,self.name,self.version=base,name,version
        self.files=[Path(name)/'__init__.py',Path(name)/'implementation.py',Path(name)/'fake_native.pyd']
    def locate_file(self,entry):return self.base/entry


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.distributions={}
        for name,version in (('tokenizers','0.22.2'),('jinja2','3.1.6'),('markupsafe','3.0.3')):
            base=self.root/name;base.mkdir()
            for filename in ('__init__.py','implementation.py','fake_native.pyd'):(base/filename).write_bytes(('CPU_FIXTURE_NOT_EXECUTED_'+filename).encode())
            self.distributions[name]=Distribution(self.root,name,version)
        self.provider=self.distributions.__getitem__
    def test_complete_exact_declared_python_and_native_superset_CPU_fixture(self):
        value=inventory.runtime_inventory(self.provider)
        self.assertEqual(9,len(value['runtime_pins']));self.assertTrue(inventory.validate_runtime(value,self.provider))
        self.assertEqual({'tokenizers':'0.22.2','jinja2':'3.1.6','markupsafe':'3.0.3'},value['versions'])
    def test_omitted_record_and_untracked_extra_executable_denied(self):
        self.distributions['tokenizers'].files.pop()
        with self.assertRaisesRegex(ValueError,'omitted or extra'):inventory.runtime_roster(self.provider)
        self.setUp();(self.root/'jinja2/untracked.py').write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError,'omitted or extra'):inventory.runtime_roster(self.provider)
    def test_same_count_path_substitution_and_declared_extra_denied(self):
        value=inventory.runtime_inventory(self.provider);key=next(iter(value['runtime_pins']));value['runtime_pins'][key+'x']=value['runtime_pins'].pop(key)
        with self.assertRaisesRegex(ValueError,'exact encoder'):inventory.validate_runtime(value,self.provider)
        self.distributions['markupsafe'].files.append(Path('markupsafe/missing.py'))
        with self.assertRaisesRegex(ValueError,'regular'):inventory.runtime_roster(self.provider)
    def test_tokenizers_jinja_and_actual_MarkupSafe_version_drift_denied(self):
        for name,bad in (('tokenizers','0.22.1'),('jinja2','3.1.5')):
            old=self.distributions[name].version;self.distributions[name].version=bad
            with self.assertRaisesRegex(ValueError,'version'):inventory.runtime_roster(self.provider)
            self.distributions[name].version=old
        value=inventory.runtime_inventory(self.provider);self.distributions['markupsafe'].version='3.0.4'
        with self.assertRaisesRegex(ValueError,'stale candidate'):inventory.validate_runtime(value,self.provider)
    def test_hash_drift_and_self_resigned_runtime_map_denied(self):
        value=inventory.runtime_inventory(self.provider);(self.root/'tokenizers/implementation.py').write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError,'stale candidate'):inventory.validate_runtime(value,self.provider)
        value['runtime_pins']={key:'0'*64 for key in value['runtime_pins']}
        with self.assertRaises(ValueError):inventory.validate_runtime(value,self.provider)
    def test_path_escape_and_pointer_change_denied(self):
        self.distributions['tokenizers'].files[0]=Path('jinja2/__init__.py')
        with self.assertRaisesRegex(ValueError,'outside'):inventory.runtime_roster(self.provider)
        with self.assertRaisesRegex(ValueError,'regular'):inventory.file_info(self.root/'absent.pyd')
    def test_symlink_reparse_negative_without_OS_link_permission(self):
        path=self.root/'tokenizers/__init__.py'
        original=Path.is_symlink
        def symlink(item):return item==path or original(item)
        with patch.object(Path,'is_symlink',symlink):
            with self.assertRaisesRegex(ValueError,'symlink'):inventory.regular(path)
    def test_stale_package_roster_or_path_header_denied(self):
        value=inventory.runtime_inventory(self.provider);value['packages']['jinja2']['root']=str(self.root/'other')
        with self.assertRaisesRegex(ValueError,'stale candidate'):inventory.validate_runtime(value,self.provider)
        value=inventory.runtime_inventory(self.provider);value['files'][next(iter(value['files']))]['path']=str(self.root/'other.pyd')
        with self.assertRaises(ValueError):inventory.validate_runtime(value,self.provider)
    def test_candidate_needs_external_hash_and_exact_current_source_binding(self):
        body={'kind':inventory.KIND,'native_registration_written':False,'native_config_written':False,'native_authority_issued':False,
              'complete_runtime_execution_closure_established':False,
              'fixture':'CPU_NOT_AUTHORITY'};body['sha256']=inventory.digest(body)
        with self.assertRaisesRegex(ValueError,'externally'):inventory.verify_candidate(body,None)
        with patch.object(inventory,'build_candidate',return_value=deepcopy(body)):
            self.assertTrue(inventory.verify_candidate(body,body['sha256']))
        altered=deepcopy(body);altered['fixture']='source mutation';altered['sha256']=inventory.digest({k:v for k,v in altered.items() if k!='sha256'})
        with patch.object(inventory,'build_candidate',return_value=body):
            with self.assertRaisesRegex(ValueError,'entire current'):inventory.verify_candidate(altered,altered['sha256'])
    def test_no_final_registration_config_or_encoder_execution_authority(self):
        body={'CPU_fixture_only':True,'native_authority_issued':False}
        with self.assertRaisesRegex(ValueError,'staging-only'):inventory.save_staged_candidate(self.root/'registration.json',body)
        self.assertFalse(any(n.split('.',1)[0] in inventory.FORBIDDEN for n in sys.modules))


if __name__=='__main__':unittest.main()
