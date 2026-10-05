"""CPU source/JSON controls for the closed tensor-purpose contract."""
from pathlib import Path
import copy, importlib.util, sys, unittest
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
PATH=HERE/'prefix_input_owned_contract.py'
SPEC=importlib.util.spec_from_file_location('owned_closed_contract_controls',PATH)
contract=importlib.util.module_from_spec(SPEC)
exec(compile(PATH.read_text(encoding='utf-8'),str(PATH),'exec'),contract.__dict__)

class OwnedContractControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools=contract.source_tools()
        cls.candidate,_=cls.tools.json_source(contract.CANDIDATE,contract.CANDIDATE_FILE,contract.CANDIDATE_BODY)
        cls.inherited,_=cls.tools.json_source(cls.tools.INHERITED,cls.tools.INHERITED_FILE,cls.tools.INHERITED_BODY)
        cls.stdlib,_=cls.tools.json_source(cls.tools.STDLIB,cls.tools.STDLIB_FILE,cls.tools.STDLIB_BODY)

    def reject_resealed(self,value):
        value['sha256']=self.tools.digest({k:v for k,v in value.items() if k!='sha256'})
        # Even if a caller supplies a re-signed candidate body selection, original
        # external maps, populations and false-qualification predicates remain.
        with patch.object(contract,'CANDIDATE_BODY',value['sha256']),self.assertRaises(ValueError):
            contract.validate_candidate(value,self.tools,self.inherited,self.stdlib)

    def test_closed_whole_metadata_contract_without_binary_reads(self):
        original=Path.open
        def only_source_json(path,*args,**kwargs):
            if path.suffix.lower() in ('.dll','.pyd','.exe','.zip','.safetensors'):
                raise AssertionError('payload byte read attempted')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',only_source_json):
            value=contract.registration_candidate()
        self.assertEqual(value['status'],'CPU_SOURCE_METADATA_VALIDATED_NATIVE_CLOSED')
        self.assertEqual(value['expected_known_runtime_pin_union'],self.candidate['expected_known_pin_union'])
        self.assertEqual(value['resources'],self.inherited['resources'])
        self.assertEqual(value['authorization']['tensor_permission'],{'allocations':2,'dtype':'int64','shape':[1,422],'logical_bytes':6752})
        self.assertEqual(value['dose']['maximum_phases'],35+8+14+1)
        self.assertIs(value['native_entry_open'],False)
        self.assertFalse({'torch','transformers','tensorflow','jax','flax'}&{x.split('.')[0] for x in sys.modules})

    def test_native_gates_refuse_before_argument_or_source_access(self):
        class Hostile:
            def __fspath__(self):raise AssertionError('path inspected')
            def __getattribute__(self,name):raise AssertionError('argument inspected')
        with patch.object(contract,'source_tools',side_effect=AssertionError('source accessed')):
            for function in (contract.native_entry,contract.preflight_native,contract.write_owned_registration):
                with self.assertRaises(ValueError):function(Hostile(),native_launch_authorized=True)

    def test_helper_pointer_denial_before_read_and_drift_before_compile(self):
        with patch.object(contract,'pointer_source',side_effect=ValueError('fixture reparse denial')),patch.object(Path,'open',side_effect=AssertionError('read before pointer admission')):
            with self.assertRaises(ValueError):contract.source_tools()
        before=contract.HELPER.lstat()
        changed=SimpleNamespace(st_size=before.st_size,st_mtime_ns=before.st_mtime_ns+1,st_ino=before.st_ino)
        with patch.object(contract,'pointer_source',side_effect=[before,changed]),patch('builtins.compile',side_effect=AssertionError('compile before stable pointer')):
            with self.assertRaises(ValueError):contract.source_tools()

    def test_same_count_inherited_pin_substitution_after_reseal(self):
        value=copy.deepcopy(self.candidate);name=next(iter(value['inherited65_map']))
        value['inherited65_map'][name]='a'*64
        value['inherited65_map_sha256']=self.tools.digest(value['inherited65_map'])
        self.reject_resealed(value)

    def test_same_count_stdlib_descriptor_substitution_after_reseal(self):
        value=copy.deepcopy(self.candidate);name=next(iter(value['stdlib779_inventory']['files']))
        value['stdlib779_inventory']['files'][name]['bytes']+=1
        self.reject_resealed(value)

    def test_missing_source_and_conflicting_known_source_pin(self):
        value=copy.deepcopy(self.candidate);name=next(iter(value['torch_source_hashes']))
        del value['torch_source_hashes'][name];self.reject_resealed(value)
        value=copy.deepcopy(self.candidate)
        name=next(p for p in value['torch_source_hashes'] if p in value['inherited65_map'])
        value['torch_source_hashes'][name]='b'*64;self.reject_resealed(value)

    def test_current_binary_or_authority_flags_cannot_be_rebranded(self):
        for key in contract.CLOSED:
            value=copy.deepcopy(self.candidate);value[key]=True
            with self.subTest(key=key):self.reject_resealed(value)
        value=copy.deepcopy(self.candidate);name=next(iter(value['torch_UNREAD_binary_obligations']))
        value['torch_UNREAD_binary_obligations'][name]['current_binary_bytes_hashed']=True
        self.reject_resealed(value)

    def test_bootstrap_pin_and_union_population_join(self):
        value=copy.deepcopy(self.candidate);value['bootstrap_source_import_observations']['_tensor.py']['source_sha256']='c'*64
        self.reject_resealed(value)
        value=copy.deepcopy(self.candidate);name=next(iter(value['expected_known_pin_union']))
        del value['expected_known_pin_union'][name]
        value['expected_known_pin_union_sha256']=self.tools.digest(value['expected_known_pin_union'])
        self.reject_resealed(value)

if __name__=='__main__':unittest.main()
