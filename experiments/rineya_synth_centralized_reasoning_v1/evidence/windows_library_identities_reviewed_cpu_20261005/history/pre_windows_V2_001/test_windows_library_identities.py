"""Real inert JSON/DLL bytes; no actual Windows/runtime qualification.

Only the fixed System32 root is replaced by an explicit temporary CPU root.
Historical65/779 descriptors, Git and interpreter layout remain the existing
OuterContextFixture seams. Actual selected readers, joins, rosters and hashes
run. This optional schema supplies no observer, native permission or host proof.
"""
from pathlib import Path
from unittest.mock import patch
import builtins,copy,hashlib,tempfile,unittest
from outer_context_fixture import OuterContextFixture,HERE,descriptor,write_new


class WindowsLibraryIdentityControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_windows_identities_',dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.forbidden=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','transformers','safetensors','native_worker','native_process','native_nvml'}:
                self.forbidden.append(name);raise AssertionError('native/runtime import forbidden: '+name)
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop)
        self.addCleanup(lambda:self.assertEqual(self.forbidden,[]))
        self.f=OuterContextFixture(self.temp.name);self.C=self.f.C
        self.system=self.f.root/'inert_windows/System32';self.system.mkdir(parents=True)
        self.paths=[]
        for name,raw in (('CPU_alpha.dll',b'inert selected Windows dependency alpha\n'),('CPU_beta.dll',b'inert selected Windows dependency beta\n')):
            p=self.system/name;write_new(p,raw);self.paths.append(str(p))
        self.previous65=copy.deepcopy(self.f.runtime['inherited65_map'])
        self.previous779=copy.deepcopy(self.f.runtime['stdlib779_map'])

    def add(self):
        f=self.f
        windows={p:descriptor(Path(p).read_bytes(),'windows_system_binary') for p in self.paths}
        f.runtime=copy.deepcopy(f.runtime);f.runtime['files'].update(windows)
        f.runtime['windows_library_identities']=copy.deepcopy(windows)
        interpreter=str(f.base/'python.exe');provider=str(f.package/'prefix_library_policy.py')
        f.runtime['library_search_policy']['enforced_boundary']={
            'kind':'FINITE_WINDOWS_DLL_BOUNDARY_V1','search_directories':[str(f.base)],
            'system_directory':str(self.system),
            'library_files':{interpreter:copy.deepcopy(f.inert_files[interpreter]),**copy.deepcopy(windows)},
            'initial_loaded_libraries':[interpreter,*self.paths],
            'extension_transitions':{},'framework_transition':[],
            'provider_source':{'path':provider,'descriptor':copy.deepcopy(f.sources[provider])}}
        return windows

    def reseal(self):
        """Change every dependent external anchor; test joins beyond checksums."""
        f=self.f;C=self.C
        f.runtime=f._seal(f.runtime)
        f.config=copy.deepcopy(f.config);f.config['runtime_files']=copy.deepcopy(f.runtime['files']);f.config=f._seal(f.config)
        f.review=copy.deepcopy(f.review)
        f.review.update(runtime_closure_sha256=C.digest(f.runtime['files']),
            library_search_policy_sha256=C.digest(f.runtime['library_search_policy']))
        f.review=f._seal(f.review)
        f.registration=copy.deepcopy(f.registration)
        f.registration.update(config_body_sha256=f.config['sha256'],runtime_body_sha256=f.runtime['sha256'],
            runtime_files=copy.deepcopy(f.runtime['files']),
            qualification_review_file_sha256=f.anchor(f.review_path,f.review)['file_sha256'],
            qualification_review_body_sha256=f.review['sha256'])
        f.registration=f._seal(f.registration)
        for path,value in ((f.config_path,f.config),(f.runtime_path,f.runtime),(f.registration_path,f.registration),(f.review_path,f.review)):
            path.write_bytes(f._raw(value));f.documents[str(path)]=value
        f.anchors.update(expected_config_file=f.anchor(f.config_path,f.config)['file_sha256'],expected_config_body=f.config['sha256'],
            expected_runtime_file=f.anchor(f.runtime_path,f.runtime)['file_sha256'],expected_runtime_body=f.runtime['sha256'],
            expected_registration_file=f.anchor(f.registration_path,f.registration)['file_sha256'],expected_registration_body=f.registration['sha256'])

    def context(self):
        with patch.object(self.C,'WINDOWS_SYSTEM_DIRECTORY',self.system):return self.f.context()

    def reject(self,pattern):
        self.reseal()
        with self.assertRaisesRegex(ValueError,pattern):self.context()

    def test_optional_absence_keeps_existing_CPU_context_and_native_gate(self):
        value=self.C.context_value(self.context())
        self.assertNotIn('windows_library_identities',value['runtime'])
        self.assertTrue(value['fixture_only']);self.assertFalse(value['native_PASS'])
        with patch.object(self.C,'read_bytes',side_effect=AssertionError('public gate must precede paths')):
            with self.assertRaisesRegex(ValueError,'stage has no native authority'):
                self.C.validate_native_manifest(self.f.config_path,**self.f.anchors)

    def test_additions_actual_hashes_join_all_maps_without_changing65_779(self):
        windows=self.add();self.reseal();value=self.C.context_value(self.context())
        self.assertEqual(value['runtime']['windows_library_identities'],windows)
        self.assertEqual(value['runtime']['inherited65_map'],self.previous65)
        self.assertEqual(value['runtime']['stdlib779_map'],self.previous779)
        self.assertTrue(set(self.paths)<=set(self.f.actual_file_checks))
        self.assertTrue(value['fixture_only']);self.assertFalse(value['native_PASS']);self.assertFalse(value['native_authority_issued'])

    def test_resealed_omitted_roster_cannot_hide_extra_runtime_DLL(self):
        self.add();del self.f.runtime['windows_library_identities']
        self.reject('only exact inherited Windows binary leaves')

    def test_resealed_same_count_descriptor_substitution_rejected(self):
        self.add();self.f.runtime['windows_library_identities'][self.paths[0]]['sha256']='0'*64
        self.reject('joins runtime and qualified library maps')

    def test_resealed_missing_qualified_library_identity_rejected(self):
        self.add();del self.f.runtime['library_search_policy']['enforced_boundary']['library_files'][self.paths[0]]
        self.reject('joins runtime and qualified library maps')

    def test_resealed_wrong_role_and_outside_root_rejected(self):
        self.add();self.f.runtime['windows_library_identities'][self.paths[0]]['role']='torch_binary'
        self.reject('typed Windows DLL descriptor')
        self.f.runtime['windows_library_identities'][self.paths[0]]['role']='windows_system_binary'
        wrong=self.f.root/'inert_windows/CPU_outside.dll';write_new(wrong,b'inert outside bytes\n')
        d=descriptor(wrong.read_bytes(),'windows_system_binary');old=self.paths[0]
        for mapping in (self.f.runtime['files'],self.f.runtime['windows_library_identities'],self.f.runtime['library_search_policy']['enforced_boundary']['library_files']):
            del mapping[old];mapping[str(wrong)]=copy.deepcopy(d)
        self.reject('only exact absolute System32 DLL leaves')

    def test_resealed_historical65_or779_substitutions_still_rejected(self):
        self.add()
        for name,pattern in (('inherited65_map','original exact65 preserved'),('stdlib779_map','preserved exact779')):
            with self.subTest(map=name):
                before=copy.deepcopy(self.f.runtime[name]);key=next(iter(before));self.f.runtime[name][key]='0'*64
                self.reject(pattern);self.f.runtime[name]=before

    def test_earlier_selected_DLL_byte_drift_rejected_after_reseal(self):
        self.add();self.reseal()
        p=Path(self.paths[0]);raw=p.read_bytes();p.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'):self.context()

    def test_nonregular_and_reparse_leaf_rejected_before_hash(self):
        self.add();p=Path(self.paths[0]);p.unlink();p.mkdir()
        self.reject('exact regular physical size')
        p.rmdir();write_new(p,b'inert selected Windows dependency alpha\n')
        real=Path.lstat
        def metadata(path,*args,**kwargs):
            info=real(path,*args,**kwargs)
            if path==p:
                from types import SimpleNamespace
                return SimpleNamespace(st_file_attributes=0x400,st_mode=info.st_mode,st_size=info.st_size,st_ino=info.st_ino,st_mtime_ns=info.st_mtime_ns)
            return info
        with patch.object(Path,'lstat',metadata):self.reject('no reparse alias')

    def test_oversized_or_wildcard_roster_rejected_before_path_inspection(self):
        self.add();runtime=self.f.runtime
        runtime['windows_library_identities']={str(self.system/('CPU_%s.dll'%n)):runtime['windows_library_identities'][self.paths[0]] for n in range(4097)}
        with patch.object(self.C,'qualified_path',side_effect=AssertionError('must bound before paths')):
            with self.assertRaisesRegex(ValueError,'finite selected Windows DLL identity roster'):self.C._windows_library_identities(runtime)
        runtime['windows_library_identities']={str(self.system/'*.dll'):descriptor(b'one','windows_system_binary')}
        with patch.object(self.C,'qualified_path',side_effect=AssertionError('wildcard before paths')):
            with self.assertRaisesRegex(ValueError,'without wildcard'):self.C._windows_library_identities(runtime)


if __name__=='__main__':unittest.main()
