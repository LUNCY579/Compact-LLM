"""Finite actual inert-file hashing; no host binary/process/framework action."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import builtins,hashlib,stat,tempfile,unittest
import inventory as I

HERE=Path(__file__).resolve().parent
class PassiveInventoryControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_passive_inventory_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.paths=[];self.files={}
        for name,data,role in (('source.py',b'# actual inert source\n','python_source'),('empty.py',b'','python_source'),('observer.dll',b'inert collector bytes','observer_binary'),('fixture.dll',b'inert fixture bytes','observer_fixture_binary')):
            p=self.root/name;p.write_bytes(data);text=str(p);self.paths.append(text);self.files[text]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'role':role}
        self.attempts=[];old=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','native_worker','native_process','native_nvml','numpy','transformers','safetensors'}:
                self.attempts.append(name);raise AssertionError('native/framework import forbidden: '+name)
            return old(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop);self.addCleanup(lambda:self.assertEqual(self.attempts,[]))

    def manifest(self):
        pinned=I.metadata()
        v={'kind':'ROOT_SELECTED_OBSERVER_ONLY_PASSIVE_INVENTORY_V1','source_commit':'a'*40,'files':deepcopy(self.files),'source_paths':self.paths[:2],
           'observer_binary_path':self.paths[2],'fixture_binary_path':self.paths[3],'metadata_map_digests':{'inherited65':I.digest(pinned['inherited65_map']),'stdlib779':I.digest(pinned['stdlib779_map'])}}
        return {**v,'sha256':I.digest(v)}

    def test_actual_finite_hashes_empty_source_and_anchored_maps_remain_unqualified(self):
        m=self.manifest();v=I.build_inventory(m,expected_manifest_sha256=m['sha256'])
        self.assertEqual(len(v['files']),4);self.assertEqual(len(v['inherited65_map']),65);self.assertEqual(len(v['stdlib779_map']),779)
        self.assertEqual(v['files'][self.paths[1]]['sha256'],hashlib.sha256(b'').hexdigest())
        self.assertFalse(v['native_PASS']);self.assertFalse(v['host_inventory_complete']);self.assertFalse(v['loaded_module_baseline_observed'])

    def test_changed_same_size_pin_and_partial_failure_retained(self):
        p=Path(self.paths[0]);raw=p.read_bytes();p.write_bytes(b'X'+raw[1:])
        try:I._collect(self.paths,self.files)
        except ValueError as error:
            d=BaseException.__dict__['__dict__'].__get__(error,type(error))['__inventory_partial__']
            self.assertEqual(d['attempted_paths'],[self.paths[0]]);self.assertFalse(d['observed_files'][self.paths[0]]['expected_pin_matched'])
        else:self.fail('changed same-size bytes must fail original pin')

    def test_exact_same_count_path_substitution_and_source_head_refused(self):
        m=self.manifest();wrong=deepcopy(m);p=self.root/'other.py';p.write_bytes(Path(self.paths[0]).read_bytes())
        wrong['files'][str(p)]=wrong['files'].pop(self.paths[0])
        with self.assertRaisesRegex(ValueError,'selected manifest/body'):I.build_inventory(wrong,expected_manifest_sha256=m['sha256'])
        for head in ('abc123','z'*40):
            wrong=deepcopy(m);wrong['source_commit']=head;wrong['sha256']=I.digest({k:v for k,v in wrong.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'full source HEAD'):I.build_inventory(wrong,expected_manifest_sha256=wrong['sha256'])
        wrong=deepcopy(m);wrong['source_paths']=['absent'];wrong['sha256']=I.digest({k:v for k,v in wrong.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'source closure'):I.build_inventory(wrong,expected_manifest_sha256=wrong['sha256'])

    def test_exact_known_alias_mapping_and_bad_binding_refused(self):
        logical=self.root/'logical';logical.mkdir();physical=self.root/'physical';physical.mkdir();p=physical/'same.py';p.write_bytes(b'alias bytes')
        selected=str(logical/'same.py');d={selected:{'bytes':11,'sha256':hashlib.sha256(b'alias bytes').hexdigest(),'role':'python_source'}}
        table={'logical_base_prefix':str(logical),'physical_base_prefix':str(physical),'file_attributes':1040,'reparse_tag':2684354563,'is_symlink':False}
        class Alias(I.Readers):
            wrong=False
            def resolve(self,path):
                p=Path(path)
                return physical/p.relative_to(logical) if p.is_relative_to(logical) else super().resolve(p)
            def metadata(self,path):
                d=super().metadata(path)
                if Path(path)==logical:d.update(file_attributes=1040,reparse_tag=7 if self.wrong else 2684354563,is_symlink=False)
                return d
        r=Alias();v=I._collect([selected],d,r,alias_table=table);self.assertEqual(v['files'][selected]['physical_path'],str(p))
        r.wrong=True
        with self.assertRaisesRegex(ValueError,'approved alias binding'):I._collect([selected],d,r,alias_table=table)

    def test_truncation_growth_and_deadline_bounds_use_finite_reads(self):
        real=I.Readers()
        class Fake(I.Readers):
            mode='truncated';ticks=0
            def clock(self):self.ticks+=1;return self.ticks*601 if self.mode=='deadline' else 0
            def opened_metadata(self,stream):
                d={k:self.metadata(Path(self_path))[k] for k in ('bytes','mtime_ns','inode','mode')};d['mode']=stat.S_IFMT(d['mode']);return d
            def open(self,path):
                if str(path)!=self_path:return super().open(path)
                mode=self.mode
                class Stream:
                    reads=[]
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def read(self,n):
                        self.reads.append(n)
                        if mode=='truncated':return b''
                        return b'# actual inert source\n' if len(self.reads)==1 else b'X'
                return Stream()
        self_path=self.paths[0]
        for mode in ('truncated','grown','deadline'):
            r=Fake();r.mode=mode
            with self.subTest(mode=mode):
                try:I._collect([self_path],{self_path:self.files[self_path]},r)
                except ValueError as error:
                    partial=BaseException.__dict__['__dict__'].__get__(error,type(error))['__inventory_partial__']
                    if mode=='truncated':
                        self.assertFalse(partial['current_read']['hash_complete']);self.assertEqual(partial['current_read']['consumed_bytes'],0)
                        self.assertEqual(partial['current_read']['remaining_bytes'],self.files[self_path]['bytes'])
                else:self.fail('invalid selected stream must fail')

    def test_late_earlier_file_pointer_mutation_and_unknown_alias_refused(self):
        first=Path(self.paths[0]);second=Path(self.paths[1]);root=self.root
        class Mutate(I.Readers):
            done=False
            def open(self,path):
                if Path(path)==second and not self.done:first.write_bytes(first.read_bytes());self.done=True
                return super().open(path)
        with self.assertRaisesRegex(ValueError,'final boundary'):I._collect(self.paths,self.files,Mutate())
        class Reparse(I.Readers):
            def metadata(self,path):
                d=super().metadata(path)
                if Path(path)==root:d['file_attributes']=0x400
                return d
        with self.assertRaisesRegex(ValueError,'other reparse alias'):I._collect(self.paths,self.files,Reparse())

    def test_model_role_population_metadata_parser_and_selection_file_anchor(self):
        for paths,mapping in (([],{}),(self.paths+self.paths,self.files)):
            with self.assertRaisesRegex(ValueError,'population'):I._collect(paths,mapping)
        d=deepcopy(self.files);d[self.paths[2]]['role']='model_payload'
        with self.assertRaisesRegex(ValueError,'role/size/pin'):I._collect(self.paths,d)
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.assertRaises(ValueError):I.parse(raw)
        m=self.manifest();path=self.root/'selection.json';raw=(I.canonical(m)+'\n').encode();path.write_bytes(raw)
        self.assertEqual(I.read_selected_manifest(path,file_sha256=hashlib.sha256(raw).hexdigest(),body_sha256=m['sha256']),m)
        path.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'manifest bytes'):I.read_selected_manifest(path,file_sha256=hashlib.sha256(raw).hexdigest(),body_sha256=m['sha256'])

    def test_opened_handle_pathname_swap_rejected_before_bytes(self):
        p=Path(self.paths[0]);other=self.root/'other.py';other.write_bytes(p.read_bytes());opened=[]
        class Swapped(I.Readers):
            def open(self,path):opened.append(str(path));return other.open('rb')
        with self.assertRaisesRegex(ValueError,'opened handle matches'):I._collect([str(p)],{str(p):self.files[str(p)]},Swapped())
        self.assertEqual(opened,[str(p)])
        exe=self.root/'CPU_python.exe';exe.write_bytes(b'inert executable fixture')
        expected={str(exe):{'bytes':exe.stat().st_size,'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'role':'interpreter_binary'}}
        class ExecutableBits(I.Readers):
            def metadata(self,path):
                d=super().metadata(path)
                if Path(path)==exe:d['mode']=stat.S_IFREG|0o777
                return d
            def opened_metadata(self,stream):
                d=super().opened_metadata(stream)
                self.assert_regular=d['mode']==stat.S_IFREG
                return d
        r=ExecutableBits();value=I._collect([str(exe)],expected,r)
        self.assertTrue(r.assert_regular);self.assertEqual(value['files'][str(exe)]['sha256'],expected[str(exe)]['sha256'])


if __name__=='__main__':unittest.main()
