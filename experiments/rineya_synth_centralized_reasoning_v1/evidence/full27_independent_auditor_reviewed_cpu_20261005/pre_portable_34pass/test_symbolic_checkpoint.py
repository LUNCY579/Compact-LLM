"""Synthetic protocol2/ZIP state controls only; no framework or native authority."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import symbolic_checkpoint as s


def emit(value):
    """Small explicit fixture writer for the actual inspected opcode subset."""
    if type(value) is s.Global:
        name=next(key for key,kind in s.GLOBALS.items() if kind==value.name);module,symbol=name.split()
        return b'c'+module.encode()+b'\n'+symbol.encode()+b'\n'
    if type(value) is s.Tensor:
        storage=value.storage
        pid=('storage',s.Global(storage.dtype),storage.key,'cpu',storage.count)
        args=emit(pid)+b'Q'+emit(0)+emit(value.shape)+emit(value.stride)+emit(False)+emit({})
        return emit(s.Global('tensor'))+b'('+args+b'tR'
    if value is None:return b'N'
    if type(value) is bool:return b'\x88' if value else b'\x89'
    if type(value) is int:
        raw=value.to_bytes(max(1,(value.bit_length()+8)//8),'little',signed=True)
        return b'\x8a'+bytes([len(raw)])+raw
    if type(value) is float:return b'G'+struct.pack('>d',value)
    if type(value) is str:
        raw=value.encode();return b'X'+struct.pack('<I',len(raw))+raw
    if type(value) is tuple:return b'('+b''.join(emit(v) for v in value)+b't' if value else b')'
    if type(value) is list:return b']('+b''.join(emit(v) for v in value)+b'e'
    if type(value) is dict:return b'}('+b''.join(emit(k)+emit(v) for k,v in value.items())+b'u'
    raise AssertionError(type(value))


def fixture(completed=9,arm='M1'):
    registration={'arms':{}}
    for role in ('M1','M2'):
        updates=[{'occurrence_row_ids':[s.digest([role,i,j]) for j in range(20)],'input_tokens':1000,
                  'supervised_tokens':400} for i in range(1,28)]
        registration['arms'][role]={'updates':updates,'occurrence_order_sha256':s.digest([row for update in updates for row in update['occurrence_row_ids']])}
    registration['sha256']=s.digest(registration);binding,order=s.boundary(registration,arm,completed)
    names=list(s.inventory());data={}
    def tensor(shape,dtype,value):
        key=str(len(data));count=1
        for size in shape:count*=size
        data[key]=struct.pack('<f',value)*count if dtype=='float32' else bytes([value])*count
        return s.Tensor(s.Storage(key,dtype,count),shape,s._contiguous(shape))
    adapters={name:tensor(shape,'float32',.125) for name,shape in s.inventory().items()}
    moments={index:{'step':tensor((),'float32',float(completed)),'exp_avg':tensor(s.inventory()[name],'float32',.25),
                    'exp_avg_sq':tensor(s.inventory()[name],'float32',.5)} for index,name in enumerate(names)}
    group={'lr':binding['learning_rate'],'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'foreach':False,
           'maximize':False,'capturable':False,'differentiable':False,'fused':False,'params':list(range(112))}
    rng={'python':random.Random(101).getstate(),'cpu':tensor((5056,),'uint8',1),'cuda0':tensor((16,),'uint8',2)}
    metadata={'schema_version':1,'kind':'SYNTH_STUDY_TORCH_BOUNDARY_V1','experiment':'SYNTH-CRS-001',
        'scope':'UNQUALIFIED_NATIVE_IMPLEMENTATION','arm':arm,'seed':101,'recipe_sha256':s.digest('SYNTHETIC_RECIPE'),
        'cohort_sha256':registration['arms'][arm]['occurrence_order_sha256'],'backend_source_sha256':s.digest('SYNTHETIC_SOURCE'),
        'fresh_start_attestation_sha256':s.digest('SYNTHETIC_FRESH'),'adapter_parameter_order':names,
        'adapter_inventory_sha256':s.inventory_digest(),'binding':binding,'native_launch_authorized':False,
        'training_authorized':False,'native_resume_qualified':False,'qualification_STATE1_used':False}
    payload={'metadata':metadata,'state':{'schema_version':1,'adapters':adapters,'optimizer':{'state':moments,'param_groups':[group]},
        'gradients':{name:None for name in names},'rng':rng},'occurrence_order':order}
    def spec(t):return {'shape':list(t.shape),'dtype':'torch.'+t.storage.dtype,'bytes':len(data[t.storage.key]),
                       'sha256':hashlib.sha256(data[t.storage.key]).hexdigest()}
    witness={'schema_version':1,'kind':'SYNTH_BACKEND_QUAL_WITNESS_NOT_STUDY_CHECKPOINT','arm':arm,'seed':101,
        'scope':'UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER','maximum_tensor_hash_chunk_bytes':64*1024**2,
        **{key:metadata[key] for key in ('recipe_sha256','cohort_sha256','backend_source_sha256')},
        'binding':binding,'parameter_order':names,'adapters':{name:spec(t) for name,t in adapters.items()},
        'optimizer_moments_and_steps':{names[i]:{key:spec(t) for key,t in part.items()} for i,part in moments.items()},
        'optimizer_groups':[deepcopy(group)],'gradients_all_None':True,'rng':{'python_sha256':s.digest(rng['python']),
        'cpu':spec(rng['cpu']),'cuda0':spec(rng['cuda0'])},'occurrence_order_sha256':s.digest(order),
        'base_frozen_source_tensors':310,'base_parameters':596049920,'base_source_tensor_hash_map_sha256':s.digest('SYNTHETIC_BASE'),
        'native_launch_authorized':False,'study_warmstart_authorized':False}
    witness['optimizer_groups'][0]['betas']=list(group['betas']);witness['sha256']=s.digest(witness)
    return registration,payload,data,witness


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'fixture_C9.pt';self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
    def write(self,*,raw=None,extra=None,omit=None):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('archive/data.pkl',raw if raw is not None else b'\x80\x02'+emit(self.payload)+b'.')
            archive.writestr('archive/byteorder',b'little');archive.writestr('archive/version',b'3\n')
            archive.writestr('archive/.data/serialization_id',b'0'*40)
            for key,value in self.data.items():
                if key!=omit:archive.writestr('archive/data/'+key,value)
            if extra:archive.writestr(*extra)
        raw=stream.getvalue();self.path.write_bytes(raw)
        return {'schema_version':1,'kind':'SYNTH_STUDY_TORCH_BOUNDARY_V1','path':str(self.path),'bytes':len(raw),
                'sha256':hashlib.sha256(raw).hexdigest(),'metadata':deepcopy(self.payload['metadata']),
                'metadata_sha256':s.digest(self.payload['metadata'])}
    def verify(self,descriptor=None):return s.verify_checkpoint(self.path,descriptor or self.write(),self.witness,self.registration,'M1',9)
    def reseal(self):self.witness['sha256']=s.digest({k:v for k,v in self.witness.items() if k!='sha256'})
    def resign_tensor(self,tensor,expected):
        raw=self.data[tensor.storage.key];expected['sha256']=hashlib.sha256(raw).hexdigest();expected['bytes']=len(raw);self.reseal()
    def test_full450_storage_synthetic_boundaries9_18_27_both_arms_no_authority(self):
        for completed,arm in ((9,'M1'),(18,'M2'),(27,'M1')):
            self.registration,self.payload,self.data,self.witness=fixture(completed,arm);descriptor=self.write()
            result=s.verify_checkpoint(self.path,descriptor,self.witness,self.registration,arm,completed)
            self.assertEqual(450,result['storage_count']);self.assertEqual(3440752,result['finite_FP32_values'])
            self.assertFalse(result['native_audit_PASS_issued']);self.assertFalse(result['Torch_RNG_encoding_probed'])
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers')))
    def test_byte_hash_and_metadata_tamper_rejected(self):
        descriptor=self.write();descriptor['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'bytes/hash'):self.verify(descriptor)
        descriptor=self.write();descriptor['metadata']['arm']='M2';descriptor['metadata_sha256']=s.digest(descriptor['metadata'])
        with self.assertRaisesRegex(ValueError,'metadata'):self.verify(descriptor)
    def test_unsafe_global_opcode_and_trailing_pickle_rejected(self):
        for raw in (b'\x80\x02cos\nsystem\n.',b'\x80\x02\x82\x01.',b'\x80\x02N.extra'):
            with self.assertRaises(ValueError):self.verify(self.write(raw=raw))
    def test_missing_first_protocol_marker_rejected(self):
        raw=emit(self.payload)+b'.'
        with self.assertRaisesRegex(ValueError,'first opcode'):self.verify(self.write(raw=raw))
    def test_resigned_wrong_native_inventory_digest_rejected(self):
        self.payload['metadata']['adapter_inventory_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'base/source metadata'):self.verify()
    def test_duplicate_escaped_missing_and_unused_zip_members(self):
        for extra,omit in ((('archive/data/0',b'x'),None),(('../escape',b'x'),None),(None,'0'),(('archive/data/450',b'x'),None)):
            with self.assertRaises(ValueError):self.verify(self.write(extra=extra,omit=omit))
    def test_finite_and_nonnegative_checks_survive_resigned_hashes(self):
        name=self.witness['parameter_order'][0]
        for value in (float('nan'),-1.0):
            self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
            tensor=self.payload['state']['optimizer']['state'][0]['exp_avg_sq']
            self.data[tensor.storage.key]=struct.pack('<f',value)*tensor.storage.count
            self.resign_tensor(tensor,self.witness['optimizer_moments_and_steps'][name]['exp_avg_sq'])
            with self.assertRaisesRegex(ValueError,'finite|nonnegative'):self.verify()
    def test_resigned_wrong_step_groups_and_parameter_order_rejected(self):
        name=self.witness['parameter_order'][0];tensor=self.payload['state']['optimizer']['state'][0]['step']
        self.data[tensor.storage.key]=struct.pack('<f',10);self.resign_tensor(tensor,self.witness['optimizer_moments_and_steps'][name]['step'])
        with self.assertRaisesRegex(ValueError,'Adam step'):self.verify()
        self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
        self.payload['state']['optimizer']['param_groups'][0]['betas']=(.8,.999)
        with self.assertRaisesRegex(ValueError,'AdamW'):self.verify()
        self.payload['metadata']['adapter_parameter_order']=list(reversed(self.payload['metadata']['adapter_parameter_order']))
        with self.assertRaisesRegex(ValueError,'parameter order'):self.verify()
    def test_shape_storage_size_dtype_and_alias_rejected(self):
        name=self.witness['parameter_order'][0];original=self.payload['state']['adapters'][name]
        for changed in (s.Tensor(original.storage,(16,512),(512,1)),s.Tensor(s.Storage('0','uint8',8192),(8,1024),(1024,1))):
            self.payload['state']['adapters'][name]=changed
            with self.assertRaises(ValueError):self.verify()
        self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
        self.payload['state']['adapters'][self.witness['parameter_order'][1]]=self.payload['state']['adapters'][name]
        with self.assertRaisesRegex(ValueError,'shared storage'):self.verify()
    def test_rng_bytes_shape_python_state_and_gradients_rejected(self):
        self.data['448']=b'x'*5056
        with self.assertRaisesRegex(ValueError,'tensor hash'):self.verify()
        self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
        original=self.payload['state']['rng']['cpu']
        self.payload['state']['rng']['cpu']=s.Tensor(original.storage,(2528,2),(2,1))
        with self.assertRaisesRegex(ValueError,'shape'):self.verify()
        self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
        self.payload['state']['rng']['python']=(3,(1,2),None)
        with self.assertRaises((ValueError,TypeError)):self.verify()
        self.registration,self.payload,self.data,self.witness=deepcopy(self.base)
        self.payload['state']['gradients'][self.witness['parameter_order'][0]]=1
        with self.assertRaisesRegex(ValueError,'empty gradients'):self.verify()
    def test_wrong_arm_cursor_order_cohort_and_STATE1_rejected(self):
        descriptor=self.write()
        with self.assertRaises(ValueError):s.verify_checkpoint(self.path,descriptor,self.witness,self.registration,'M2',9)
        with self.assertRaises(ValueError):s.verify_checkpoint(self.path,descriptor,self.witness,self.registration,'M1',18)
        self.payload['occurrence_order'].reverse()
        with self.assertRaisesRegex(ValueError,'order'):self.verify()
        descriptor=self.write();descriptor['path']=str(self.path.parent/'STATE1.pt')
        with self.assertRaises(ValueError):s.verify_checkpoint(Path(descriptor['path']),descriptor,self.witness,self.registration,'M1',9)
    def test_container_cycles_limits_and_unknown_globals_rejected(self):
        for raw in (b'\x80\x02]q\x00h\x00a.',b'\x80\x02csubprocess\nPopen\n.',b'\x80\x04N.'):
            with self.assertRaises(ValueError):s.interpret(raw)
        with self.assertRaisesRegex(ValueError,'bounded'):s.interpret(b'x'*(s.MAX_PICKLE+1))


if __name__=='__main__':unittest.main()
