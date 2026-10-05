"""Bounded inert protocol-2 Torch archive inspection; no framework or unpickle.

The four observed GLOBAL names become markers. REDUCE is interpreted locally
for empty OrderedDict and tensor descriptors only; no callback is executed.
This checks saved bytes against externally selected descriptor/witness anchors.
It grants no native audit, restoration, training or checkpoint reuse authority.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import pickletools
import random
import re
import struct
import zipfile

MAX_BYTES=32*1024**2
MAX_PICKLE=1024**2
MAX_OPS=100000
MAX_ITEMS=20000
GLOBALS={'torch._utils _rebuild_tensor_v2':'tensor','torch FloatStorage':'float32',
         'torch ByteStorage':'uint8','collections OrderedDict':'ordered'}
FIELDS={'schema_version','kind','path','bytes','sha256','metadata','metadata_sha256'}


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def hash64(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class Global:
    name:str


@dataclass(frozen=True)
class Storage:
    key:str
    dtype:str
    count:int


@dataclass(frozen=True)
class Tensor:
    storage:Storage
    shape:tuple
    stride:tuple


def _contiguous(shape):
    value=1;stride=[]
    for size in reversed(shape):stride.append(value);value*=size
    return tuple(reversed(stride))


def interpret(raw):
    """Return plain bounded containers and inert Tensor references only."""
    require(type(raw) is bytes and 0<len(raw)<=MAX_PICKLE,'bounded pickle metadata')
    stack=[];memo={};storages={};mark=object();stopped=False;items=0
    def push(value):
        stack.append(value);require(len(stack)<=10000,'pickle stack limit')
    def pop():
        require(bool(stack),'pickle stack underflow');return stack.pop()
    def marked():
        require(mark in stack,'pickle MARK required');index=len(stack)-1-stack[::-1].index(mark)
        value=stack[index+1:];del stack[index:];return value
    def setitems(target,values):
        nonlocal items
        require(type(target) in (dict,OrderedDict) and len(values)%2==0,'typed mapping SETITEMS')
        for key,value in zip(values[::2],values[1::2]):
            require(type(key) in (str,int) and key not in target,'unique typed pickle mapping key')
            target[key]=value;items+=1
        require(items<=MAX_ITEMS,'pickle collection item limit')
    for index,(op,arg,pos) in enumerate(pickletools.genops(raw)):
        require(index<MAX_OPS and not stopped,'pickle opcode limit or trailing operations')
        name=op.name
        if name=='PROTO':require(index==0 and arg==2,'only inspected protocol2 is supported')
        elif name=='MARK':push(mark)
        elif name=='EMPTY_DICT':push({})
        elif name=='EMPTY_LIST':push([])
        elif name=='EMPTY_TUPLE':push(())
        elif name=='NONE':push(None)
        elif name in ('NEWTRUE','NEWFALSE'):push(name=='NEWTRUE')
        elif name in ('BININT','BININT1','BININT2','LONG1','LONG4'):
            require(type(arg) is int and abs(arg)<=2**64,'bounded pickle integer');push(arg)
        elif name=='BINFLOAT':require(math.isfinite(arg),'finite pickle float');push(arg)
        elif name=='BINUNICODE':
            require(type(arg) is str and len(arg.encode())<=262144,'bounded pickle string');push(arg)
        elif name=='GLOBAL':
            require(arg in GLOBALS,'unsafe or unsupported pickle GLOBAL');push(Global(GLOBALS[arg]))
        elif name in ('BINPUT','LONG_BINPUT'):
            require(type(arg) is int and 0<=arg<10000 and arg not in memo and stack,'unique bounded pickle memo')
            memo[arg]=stack[-1]
        elif name in ('BINGET','LONG_BINGET'):
            require(arg in memo,'undefined pickle memo');push(memo[arg])
        elif name=='TUPLE':push(tuple(marked()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            count=int(name[-1]);require(len(stack)>=count,'tuple stack underflow')
            values=stack[-count:];del stack[-count:];push(tuple(values))
        elif name=='APPEND':
            value=pop();require(stack and type(stack[-1]) is list,'typed list APPEND');stack[-1].append(value);items+=1
        elif name=='APPENDS':
            values=marked();require(stack and type(stack[-1]) is list,'typed list APPENDS');stack[-1].extend(values);items+=len(values)
        elif name=='SETITEMS':
            values=marked();require(stack,'mapping stack');setitems(stack[-1],values)
        elif name=='SETITEM':
            value,key=pop(),pop();require(stack,'mapping stack');setitems(stack[-1],[key,value])
        elif name=='BINPERSID':
            pid=pop()
            require(type(pid) is tuple and len(pid)==5 and pid[0]=='storage' and type(pid[1]) is Global
                    and pid[1].name in ('float32','uint8') and type(pid[2]) is str and re.fullmatch('0|[1-9][0-9]{0,2}',pid[2])
                    and pid[3]=='cpu' and type(pid[4]) is int and 0<pid[4]<=MAX_BYTES,'exact bounded CPU storage reference')
            value=Storage(pid[2],pid[1].name,pid[4]);require(value.key not in storages,'shared storage reference forbidden')
            storages[value.key]=value;push(value)
        elif name=='REDUCE':
            args,func=pop(),pop();require(type(func) is Global and type(args) is tuple,'inert allowed REDUCE only')
            if func.name=='ordered':require(args==(),'only empty OrderedDict constructor');push(OrderedDict())
            else:
                require(func.name=='tensor' and len(args)==6 and type(args[0]) is Storage and type(args[1]) is int and args[1]==0
                        and type(args[2]) is tuple and len(args[2])<=2 and all(type(n) is int and 0<n<=8192 for n in args[2])
                        and type(args[3]) is tuple and args[3]==_contiguous(args[2]) and args[4] is False
                        and type(args[5]) in (dict,OrderedDict) and not args[5]
                        and math.prod(args[2])==args[0].count,'exact detached contiguous whole-storage tensor descriptor')
                push(Tensor(args[0],args[2],args[3]))
        elif name=='STOP':
            require(len(stack)==1 and pos==len(raw)-1,'exact single-root pickle STOP');stopped=True
        else:raise ValueError('unsafe or unsupported pickle opcode '+name)
        require(items<=MAX_ITEMS,'pickle collection item limit')
    require(stopped and len(storages)==450,'complete exact450-storage archive')
    nodes=0
    def walk(value,active,depth):
        nonlocal nodes
        nodes+=1;require(nodes<=50000 and depth<=32,'bounded acyclic pickle graph')
        if type(value) in (dict,OrderedDict,list,tuple):
            require(id(value) not in active,'cyclic pickle graph');active=active|{id(value)}
            for child in (list(value.keys())+list(value.values()) if type(value) in (dict,OrderedDict) else value):walk(child,active,depth+1)
        else:require(value is None or type(value) in (str,int,float,bool,Tensor),'plain inert graph values only')
    walk(stack[0],set(),0)
    return stack[0],storages


def read_archive(path,descriptor):
    """Read same bounded hashed bytes, validate ZIP, interpret metadata only."""
    require(type(descriptor) is dict and set(descriptor)==FIELDS and type(descriptor['schema_version']) is int and descriptor['schema_version']==1
            and descriptor['kind']=='SYNTH_STUDY_TORCH_BOUNDARY_V1','exact backend descriptor')
    path=Path(path);require(path.is_absolute() and path==Path(descriptor['path']) and path.is_file()
        and not any(p.is_symlink() for p in (path,*path.parents)) and path.resolve()==path.absolute()
        and 'state1' not in str(path).lower(),'absolute regular checkpoint; no path indirection or STATE1')
    require(type(descriptor['bytes']) is int and 0<descriptor['bytes']<=MAX_BYTES and hash64(descriptor['sha256'])
            and descriptor['metadata_sha256']==digest(descriptor['metadata']),'bounded descriptor bytes and metadata hash')
    with path.open('rb') as stream:raw=stream.read(descriptor['bytes']+1)
    require(len(raw)==descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'exact checkpoint bytes/hash')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=archive.infolist();names=[item.filename for item in members]
        fixed={'archive/data.pkl','archive/byteorder','archive/version','archive/.data/serialization_id'}
        require(len(names)==len(set(names))==454 and not archive.comment,'exact unique ZIP member roster')
        require(all(not item.is_dir() and item.compress_type==zipfile.ZIP_STORED and not item.flag_bits&1
                    and 0<item.file_size<=MAX_BYTES and item.file_size==item.compress_size
                    and '\\' not in item.filename and '..' not in item.filename.split('/') for item in members),'stored bounded nonencrypted regular ZIP members')
        require(sum(item.file_size for item in members)<=MAX_BYTES,'ZIP expanded-byte limit')
        require(archive.getinfo('archive/data.pkl').file_size<=MAX_PICKLE,'bounded pickle member before read')
        require(archive.read('archive/byteorder')==b'little' and archive.read('archive/version')==b'3\n'
                and re.fullmatch(b'[0-9]{40}',archive.read('archive/.data/serialization_id')),'inspected archive version/byte order')
        payload,storages=interpret(archive.read('archive/data.pkl'))
        require(set(names)==fixed|{'archive/data/'+key for key in storages},'no unused/missing/escaped ZIP storage')
        data={}
        for key,storage in storages.items():
            width=4 if storage.dtype=='float32' else 1;name='archive/data/'+key
            require(archive.getinfo(name).file_size==storage.count*width,'exact storage dtype/element byte count')
            data[key]=archive.read(name)
    require(type(payload) is dict and set(payload)=={'metadata','state','occurrence_order'}
            and digest(payload['metadata'])==digest(descriptor['metadata']),'serialized metadata equals descriptor')
    return payload,data


def inventory():
    return {f'model.layers.{i}.self_attn.{projection}.lora_{factor}':
            (8,1024) if factor=='A' else (2048 if projection=='q_proj' else 1024,8)
            for i in range(28) for projection in ('q_proj','v_proj') for factor in ('A','B')}


def boundary(registration,arm,completed):
    require(arm in ('M1','M2') and type(completed) is int and completed in (9,18,27),'full completed9/18/27 adapted arm')
    require(registration['sha256']==digest({k:v for k,v in registration.items() if k!='sha256'}),'registration checksum')
    updates=registration['arms'][arm]['updates'][:completed];order=[identity for u in updates for identity in u['occurrence_row_ids']]
    require(len(updates)==completed and len(order)==completed*20,'exact20 per global update')
    lr=float(Fraction(1,10000)*Fraction(27-completed,25))
    return {'completed_updates':completed,'consumed_occurrences':completed*20,'input_tokens':sum(u['input_tokens'] for u in updates),
        'supervised_tokens':sum(u['supervised_tokens'] for u in updates),'consumed_order_sha256':digest(order),
        'next_update_ordinal':completed+1,'learning_rate':lr},order


def compare_state(payload,data,descriptor,witness,binding,order,arm,completed):
    """Pure exact state comparison; caller authenticates provenance anchors."""
    witness_fields={'schema_version','kind','scope','arm','seed','recipe_sha256','cohort_sha256','backend_source_sha256',
        'binding','parameter_order','adapters','optimizer_moments_and_steps','optimizer_groups','gradients_all_None','rng',
        'occurrence_order_sha256','base_source_tensor_hash_map_sha256','base_frozen_source_tensors','base_parameters',
        'maximum_tensor_hash_chunk_bytes','native_launch_authorized','study_warmstart_authorized','sha256'}
    require(type(witness) is dict and set(witness)==witness_fields and type(witness['schema_version']) is int
            and witness['schema_version']==1 and witness['scope']=='UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER'
            and witness['maximum_tensor_hash_chunk_bytes']==64*1024**2
            and witness['sha256']==digest({k:v for k,v in witness.items() if k!='sha256'})
            and witness['kind']=='SYNTH_BACKEND_QUAL_WITNESS_NOT_STUDY_CHECKPOINT'
            and witness['binding']==binding and witness['arm']==arm and witness['seed']==101,'bound detached witness')
    metadata=payload['metadata'];shapes=inventory();names=list(shapes)
    metadata_fields={'schema_version','kind','experiment','scope','arm','seed','recipe_sha256','cohort_sha256',
        'backend_source_sha256','fresh_start_attestation_sha256','adapter_parameter_order','adapter_inventory_sha256',
        'binding','native_launch_authorized','training_authorized','native_resume_qualified','qualification_STATE1_used'}
    require(set(metadata)==metadata_fields and type(metadata['schema_version']) is int and metadata['schema_version']==1
            and metadata['kind']=='SYNTH_STUDY_TORCH_BOUNDARY_V1' and metadata['experiment']=='SYNTH-CRS-001'
            and metadata['binding']==binding and metadata['arm']==arm and metadata['seed']==101
            and metadata['scope']=='UNQUALIFIED_NATIVE_IMPLEMENTATION'
            and metadata['adapter_parameter_order']==witness['parameter_order']==names,'exact112 parameter order/cursor/arm/seed')
    require(all(metadata.get(k) is False for k in ('native_launch_authorized','training_authorized','native_resume_qualified','qualification_STATE1_used'))
            and witness.get('native_launch_authorized') is False and witness.get('study_warmstart_authorized') is False,'closed checkpoint permissions')
    for key in ('recipe_sha256','cohort_sha256','backend_source_sha256'):
        require(hash64(metadata[key]) and metadata[key]==witness[key],'same witness/metadata source '+key)
    require(hash64(metadata['fresh_start_attestation_sha256']) and hash64(metadata['adapter_inventory_sha256'])
            and witness['base_frozen_source_tensors']==310 and witness['base_parameters']==596049920
            and hash64(witness['base_source_tensor_hash_map_sha256']),'frozen base/source metadata')
    state=payload['state'];require(type(state) is dict and set(state)=={'schema_version','adapters','optimizer','gradients','rng'}
        and type(state['schema_version']) is int and state['schema_version']==1 and set(state['adapters'])==set(state['gradients'])==set(names)
        and all(value is None for value in state['gradients'].values()) and witness['gradients_all_None'] is True
        and payload['occurrence_order']==order and witness['occurrence_order_sha256']==digest(order),'complete state/order/empty gradients')
    optimizer=state['optimizer'];require(type(optimizer) is dict and set(optimizer)=={'state','param_groups'}
        and list(optimizer['state'])==list(range(112)) and len(optimizer['param_groups'])==1,'full optimizer state/index order')
    expected_group={'lr':binding['learning_rate'],'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'foreach':False,
        'maximize':False,'capturable':False,'differentiable':False,'fused':False,'params':list(range(112))}
    require(digest(optimizer['param_groups'])==digest([expected_group])==digest(witness['optimizer_groups']),'fixed AdamW/globalLR/groups')
    seen=set();finite=nonnegative=0
    def check_tensor(value,expected,shape,dtype,*,second=False,step=False):
        nonlocal finite,nonnegative
        require(type(value) is Tensor and value.shape==shape and value.storage.dtype==dtype and value.storage.key not in seen,'unique exact tensor shape/dtype/storage')
        seen.add(value.storage.key);raw=data[value.storage.key];spec={'shape':list(shape),'dtype':'torch.'+dtype,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        require(expected==spec,'actual storage bytes match detached tensor hash')
        if dtype=='float32':
            for (number,) in struct.iter_unpack('<f',raw):
                require(math.isfinite(number),'finite FP32 tensor');finite+=1
                if second:require(number>=0,'nonnegative second moment');nonnegative+=1
                if step:require(number==float(completed),'exact completed Adam step')
    require(set(witness['adapters'])==set(witness['optimizer_moments_and_steps'])==set(names),'complete detached tensor inventory')
    for index,name in enumerate(names):
        check_tensor(state['adapters'][name],witness['adapters'][name],shapes[name],'float32')
        moments=optimizer['state'][index];require(type(moments) is dict and set(moments)=={'step','exp_avg','exp_avg_sq'},'exact Adam moment fields')
        expected=witness['optimizer_moments_and_steps'][name];require(set(expected)==set(moments),'complete detached moment fields')
        check_tensor(moments['step'],expected['step'],(),'float32',step=True)
        check_tensor(moments['exp_avg'],expected['exp_avg'],shapes[name],'float32')
        check_tensor(moments['exp_avg_sq'],expected['exp_avg_sq'],shapes[name],'float32',second=True)
    rng=state['rng'];require(type(rng) is dict and set(rng)=={'python','cpu','cuda0'} and type(rng['python']) is tuple
        and len(rng['python'])==3 and type(rng['python'][0]) is int and rng['python'][0]==3,'exact three RNG state fields')
    random.Random().setstate(rng['python'])
    require(digest(rng['python'])==witness['rng']['python_sha256'],'actual saved Python RNG hash')
    check_tensor(rng['cpu'],witness['rng']['cpu'],(5056,),'uint8')
    check_tensor(rng['cuda0'],witness['rng']['cuda0'],(16,),'uint8')
    require(len(seen)==len(data)==450,'all450 saved storages matched without aliases/unused data')
    return {'status':'SYMBOLIC_ARCHIVE_MATCHES_BOUND_WITNESS_NO_NATIVE_AUTHORITY','storage_count':450,'adapter_tensors':112,
        'adapter_parameters':sum(math.prod(s) for s in shapes.values()),'Adam_moment_tensors':224,'Adam_step_tensors':112,
        'completed_updates':completed,'finite_FP32_values':finite,'nonnegative_second_moment_values':nonnegative,
        'Python_RNG_private_probe_valid':True,'CPU_CUDA0_RNG_bytes_match':True,'Torch_RNG_encoding_probed':False,
        'metadata_order_cursor_groups_empty_gradients_match':True,'checkpoint_sha256':descriptor['sha256'],
        'native_qualification_established':False,'native_audit_PASS_issued':False,'framework_model_GPU_restore_replay_executed':False,
        'study_training_authorized':False,'study_warmstart_allowed':False}


def verify_checkpoint(path,descriptor,witness,registration,arm,completed):
    binding,order=boundary(registration,arm,completed)
    require(descriptor['metadata']['cohort_sha256']==registration['arms'][arm]['occurrence_order_sha256'],'full27 qualification cohort')
    payload,data=read_archive(path,descriptor)
    return compare_state(payload,data,descriptor,witness,binding,order,arm,completed)
