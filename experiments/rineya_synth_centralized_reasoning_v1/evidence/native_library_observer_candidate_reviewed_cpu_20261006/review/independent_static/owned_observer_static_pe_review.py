"""Read owned PE/object disassembly only; never load or register the DLL."""
from pathlib import Path
import hashlib,json,re,struct,time
ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work/observer_source_work'
BUILD=SRC/'native_build_001';OUT=Path(__file__).resolve().parent
names=('native_library_observer.dll','OBJECT_DISASSEMBLY.txt','DLL_DISASSEMBLY.txt','SYMBOLS.txt','IMPORTS.txt','EXPORTS.txt')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={name:sha(BUILD/name) for name in names}
assert before['native_library_observer.dll']=='8f6ac9f0edcd716b62e06054ec6ca503d961b598177144ffcb1493656e24c679'
raw=(BUILD/names[0]).read_bytes();assert len(raw)==5632 and raw[:2]==b'MZ'
u16=lambda offset:struct.unpack_from('<H',raw,offset)[0]
u32=lambda offset:struct.unpack_from('<I',raw,offset)[0]
nt=u32(0x3c);assert raw[nt:nt+4]==b'PE\0\0';machine=u16(nt+4);count=u16(nt+6);optional=nt+24;optional_size=u16(nt+20)
assert machine==0x8664 and u16(optional)==0x20b
entrypoint=u32(optional+16);assert entrypoint==0
directory=optional+112
imports=(u32(directory+8),u32(directory+12));delay=(u32(directory+13*8),u32(directory+13*8+4))
assert imports==(0,0) and delay==(0,0)
sections=[]
for i in range(count):
    base=optional+optional_size+i*40
    name=raw[base:base+8].split(b'\0')[0].decode('ascii')
    sections.append({'name':name,'virtual_size':u32(base+8),'rva':u32(base+12),'raw_size':u32(base+16),'raw_offset':u32(base+20)})
def offset(rva):
    for section in sections:
        if section['rva']<=rva<section['rva']+max(section['virtual_size'],section['raw_size']):
            answer=section['raw_offset']+rva-section['rva'];assert answer<len(raw);return answer
    raise AssertionError('RVA outside owned PE sections')
def text(rva):
    start=offset(rva);end=raw.find(b'\0',start,start+4096);assert end>=start;return raw[start:end].decode('ascii')
export_rva,export_size=u32(directory),u32(directory+4);ex=offset(export_rva)
functions=u32(ex+20);number_names=u32(ex+24);names_rva=u32(ex+32);ordinals=u32(ex+36);addresses=u32(ex+28)
assert functions==number_names==5
exports={}
for i in range(number_names):
    name=text(u32(offset(names_rva)+4*i));ordinal=u16(offset(ordinals)+2*i)
    rva=u32(offset(addresses)+4*ordinal);assert not export_rva<=rva<export_rva+export_size
    exports[name]=rva
expected={'plobs_abi_version','plobs_start','plobs_snapshot','plobs_copy_rows','plobs_stop'}
assert set(exports)==expected
obj=(BUILD/'OBJECT_DISASSEMBLY.txt').read_text('utf-8');functions_text={};current=None
for line in obj.splitlines():
    match=re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*):',line)
    if match:current=match.group(1);functions_text[current]=[]
    elif current is not None:functions_text[current].append(line)
graph={};visited=set();todo=['notification']
while todo:
    name=todo.pop();assert name in functions_text;visited.add(name)
    calls=[]
    for line in functions_text[name]:
        match=re.search(r'\b(call|jmp)\s+(.+)$',line)
        if not match:continue
        target=match.group(2).strip()
        if match.group(1)=='jmp' and re.fullmatch(r'[0-9A-Fa-f]+',target):continue
        assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',target) and target in functions_text, ('callback indirect/external target',name,target)
        calls.append(target)
        if target not in visited:todo.append(target)
    graph[name]=calls
assert set(graph)=={'notification','plo_core_record','copy_name'}
assert graph['copy_name']==[] and set(graph['notification'])=={'plo_core_record'} and set(graph['plo_core_record'])=={'copy_name'}
symbols=(BUILD/'SYMBOLS.txt').read_text('utf-8');assert not re.search(r'UNDEF.*External',symbols)
source_hashes={name:sha(SRC/name) for name in ('prefix_library_observer.c','prefix_library_observer.h')}
assert source_hashes['prefix_library_observer.c']=='436d26bd0dfc71e659295e9a87a9bb48b73ead68602b4b46077bd5c003618855'
assert source_hashes['prefix_library_observer.h']=='7a6a8a33fafd53e279f5c0aec6eab00b38add190cc2f0b9e99b5536b02ebd9d7'
after={name:sha(BUILD/name) for name in names};assert before==after
value={'kind':'INDEPENDENT_OWNED_OBSERVER_PE_CALLBACK_GRAPH_STATIC_REVIEW','passed':True,
 'source_hashes':source_hashes,'owned_artifact_before':before,'owned_artifact_after':after,
 'PE':{'bytes':len(raw),'machine':machine,'entrypoint':entrypoint,'imports':imports,'delay_imports':delay,'exports':exports,'sections':sections},
 'callback_reachable_graph':graph,'callback_indirect_external_targets':[],
 'DLL_loaded':False,'Windows_API_called':False,'observer_registered':False,'native_PASS':False,
 'native_authority_issued':False,'unregister_quiescence_qualified':False,
 'limits':['PE bytes independently show zero entry/import/delay-import directories and five nonforwarded exports.',
 'Callback/local call reachability uses the saved object disassembly and symbol provenance; static review is not dynamic execution/lifetime proof.',
 'Start/stop intentionally invoke supplied registration/unregister function pointers outside the callback graph.',
 'Native callback scheduling, actual unregister/quiescence, loader-lock behavior, complete runtime/bootstrap/ownership/resource evidence and launch permission remain unqualified.']}
value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()
path=OUT/('OWNED_OBSERVER_PE_STATIC_REVIEW_'+str(time.time_ns())+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':value['sha256'],'owned_DLL_bytes':len(raw),'entrypoint':entrypoint,'imports':imports,'exports':list(exports),'callback_graph':graph,'source_stable':True}))
