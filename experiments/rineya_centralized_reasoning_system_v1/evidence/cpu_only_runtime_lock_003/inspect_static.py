"""One bounded, data-only CPU artifact/lock inspection. Never import wheel code."""
import base64
import csv
import email.parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

BEGIN = time.monotonic()
ROOT = Path(__file__).resolve().parents[4]
PKG = ROOT / 'experiments/rineya_centralized_reasoning_system_v1'
OUT = Path(__file__).resolve().parent
ART = PKG / '.venv/artifacts'
BASE = Path(r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none')
TOOL = Path(r'C:\Users\clclc\BusinessProject\.venv-mq001\Lib\site-packages\packaging')
SCOPE_SHA = 'eef6b84f5fd7d69e5d63160508a4aee6ec116dceecd2dc7dfb66ac674a90d498'
AMENDMENT_SHA = '7ddb209c5b6bba259ba4507fb20d92410828f4117726fc1f600ad29d05d142be'
TORCH_SHA = '17a09465bab2aab8f0f273410297133d8d8fb6dd84dccbd252ca4a4f3a111847'
events = []
cost = dict(metadata_http_responses=0, metadata_http_bytes=0, wheel_transfers=0,
            new_wheel_bytes=0, combined_wheel_bytes=113636478,
            selected_source_metadata_files=0, selected_source_metadata_bytes=0,
            selected_PE_binaries=0, selected_PE_bytes=0)
report = dict(status='IN_PROGRESS_NOT_QUALIFIED', scope_sha256=SCOPE_SHA,
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              limits={'seconds':600, 'wheels':40, 'compressed_bytes':805306368,
                      'metadata_responses':80, 'metadata_bytes':33554432,
                      'selected_files':100, 'selected_source_bytes':8388608,
                      'selected_PE':64, 'selected_PE_bytes':1073741824},
              prohibited_actions={'installation':0,'venv_creation':0,'target_code_import_execution':0,
                                  'member_extraction':0,'model_assets':0,'GPU_process_DB_observation':0},
              cost=cost, events=events, wheels=[], interpreter=[], dependencies=[])

class Stop(Exception):
    pass

def check(condition, reason):
    if not condition:
        raise Stop(reason)
    if time.monotonic()-BEGIN > 600:
        raise Stop('stage_elapsed_600_second_cap')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def hashfile(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):
            h.update(chunk)
            check(True,'deadline')
    return h.hexdigest()

def write(name, value):
    path=OUT/name
    check(not path.exists(), 'refuse_overwrite:'+name)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=True); f.write('\n')

def safe_name(name):
    check(bool(name) and '\\' not in name and '\x00' not in name,
          'unsafe_zip_name:'+name)
    check(not name.startswith('/') and not re.match(r'^[A-Za-z]:',name), 'absolute_zip_name:'+name)
    parts=PurePosixPath(name).parts
    check(not any(p in ('..','.') or ':' in p for p in parts), 'traversal_drive_zip_name:'+name)
    # Reject Windows trailing-dot/space and device path ambiguity as well.
    for p in parts:
        check(p.rstrip(' .')==p and p.split('.')[0].upper() not in
              {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))},
              'ambiguous_windows_zip_name:'+name)

def validate_zip(z):
    infos=z.infolist(); check(len(infos)<=40000, 'zip_member_cap')
    total=sum(x.file_size for x in infos); check(total<=2147483648,'zip_declared_uncompressed_cap')
    exact=set(); case=set()
    for x in infos:
        safe_name(x.filename)
        key=x.filename.rstrip('/'); fold=key.casefold()
        check(key not in exact and fold not in case, 'zip_duplicate_or_case_collision:'+x.filename)
        exact.add(key); case.add(fold)
        mode=x.external_attr>>16
        check(not stat.S_ISLNK(mode), 'zip_symlink:'+x.filename)
        check(not (x.flag_bits & 1), 'encrypted_zip_member:'+x.filename)
        check(x.file_size<=1073741824 and (x.file_size<1024*1024 or x.file_size/max(1,x.compress_size)<=1000),
              'zip_bomb_ratio_or_member_size:'+x.filename)
    return dict(members=len(infos), declared_uncompressed_bytes=total, safety='PASS_DATA_ONLY_NO_EXTRACTION')

def selected(z,name,pe=False):
    size=z.getinfo(name).file_size
    prefix='selected_PE' if pe else 'selected_source_metadata'
    cost[prefix+('_binaries' if pe else '_files')]+=1
    cost[prefix+'_bytes']+=size
    check(cost[prefix+('_binaries' if pe else '_files')] <= (64 if pe else 100), 'selected_file_cap')
    check(cost[prefix+'_bytes'] <= (1073741824 if pe else 8388608),'selected_bytes_cap')
    b=z.read(name); check(len(b)==size,'zip_selected_size_mismatch')
    return b

def parse_pe(b):
    check(b[:2]==b'MZ' and len(b)>=64,'not_PE')
    pe=struct.unpack_from('<I',b,0x3c)[0]
    check(pe+24<=len(b) and b[pe:pe+4]==b'PE\0\0','invalid_PE_header')
    machine,nsec,_,_,_,optsize,flags=struct.unpack_from('<HHIIIHH',b,pe+4)
    opt=pe+24; magic=struct.unpack_from('<H',b,opt)[0]
    check(magic in (0x20b,0x10b), 'invalid_PE_optional_magic')
    dd=opt+(112 if magic==0x20b else 96)
    imagebase=struct.unpack_from('<Q' if magic==0x20b else '<I',b,opt+(24 if magic==0x20b else 28))[0]
    sections=[]
    for i in range(nsec):
        off=opt+optsize+40*i; check(off+40<=len(b),'truncated_PE_section')
        name=b[off:off+8].rstrip(b'\0').decode('ascii','replace')
        vs,va,rs,raw=struct.unpack_from('<IIII',b,off+8)
        check(raw+rs<=len(b),'PE_section_outside_file')
        sections.append((name,vs,va,rs,raw))
    def rva(x):
        for _,vs,va,rs,raw in sections:
            if va<=x<va+max(vs,rs):
                off=raw+x-va; check(off<len(b),'PE_RVA_outside_file'); return off
        if 0<=x<struct.unpack_from('<I',b,opt+60)[0]: return x
        raise Stop('unmapped_PE_RVA:'+hex(x))
    def string(x):
        off=rva(x); end=b.find(b'\0',off,min(off+4096,len(b)))
        check(end>=off,'unterminated_PE_import_string')
        return b[off:end].decode('ascii','strict')
    def directory(index):
        check(dd+index*8+8<=opt+optsize,'PE_directory_outside_optional_header')
        return struct.unpack_from('<II',b,dd+index*8)
    imports=[]; functions={}; delays=[]
    va,sz=directory(1)
    if va:
        off=rva(va)
        for i in range(4096):
            pos=off+20*i; check(pos+20<=len(b),'truncated_PE_imports')
            fields=struct.unpack_from('<IIIII',b,pos)
            if not any(fields):break
            dll=string(fields[3]); imports.append(dll)
            thunk=fields[0] or fields[4]
            width=8 if magic==0x20b else 4; names=[]
            if thunk:
                to=rva(thunk)
                for j in range(20000):
                    tv=struct.unpack_from('<Q' if width==8 else '<I',b,to+width*j)[0]
                    if not tv:break
                    if not(tv & (1 << (width*8-1))): names.append(string(tv+2))
                else: raise Stop('PE_import_thunk_cap')
            functions[dll]=[x for x in names if re.search(r'LoadLibrary|GetProcAddress|DllDirectory|CreateProcess|OpenCL|CUDA|cuInit|nvcuda',x,re.I)]
        else: raise Stop('PE_import_descriptor_cap')
    va,sz=directory(13)
    if va:
        off=rva(va)
        for i in range(4096):
            pos=off+32*i; check(pos+32<=len(b),'truncated_PE_delay_imports')
            f=struct.unpack_from('<IIIIIIII',b,pos)
            if not any(f):break
            delays.append(string(f[1] if f[0]&1 else f[1]-imagebase))
        else: raise Stop('PE_delay_descriptor_cap')
    return dict(machine=hex(machine), machine_name='AMD64' if machine==0x8664 else 'OTHER',
                optional_magic=hex(magic), imports=imports, delay_imports=delays,
                selected_dynamic_loader_import_symbols=functions)

class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check(urllib.parse.urlparse(newurl).scheme=='https' and
              urllib.parse.urlparse(newurl).hostname in {'pypi.org','files.pythonhosted.org'},'unapproved_redirect:'+newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def http_metadata(url):
    check(cost['metadata_http_responses']<80,'metadata_response_cap')
    check(urllib.parse.urlparse(url).hostname=='pypi.org','metadata_origin')
    event=dict(kind='metadata',url=url,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),attempt=1)
    events.append(event); t=time.monotonic()
    try:
        with urllib.request.build_opener(OfficialRedirect()).open(
             urllib.request.Request(url,headers={'User-Agent':'CRS-001-static-lock/003'}),timeout=min(45,max(1,600-(time.monotonic()-BEGIN)))) as response:
            cost['metadata_http_responses']+=1
            data=response.read(min(33554432-cost['metadata_http_bytes']+1,8388609))
            cost['metadata_http_bytes']+=len(data)
            check(cost['metadata_http_bytes']<=33554432 and len(data)<=8388608,'metadata_byte_cap')
            event.update(status=response.status,final_url=response.url,bytes=len(data),sha256=digest(data))
        value=json.loads(data)
        write('PYPI_'+urllib.parse.quote(value['info']['name'],safe='')+'_'+value['info']['version']+'.json',value)
        return value
    except Exception as e:
        event.update(outcome='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e))
        raise
    finally:event['elapsed_seconds']=time.monotonic()-t

def main():
    check(ROOT.name=='ChatBubble-rineya-centralized-reasoning-system-v1','workspace_binding')
    scope=PKG/'authorization/DELEGATED_CPU_ONLY_RUNTIME_LOCK_INSPECTION_003.json'
    check(hashfile(scope)==SCOPE_SHA,'scope_hash_drift')
    amendment=PKG/'authorization/DELEGATED_CPU_ONLY_RUNTIME_METADATA_SELECTION_004.json'
    check(hashfile(amendment)==AMENDMENT_SHA,'metadata_amendment_hash_drift')
    report['metadata_selection_amendment']={'path':str(amendment),'sha256':AMENDMENT_SHA}
    pins={'OPS_scope':('audits/CPU_ONLY_RUNTIME_CONTROL_OPS_SCOPE_001.md','f667f6be70cd4f5bd582c7e05e8427831a1f3a4e0d1c8b299f92feb2ea186a4f'),
          'feasibility':('audits/CPU_ONLY_RUNTIME_CONTROL_FEASIBILITY_001.md','bac0688324183840e71b6447794bc238af92c9d15514941d02d90144fb3fb859'),
          'baseline':('evidence/cpu_only_runtime_artifact_002/BASELINE_DISTRIBUTIONS.json','c3f09cfb8161a835ae0beff6f3a77afcf88670aaa34b40ac4c80429cfef0d7a7'),
          'wheel_inspection':('evidence/cpu_only_runtime_artifact_002/WHEEL_INSPECTION.json','e6172d2cba146a0de39ebbcd483ebea98bc0757a8921adb98b0582447795c756')}
    report['bindings']={}
    for key,(path,sha) in pins.items():
        check(hashfile(PKG/path)==sha,'binding_drift:'+key)
        report['bindings'][key]={'path':str(PKG/path),'sha256':sha}
    report['tool']={'script_sha256':hashfile(Path(__file__)), 'python_version':sys.version,
                    'isolated_startup':sys.flags.isolated==1,'site_disabled':sys.flags.no_site==1,
                    'executable':str(Path(sys.executable).resolve()),'packaging_source':str(TOOL),'packaging_files':[]}
    check(Path(sys.executable).resolve()==BASE/'python.exe','wrong_inspection_interpreter')
    for name in ['python.exe','python3.dll','python311.dll','vcruntime140.dll','vcruntime140_1.dll']:
        p=BASE/name; b=p.read_bytes()
        report['interpreter'].append({'path':str(p),'bytes':len(b),'sha256':digest(b),'PE':parse_pe(b)})
    for p in sorted(TOOL.rglob('*.py')):
        report['tool']['packaging_files'].append({'path':str(p),'sha256':hashfile(p),'bytes':p.stat().st_size})
    report['tool']['packaging_tree_manifest_sha256']=digest(json.dumps(report['tool']['packaging_files'],sort_keys=True).encode())
    # This existing pure-Python parser is explicitly approved inspection tooling,
    # not target wheel code. The old environment is read-only and never launched.
    sys.path.insert(0,str(TOOL.parent))
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.specifiers import SpecifierSet
    env=default_environment(); env.update(python_version='3.11',python_full_version='3.11.15',
         sys_platform='win32',os_name='nt',platform_system='Windows',platform_machine='AMD64',
         implementation_name='cpython',platform_python_implementation='CPython',extra='')
    report['marker_environment']=env
    tags=list(cpython_tags(python_version=(3,11),platforms=['win_amd64']))+list(compatible_tags(python_version=(3,11),interpreter='cp311',platforms=['win_amd64']))
    tag_order={t:i for i,t in enumerate(tags)}
    versions={canonicalize_name(k):v for k,v in json.loads((PKG/pins['baseline'][0]).read_text())['versions'].items()}
    versions['torch']='2.10.0+cpu'; report['baseline_versions']=versions
    torch=ART/'torch-2.10.0+cpu-cp311-cp311-win_amd64.whl'
    check(torch.stat().st_size==113636478 and hashfile(torch)==TORCH_SHA,'torch_physical_hash_or_size_drift')
    torch_item={'name':'torch','version':'2.10.0+cpu','filename':torch.name,'path':str(torch),
                'sha256':TORCH_SHA,'bytes':torch.stat().st_size,'origin':'https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp311-cp311-win_amd64.whl',
                'retained_official_publication_binding':'authorization/DELEGATED_CPU_ONLY_RUNTIME_ARTIFACT_001.json; artifact_002 frozen evidence',
                'new_transfer':False}
    report['wheels'].append(torch_item)
    with zipfile.ZipFile(torch) as z:
        torch_item['ZIP']=validate_zip(z)
        names=z.namelist(); dist='torch-2.10.0+cpu.dist-info/'
        meta=selected(z,dist+'METADATA'); wheel=selected(z,dist+'WHEEL'); record=selected(z,dist+'RECORD')
        m=email.parser.BytesParser().parsebytes(meta); w=email.parser.BytesParser().parsebytes(wheel)
        records={row[0]:row[1:] for row in csv.reader(io.StringIO(record.decode()))}
        for name in records:safe_name(name)
        check(all(n in records for n in names if not n.endswith('/')),'torch_RECORD_missing_member')
        torch_item['metadata']={'name':m['Name'],'version':m['Version'],'requires_python':m['Requires-Python'],
                                'requires_dist':m.get_all('Requires-Dist',[]),'tags':w.get_all('Tag',[]),
                                'METADATA_sha256':digest(meta),'WHEEL_sha256':digest(wheel),'RECORD_sha256':digest(record),
                                'RECORD_rows':len(records)}
        check(m['Version']=='2.10.0+cpu' and 'cp311-cp311-win_amd64' in w.get_all('Tag',[]),'torch_metadata_binding')
        native=[n for n in names if n.endswith(('.pyd','.dll'))]
        check(len(native)==10,'unexpected_Torch_native_selection_count')
        torch_item['selected_PE']=[]
        selected_hashes={dist+'METADATA':meta,dist+'WHEEL':wheel}
        for n in native:
            b=selected(z,n,True); sha=digest(b); pe=parse_pe(b)
            check(pe['machine']=='0x8664','torch_non_AMD64_binary')
            build=[]
            for match in re.finditer(rb'(?:USE_CUDA|USE_CUDNN|USE_XPU|USE_ROCM|USE_MKLDNN|USE_MKL|BLAS_INFO|Build settings)[=:][ -~]{0,512}',b):
                build.append(match.group().decode('ascii'))
                if len(build)>=24:break
            torch_item['selected_PE'].append({'member':n,'bytes':len(b),'sha256':sha,'PE':pe,'bounded_build_strings':build})
            rec=records[n]; expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')
            check(rec==[expected,str(len(b))],'torch_selected_PE_RECORD_mismatch:'+n)
        torch_item['selected_source']=[]
        for n in ['torch/version.py','torch/__init__.py','torch/__config__.py','torch/_utils_internal.py','torch/cuda/__init__.py']:
            b=selected(z,n); selected_hashes[n]=b
            text=b.decode('utf-8')
            patterns=['cuda:', 'hip:', 'rocm:', 'xpu:', 'USE_CUDA', 'USE_XPU','LoadLibrary','AddDllDirectory','_load_dll_libraries','os.environ','PATH','from torch._C','_show_config']
            lines=[{'line':i,'text':line[:400]} for i,line in enumerate(text.splitlines(),1) if any(p in line for p in patterns)]
            torch_item['selected_source'].append({'member':n,'bytes':len(b),'sha256':digest(b),'relevant_lines':lines[:100]})
        for n,b in selected_hashes.items():
            expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')
            check(records[n]==[expected,str(len(b))],'torch_selected_metadata_source_RECORD_mismatch:'+n)
    report['native_proof_limits']={
        'selection':'All ten Torch .pyd/.dll payloads, load entry/source/version/config and five Python base executable/DLL files.',
        'no_target_execution':True,'all_C_zero_hardware_guarantee':False,
        'unresolved':['PE imports do not enumerate computed LoadLibrary/GetProcAddress targets or internal C dispatch.',
                      'torch.__config__ calls _C._show_config; this static stage does not execute it.',
                      'torch loader source includes Python-base, userbase/Library/bin and PATH search branches; future containment must bind/refuse inherited roots.',
                      'Actual loaded-module identities, startup hooks/search paths, optional plugin routes and no-device dispatch require fresh independently reviewed runtime source and observation.',
                      'Helper native wheels and all transitive artifact hashes must be bound before setup.' ],
        'readiness':'NOT_READY_CPU_RUNTIME_PROVENANCE_BOUNDARY_UNTIL_COMPLETE_LOCK_AND_INDEPENDENT_REVIEW'}
    # Resolve exact official metadata first. A mandatory absent baseline entry
    # stops before artifact transfer; no fallback version, optional CUDA extras,
    # installer resolver, or target code is used.
    queue=['transformers','safetensors','tokenizers','psutil','torch']; seen=set(); metadata={}
    while queue:
        name=queue.pop(0)
        if name in seen:continue
        seen.add(name); check(name in versions,'mandatory_dependency_absent_baseline:'+name)
        if name=='torch': reqs=torch_item['metadata']['requires_dist']
        else:
            url='https://pypi.org/pypi/'+urllib.parse.quote(name,safe='')+'/'+urllib.parse.quote(versions[name],safe='')+'/json'
            info=http_metadata(url); metadata[name]=info
            check(canonicalize_name(info['info']['name'])==name and info['info']['version']==versions[name], 'PyPI_identity_conflict:'+name)
            check(not info['info'].get('requires_python') or SpecifierSet(info['info']['requires_python']).contains('3.11.15'), 'PyPI_python_conflict:'+name)
            reqs=info['info'].get('requires_dist') or []
        for raw in reqs:
            req=Requirement(raw); active=req.marker is None or req.marker.evaluate(env)
            dep=canonicalize_name(req.name)
            row=dict(parent=name,requires_dist=raw,active_without_extras=active,dependency=dep)
            report['dependencies'].append(row)
            if not active:continue
            check(not req.url and not req.extras,'unsupported_dependency_URL_or_extra:'+raw)
            if dep not in versions:
                row['conflict']='MANDATORY_DEPENDENCY_ABSENT_BASELINE'
                report['blocking_dependency']=row
                raise Stop('mandatory_dependency_absent_baseline:'+dep)
            row['baseline_version']=versions[dep]
            check(req.specifier.contains(versions[dep],prereleases=True),'mandatory_baseline_version_conflict:'+raw+' against '+versions[dep])
            queue.append(dep)
    report['resolved_closure']=sorted(seen)
    # This branch can only be reached after complete exact metadata closure.
    # Refuse an aggregate metadata selection that cannot cover three files per
    # wheel plus the selected Torch source within the amended 100-file cap.
    check(3*len(seen)+5<=100,'complete_wheel_metadata_selection_exceeds_amended_100_file_cap')
    raise Stop('complete_metadata_closure_requires_registered_one_batch_artifact_transfer_and_wheel_metadata_proof_before_readiness')

try:
    main()
except Stop as e:
    report['status']='NOT_READY_CPU_RUNTIME_PROVENANCE_BOUNDARY'
    report['stop_reason']=str(e)
except Exception as e:
    report['status']='STOPPED_FAILURE_NO_RETRY_NOT_READY'
    report['stop_reason']=type(e).__name__+': '+str(e)
    report['traceback']=traceback.format_exc()
finally:
    cost['elapsed_seconds']=time.monotonic()-BEGIN
    report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write('STATIC_INSPECTION_AND_CLOSURE.json',report)
    manifest=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file():manifest.append({'path':p.name,'bytes':p.stat().st_size,'sha256':hashfile(p)})
    write('FREEZE.json',{'status':'FROZEN_STOP_FOR_INDEPENDENT_ARTIFACT_REVIEW_NO_INSTALL',
                         'files':manifest,'scope_sha256':SCOPE_SHA,'artifact_sha256':TORCH_SHA,
                         'result_status':report['status'],'stop_reason':report.get('stop_reason'),
                         'cost':cost,'not_native_model_or_bootstrap_result':True})
    print(json.dumps({'status':report['status'],'stop_reason':report.get('stop_reason'),'cost':cost,'freeze':str(OUT/'FREEZE.json')}))
