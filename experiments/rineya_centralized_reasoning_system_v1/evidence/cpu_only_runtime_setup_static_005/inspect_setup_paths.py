"""One 60-second read-only setup-path/tool proof; no installer or target code."""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
import time
import traceback

START=time.monotonic()
OUT=Path(__file__).resolve().parent
PKG=OUT.parents[1]
SCOPE_SHA='c319e24a1d0066c2735266824990b863a4c08faf4c4c077d55ad649712a0f64e'
PLAN_SHA='57397ff857168d2ae26cd716665f6356e1b3f63a16ae0fff39ff6d130d738d25'
PROPOSAL_SHA='46192021cf96bad9f3d291e5326a22886d2652b39e10577815169f9c7f86caf3'
HELPER_SHA='e2608322bb848d1a16c6f53aaea848ada7475dc5fb4c02d87cd8caf994d23c9d'
cost={'selected_tool_binary_files':0,'selected_tool_binary_bytes':0,
      'selected_startup_source_data_files':0,'selected_startup_source_data_bytes':0,'import_API_set_rows':0}
report={'status':'IN_PROGRESS_NOT_READY','scope_sha256':SCOPE_SHA,'plan_sha256':PLAN_SHA,
        'proposal_sha256':PROPOSAL_SHA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'began_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cost':cost,
        'selected_files':[],'presence_checks':[],'path_checks':[],'import_API_set_dispositions':[],
        'startup_findings':[],'UV_embedded_query_evidence':{},'unknowns':[],
        'prohibited_actions':{'UV_invocations':0,'venv_creation_install':0,'target_code_execution':0,
                             'framework_model_GPU_process_DB_observation':0,'network_transfers_tests':0},
        'inspection_interpreter':{'path':sys.executable,'version':sys.version,'isolated':sys.flags.isolated,
                                  'site_disabled':sys.flags.no_site,'tool_mode':'stdlib-data-only, not an interpreter/base qualification test'}}

class Stop(Exception):pass

def check(value,reason):
    if not value:raise Stop(reason)
    if time.monotonic()-START>60:raise Stop('60_second_static_proof_cap')

def sha(data):return hashlib.sha256(data).hexdigest()

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b);check(True,'deadline')
    return h.hexdigest()

def read_selected(p,source=False):
    check(p.exists(),'selected_file_absent:'+str(p))
    check(not (p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT),'selected_reparse:'+str(p))
    size=p.stat().st_size
    prefix='selected_startup_source_data' if source else 'selected_tool_binary'
    cost[prefix+'_files']+=1;cost[prefix+'_bytes']+=size
    check(cost[prefix+'_files'] <= (32 if source else 64),'selected_file_cap')
    check(cost[prefix+'_bytes'] <= (8388608 if source else 536870912),'selected_byte_cap')
    b=p.read_bytes()
    report['selected_files'].append({'path':str(p.resolve()),'bytes':len(b),'sha256':sha(b),'kind':'startup_source_data' if source else 'tool_binary'})
    return b

def pe_sections(b):
    check(b[:2]==b'MZ','non_PE_file')
    po=struct.unpack_from('<I',b,60)[0];check(b[po:po+4]==b'PE\0\0','PE_signature')
    count=struct.unpack_from('<H',b,po+6)[0];opt=struct.unpack_from('<H',b,po+20)[0]
    sections={}
    for i in range(count):
        off=po+24+opt+40*i
        name=b[off:off+8].rstrip(b'\0').decode('ascii')
        vs,va,rs,raw=struct.unpack_from('<IIII',b,off+8)
        check(raw+rs<=len(b),'PE_section_bounds')
        sections[name]={'virtual_size':vs,'VA':va,'bytes':rs,'offset':raw}
    return sections

def api_map(b):
    sec=pe_sections(b).get('.apiset');check(sec is not None,'API_set_schema_section_absent')
    data=b[sec['offset']:sec['offset']+sec['bytes']]
    version,size,flags,count,entry,hashoff,hashfactor=struct.unpack_from('<7I',data)
    check(version==6 and size<=len(data) and count<10000,'unhandled_API_set_schema_header')
    def string(off,length):
        check(off+length<=size and length%2==0,'API_set_string_bounds')
        return data[off:off+length].decode('utf-16-le')
    mapping={}
    for i in range(count):
        pos=entry+24*i;check(pos+24<=size,'API_set_entry_bounds')
        _,no,nl,hl,vo,vc=struct.unpack_from('<6I',data,pos);name=string(no,nl).lower()
        values=[]
        for j in range(vc):
            pos=vo+20*j;check(pos+20<=size,'API_set_value_bounds')
            _,ao,al,ho,hl=struct.unpack_from('<5I',data,pos)
            values.append({'alias':string(ao,al),'host':string(ho,hl)})
        mapping[name]=values
    return mapping,{'version':version,'namespace_bytes':size,'entries':count,'selected_section_sha256':sha(data[:size])}

def path_check(p,allow_absent=False):
    for ancestor in reversed(p.parents):
        if ancestor.exists():
            check(not(ancestor.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT),'ancestor_reparse:'+str(ancestor))
    exists=p.exists()
    if exists:check(not(p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT),'target_reparse:'+str(p))
    report['path_checks'].append({'path':str(p),'exists':exists,'canonical_when_present':str(p.resolve()) if exists else None,'existing_ancestor_reparse':False})
    if not allow_absent:check(exists,'required_path_absent:'+str(p))

def main():
    check(hashfile(PKG/'authorization/DELEGATED_CPU_ONLY_RUNTIME_SETUP_STATIC_005.json')==SCOPE_SHA,'scope_hash_drift')
    proposed=PKG/'resolution_proposals/cpu_only_runtime_setup_001'
    check(hashfile(proposed/'PLAN.md')==PLAN_SHA and hashfile(proposed/'PROPOSAL.json')==PROPOSAL_SHA,'prospective_design_drift')
    proposal=json.loads((proposed/'PROPOSAL.json').read_text())
    helper=PKG/'evidence/cpu_only_runtime_lock_003/inspect_static.py'
    check(hashfile(helper)==HELPER_SHA,'hash_bound_PE_parser_drift')
    # Reuse only the locally authored, frozen data-only PE parser definitions.
    # Target wheel/source code is never evaluated, imported or compiled.
    namespace={'__file__':str(helper),'__name__':'hash_bound_static_PE_tool'}
    exec(compile(helper.read_text().split('\ntry:\n    main()\n')[0],str(helper),'exec'),namespace)
    namespace['check']=check;parse_pe=namespace['parse_pe']
    report['PE_parser_binding']={'source':str(helper),'sha256':HELPER_SHA,'target_code':False}
    bases=proposal['current_read_only_proof']['selected_files']
    base=Path(bases[1]['path']).parent;uv=Path(bases[0]['path']);system=Path('C:/Windows/System32')
    check(Path(sys.executable).resolve()==base/'python.exe' and sys.flags.isolated and sys.flags.no_site,'inspection_tool_startup_binding')
    for p in [uv.parent,base,system,PKG/'.venv',PKG/'.venv/artifacts']:
        path_check(p)
    for p in [PKG/'.venv/runtime',PKG/'.venv/setup-cache-001']:
        path_check(p,True);check(not p.exists(),'future_setup_destination_not_absent')
    binaries={};uv_bytes=None
    for pin in bases:
        p=Path(pin['path']);b=read_selected(p)
        check(sha(b)==pin['sha256'] and len(b)==pin['bytes'],'UV_or_base_pin_drift')
        pe=parse_pe(b);check(pe['machine']=='0x8664','non_AMD64_selected_binary')
        report['selected_files'][-1]['PE']=pe;binaries[str(p)]=pe
        if p==uv:uv_bytes=b
    expected=proposal['proposed_prerequisite_static_proof']['expected_paths']
    source_text={}
    for raw in expected:
        p=Path(raw)
        if p in [Path(x['path']) for x in bases] or p.parent==system:continue
        exists=p.exists();report['presence_checks'].append({'path':str(p),'exists':exists})
        if not exists:continue
        if p.suffix.lower() in ('.exe','.dll','.pyd'):
            b=read_selected(p);report['selected_files'][-1]['PE']=parse_pe(b)
        else:
            b=read_selected(p,True);text=b.decode('utf-8-sig');source_text[str(p)]=text
            relevant=[{'line':i,'text':line[:500]} for i,line in enumerate(text.splitlines(),1)
                      if any(x in line for x in ['sitecustomize','usercustomize','ENABLE_USER_SITE','exec(','.pth','sys.path','include-system-site-packages','sys.prefix','sys.base_prefix','venvlauncher','symlink','SETUPTOOLS_USE_DISTUTILS'])]
            report['startup_findings'].append({'path':str(p),'relevant_lines':relevant[:120]})
    site=base/'Lib/site-packages'
    hooks=sorted(str(p) for p in site.iterdir() if p.name.endswith('.pth') or p.name in {'sitecustomize.py','usercustomize.py'})
    report['base_startup_direct_inventory']={'directory':str(site),'hooks':hooks,'recursive':False}
    allowed={str(base/'Lib/site-packages/distutils-precedence.pth')}
    check(set(hooks)<=allowed,'unexpected_base_startup_hook')
    for hook in hooks:
        text=source_text.get(hook)
        check(text is not None,'startup_hook_not_in_finite_source_selection')
        if Path(hook).name=='distutils-precedence.pth':
            check("SETUPTOOLS_USE_DISTUTILS" in text and "'local'" in text,'unhandled_distutils_hook_condition')
            report['startup_findings'].append({'path':hook,'SETUPTOOLS_USE_DISTUTILS_stdlib':'conditional static interpretation only; exact source requires independent review'})
    sysfiles={}
    for raw in expected:
        p=Path(raw)
        if p.parent!=system:continue
        path_check(p)
        b=read_selected(p);pe=parse_pe(b);check(pe['machine']=='0x8664','non_AMD64_System32_candidate')
        report['selected_files'][-1]['PE']=pe;sysfiles[p.name.lower()]={'path':str(p),'sha256':sha(b),'PE':pe}
        if p.name.lower()=='apisetschema.dll':schema=b
    mapping,schema_info=api_map(schema);report['API_set_schema']=schema_info
    direct={}
    for source,pe in binaries.items():
        for name in pe['imports']+pe['delay_imports']:direct.setdefault(name.lower(),[]).append(source)
    for name,origins in sorted(direct.items()):
        cost['import_API_set_rows']+=1;check(cost['import_API_set_rows']<=64,'import_API_set_disposition_cap')
        row={'import_name':name,'selected_origins':origins,'recursive_expansion':False}
        if name.startswith(('api-','ext-')):
            values=mapping.get(name.removesuffix('.dll'))
            check(values is not None,'unresolved_selected_API_set:'+name)
            hosts=sorted(set(x['host'].lower() for x in values if not x['alias']))
            check(bool(hosts) and all(h in sysfiles for h in hosts),'API_set_host_outside_selected_literal_system_paths:'+name)
            row.update(disposition='EXACT_INSTALLED_API_SET_SCHEMA_DEFAULT_HOST',schema_values=values,
                       hosts=[{'path':sysfiles[h]['path'],'sha256':sysfiles[h]['sha256']} for h in hosts])
        elif name in {'python311.dll','vcruntime140.dll','vcruntime140_1.dll'}:
            pin=next((x for x in bases if Path(x['path']).name.lower()==name),None)
            check(pin is not None,'missing_selected_base_import_pin')
            row.update(disposition='EXACT_SELECTED_BASE_ROOT',path=pin['path'],sha256=pin['sha256'])
        else:
            check(name in sysfiles,'direct_named_import_outside_selected_literal_system_paths:'+name)
            row.update(disposition='EXACT_LITERAL_SYSTEM32_FILE',path=sysfiles[name]['path'],sha256=sysfiles[name]['sha256'])
        report['import_API_set_dispositions'].append(row)
    # UV executable-directory shadow candidates: name checks only, no widening
    # of the selected file set or DLL recursion.
    shadows=[]
    for name in direct:
        p=uv.parent/name
        if p.exists():shadows.append(str(p))
    report['UV_directory_direct_import_shadow_candidates']=shadows
    check(not shadows,'unexpected_UV_executable_directory_DLL_shadow')
    tokens=['no-config','offline','cache-dir','venv','python','no-python-downloads','no-seed','no-project','target','no-index','find-links','require-hashes','no-deps','only-binary','no-build','no-compile-bytecode','link-mode','requirement','compile-bytecode','UV_COMPILE_BYTECODE']
    report['UV_flag_literal_evidence']=[{'token':t,'present':t.encode() in uv_bytes,'not_semantic_verification':True} for t in tokens]
    snippets=[]
    for needle in [b'import sys\n',b'import os\n',b'import json\n',b'sysconfig.get_config_var(',b'sysconfig.get_paths(',b'sys.flags.isolated',b'python_version_tuple']:
        offsets=[m.start() for m in re.finditer(re.escape(needle),uv_bytes)][:4]
        for offset in offsets:
            snippet=uv_bytes[max(0,offset-200):min(len(uv_bytes),offset+1400)]
            snippets.append({'anchor':needle.decode(),'offset':offset,'selected_context':snippet.decode('utf-8','backslashreplace')})
    report['UV_embedded_query_evidence']={'selected_anchor_contexts':snippets,
                                        'entire_query_and_actual_argument_construction_bound':False,
                                        'context_is_binary_source_data_not_executed':True}
    report['unknowns']=[
        'Selected UV embedded text/flag literals do not establish the complete actual interpreter-query argv, isolated/site behavior or exact proposed CLI flag semantics.',
        'UV may create its own Windows launcher and _virtualenv startup helpers; their exact generated bytes/placement cannot be inferred from selected base launcher source alone.',
        'The --target install scheme and no-compile-bytecode behavior require independent tool/source disposition before the one setup, without trying commands until one works.',
        'Selected direct System32/API-set rows are not a recursive Windows/internal-C proof. Target helper/framework paths and actual loaded module evidence remain pre-bootstrap prerequisites.'
    ]
    report['next_prerequisite']='Independent SCI/OPS review of this frozen selected proof and exact command design; resolve UV query/CLI/generated-launcher semantics through separately scoped hash-bound official tool source/document evidence before setup source/registration. No installer/help/venv/install invocation follows automatically.'
    report['status']='NOT_READY_SELECTED_SETUP_PATHS_BOUND_BUT_INSTALLER_STARTUP_COMMAND_SEMANTICS_UNRESOLVED'

try:main()
except Exception as e:
    report['status']='NOT_READY_STATIC_PROOF_STOPPED_NO_EXPANSION'
    report['stop_reason']=type(e).__name__+': '+str(e)
    report['traceback']=traceback.format_exc()
    report['next_prerequisite']='Independent review of exact frozen failure/partial proof; any unresolved route needs separately scoped evidence disposition. No automatic expansion, retry or setup.'
finally:
    cost['elapsed_seconds']=time.monotonic()-START
    check(not(OUT/'RESULT.json').exists(),'refuse_overwrite_result')
    (OUT/'RESULT.json').write_text(json.dumps(report,indent=2,ensure_ascii=True)+'\n',encoding='utf-8',newline='\n')
    manifest=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashfile(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT/'FREEZE.json').write_text(json.dumps({'status':'FROZEN_NOT_READY_STOP_FOR_INDEPENDENT_REVIEW_NO_SETUP',
        'scope_sha256':SCOPE_SHA,'files':manifest,'result_status':report['status'],'stop_reason':report.get('stop_reason'),
        'cost':cost,'prohibited_actions':report['prohibited_actions']},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'stop_reason':report.get('stop_reason'),'cost':cost,'freeze':str(OUT/'FREEZE.json')}))
