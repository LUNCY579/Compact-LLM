"""Complete the already-authorized one wheel batch from frozen exact metadata.

Preserves STATIC_INSPECTION_AND_CLOSURE.json and its FREEZE.json. No metadata
HTTP repetition, version search, installation, extraction, or target execution.
"""
import calendar
import copy
import csv
import email.parser
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
prior=json.loads((HERE/'STATIC_INSPECTION_AND_CLOSURE.json').read_text())
freeze=json.loads((HERE/'FREEZE.json').read_text())
helper=HERE/'inspect_static.py'
helper_hash=hashlib.sha256(helper.read_bytes()).hexdigest()
assert helper_hash==next(x['sha256'] for x in freeze['files'] if x['path']=='inspect_static.py')
# Execute this hash-bound, locally authored data-only inspection tool's
# definitions. Its terminal invocation block is excluded. No wheel source is
# imported, evaluated, compiled, installed, or extracted.
ns={'__file__':str(helper),'__name__':'hash_bound_static_inspection_tool'}
exec(compile(helper.read_text().split('\ntry:\n    main()\n')[0],str(helper),'exec'),ns)
check=ns['check']; digest=ns['digest']; hashfile=ns['hashfile']; write=ns['write']
selected=ns['selected']; validate_zip=ns['validate_zip']; parse_pe=ns['parse_pe']; safe_name=ns['safe_name']
Stop=ns['Stop']; ROOT=ns['ROOT']; PKG=ns['PKG']; ART=ns['ART']; BASE=ns['BASE']; TOOL=ns['TOOL']
batch_begin=time.monotonic()
stage_start=calendar.timegm(time.strptime(prior['started_utc'],'%Y-%m-%dT%H:%M:%SZ'))
stage_elapsed_before_batch=max(0,time.time()-stage_start)
ns['BEGIN']=batch_begin-stage_elapsed_before_batch
cost=copy.deepcopy(prior['cost']); cost.pop('elapsed_seconds',None)
# Count the five interpreter/base-runtime PE files in the aggregate selection
# budget as well as the ten Torch wheel binaries (the checkpoint kept them in
# its separate interpreter inventory).
cost['selected_PE_binaries']+=len(prior['interpreter'])
cost['selected_PE_bytes']+=sum(x['bytes'] for x in prior['interpreter'])
ns['cost']=cost
report={'status':'IN_PROGRESS_NO_INSTALL', 'prior_checkpoint_sha256':hashfile(HERE/'FREEZE.json'),
        'scope_sha256':ns['SCOPE_SHA'],'metadata_amendment_sha256':ns['AMENDMENT_SHA'],
        'helper_sha256':helper_hash,'acquisition_script_sha256':hashfile(Path(__file__)),
        'stage_original_started_utc':prior['started_utc'],'stage_elapsed_before_batch':stage_elapsed_before_batch,
        'cost':cost,'events':[],'wheels':[prior['wheels'][0]],'dependencies':[],
        'base_interpreter_binding':prior['interpreter'],'inspection_tool_binding':prior['tool'],
        'marker_environment':prior['marker_environment'],'new_source_build_evidence':[],
        'no_install_venv_extraction_target_import_models_GPU_process_DB_actions':True}

def main():
    check(hashfile(PKG/'authorization/DELEGATED_CPU_ONLY_RUNTIME_LOCK_INSPECTION_003.json')==ns['SCOPE_SHA'],'scope_drift')
    check(hashfile(PKG/'authorization/DELEGATED_CPU_ONLY_RUNTIME_METADATA_SELECTION_004.json')==ns['AMENDMENT_SHA'],'amendment_drift')
    for entry in prior['tool']['packaging_files']:
        check(hashfile(Path(entry['path']))==entry['sha256'],'packaging_tool_drift')
    for entry in prior['interpreter']:
        check(hashfile(Path(entry['path']))==entry['sha256'],'base_runtime_drift')
    sys.path.insert(0,str(TOOL.parent))
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import cpython_tags,compatible_tags
    from packaging.utils import canonicalize_name,parse_wheel_filename
    versions=prior['baseline_versions']; env=prior['marker_environment']
    tags=list(cpython_tags(python_version=(3,11),platforms=['win_amd64']))+list(compatible_tags(python_version=(3,11),interpreter='cp311',platforms=['win_amd64']))
    rank={t:i for i,t in enumerate(tags)}
    planned=[]
    for name in prior['resolved_closure']:
        if name=='torch':continue
        json_files=[p for p in HERE.glob('PYPI_*.json') if canonicalize_name(json.loads(p.read_text())['info']['name'])==name]
        check(len(json_files)==1,'missing_or_ambiguous_frozen_metadata:'+name)
        source=json_files[0]; value=json.loads(source.read_text())
        check(hashfile(source)==next(x['sha256'] for x in freeze['files'] if x['path']==source.name),'frozen_metadata_drift:'+name)
        candidates=[]
        for f in value['urls']:
            if f['packagetype']!='bdist_wheel' or f.get('yanked'):continue
            nm,ver,_,ftags=parse_wheel_filename(f['filename'])
            if canonicalize_name(nm)!=name or str(ver)!=versions[name]:continue
            compatible=ftags & rank.keys()
            if not compatible:continue
            if f.get('requires_python') and not SpecifierSet(f['requires_python']).contains('3.11.15'):continue
            check(urllib.parse.urlparse(f['url']).scheme=='https' and urllib.parse.urlparse(f['url']).hostname=='files.pythonhosted.org','unapproved_wheel_origin')
            check(bool(f['digests'].get('sha256')),'missing_publisher_hash')
            candidates.append((min(rank[t] for t in compatible),f['filename'],f))
        check(bool(candidates),'no_compatible_exact_official_wheel:'+name)
        candidates.sort(key=lambda x:(x[0],x[1])); f=candidates[0][2]
        planned.append({'name':name,'version':versions[name],'filename':f['filename'],'url':f['url'],
                        'published_sha256':f['digests']['sha256'],'published_bytes':f['size'],
                        'publisher_metadata':str(source),'publisher_metadata_sha256':hashfile(source),
                        'path':str(ART/f['filename'])})
    check(len(planned)+1<=40,'wheel_total_cap')
    check(sum(x['published_bytes'] for x in planned)+113636478<=805306368,'compressed_batch_cap')
    write('ACQUISITION_PLAN.json',{'one_batch':True,'attempts_per_artifact':1,'no_fallback_no_retry':True,
          'scope_sha256':ns['SCOPE_SHA'],'metadata_amendment_sha256':ns['AMENDMENT_SHA'],
          'prior_checkpoint_sha256':report['prior_checkpoint_sha256'],'script_sha256':report['acquisition_script_sha256'],
          'files':planned,'total_bytes_including_torch':sum(x['published_bytes'] for x in planned)+113636478})
    for item in planned:
        path=Path(item['path']); check(not path.exists(),'refuse_existing_dependency_artifact:'+str(path))
        event={'filename':item['filename'],'URL':item['url'],'attempt':1,'expected_sha256':item['published_sha256'],
               'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'bytes':0}
        report['events'].append(event); t=time.monotonic(); cost['wheel_transfers']+=1
        try:
            with urllib.request.build_opener(ns['OfficialRedirect']()).open(
                 urllib.request.Request(item['url'],headers={'User-Agent':'CRS-001-static-lock/003'}),
                 timeout=min(45,max(1,600-(time.monotonic()-ns['BEGIN'])))) as response:
                check(urllib.parse.urlparse(response.url).hostname=='files.pythonhosted.org','wheel_redirect_origin')
                event.update(status=response.status,final_url=response.url)
                with path.open('xb') as f:
                    while chunk:=response.read(1024*1024):
                        check(event['bytes']+len(chunk)<=item['published_bytes'],'wheel_publisher_size_cap')
                        check(cost['combined_wheel_bytes']+len(chunk)<=805306368,'wheel_aggregate_bytes_cap')
                        f.write(chunk); event['bytes']+=len(chunk)
                        cost['new_wheel_bytes']+=len(chunk); cost['combined_wheel_bytes']+=len(chunk)
            check(path.stat().st_size==item['published_bytes'],'wheel_size_mismatch')
            sha=hashfile(path); event['physical_sha256']=sha
            check(sha==item['published_sha256'],'wheel_hash_mismatch')
            item['physical_sha256']=sha; item['bytes']=path.stat().st_size; event['outcome']='HASH_VERIFIED'
        except Exception as e:
            event.update(outcome='FAILED_OR_PARTIAL_STOP_NO_RETRY',error_type=type(e).__name__,error=str(e))
            raise
        finally:event['elapsed_seconds']=time.monotonic()-t
        with zipfile.ZipFile(path) as z:
            item['ZIP']=validate_zip(z); names=z.namelist()
            metas=[n for n in names if n.endswith('.dist-info/METADATA')]
            check(len(metas)==1,'ambiguous_dist_info:'+item['name'])
            dist=metas[0].rsplit('/',1)[0]+'/'
            meta=selected(z,dist+'METADATA'); wheel=selected(z,dist+'WHEEL'); rec=selected(z,dist+'RECORD')
            m=email.parser.BytesParser().parsebytes(meta); w=email.parser.BytesParser().parsebytes(wheel)
            check(canonicalize_name(m['Name'])==item['name'] and m['Version']==item['version'],'physical_metadata_identity')
            check(not m['Requires-Python'] or SpecifierSet(m['Requires-Python']).contains('3.11.15'),'physical_python_version_conflict')
            filename_tags=parse_wheel_filename(item['filename'])[3]
            physical_tags=w.get_all('Tag',[])
            check(bool(physical_tags) and all(wt in {str(t) for t in filename_tags} for wt in physical_tags)
                  and any(wt in {str(t) for t in tags} for wt in physical_tags),'physical_WHEEL_tag_mismatch')
            rows=list(csv.reader(io.StringIO(rec.decode()))); records={row[0]:row[1:] for row in rows}
            check(len(records)==len(rows),'duplicate_RECORD_path')
            for n in records:safe_name(n)
            check(all(n in records for n in names if not n.endswith('/')),'physical_RECORD_missing_member')
            item['metadata']={'name':m['Name'],'version':m['Version'],'requires_python':m['Requires-Python'],
                              'requires_dist':m.get_all('Requires-Dist',[]),'tags':w.get_all('Tag',[]),
                              'METADATA_sha256':digest(meta),'WHEEL_sha256':digest(wheel),'RECORD_sha256':digest(rec),
                              'RECORD_rows':len(rows),'startup_hook_members':[n for n in names if n.endswith('.pth') or n.rsplit('/',1)[-1] in ('sitecustomize.py','usercustomize.py')]}
            native=[n for n in names if n.endswith(('.pyd','.dll'))]; item['selected_PE']=[]
            selected_members={dist+'METADATA':meta,dist+'WHEEL':wheel}
            for n in native:
                b=selected(z,n,True); pe=parse_pe(b); check(pe['machine']=='0x8664','dependency_PE_non_AMD64')
                item['selected_PE'].append({'member':n,'sha256':digest(b),'bytes':len(b),'PE':pe})
                selected_members[n]=b
            for n,b in selected_members.items():
                encoded=ns['base64'].urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')
                check(records[n]==['sha256='+encoded,str(len(b))],'selected_RECORD_hash_size_mismatch:'+n)
        report['wheels'].append(item)
        for raw in item['metadata']['requires_dist']:
            req=Requirement(raw); active=req.marker is None or req.marker.evaluate(env); dep=canonicalize_name(req.name)
            row={'parent':item['name'],'requires_dist':raw,'active_without_extras':active,'dependency':dep}
            report['dependencies'].append(row)
            if active:
                check(dep in versions and dep in prior['resolved_closure'],'mandatory_wheel_dependency_absent_baseline:'+dep)
                check(not req.url and not req.extras,'active_extra_or_direct_URL_requires_separate_disposition')
                check(req.specifier.contains(versions[dep],prereleases=True),'physical_wheel_baseline_version_conflict:'+raw)
                row['locked_version']=versions[dep]
    report['dependencies'].extend(row for row in prior['dependencies'] if row['parent']=='torch')
    # CMake files are generated build configuration data, inspected without
    # executing _C._show_config or any target package.
    with zipfile.ZipFile(ART/prior['wheels'][0]['filename']) as z:
        for n in ['torch/share/cmake/Caffe2/Caffe2Config.cmake','torch/share/cmake/Torch/TorchConfig.cmake']:
            if n not in z.namelist():continue
            b=selected(z,n); text=b.decode('utf-8'); lines=text.splitlines()
            matches=[]
            for i,line in enumerate(lines):
                if any(k in line.lower() for k in ['cuda','rocm','xpu','hip','mkl','onednn','caffe2_use','if(off)','if(on)']):
                    matches.append({'line':i+1,'context':'\n'.join(lines[max(0,i-1):min(len(lines),i+3)])[:1000]})
            report['new_source_build_evidence'].append({'member':n,'sha256':digest(b),'bytes':len(b),'selected_context':matches})
    lock=['# CPU-only runtime artifact lock; Windows CPython 3.11.15 AMD64; no extras.',
          '# Generated without installation or target package execution. Requires independent artifact/path/ABI review.']
    for item in sorted(report['wheels'],key=lambda x:x['name']):
        lock.append(item['name']+'=='+item['version']+' --hash=sha256:'+(item.get('physical_sha256') or item['sha256']))
    lockpath=HERE/'requirements-cpu-lock.txt'; check(not lockpath.exists(),'lock_overwrite')
    lockpath.write_text('\n'.join(lock)+'\n',encoding='utf-8',newline='\n')
    report['complete_hash_lock']={'path':str(lockpath),'sha256':hashfile(lockpath),'wheels':len(report['wheels']),
                                  'all_non_Torch_versions_unchanged':True,'wheel_requires_dist_markers_checked':True}
    report['native_boundary_limits']=prior['native_proof_limits']
    report['readiness']={'artifact_hash_lock':'COMPLETE_STATIC_REVIEW_READY',
                        'installation':'NOT_AUTHORIZED_BY_THIS_RECEIPT',
                        'actual_CPU_runtime_boundary':'NOT_READY_REQUIRES_INDEPENDENT_PATH_ABI_SOURCE_AND_ACTUAL_LOADED_MODULE_EVIDENCE',
                        'all_C_hardware_zero_guarantee':False,
                        'unbound_system_runtime_imports':'Windows API-set resolution/MSVCP140/system DLL canonical hashes and future exact DLL search roots remain required before setup/bootstrap.'}
    report['status']='COMPLETE_ARTIFACT_HASH_LOCK_STATIC_REVIEW_READY_NO_INSTALL'

try:main()
except Exception as e:
    report['status']='NOT_READY_STOPPED_NO_RETRY'
    report['stop_reason']=type(e).__name__+': '+str(e)
    report['traceback']=ns['traceback'].format_exc()
finally:
    cost['stage_wall_elapsed_seconds']=time.monotonic()-ns['BEGIN']
    cost['batch_active_elapsed_seconds']=time.monotonic()-batch_begin
    cost['cumulative_active_inspection_seconds']=prior['cost']['elapsed_seconds']+cost['batch_active_elapsed_seconds']
    write('ARTIFACT_BATCH_AND_HASH_LOCK.json',report)
    entries=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name!='FINAL_FREEZE.json':entries.append({'path':p.name,'bytes':p.stat().st_size,'sha256':hashfile(p)})
    write('FINAL_FREEZE.json',{'status':'FROZEN_STOP_FOR_INDEPENDENT_ARTIFACT_REVIEW_NO_INSTALL','files':entries,
                              'result_status':report['status'],'stop_reason':report.get('stop_reason'),
                              'scope_sha256':ns['SCOPE_SHA'],'metadata_amendment_sha256':ns['AMENDMENT_SHA'],
                              'cost':cost,'not_native_model_or_bootstrap_result':True})
    print(json.dumps({'status':report['status'],'stop_reason':report.get('stop_reason'),'cost':cost,'freeze':str(HERE/'FINAL_FREEZE.json')}))
