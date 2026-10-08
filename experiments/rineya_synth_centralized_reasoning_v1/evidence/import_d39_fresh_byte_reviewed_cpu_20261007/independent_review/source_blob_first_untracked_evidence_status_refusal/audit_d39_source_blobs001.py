"""Exact selected source disk/Git-blob comparison, bounded batch read only."""
from pathlib import Path
import hashlib,json,subprocess,time
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1');ROOT=PACKAGE.parents[1];SELECTION=PACKAGE/'artifacts/import_exact_registration_work/ACTUAL_SOURCE_RUNTIME_SELECTION002.json';EXPECTED='d39d64b1447c8f0baffdebcc4a0a1092f76bbd88';started=time.monotonic()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def git(*args,input=None):
    assert time.monotonic()-started<600;return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=ROOT,input=input,capture_output=True,check=True,timeout=max(.001,600-(time.monotonic()-started))).stdout
raw=SELECTION.read_bytes();selection=json.loads(raw);assert selection['sha256']==digest({k:v for k,v in selection.items() if k!='sha256'}) and selection['implementation_commit']==EXPECTED;sources=selection['source_files'];assert len(sources)==196;assert sum(d['bytes'] for d in sources.values())<=16*1024**3;head_before=git('rev-parse','HEAD').decode().strip();status_before=git('status','--porcelain=v1','-z');assert head_before==EXPECTED and not status_before
config={}
for name in ('core.autocrlf','core.eol','core.safecrlf'):
    result=subprocess.run(['git','config','--get',name],cwd=ROOT,capture_output=True,timeout=30);assert result.returncode in (0,1);config[name]=result.stdout.decode().strip() if result.returncode==0 else None
paths=sorted(sources);requests=[];disk={};fingerprints={};disk_total=0
for text in paths:
    p=Path(text);d=sources[text];assert p.is_relative_to(ROOT) and p.resolve()==p and 0<=d['bytes']<=16*1024**2;before=p.stat()
    with p.open('rb') as stream:opened=__import__('os').fstat(stream.fileno());b=stream.read(d['bytes']+1);after_open=__import__('os').fstat(stream.fileno())
    after=p.stat();fp=lambda m:(m.st_size,m.st_mtime_ns,m.st_ino);assert fp(before)==fp(opened)==fp(after_open)==fp(after) and len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256'];disk[text]=b;fingerprints[text]=fp(after);disk_total+=len(b);requests.append(EXPECTED+':'+p.relative_to(ROOT).as_posix())
output=git('cat-file','--batch',input=('\n'.join(requests)+'\n').encode('utf-8'));assert len(output)<=16*1024**3;offset=0;rows=[];mismatches=[];blob_total=0
for text in paths:
    end=output.index(b'\n',offset);header=output[offset:end].decode('ascii');offset=end+1;parts=header.split(' ');assert len(parts)==3 and parts[1]=='blob' and parts[2].isdigit();size=int(parts[2]);assert size<=16*1024**2;blob=output[offset:offset+size];offset+=size;assert output[offset:offset+1]==b'\n';offset+=1;blob_total+=len(blob);same=blob==disk[text];row={'path':text,'git_object_id':parts[0],'disk_bytes':len(disk[text]),'blob_bytes':len(blob),'disk_sha256':hashlib.sha256(disk[text]).hexdigest(),'blob_sha256':hashlib.sha256(blob).hexdigest(),'exact_bytes_equal':same};rows.append(row)
    if not same:mismatches.append(row)
assert offset==len(output);assert disk_total+blob_total<=16*1024**3
attributes=git('check-attr','-z','text','eol','filter','working-tree-encoding','--',*(str(Path(p).relative_to(ROOT)).replace('\\','/') for p in paths));assert len(attributes)<=16*1024**2;triplets=attributes.split(b'\0');assert triplets[-1]==b'';triplets.pop();assert len(triplets)%3==0;attr={}
for n in range(0,len(triplets),3):p,key,value=(x.decode('utf-8') for x in triplets[n:n+3]);attr.setdefault(p,{})[key]=value
for text in paths:
    p=Path(text);m=p.stat();assert (m.st_size,m.st_mtime_ns,m.st_ino)==fingerprints[text];assert p.read_bytes()==disk[text]
head_after=git('rev-parse','HEAD').decode().strip();status_after=git('status','--porcelain=v1','-z');assert head_after==head_before and status_after==status_before
result={'kind':'INDEPENDENT_ACTUAL_D39_SOURCE_DISK_GIT_BLOB_AUDIT001','implementation_commit':EXPECTED,'selection_file_sha256':hashlib.sha256(raw).hexdigest(),'selection_body_sha256':selection['sha256'],'source_inventory_sha256':digest(sources),'selected_source_count':len(paths),'disk_bytes_read':disk_total,'immutable_blob_bytes_read':blob_total,'exact_source_blob_matches':len(paths)-len(mismatches),'mismatches':mismatches,'rows':rows,'actual_read_only_Git_config':config,'actual_selected_attributes':attr,'HEAD_before':head_before,'HEAD_after':head_after,'Git_status_clean_before_after':True,'seconds':time.monotonic()-started,'deadline_seconds':600,'combined_byte_ceiling':16*1024**3,'actual_source_bytes_stable_before_after':True,'Git_config_refs_or_endings_changed':False,'configs_runtime_registration_dispatch_join_checked':False,'new_staged_fix_sources_covered':False,'native_API_framework_or_process_launched':False,'native_authority_issued':False};result['sha256']=digest(result)
with (HERE/'INDEPENDENT_D39_SOURCE_BLOBS001.json').open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(canonical({'sources':len(paths),'exact_matches':len(paths)-len(mismatches),'mismatches':len(mismatches),'disk_bytes':disk_total,'blob_bytes':blob_total,'config':config,'body_sha256':result['sha256']}))
