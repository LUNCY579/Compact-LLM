"""Low-priority streaming preservation. Does not remove or change source checkouts."""
import ctypes,datetime,hashlib,io,json,re,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath

ROOT=Path(r'C:\Users\clclc\BusinessProject').resolve()
OUT=ROOT/'Rineya-Worktree-Audit-20261009'
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
CACHE=re.compile(r'(^|/)(__pycache__|\.pytest_cache)(/|$)|\.py[co]$')
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b)')

def git(p,*args,binary=False):
 r=subprocess.run(['git','-c','core.longpaths=true','-C',str(p),*args],capture_output=True,timeout=120)
 if r.returncode or r.stderr:raise RuntimeError(r.stderr.decode('utf-8','replace'))
 return r.stdout if binary else r.stdout.decode('utf-8','replace').strip()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def record(root,name,category):
 p=root/PurePosixPath(name);q=p.resolve(strict=True)
 if not q.is_relative_to(root) or not p.is_file() or p.is_symlink() or p.lstat().st_file_attributes&0x400:raise RuntimeError('Unsafe preservation member: '+name)
 if p.suffix.lower() in {'.env','.pem','.key','.p12','.pfx','.sqlite','.db'}:raise RuntimeError('Sensitive file type requires separate reviewed custody: '+name)
 h=hashlib.sha256();tail=b''
 with p.open('rb') as f:
  while b:=f.read(1024*1024):
   h.update(b)
   if SECRET.search(tail+b):raise RuntimeError('Credential-like content; preserve locally and keep checkout: '+name)
   tail=b[-256:]
 return {'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest(),'category':category}

def archive(spec):
 name=spec['name'];root=(ROOT/name).resolve(strict=True)
 if not root.is_relative_to(ROOT) or root.parent!=ROOT:raise RuntimeError('Unexpected checkout path')
 head=git(root,'rev-parse','HEAD')
 if head!=spec['head']:raise RuntimeError('Source HEAD changed')
 untracked=[x for x in git(root,'ls-files','--others','--exclude-standard','-z',binary=True).decode().split('\0') if x]
 ignored=[x for x in git(root,'ls-files','--others','--ignored','--exclude-standard','-z',binary=True).decode().split('\0') if x]
 changed=[x for x in git(root,'diff','--name-only','HEAD','-z',binary=True).decode().split('\0') if x]
 category={n:'untracked source/evidence' for n in untracked}
 category.update({n:'ignored scientific evidence' for n in ignored if not CACHE.search(n)})
 deleted=[]
 for n in changed:
  if (root/n).exists():category[n]='modified tracked file'
  else:deleted.append(n)
 records=[record(root,n,c) for n,c in sorted(category.items())]
 patch=git(root,'diff','--binary','HEAD',binary=True)
 state={'schema':'rineya.retired-checkout-preservation/2','source_root':str(root),'branch':git(root,'symbolic-ref','--short','HEAD'),'source_head':head,'remote_source_tag':spec['tag'],'archived_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retirement':'Remove dormant checkout after exact remote source tag and downloaded artifact verification; retain study, open questions, original branch and negative evidence.','restoration':'Add source worktree at original path from source_head/tag. Extract archive at checkout root; material files restore exact bytes. Apply recorded deleted_tracked_files only if restoring deleted working-tree paths. HEAD.patch records complete tracked difference.','status_porcelain':git(root,'status','--porcelain=v1','--untracked-files=all'),'diff_name_status':git(root,'diff','--name-status','HEAD'),'deleted_tracked_files':deleted,'files':records,'material_bytes':sum(f['bytes'] for f in records),'excluded_disposable_cache_files':[n for n in ignored if CACHE.search(n)],'patch_sha256':hashlib.sha256(patch).hexdigest(),'hardware_policy':{'priority':'BelowNormal','compression_level':1,'workers':1,'stream_buffer_bytes':1048576,'model_or_gpu_imports':False}}
 stem='phase2-'+spec['slug'];archive_path=OUT/(stem+'.tar.gz');manifest_path=OUT/(stem+'-manifest.json')
 encoded=(json.dumps(state,indent=2)+'\n').encode();manifest_path.write_bytes(encoded)
 with tarfile.open(archive_path,'w:gz',compresslevel=1) as tar:
  for f in records:tar.add(root/f['path'],arcname=f['path'],recursive=False)
  for member_name,content in [('_WORKTREE_STATE.json',encoded),('_HEAD.patch',patch)]:
   info=tarfile.TarInfo(member_name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
 expected={f['path']:f for f in records};seen=set()
 with tarfile.open(archive_path,'r:gz') as tar:
  for member in tar:
   if member.name in ('_WORKTREE_STATE.json','_HEAD.patch'):continue
   f=expected.get(member.name)
   if not f or not member.isfile() or member.size!=f['bytes']:raise RuntimeError('Archive member mismatch')
   h=hashlib.sha256();stream=tar.extractfile(member)
   while b:=stream.read(1024*1024):h.update(b)
   if h.hexdigest()!=f['sha256']:raise RuntimeError('Archive bytes mismatch')
   seen.add(member.name)
 if seen!=set(expected):raise RuntimeError('Archive member inventory mismatch')
 if git(root,'rev-parse','HEAD')!=head or git(root,'status','--porcelain=v1','--untracked-files=all')!=state['status_porcelain'] or git(root,'diff','--binary','HEAD',binary=True)!=patch:raise RuntimeError('Source changed during archive')
 for f in records:
  if digest(root/f['path'])!=f['sha256']:raise RuntimeError('Material file changed during archive')
 receipt={'status':'VERIFIED_LOCAL_STREAMING_ARCHIVE','checkout':name,'source_head':head,'source_tag':spec['tag'],'archive':str(archive_path),'archive_bytes':archive_path.stat().st_size,'archive_sha256':digest(archive_path),'manifest':str(manifest_path),'manifest_sha256':digest(manifest_path),'material_files':len(records),'material_bytes':state['material_bytes'],'tracked_deletions':deleted,'discardable_cache_count':len(state['excluded_disposable_cache_files']),'source_unchanged':True,'roundtrip_verified_files':len(seen)}
 (OUT/(stem+'-verification.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 (OUT/(stem+'-SHA256SUMS.txt')).write_text(receipt['archive_sha256']+'  '+archive_path.name+'\n'+receipt['manifest_sha256']+'  '+manifest_path.name+'\n',encoding='utf-8')
 print(name,len(records),'material files',receipt['archive_bytes'],'archive bytes',flush=True)
 return receipt

if __name__=='__main__':
 specs=json.loads((OUT/'phase2-approved-checkouts.json').read_text())
 results=[]
 for spec in specs:
  results.append(archive(spec));(OUT/'phase2-archive-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
