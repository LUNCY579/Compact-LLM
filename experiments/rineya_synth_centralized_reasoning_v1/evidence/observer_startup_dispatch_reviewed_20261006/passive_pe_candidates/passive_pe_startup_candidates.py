"""Bounded passive PE imports: prospective candidates, never loader closure."""
from pathlib import Path
import argparse, hashlib, json, os, stat, struct, sys, time

HERE = Path(__file__).resolve().parent
BASE = Path(r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none')
SYSTEM = Path(r'C:\Windows\System32')
SELECTION = HERE.parent/'observer_only_inventory_work/ROOT_SELECTION_001.json'
SELECTION_FILE_SHA = '417430b7712664e80a733c3e78929b629f79cee91f578c7dc7951eac572e66fa'
SELECTION_BODY_SHA = '0d4a5cb1ebd4030f5f105a809068ab98d74acc27675155079b4588d2227de703'
SEEDS = [BASE/n for n in ('python.exe','python311.dll','python3.dll')]+[BASE/'DLLs'/n for n in ('_ctypes.pyd','_hashlib.pyd','_socket.pyd','select.pyd','_ssl.pyd')]+[SYSTEM/n for n in ('ntdll.dll','kernel32.dll','psapi.dll')]
LIMITS = {'files':128,'total_bytes':512*1024**2,'file_bytes':64*1024**2,'seconds':600,'chunk_bytes':1024**2}

def require(ok, text):
 if not ok: raise ValueError(text)
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def seal(v): return {**v,'sha256':hashlib.sha256(canonical(v).encode('ascii')).hexdigest()}
def error(e): return {'type':type(e).__name__,'message':str(e)[:4096]}
def fingerprint(m): return (stat.S_IFMT(m.st_mode),m.st_size,m.st_mtime_ns,m.st_ino)
def regular(path):
 p=Path(path).absolute();require(p.resolve()==p,'unqualified pathname alias')
 for part in (p,*p.parents):
  m=part.lstat();require(not stat.S_ISLNK(m.st_mode) and not (getattr(m,'st_file_attributes',0)&0x400),'reparse/alias refused')
 m=p.lstat();require(stat.S_ISREG(m.st_mode),'regular selected file');return p,m

def pe_imports(raw):
 def span(offset,length):
  require(type(offset) is int and 0<=offset and 0<=length and offset+length<=len(raw),'PE field outside actual file')
  return raw[offset:offset+length]
 def u16(n): return struct.unpack('<H',span(n,2))[0]
 def u32(n): return struct.unpack('<I',span(n,4))[0]
 require(span(0,2)==b'MZ','DOS signature');pe=u32(0x3c)
 require(span(pe,4)==b'PE\0\0','PE signature');count=u16(pe+6);optional_size=u16(pe+20)
 require(0<count<=96,'bounded PE sections');opt=pe+24;magic=u16(opt)
 require(magic in (0x10b,0x20b),'PE32/PE32+ optional header')
 table=opt+(112 if magic==0x20b else 96);number=u32(opt+(108 if magic==0x20b else 92))
 require(table+min(number,16)*8<=opt+optional_size,'bounded complete PE directories')
 imagebase=struct.unpack('<Q',span(opt+24,8))[0] if magic==0x20b else u32(opt+28)
 headers=u32(opt+60);sections=[]
 for i in range(count):
  n=opt+optional_size+40*i;span(n,40)
  sections.append((u32(n+12),u32(n+8),u32(n+16),u32(n+20)))
 def address(rva,length=1):
  if rva<headers:span(rva,length);return rva
  matches=[s for s in sections if s[0]<=rva<s[0]+max(s[1],s[2])]
  require(len(matches)==1,'exact section for PE RVA');va,virtual,size,start=matches[0];delta=rva-va
  require(delta+length<=size,'PE RVA requires actual initialized bytes');span(start+delta,length);return start+delta
 def name(rva):
  out=bytearray()
  for i in range(260):
   b=raw[address(rva+i)]
   if b==0:
    require(out and all(32<=v<127 for v in out),'bounded plain ASCII import name');return out.decode('ascii')
   out.append(b)
  raise ValueError('unterminated or oversized PE import name')
 result=[]
 for index,width,kind in ((1,20,'import'),(13,32,'delay_import')):
  if index>=number:continue
  directory_rva=u32(table+index*8);directory_size=u32(table+index*8+4)
  if not directory_rva and not directory_size:continue
  require(directory_rva and width<=directory_size<=4096*width,'bounded actual import directory')
  terminated=False
  for i in range(min(directory_size//width,4096)):
   offset=address(directory_rva+i*width,width);values=struct.unpack('<'+'I'*(width//4),span(offset,width))
   if not any(values):terminated=True;break
   if kind=='import':rva=values[3]
   else:
    require(values[0] in (0,1),'known delay-import attribute');rva=values[1] if values[0]&1 else values[1]-imagebase
   result.append({'kind':kind,'name':name(rva)})
  require(terminated,'import descriptor terminator within declared directory')
 return {'machine':u16(pe+4),'optional_magic':magic,'imports':result}

def main():
 selection_raw=SELECTION.read_bytes();require(hashlib.sha256(selection_raw).hexdigest()==SELECTION_FILE_SHA,'external exact selection bytes')
 selection=json.loads(selection_raw);require(selection['sha256']==SELECTION_BODY_SHA==hashlib.sha256(canonical({k:v for k,v in selection.items() if k!='sha256'}).encode('ascii')).hexdigest(),'external selection body')
 original_maps=dict(selection['metadata_map_digests']);known=selection['files']
 by_name={}
 for text,d in known.items():
  p=Path(text)
  if p.is_relative_to(BASE) and p.suffix.lower() in {'.exe','.dll','.pyd'}:by_name.setdefault(p.name.casefold(),[]).append(str(p))
 started=time.monotonic();allowed=set();forbidden=[]
 def audit(event,args):
  if event in ('ctypes.dlopen','ctypes.dlsym','subprocess.Popen','os.system'):forbidden.append(event);raise AssertionError('native/process call prohibited')
  if event=='import' and args[0].split('.',1)[0] in {'ctypes','torch','psutil','native_process','native_nvml','native_worker','transformers','safetensors'}:
   forbidden.append('import:'+args[0]);raise AssertionError('native/framework import prohibited')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).suffix.lower() in {'.exe','.pyd','.dll','.drv'}:
   p=str(Path(os.fsdecode(args[0])).absolute()).casefold()
   if p not in allowed:forbidden.append('read:'+p);raise AssertionError('not an explicit selected/import-derived file')
 sys.addaudithook(audit)
 records=[];edges=[];queue=[(p,'explicit_seed') for p in SEEDS];visited=set();total=0;current=None;errors=[];fingerprints={}
 def elapsed():require(time.monotonic()-started<=LIMITS['seconds'],'whole passive deadline')
 try:
  while queue:
   elapsed();path,reason=queue.pop(0);key=str(path).casefold()
   if key in visited:continue
   require(len(visited)<LIMITS['files'],'finite import-derived file budget');visited.add(key)
   current={'path':str(path),'reason':reason,'bytes_consumed':0,'hash_complete':False,'opened_handle_verified':False}
   try:
    path,before=regular(path);require(0<before.st_size<=LIMITS['file_bytes'] and total+before.st_size<=LIMITS['total_bytes'],'passive byte budgets')
    current['declared_bytes']=before.st_size;allowed.add(str(path).casefold());chunks=[];hasher=hashlib.sha256();remaining=before.st_size
    with path.open('rb') as f:
     require(fingerprint(os.fstat(f.fileno()))==fingerprint(before),'opened handle equals selected pathname identity');current['opened_handle_verified']=True
     while remaining:
      elapsed();block=f.read(min(LIMITS['chunk_bytes'],remaining));require(block,'selected PE early EOF')
      chunks.append(block);hasher.update(block);remaining-=len(block);current['bytes_consumed']+=len(block)
     elapsed();require(f.read(1)==b'','selected PE grew after declared bytes');require(fingerprint(os.fstat(f.fileno()))==fingerprint(before),'opened PE fingerprint changed')
    require(fingerprint(path.lstat())==fingerprint(before),'pathname PE fingerprint changed')
    raw=b''.join(chunks);actual=hasher.hexdigest();current.update(hash_complete=True,sha256=actual);total+=len(raw);fingerprints[str(path)]=fingerprint(before)
    expected=known.get(str(path),{}).get('sha256');current['expected_sha256']=expected
    require(expected is None or actual==expected,'original known pin differs')
    parsed=pe_imports(raw);current.update(parsed)
    for item in parsed['imports']:
     name=item['name'];lower=name.casefold();edge={'from':str(path),**item,'candidate_path':None,'status':None}
     if lower.startswith(('api-ms-','ext-ms-')):edge['status']='API_SET_RESOLUTION_UNOBSERVED'
     elif Path(name).name!=name or not lower.endswith('.dll'):edge['status']='NON_DLL_OR_PATH_NAME_UNADMITTED'
     elif len(by_name.get(lower,[]))>1:edge.update(status='AMBIGUOUS_KNOWN_ALIAS_UNOBSERVED',candidates=by_name[lower])
     else:
      candidate=Path(by_name[lower][0]) if lower in by_name else SYSTEM/name
      edge.update(candidate_path=str(candidate),status='PROSPECTIVE_LITERAL_PATH')
      # Existence/physical identity is a candidate observation, not loader resolution.
      try:regular(candidate)
      except (OSError,ValueError) as failure:edge.update(status='MISSING_OR_ALIAS_UNOBSERVED',error=error(failure))
      else:queue.append((candidate,'literal_'+item['kind']))
     edges.append(edge)
    current['status']='PASSIVE_PE_PARSED'
   except (OSError,ValueError,AssertionError) as failure:
    current.update(status='PASSIVE_FILE_FAILED',error=error(failure));errors.append({'path':str(path),**error(failure)})
   records.append(current);current=None
  for text,before in fingerprints.items():elapsed();require(fingerprint(regular(text)[1])==before,'final selected pathname fingerprint drift')
 except BaseException as failure:errors.append({'stage':'whole_scope','current_read':current,**error(failure)})
 windows={r['path']:{'bytes':r['declared_bytes'],'sha256':r['sha256'],'role':'windows_system_binary'} for r in records if r.get('hash_complete') and r.get('status')=='PASSIVE_PE_PARSED' and Path(r['path']).parent==SYSTEM}
 value={'kind':'PASSIVE_PE_STARTUP_DEPENDENCY_CANDIDATES_V1','limits':LIMITS,'explicit_seeds':[str(p) for p in SEEDS],'files':records,'edges':edges,'windows_candidate_files':windows,'original_metadata_map_digests':original_maps,'known65_779_changed':False,'files_attempted':len(visited),'bytes_hashed':total,'seconds':time.monotonic()-started,'errors':errors,'pending_literal_paths':[str(p) for p,_ in queue],'forbidden_actions':forbidden,'actual_loader_resolution_observed':False,'API_set_resolution_observed':False,'complete_Windows_closure':False,'native_launch_authorized':False,'native_PASS':False,'native_authority_issued':False}
 value=seal(value);stamp=time.time_ns();path=HERE/('PASSIVE_PE_STARTUP_CANDIDATES_'+str(stamp)+'.json');raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode('ascii')
 with path.open('xb') as f:require(f.write(raw)==len(raw),'complete exclusive receipt');f.flush();os.fsync(f.fileno())
 require(path.read_bytes()==raw,'receipt exact readback')
 print(json.dumps({'receipt':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256'],'files_attempted':len(visited),'bytes_hashed':total,'windows_candidates':len(windows),'errors':errors,'unobserved_edges':sum(e['status']!='PROSPECTIVE_LITERAL_PATH' for e in edges),'native_PASS':False}))
 return bool(errors or forbidden)

if __name__=='__main__':raise SystemExit(main())
