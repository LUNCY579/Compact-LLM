import hashlib,json,pathlib,subprocess
R='C:/Users/clclc/BusinessProject/SupaExperiment'
O=pathlib.Path('C:/Users/clclc/BusinessProject/model-quality-audit-20260915')
def git(*a):return subprocess.check_output(['git','-C',R,*a],stderr=subprocess.DEVNULL)
ref='origin/research/supa-a1.1-nonrecurrent-adaptation-v1'
base='experiments/nonrecurrent_adaptation_v1/evidence/'
out=[]
for d in ['2026-09-10-capability1','2026-09-13-transfer3','2026-09-13-symmetric-adoption','2026-09-14-floor1']:
    path=base+d+'/SHA256SUMS'
    for line in git('show',f'{ref}:{path}').decode().splitlines():
        if not line.strip():continue
        expected,name=line.split(maxsplit=1)
        name=name.removeprefix('*').removeprefix('./')
        target=base+d+'/'+name
        try:
            raw=git('show',f'{ref}:{target}')
            actual=hashlib.sha256(raw).hexdigest()
            out.append(dict(path=target,expected=expected,actual=actual,match=expected==actual))
        except subprocess.CalledProcessError:out.append(dict(path=target,expected=expected,tracked=False))
crosspath=base+'2026-09-13-symmetric-adoption/crossvalidation.json'
crossraw=git('show',f'{ref}:{crosspath}')
result={'ref':ref,'sha':git('rev-parse',ref).decode().strip(),'manifest_checks':out,'crossvalidation':{'path':crosspath,'sha256':hashlib.sha256(crossraw).hexdigest(),'json':json.loads(crossraw)}}
with (O/'supa-manifest-verification.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'checks':len(out),'passed':sum(x.get('match',False) for x in out),'unmatched':[x for x in out if not x.get('match')],'crossvalidation_keys':list(result['crossvalidation']['json'])}))
