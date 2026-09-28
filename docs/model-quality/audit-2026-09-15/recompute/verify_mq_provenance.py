import hashlib,json,subprocess
from pathlib import Path
B=Path('C:/Users/clclc/BusinessProject'); R=B/'ChatBubble-ced-native-loader-qualification'; O=B/'model-quality-audit-20260915'
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
out={'scope':'Offline historical code/source integrity; no model execution','code':{},'source_weights':{}}
for commit in ['131f0af71536489308248bb2bca52c76e157821c','1e41cc5d9f8f59b62559c0e58384fce77423b8f4']:
    prefix='experiments/model_quality_gpu_v1/'
    paths=git('ls-tree','-r','--name-only',commit,'--',prefix).decode().splitlines(); files={}
    for path in paths:
        rel=path.removeprefix(prefix)
        if set(Path(rel).parts)&{'__pycache__','.pytest_cache','outputs','evidence'}:continue
        files[rel]=hashlib.sha256(git('show',f'{commit}:{path}').replace(b'\r\n',b'\n')).hexdigest()
    calculated=hashlib.sha256(canonical(files)).hexdigest()
    bindings=[json.loads(p.read_text())['binding'] for p in (B/'mq-runs').glob('cell-*/result.json')]
    expected={v['code_sha256'] for v in bindings if v['checkout']==commit}
    out['code'][commit]={'calculated_sha256':calculated,'recorded_sha256':sorted(expected),'matches':expected=={calculated},'files':files}
for rel,expected in [('Supa/model.safetensors','962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b'),('Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/model.safetensors','e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b')]:
    h=hashlib.sha256()
    with (R/rel).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    out['source_weights'][rel]={'sha256':h.hexdigest(),'expected':expected,'matches':h.hexdigest()==expected,'bytes':(R/rel).stat().st_size}
with (O/'mq-provenance-recheck.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps({'code_matches':{c:v['matches'] for c,v in out['code'].items()},'source_matches':{c:v['matches'] for c,v in out['source_weights'].items()}}))
