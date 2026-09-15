import datetime,json,subprocess
from pathlib import Path
B=Path('C:/Users/clclc/BusinessProject'); O=B/'model-quality-audit-20260915'
repos={'ChatBubble':B/'ChatBubble-ced-native-loader-qualification','SupaExperiment':B/'SupaExperiment','RineyaArchitecture':B/'RineyaArchitecture','RineyaBeta':O/'RineyaBeta.git','RineyaRAG':O/'RineyaRAG.git'}
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repos':{}}
for name,p in repos.items():
    remote=subprocess.check_output(['git','ls-remote','--heads',f'https://github.com/LUNCY579/{name}.git'],text=True)
    heads={}
    for line in remote.splitlines():
        sha,ref=line.split();parents=subprocess.check_output(['git','-C',str(p),'show','-s','--format=%P',sha],text=True).strip().split()
        heads[ref.removeprefix('refs/heads/')]={'sha':sha,'parents':parents}
    out['repos'][name]={'heads':heads}
with (O/'remote-heads-before.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print({k:len(v['heads']) for k,v in out['repos'].items()})
