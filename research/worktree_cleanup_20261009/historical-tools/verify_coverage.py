import json, subprocess
from pathlib import Path

OUT=Path(r'C:\Users\clclc\BusinessProject\Rineya-Worktree-Audit-20261009')
data=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))

def call(repo,args,stdin=None):
    return subprocess.run(['git','-c','core.longpaths=true','-C',repo,*args],input=stdin,capture_output=True,timeout=180)

for repo in data['repositories']:
    path=repo['path']
    r=call(path,['ls-remote','--heads','--tags','origin'])
    remote={line.split('\t',1)[1]:line.split('\t',1)[0] for line in r.stdout.decode().splitlines() if '\t' in line}
    repo['live_remote_refs']=remote
    repo['live_remote_verified']=r.returncode==0
    if r.returncode: repo['remote_query_error']=r.stderr.decode('utf-8','replace')
    exclusion_ids=set(remote.values())
    available=[]; missing=[]
    for oid in exclusion_ids:
        check=call(path,['cat-file','-e',oid+'^{commit}'])
        (available if check.returncode==0 else missing).append(oid)
    repo['remote_commit_objects_missing_locally']=missing
    stdin=('\n'.join('^'+oid for oid in available)+'\n').encode() if available else None
    for branch in repo['branches']:
        args=['rev-list','--count',branch['oid']]
        if available: args+=['--stdin']
        rc=call(path,args,stdin)
        branch['live_uncovered_commits']=int(rc.stdout) if rc.returncode==0 else None
        branch['live_remote_preserved']=rc.returncode==0 and int(rc.stdout)==0
        branch['live_exact_refs']=[n for n,o in remote.items() if o==branch['oid']]
        if branch['live_uncovered_commits']:
            print('UNPRESERVED',Path(path).name,branch['ref'],branch['oid'][:12],branch['live_uncovered_commits'],flush=True)
    local_names={x['ref'] for x in repo['branches']}
    repo['remote_only_branches']=[{'ref':n,'oid':o,'disposition':'Preserved remotely; no local branch. Retain remote research history.'} for n,o in remote.items() if n.startswith('refs/heads/') and n not in local_names]
    for wt in repo['worktrees']:
        if not wt.get('exists') or repo.get('bare'): continue
        p=Path(wt['worktree'])
        for key,args in [('untracked',['ls-files','--others','--exclude-standard','-z']),('ignored',['ls-files','--others','--ignored','--exclude-standard','-z'])]:
            raw=call(str(p),args)
            names=[n for n in raw.stdout.decode('utf-8','replace').split('\0') if n]
            entries=[]
            for n in names:
                q=p/n
                try:
                    s=q.lstat(); entries.append({'path':n,'bytes':s.st_size,'mtime':s.st_mtime,'directory':q.is_dir(),'symlink':q.is_symlink()})
                except OSError as e: entries.append({'path':n,'error':str(e)})
            wt[key]=entries
            if raw.stderr: wt[key+'_inventory_warning']=raw.stderr.decode('utf-8','replace')
        wt['live_remote_head_exact_refs']=[n for n,o in remote.items() if o==wt.get('HEAD')]
        branch=next((b for b in repo['branches'] if b['oid']==wt.get('HEAD')),None)
        wt['live_remote_head_preserved']=branch['live_remote_preserved'] if branch else bool(wt['live_remote_head_exact_refs'])
    print('verified',Path(path).name,len(repo['branches']),len(repo['remote_only_branches']),flush=True)
    (OUT/'inventory.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print('Coverage verification complete',flush=True)
