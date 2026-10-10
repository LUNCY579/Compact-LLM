import concurrent.futures, datetime, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(r'C:\Users\clclc\BusinessProject')
OUT = ROOT / 'Rineya-Worktree-Audit-20261009'
REPOS = [ROOT/'Chat-Bubble', ROOT/'SupaExperiment', ROOT/'RineyaArchitecture', ROOT/'rineya-dataset-retention-records', ROOT/'Rineya-Research-Git/self-distillation-axes.git', ROOT/'WorldModel/rineya-world-model-research', ROOT/'WorldModel/Rineya_BWM_001_Research_Bundle/gh_export']

def git(repo, *args, allow=False, binary=False):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, timeout=180)
    if p.returncode and not allow:
        raise RuntimeError(f'{repo}: {args}: {p.stderr.decode("utf-8", "replace")}')
    return p.stdout if binary else p.stdout.decode('utf-8', 'replace').strip()

def refs(repo):
    fmt = '%(refname)%09%(objectname)%09%(*objectname)%09%(upstream)%09%(committerdate:iso-strict)%09%(subject)'
    return [dict(zip(['ref','oid','peeled','upstream','date','subject'], l.split('\t',5))) for l in git(repo,'for-each-ref',f'--format={fmt}','refs/heads','refs/remotes','refs/tags',allow=True).splitlines()]

def worktrees(repo):
    result=[]
    for block in git(repo,'worktree','list','--porcelain').split('\n\n'):
        d={}
        for line in block.splitlines():
            k,_,v=line.partition(' ')
            d[k]=v or True
        if d.get('worktree'): result.append(d)
    return result

def file_info(path, name):
    p=path/name
    try:
        st=p.lstat()
        return {'path':name,'bytes':st.st_size,'mtime':st.st_mtime,'directory':p.is_dir(),'symlink':p.is_symlink()}
    except OSError as e: return {'path':name,'error':str(e)}

def scan_wt(repo,w,refdata,remote):
    p=pathlib.Path(w['worktree'])
    d=dict(w); d['exists']=p.exists(); d['inside_requested_directory']=str(p).lower().startswith(str(ROOT).lower()+os.sep)
    if not p.exists(): return d
    d['primary']=p.resolve()==repo.resolve()
    d['status_z']=git(p,'status','--porcelain=v1','-z','--untracked-files=all',binary=True).decode('utf-8','replace').split('\0')
    d['status_z']=[x for x in d['status_z'] if x]
    d['tracked_changes']=git(p,'diff','--name-status','HEAD',allow=True).splitlines()
    d['untracked']=[file_info(p,n) for n in git(p,'ls-files','--others','--exclude-standard','-z',binary=True).decode('utf-8','replace').split('\0') if n]
    d['ignored']=[file_info(p,n) for n in git(p,'ls-files','--others','--ignored','--exclude-standard','-z',binary=True).decode('utf-8','replace').split('\0') if n]
    d['diff_stat']=git(p,'diff','--stat','HEAD',allow=True)
    d['submodules']=git(p,'submodule','status',allow=True)
    d['root_instructions']=(p/'AGENTS.md').read_text(encoding='utf-8',errors='replace') if (p/'AGENTS.md').exists() else ''
    d['last_commit']=git(p,'show','-s','--format=%H%n%cI%n%s','HEAD')
    head=w.get('HEAD')
    d['same_head_local_refs']=[r['ref'] for r in refdata if (r['peeled'] or r['oid'])==head]
    d['same_head_live_remote_refs']=[r for r,oid in remote.items() if oid==head]
    d['remote_containing_head']=git(repo,'for-each-ref',f'--contains={head}','--format=%(refname)','refs/remotes/origin','refs/tags',allow=True).splitlines() if head else []
    total=0; count=0; nested=[]; reparse=[]; latest=0
    def onerror(e): d.setdefault('walk_errors',[]).append(str(e))
    for base,dirs,files in os.walk(p,followlinks=False,onerror=onerror):
        for name in list(dirs):
            q=pathlib.Path(base)/name
            if name=='.git':
                if q.parent!=p: nested.append(str(q))
                dirs.remove(name)
            elif q.is_symlink() or (q.lstat().st_file_attributes & 0x400):
                reparse.append(str(q)); dirs.remove(name)
        for name in files:
            q=pathlib.Path(base)/name
            if name=='.git' and q.parent!=p: nested.append(str(q))
            try:
                st=q.lstat(); total+=st.st_size; count+=1; latest=max(latest,st.st_mtime)
            except OSError: pass
    d.update(total_bytes=total,total_files=count,nested_git=nested,reparse_paths=reparse,latest_file_mtime=latest)
    print(f'worktree {p.name}: {len(d["status_z"])} status, {len(d["ignored"])} ignored, {total/1024**3:.2f} GiB',flush=True)
    return d

def scan_repo(repo):
    d={'path':str(repo),'remotes':git(repo,'remote','-v'),'bare':git(repo,'rev-parse','--is-bare-repository')=='true'}
    d['refs']=refs(repo)
    lr=git(repo,'ls-remote','--heads','--tags','origin',allow=True)
    d['live_remote_refs']={line.split('\t',1)[1]:line.split('\t',1)[0] for line in lr.splitlines() if '\t' in line}
    d['remote_default']=git(repo,'ls-remote','--symref','origin','HEAD',allow=True)
    remote=d['live_remote_refs']
    branches=[]
    for r in d['refs']:
        if not r['ref'].startswith('refs/heads/'): continue
        b=dict(r)
        b['exact_remote_refs']=[name for name,oid in remote.items() if oid==r['oid']]
        b['containing_remote_refs']=git(repo,'for-each-ref',f'--contains={r["oid"]}','--format=%(refname)','refs/remotes/origin','refs/tags',allow=True).splitlines()
        b['uncovered_commits']=git(repo,'rev-list','--count',r['ref'],'--not','--remotes=origin','--tags',allow=True)
        if r['upstream']:
            b['ahead_behind']=git(repo,'rev-list','--left-right','--count',r['ref']+'...'+r['upstream'],allow=True)
        branches.append(b)
    d['branches']=branches
    ws=worktrees(repo)
    if not d['bare']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            d['worktrees']=list(pool.map(lambda w:scan_wt(repo,w,d['refs'],remote),ws))
    else: d['worktrees']=ws
    (OUT/(repo.name.replace('.git','')+'-inventory.json')).write_text(json.dumps(d,indent=2),encoding='utf-8')
    print(f'repo {repo}: {len(branches)} local branches, {len(remote)} live remote refs, {len(ws)} worktrees',flush=True)
    return d

def main():
    OUT.mkdir(exist_ok=True)
    data={'scan_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_root':str(ROOT),'repositories':[]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(scan_repo,repo):repo for repo in REPOS}
        for task in concurrent.futures.as_completed(tasks):
            repo=tasks[task]
            try: data['repositories'].append(task.result())
            except Exception as e:
                data['repositories'].append({'path':str(repo),'error':str(e)}); print(str(e),flush=True)
            (OUT/'inventory.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('Inventory complete',flush=True)

if __name__=='__main__': main()
