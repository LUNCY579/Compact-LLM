"""Read-only, metadata-only Supa storage review; no imports from research runtimes."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import subprocess

BASE=Path(r'C:\Users\clclc\BusinessProject')
OUT=BASE/'Rineya-Worktree-Audit-20261009'
inventory=json.loads((OUT/'supa-audit.json').read_text(encoding='utf-8-sig'))
processes=json.loads((OUT/'phase2-storage-processes.json').read_text(encoding='utf-8-sig'))
roots={record['name']:record for record in inventory['repositories']}
cache_files=[]
skipped=[]
for name, record in roots.items():
    root=Path(record['path'])
    if not root.exists():
        continue
    candidates=[f for f in record['ignored'] if '/__pycache__/' in f['path'] and f['path'].endswith('.pyc')]
    for f in candidates:
        pyc=root/f['path']
        match=re.fullmatch(r'(.+)\.(?:cpython-\d+|pypy\d+)(?:\.opt-\d+)?\.pyc',pyc.name)
        source=pyc.parent.parent/(match.group(1)+'.py') if match else None
        if not pyc.is_file() or pyc.is_symlink() or source is None or not source.is_file() or source.is_symlink():
            skipped.append({'path':str(pyc),'reason':'Missing/nonregular cached file or matching retained source; do not delete'})
            continue
        cache_files.append({'repository':name,'path':str(pyc),'relativePath':f['path'],'bytes':pyc.stat().st_size,
            'retainedSource':str(source),'retainedSourceBytes':source.stat().st_size,
            'modifiedUtc':datetime.fromtimestamp(pyc.stat().st_mtime,tz=timezone.utc).isoformat(),
            'kind':'SOURCE_BACKED_INTERPRETER_BYTECODE_CACHE_ONLY','archiveRequired':False})
cache_dirs=[]
for path in sorted({str(Path(f['path']).parent) for f in cache_files}):
    expected={f['path'] for f in cache_files if str(Path(f['path']).parent)==path}
    actual={str(p) for p in Path(path).iterdir()}
    eligible=actual==expected and all(p.is_file() and not p.is_symlink() for p in Path(path).iterdir())
    cache_dirs.append({'path':path,'files':len(expected),'bytes':sum(f['bytes'] for f in cache_files if f['path'] in expected),
        'wholeDirectoryContainsOnlyListedSourceBackedPyc':eligible,
        'recommendation':'LOW_RISK_BYTECODE_CACHE_ONLY_CLEANUP' if eligible else 'ONLY_LISTED_FILES_ELIGIBLE_KEEP_OTHER_DIRECTORY_CONTENT'})
checks=[]
for name in ('SupaExperiment','SupaExperiment-nonrecurrent-adaptation-v1','rineya-dataset-retention-records'):
    r=roots[name]
    paths=[f['relativePath'] for f in cache_files if f['repository']==name]
    if paths:
        cp=subprocess.run(['git','-c','core.longpaths=true','-C',r['path'],'check-ignore','-z','--stdin'],input=('\0'.join(paths)+'\0').encode('utf-8'),capture_output=True)
        checked={value.decode('utf-8') for value in cp.stdout.split(b'\0') if value}
        checks.append({'repository':name,'listedFiles':len(paths),'ignoredFilesConfirmed':len(checked),'allIgnored':checked==set(paths),'exitCode':cp.returncode})
        if cp.stderr or checked!=set(paths):
            raise RuntimeError('Ignored file verification failed')
envs=[]
for name,env,role in (
    ('SupaExperiment','.venv-cuda','DORMANT_VARIANT_KEEP_PENDING_WIDER_CALLER_RECONCILIATION'),
    ('SupaExperiment-nonrecurrent-adaptation-v1','.venv','KEEP_CURRENT_OWNER_AND_EXPLICIT_DONOR')):
    r=roots[name]
    path=Path(r['path'])/env
    facts=[f for f in r['ignored'] if f['path'].startswith(env+'/')]
    package_dir=path/'Lib/site-packages'
    versions=[p.name for p in package_dir.glob('*.dist-info') if p.name.startswith(('torch-','transformers-','numpy-','tokenizers-'))]
    matching=next((p['matchingProcesses'] for p in processes['candidates'] if p['candidate']==str(path)),[])
    envs.append({'path':str(path),'inventoryBytes':sum(f['bytes'] for f in facts),'inventoryFiles':len(facts),
        'installedPackageMetadataDirectories':sorted(versions),'processReferencesAtSnapshot':matching,
        'role':role,'recommendWholeEnvironmentDeletionNow':False,
        'disposition':'Retain complete interpreter, DLLs, packages and recipe; source-backed pyc files separately eligible.',
        'dependencyEvidence': (
           ['Current inherited GN1 PROGRAMME.md:432 explicitly requires transformers5.14.1 and rejects .venv-cuda5.16.1.',
            'Completed hybrid worktree used this runtime historically; hybrid is already retired and its scientific artifacts privately archived.',
            'Narrow source/config search in retained Supa worktrees and selected M1/M2/RINQ records found no current caller for this env. This is not global caller proof.']
           if name=='SupaExperiment' else
           ['mlp-ship-001-records/20261005-m1-8ef12f4b/prepare_local_environment.py:10 explicitly names this environment as PRIMARY donor.',
            'LOCAL_RUNTIME_CUSTODY.json records non-Torch copy entries from this environment; runtime preparation and observer custody exclude __pycache__, .pyc and .pyo.',
            'Current owner remains blocked/active; deleting interpreter or package directories could break its qualified recipe.'])})
report={
    'schema':'RINEYA_PHASE2_SUPA_STORAGE_REVIEW_V1',
    'capturedUtc':datetime.now(timezone.utc).isoformat(),
    'clientDate':'2026-10-09','timezone':'America/Chicago',
    'scope':'Retained Supa registered worktrees, RineyaArchitecture primary and retention-records primary; narrow supplemental MLP/M2/RINQ runtime-reference metadata.',
    'method':'Reuse completed ignored-file inventory, stat selected existing cache/source files, inspect installed dist-info directory names and pyvenv.cfg, redacted Win32_Process snapshot, narrow rg source/config searches. No research runtime, Torch, model/tokenizer, fixture or native evaluation imported/executed.',
    'summary':{'lowRiskSourceBackedBytecodeFiles':len(cache_files),'lowRiskCacheBytes':sum(f['bytes'] for f in cache_files),
        'lowRiskWholeCacheDirectories':sum(d['wholeDirectoryContainsOnlyListedSourceBackedPyc'] for d in cache_dirs),
        'materialSizedEnvironmentDeletionRecommendedNow':False,
        'materialSizedDisposableStandaloneCacheOrDependencyDirectoriesFound':False,
        'reason':'Remaining large allocations are distinct CUDA runtime recipes or research artifacts. Disposable cache totals are only about232KB; avoid package/runtime removal for small wins.'},
    'bytecodeCandidates':cache_files,'cacheDirectoryCandidates':cache_dirs,'skippedCacheFiles':skipped,'gitIgnoreChecks':checks,
    'environmentReview':envs,'processSnapshot':processes,
    'retentionArtifactDisposition':{'root':str(BASE/'rineya-dataset-retention-records'),'ignoredBytes':roots['rineya-dataset-retention-records']['ignoredBytes'],
        'recommendDeletion':False,'reason':'12.829GB ignored storage mainly CLOCKFIX/TRISKILL trained/checkpoint tensors, tokenizer sidecars and input zip. These are scientific/model assets, not general-purpose caches. Trained results are nonreproducible; public Git/archive tags exclude ignored assets. New vault/duplicate-file verification and preservation would precede separate artifact pruning.',
        'bytecodeOnlyBytes':sum(f['bytes'] for f in cache_files if f['repository']=='rineya-dataset-retention-records')},
    'standaloneFixtureAssessment':{'disposableStandaloneFixtureDirectoriesWithinDelegatedRetainedRepositoriesFound':False,
        'excludedFromDeletion':'Historical raw outputs, execution ledgers, permits, byte-verified source-custody fixtures, native receipts, bank files and all owner active/blocked studies.',
        'sourceCustodyWarning':'External M1 thin-observer fixtures deliberately include __pycache__/excluded.py and excluded.pyc whose bytes are recorded in THIN_COMPLETE_FIXTURE_MANIFEST.json/THIN_SUCCESSOR_HANDOFF.json. Cache-like names alone are not deletion authority; those evidentiary fixture directories stay intact.'},
    'runtimeReferenceSearchPaths':str(OUT/'phase2-supa-runtime-reference-paths.txt'),
    'beforeAnyDeletion':'Root rechecks resolved absolute candidate paths, unchanged matching .py sources, directory contents and current running-process references. Delete only eligible pyc/cache directories, never complete interpreter/runtime/package paths. No archive needed for source-backed bytecode caches.',
    'limitations':['Metadata-only review; no full environment content hashing or package reinstall tests.',
        'Process state is one snapshot; no process match does not prove future use or no opaque handle.',
        'Environment recipes differ, so shared package names are not proof of interchangeable duplicate installations.',
        'A dormant primary .venv-cuda may offer4.59GB later only after wider caller/owner reconciliation and exact recipe preservation; it is not approved for removal by this review.',
        'No repository, interpreter, cache, fixture or artifact was modified or deleted in this delegated phase.']
}
path=OUT/'phase2-storage-review.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(path),'summary':report['summary'],'environmentBytes':[e['inventoryBytes'] for e in envs]},indent=2))
