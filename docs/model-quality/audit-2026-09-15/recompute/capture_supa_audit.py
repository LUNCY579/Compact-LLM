import datetime, hashlib, json, pathlib, subprocess

ROOT = pathlib.Path(r'C:/Users/clclc/BusinessProject/model-quality-audit-20260915')
REPO = pathlib.Path(r'C:/Users/clclc/BusinessProject/SupaExperiment')
def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args])
def show(ref,path):
    return git('show',f'{ref}:{path}')
def compact(v):
    if isinstance(v,dict):
        return {k:({'omitted_count':len(x)} if k in ['history','records','items','curves','per_item','before','after','outer_before','outer_after'] and isinstance(x,(dict,list)) else compact(x)) for k,x in v.items()}
    if isinstance(v,list):
        return [compact(x) for x in v] if len(v)<=20 else {'omitted_count':len(v),'first':compact(v[0]),'last':compact(v[-1])}
    return v
refs=[]
for row in git('ls-remote','--heads','origin').decode().splitlines():
    sha,ref=row.split()
    parents=git('show','-s','--format=%P',sha).decode().strip().split()
    refs.append(dict(branch=ref.removeprefix('refs/heads/'),sha=sha,parents=parents,merge_base_main=git('merge-base',sha,'origin/main').decode().strip(),subject=git('show','-s','--format=%s',sha).decode().strip()))
nra='origin/research/supa-a1.1-nonrecurrent-adaptation-v1'
nrap='experiments/nonrecurrent_adaptation_v1/'
sources=[(nra,nrap+p) for p in [
 'STATUS.json','evidence/2026-09-10-capability1/registered_training.json','evidence/2026-09-10-capability1/score.json','evidence/2026-09-10-capability1/adapter/manifest.json',
 'evidence/2026-09-13-transfer3/score.json','evidence/2026-09-14-floor1/score.json','evidence/2026-09-14-floor1/attack.json','evidence/2026-09-14-floor1/loss.json','bsel2/training/manifest.json',
 'evidence/2026-09-13-symmetric-adoption/reproduction.json']]
sources += [
 ('origin/research/supa-a1.1-wsd-schedule-v1','experiments/wsd_schedule_v1/evidence/LOCAL_MECHANICAL.json'),
 ('origin/research/supa-a1.1-wsd-native-qualification-v1','experiments/wsd_native_qualification_v1/evidence/BLOCKED_ENVIRONMENT.json'),
 ('origin/research/supa-a1.1-matched-mlp-shipping-v1','experiments/matched_mlp_shipping_v1/evidence/2026-09-14-m0/receipt.json'),
 ('origin/research/capacity-expansion-v1','experiments/capacity_expansion_v1/evidence/2026-09-05-strict-gate/audit.json'),
 ('origin/research/supa-a1.1-recurrence-a06-mechanism-resolution-v1','experiments/recurrent_a06_mechanism_resolution_v1/evidence/2026-09-13-fixture-portability/REPAIR_RECEIPT.json'),
 ('origin/research/supa-a1.1-recurrent-adapter-v1','experiments/recurrent_adapter_v1/evidence/2026-09-05-stage0-2-qualified/stage0-2-mechanical.json'),
 ('origin/research/supa-a1.1-depth3-data34-replacement-v1',nrap+'depth3_replacement_v1/MANIFEST.json'),
 ('origin/research/supa-a1.1-bsel4-qualification-v1','.github/bsel4_qualification/execution_status.json'),
 ('research/supa-a1.1-hybrid-memory-architecture-v1','experiments/hybrid_memory_architecture_v1/receipts/evaluation-recovery-summary.json'),
 ('research/supa-a1.1-hybrid-memory-architecture-v1','experiments/hybrid_memory_architecture_v1/receipts/independent-evaluation-audit.json'),
 ('research/supa-a1.1-hybrid-memory-architecture-v1','experiments/hybrid_memory_architecture_v1/receipts/registered-summary.json'),
 ('research/supa-a1.1-abstention-calibration-v1','experiments/abstention_calibration_v1/STATUS.json')]
receipts=[]
for ref,path in sources:
    try:
        raw=show(ref,path)
    except subprocess.CalledProcessError:
        receipts.append(dict(ref=ref,path=path,missing=True));continue
    receipts.append(dict(ref=ref,sha=git('rev-parse',ref).decode().strip(),path=path,blob_sha=git('rev-parse',f'{ref}:{path}').decode().strip(),sha256=hashlib.sha256(raw).hexdigest(),content=compact(json.loads(raw))))
historical=[]
for ref in ['research/supa-a1.1-abstention-calibration-v1','research/supa-a1.1-hybrid-memory-architecture-v1','research/supa-a1.1-positional-abstention-a07-v1','research/supa-a1.1-recurrence-a04-per-hook-gates-v1','research/supa-a1.1-recurrence-a05-gate-lr-v1','research/supa-a1.1-recurrent-normalized-gate-lr-v1','research/recurrence-atlas-review-v2']:
    historical.append(dict(branch=ref,local_sha=git('rev-parse',ref).decode().strip(),parents=git('show','-s','--format=%P',ref).decode().strip().split(),remote_live=False))
payload=dict(audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='LUNCY579/SupaExperiment',live_heads=refs,local_historical_refs=historical,source_receipts=receipts,scope='Read-only git/object/artifact audit; no native calls, training, sealed-bank reads, branch edits, commits or pushes.')
ROOT.mkdir(exist_ok=True)
target=ROOT/'supa-evidence.json'
with target.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps({'file':str(target),'live_heads':len(refs),'receipts':len(receipts),'missing':[x['path'] for x in receipts if x.get('missing')]}))
