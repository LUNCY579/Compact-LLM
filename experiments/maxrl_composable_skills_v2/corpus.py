"""Build and audit MaxRL composition data. No downloads, model loads or training.

python -m experiments.maxrl_composable_skills_v2.corpus build --out NEW_DIR --v1-zip V1_ZIP
python -m experiments.maxrl_composable_skills_v2.corpus audit --data EXISTING_DIR
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import random
import zipfile
from .contracts import *
from .generator import world,view,controls,boundary_pair
from .oracle import answers
from .verifier import score,compute_score

BAD_RESPONSES=('', '{}','{"answer":null}','{"answer":true}','{"answer":[false]}',
               '{"answer":[0.0]}','{"answer":NaN}','{"answer":[0],"answer":[1]}',
               '{"answer":[0],"extra":1}','```json\n{"answer":[0]}\n```',
               '{"answer":[0]} trailing','{"answer":[-1]}','{"answer":[99]}')
LEGACY_FAMILIES=('countdown','subset_sum','linear_constraints','cnf_assignment','shortest_path','precedence_order','finite_program','state_tracking')
EVAL_ROLES={'dev','test','challenge','guard_dev','guard_test','guard_challenge'}

def readrows(path):
    with open(path,encoding='utf-8') as src:
        for line in src:
            yield strict_json(line,131072)

def source_hashes():
    root=Path(__file__).parent
    return {str(p.relative_to(root)):filehash(p) for p in sorted(root.rglob('*.py'))}

def prior_hashes():
    root=Path(__file__).parent.parent/'maxrl_verifiable_data_v1'
    return {str(p.relative_to(root)):filehash(p) for p in sorted(root.rglob('*.py'))}

def v1_exclusions(path):
    if Path(path).suffix != '.zip':
        payload=json.loads(Path(path).read_text())
        require(set(payload)=={'problem_hashes','prompt_hashes','provenance'},'exclusion inventory schema')
        for key in ('problem_hashes','prompt_hashes'):
            require(type(payload[key]) is list and len(payload[key])==len(set(payload[key])) and all(type(x) is str and re.fullmatch('[0-9a-f]{64}',x) for x in payload[key]),'exclusion fingerprints')
        require(len(payload['problem_hashes'])==payload['provenance']['canonical_instances'],'inventory count')
        return set(payload['problem_hashes']),set(payload['prompt_hashes']),payload['provenance']
    with zipfile.ZipFile(path) as z:
        manifest=z.read('data/manifest.json');m=json.loads(manifest)
        hashes=set();prompts=set()
        for name,receipt in m['files'].items():
            if not name.startswith('canonical/'):continue
            raw=z.read('data/'+name)
            require(hashlib.sha256(raw).hexdigest()==receipt['sha256'],'v1 source hash mismatch')
            for line in raw.splitlines():
                row=json.loads(line);hashes.add(row['problem_hash']);prompts.add(row['prompt_hash'])
    return hashes,prompts,{'archive_sha256':filehash(path),'manifest_sha256':hashlib.sha256(manifest).hexdigest(),
                           'canonical_instances':len(hashes),'scope':'provided v1 corpus only; no complete historical or semantic exclusion'}

def quota(scale):
    require(scale in ('full','pilot'),'scale')
    if scale=='full':
        return {'parents':{'train':3000,'calibration':150,'dev':250,'test':250,'challenge':250,'warmstart':150},
                'boundary_pairs':{'train':500,'calibration':100,'dev':200,'test':200,'challenge':200},
                'legacy_per_family':{'replay_train':200,'guard_dev':150,'guard_test':200,'guard_challenge':100}}
    return {'parents':{'train':6,'calibration':2,'dev':3,'test':3,'challenge':3,'warmstart':2},
            'boundary_pairs':{'train':6,'calibration':2,'dev':4,'test':4,'challenge':4},
            'legacy_per_family':{'replay_train':2,'guard_dev':2,'guard_test':2,'guard_challenge':2}}

def make_row(spec,parent,role,level,template,kind='composition',family=None,tag='core'):
    if kind=='composition':
        prompt=render(spec,template);ph=problem_hash(spec);skills=list(signature(spec));data_source='rineya_maxrl_comp_v2'
    else:
        from experiments.maxrl_verifiable_data_v1.specs import render as old_render,fingerprint as old_fp
        prompt=old_render(family,spec,template);ph=old_fp(family,spec);skills=['legacy:'+family];data_source='rineya_maxrl_comp_v2/legacy/'+family
        spec={'family':family,'problem':spec}
    return {'schema_version':VERSION,'id':'MC2-'+digest([kind,ph]),'parent_id':parent,'role':role,
            'kind':kind,'tag':tag,'skills':skills,'difficulty':level,'template':template,
            'problem_hash':ph,'prompt_hash':digest(prompt),'prompt':prompt,
            'verifier':{'version':VERSION,'kind':kind,'spec':spec},'data_source':data_source}

def reward(row,response):
    return int(compute_score(row['data_source'],response,canonical(row['verifier'])))

def verl(row):
    return {'data_source':row['data_source'],'prompt':row['prompt'],'ability':'+'.join(row['skills']),
            'reward_model':{'style':'rule','ground_truth':canonical(row['verifier'])},
            'extra_info':{k:row[k] for k in ('id','parent_id','role','skills','problem_hash','tag')}}

def build(out:Path,v1_zip:Path,scale='full',seed=260922002):
    require(type(seed) is int and seed>=0,'seed');require(not out.exists(),'output must be new')
    excluded,old_prompts,prior=v1_exclusions(v1_zip)
    if scale=='full':
        require(prior['manifest_sha256']=='3a7f1a20d15d6c2ce26eeb02ae3c0a6353f771955230b390c35ac6739c241563' and len(excluded)==32800,'full build requires the bound v1 corpus/inventory')
    out.mkdir(parents=True)
    rows_seen=set();problems_seen={};parents_seen={};prompt_seen=set();stats=Counter();counts=Counter();multis=Counter()
    plans=defaultdict(list);lengths=[];controls_count=Counter();family_stats=Counter()
    roles=('train','calibration','dev','test','challenge','warmstart','replay_train','guard_dev','guard_test','guard_challenge')
    q=quota(scale)
    with ExitStack() as stack:
        files={}
        for directory in ('canonical','policy','verl_jsonl','oracles'):
            (out/directory).mkdir()
            for role in roles:files[directory,role]=stack.enter_context(open(out/directory/(role+'.jsonl'),'w',encoding='utf-8',newline='\n'))
        (out/'diagnostics').mkdir();(out/'sft').mkdir();(out/'manifests').mkdir()
        necessity=stack.enter_context(open(out/'diagnostics/stage_necessity.jsonl','w',encoding='utf-8',newline='\n'))
        twinfile=stack.enter_context(open(out/'diagnostics/answerability_twins.jsonl','w',encoding='utf-8',newline='\n'))
        parentfile=stack.enter_context(open(out/'diagnostics/parent_worlds.jsonl','w',encoding='utf-8',newline='\n'))
        sftfile=stack.enter_context(open(out/'sft/warmstart.messages.jsonl','w',encoding='utf-8',newline='\n'))
        lineage=stack.enter_context(open(out/'sft/warmstart.lineage.jsonl','w',encoding='utf-8',newline='\n'))

        def write(row,refs,negative,extra=None):
            require(row['id'] not in rows_seen and row['problem_hash'] not in problems_seen and row['prompt_hash'] not in prompt_seen,'duplicate primary')
            require(row['prompt_hash'] not in old_prompts,'exact v1 prompt collision')
            rows_seen.add(row['id']);problems_seen[row['problem_hash']]=row['role'];prompt_seen.add(row['prompt_hash'])
            require(row['parent_id'] not in parents_seen or parents_seen[row['parent_id']]==row['role'],'parent split leak')
            parents_seen[row['parent_id']]=row['role']
            require(refs and all(reward(row,x)==1 for x in refs),'positive certificate')
            require(reward(row,negative)==0 and all(reward(row,x)==0 for x in BAD_RESPONSES),'negative acceptance')
            oracle={'id':row['id'],'parent_id':row['parent_id'],'references':refs,'negative':negative,
                    'origin':'independent_public_spec_search','is_policy_rollout':False}
            if extra:oracle.update(extra)
            for directory,value in (('canonical',row),('policy',{'id':row['id'],'messages':row['prompt']}),('verl_jsonl',verl(row)),('oracles',oracle)):
                files[directory,row['role']].write(canonical(value)+'\n')
            stats['rows']+=1;stats['positive_certificates']+=len(refs);stats['negative_checks']+=len(BAD_RESPONSES)+1
            counts[row['role']]+=1;family_stats[row['tag']+'/'+''.join(row['skills'])]+=1
            if len(refs)>1:multis[row['tag']]+=1
            lengths.append(len(canonical(row['prompt'])))
            if row['tag']=='core':
                sig=''.join(row['skills'])
                if row['role']=='train':
                    plans['atomic_'+sig if len(sig)==1 else 'bridge_'+sig].append(row['id'])
                else:plans[row['role']+'_core'].append(row['id'])
            elif row['tag']=='boundary':plans[row['role']+'_boundary'].append(row['id'])
            else:plans[row['role']].append(row['id'])
            if row['role']=='warmstart':
                # Only atomic skills may enter the common optional warm start.
                require(len(row['skills'])==1,'compound warmstart leakage')
                sftfile.write(canonical({'messages':row['prompt']+[{'role':'assistant','content':refs[0]}]})+'\n')
                lineage.write(canonical({'id':row['id'],'parent_id':row['parent_id'],'target_sha256':hashlib.sha256(refs[0].encode()).hexdigest()})+'\n')

        for role,total in q['parents'].items():
            rng=random.Random(int(digest([seed,'core',role])[:16],16));accepted=0;attempt=0
            sigs=SKILLS if role=='warmstart' else TRAIN_SIGNATURES if role in ('train','calibration') else SIGNATURES
            while accepted<total:
                attempt+=1;require(attempt<total*150+10000,'core exhaustion')
                level=3 if role=='challenge' else accepted%2 if role=='warmstart' else accepted%3
                w=world(rng,level);parent='world-'+digest(w)
                if partition(digest(w))!=role or parent in parents_seen:continue
                bundle=[];bad=False
                template=rng.randrange(2,4) if role=='challenge' else rng.randrange(2)
                for sig in sigs:
                    spec=view(w,sig,rng,level)
                    if spec is None:bad=True;break
                    refs=answers(spec);check=controls(spec,refs)
                    if check is None:bad=True;break
                    row=make_row(spec,parent,role,level,template)
                    if row['problem_hash'] in problems_seen or any(row['problem_hash']==b[0]['problem_hash'] for b in bundle):bad=True;break
                    bundle.append((row,refs,check))
                if bad:stats['rejected_core_worlds']+=1;continue
                parentfile.write(canonical({'parent_id':parent,'role':role,'world':w,'level':level,'children':[b[0]['id'] for b in bundle]})+'\n')
                for row,refs,(control_ref,checks) in bundle:
                    negative='{"answer":"UNKNOWN"}'
                    write(row,refs,negative)
                    for control in checks:
                        require(signature(control['spec'])==''.join(row['skills']),'changed signature')
                        necessity.write(canonical({'id':row['id'],'parent_id':parent,'role':role,**control})+'\n')
                        controls_count[control['skill']]+=1
                accepted+=1
            print('core',role,accepted,'worlds',counts[role],'rows',flush=True)

        # Answerability twins are additional parents, not descendants in another split.
        for role,total in q['boundary_pairs'].items():
            rng=random.Random(int(digest([seed,'boundary',role])[:16],16));accepted=0;attempt=0
            eligible_sigs=tuple(s for s in (TRAIN_SIGNATURES if role in ('train','calibration') else SIGNATURES) if 'S' in s)
            while accepted<total:
                attempt+=1;require(attempt<total*200+10000,'boundary exhaustion')
                level=3 if role=='challenge' else accepted%3
                w=world(rng,level);parent='world-'+digest(w)
                if partition(digest(w))!=role or parent in parents_seen:continue
                sig=eligible_sigs[accepted%len(eligible_sigs)];s=view(w,sig,rng,level)
                if s is None:continue
                pair=boundary_pair(s)
                if pair is None:continue
                yes,no,yr,nr=pair
                template=rng.randrange(2,4) if role=='challenge' else rng.randrange(2)
                rows=[make_row(x,parent,role,level,template,tag='boundary') for x in (yes,no)]
                if any(r['problem_hash'] in problems_seen for r in rows):continue
                write(rows[0],yr,nr[0],{'answerability':'feasible'})
                write(rows[1],nr,yr[0],{'answerability':'infeasible'})
                twinfile.write(canonical({'parent_id':parent,'role':role,'answerable_id':rows[0]['id'],'unanswerable_id':rows[1]['id'],'changed_field':'selection.target'})+'\n')
                accepted+=1
            print('boundary',role,accepted,'pairs',flush=True)

        from experiments.maxrl_verifiable_data_v1.generators import make as legacy_make
        from experiments.maxrl_verifiable_data_v1.oracle import solutions as legacy_answers
        from experiments.maxrl_verifiable_data_v1.corpus import wrong_answer
        from experiments.maxrl_verifiable_data_v1.specs import fingerprint as old_fp
        for role,per_family in q['legacy_per_family'].items():
            destination={'replay_train':'train','guard_dev':'dev','guard_test':'test','guard_challenge':'challenge'}[role]
            for family in LEGACY_FAMILIES:
                rng=random.Random(int(digest([seed,'legacy',role,family])[:16],16));accepted=0;attempt=0
                while accepted<per_family:
                    attempt+=1;require(attempt<per_family*300+10000,'legacy exhaustion')
                    level=3 if role=='guard_challenge' else accepted%3
                    s=legacy_make(family,level,rng)
                    if s is None:continue
                    ph=old_fp(family,s)
                    if ph in excluded:stats['v1_collision_rejections']+=1;continue
                    if ph in problems_seen or partition(ph)!=destination:continue
                    refs=legacy_answers(family,s);require(bool(refs),'legacy construction unsolved')
                    template=rng.randrange(4,6) if role=='guard_challenge' else rng.randrange(4)
                    row=make_row(s,'legacy-'+ph,role,level,template,'legacy',family,'preservation')
                    write(row,refs,wrong_answer(family,s,refs[0]))
                    accepted+=1
                print('legacy',role,family,accepted,flush=True)

    # References to data, not duplicate new examples. Explicit source-usage contracts.
    curricula={}
    for name,ids in sorted(plans.items()):
        rel='manifests/'+name+'.json';payload={'name':name,'ids':ids,'rows':len(ids),'schema_version':VERSION,'is_replication':False}
        (out/rel).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
        curricula[name]={'path':rel,'rows':len(ids),'sha256':filehash(out/rel)}
    recipes={'schema_version':VERSION,'status':'DATA_RECIPES_NOT_NATIVE_LAUNCH',
             'common_anchor':None,'native_token_budget':None,'tokenizer_binding':None,
             'atomic_banks':['atomic_'+s for s in SKILLS],'optional_common_warmstart':'warmstart_core',
             'optional_replay':'replay_train','bridge_banks':['bridge_'+s for s in SEEN_PAIRS],
             'heldout_compositions':[s for s in SIGNATURES if len(s)>1 and s not in SEEN_PAIRS],
             'independent_delta_rule':'Every atomic donor starts from the same immutable anchor; train only its atomic bank. No bridge or boundary bank in the clean donor comparison.',
             'sequential_rule':'After A, train B from A; evaluate A before and after. With/without replay is a separate factor. Equal replay token budgets, not just row counts.',
             'coefficient_rule':'theta0 + sum(alpha_i * delta_i); independently chosen on DEV, with matching per-donor controls and no silent normalization of the sum.',
             'sampling_rule':'Within each fixed-N group, one prompt and one behavior-policy snapshot; all costs counted. Rehearsal sampling is not a positive-gradient guarantee on all-success groups.'}
    (out/'manifests/recipes.json').write_text(json.dumps(recipes,indent=2,sort_keys=True)+'\n')
    # Compact exact v1 exclusion inventory avoids shipping the v1 corpus again.
    (out/'diagnostics/v1_exclusion_hashes.json').write_text(json.dumps({'problem_hashes':sorted(excluded),'prompt_hashes':sorted(old_prompts),'provenance':prior},sort_keys=True)+'\n')
    files={str(p.relative_to(out)):{'sha256':filehash(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    ordered=sorted(lengths)
    manifest={'dataset':'Rineya MaxRL Composable Skills v2','experiment':'MAXRL-COMP-002','version':VERSION,
              'seed':seed,'scale':scale,'status':'GENERATED_NOT_NATIVE_QUALIFIED','quota':q,
              'counts':dict(counts),'totals':dict(stats),'stage_controls':dict(controls_count),
              'multiple_reference_rows':dict(multis),'family_rows':dict(family_stats),'parents':len(parents_seen),
              'files':files,'source_sha256':source_hashes(),'v1_source_sha256':prior_hashes(),'v1_provenance':prior,
              'curricula':curricula,'serialized_prompt_characters_NOT_TOKENS':{'min':ordered[0],'median':ordered[len(ordered)//2],'p95':ordered[int(.95*(len(ordered)-1))],'max':ordered[-1]},
              'evaluation_status':'EXPOSED; not sealed; source-correct membership unknown until actual baseline collection',
              'claims_excluded':['no forgetting guarantee','orthogonal weights by dataset construction','native token fit','measured Rineya difficulty','historical semantic disjointness','native training efficacy']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def audit(data:Path):
    m=json.loads((data/'manifest.json').read_text());require(m['source_sha256']==source_hashes(),'v2 source drift');require(m['v1_source_sha256']==prior_hashes(),'v1 source drift')
    for name,r in m['files'].items():
        p=data/name;require(p.is_file() and p.stat().st_size==r['bytes'] and filehash(p)==r['sha256'],'artifact hash/size: '+name)
    exclusions=json.loads((data/'diagnostics/v1_exclusion_hashes.json').read_text());excluded=set(exclusions['problem_hashes']);old_prompts=set(exclusions['prompt_hashes'])
    index={};parent_roles={};fps={};stats=Counter();counts=Counter();prompt_ids=set();multi=Counter()
    from experiments.maxrl_verifiable_data_v1.specs import fingerprint as old_fp,parse_visible as old_parse
    from experiments.maxrl_verifiable_data_v1.oracle import solutions as old_answers
    for path in sorted((data/'canonical').glob('*.jsonl')):
        role=path.stem
        companions=[iter(readrows(data/directory/path.name)) for directory in ('policy','verl_jsonl','oracles')]
        for row in readrows(path):
            policy,vr,oracle=[next(it,None) for it in companions]
            require(all(x is not None for x in (policy,vr,oracle)),'missing sidecar')
            require(row['role']==role and row['id'] not in index,'duplicate ID/role')
            require(row['problem_hash'] not in fps and row['prompt_hash'] not in prompt_ids,'duplicate problem/prompt')
            require(row['prompt_hash'] not in old_prompts,'v1 prompt overlap')
            require(parent_roles.setdefault(row['parent_id'],role)==role,'parent leakage')
            if row['kind']=='composition':
                s=parse_visible(row['prompt']);require(row['verifier']=={'version':VERSION,'kind':'composition','spec':s},'visible/reward drift')
                require(problem_hash(s)==row['problem_hash'] and list(signature(s))==row['skills'],'skill identity')
                refs=answers(s)
                require(partition(row['parent_id'].removeprefix('world-'))==role,'world split assignment')
                if role in ('train','calibration'):require(signature(s) in TRAIN_SIGNATURES,'heldout combination leaked')
                if role=='warmstart':require(signature(s) in SKILLS,'compound warmstart')
            else:
                f,s=old_parse(row['prompt']);require(row['verifier']['spec']=={'family':f,'problem':s},'legacy reward drift')
                require(old_fp(f,s)==row['problem_hash'] and row['problem_hash'] not in excluded,'v1 exclusion failure')
                require(partition(row['problem_hash'])=={'replay_train':'train','guard_dev':'dev','guard_test':'test','guard_challenge':'challenge'}[role],'legacy split assignment')
                refs=old_answers(f,s)
            require(row['id']=='MC2-'+digest([row['kind'],row['problem_hash']]) and digest(row['prompt'])==row['prompt_hash'],'hash identity')
            require(policy=={'id':row['id'],'messages':row['prompt']} and vr==verl(row),'export or policy leakage')
            require(oracle['id']==row['id'] and oracle['parent_id']==row['parent_id'] and oracle['is_policy_rollout'] is False and oracle['references']==refs,'independent reference mismatch')
            require(all(reward(row,a)==1 for a in refs),'positive certificate')
            require(reward(row,oracle['negative'])==0 and all(reward(row,b)==0 for b in BAD_RESPONSES),'negative accepted')
            index[row['id']]=row;fps[row['problem_hash']]=role;prompt_ids.add(row['prompt_hash'])
            counts[role]+=1;stats['rows']+=1;stats['positive_certificates']+=len(refs);stats['negative_checks']+=len(BAD_RESPONSES)+1
            if len(refs)>1:multi[row['tag']]+=1
        require(all(next(it,None) is None for it in companions),'extra sidecar')
    coverage=defaultdict(set);auxfps={}
    for c in readrows(data/'diagnostics/stage_necessity.jsonl'):
        r=index[c['id']];s=r['verifier']['spec'];key=FIELDS[c['skill']]
        require(c['parent_id']==r['parent_id'] and c['role']==r['role'],'counterfactual parent')
        require(set(c['spec'])==set(s) and c['spec'][key]!=s[key] and all(c['spec'][k]==v for k,v in s.items() if k!=key),'not single-stage change')
        require(score(s,c['transferred_response'])==1 and score(c['spec'],c['transferred_response'])==0 and score(c['spec'],c['positive_response'])==1,'vacuous stage control')
        require(answers(c['spec'],1)[0]==c['positive_response'],'control independent search')
        cf=problem_hash(c['spec'])
        require(cf not in fps or fps[cf]==r['role'],'cross-split primary/counterfactual collision')
        require(auxfps.setdefault(cf,r['role'])==r['role'],'cross-split auxiliary collision')
        require(c['skill'] not in coverage[c['id']],'duplicate stage witness')
        coverage[c['id']].add(c['skill']);stats['stage_controls']+=1
    for r in index.values():
        if r['tag']=='core':require(coverage[r['id']]==set(r['skills']),'missing skill necessity control')
    for twin in readrows(data/'diagnostics/answerability_twins.jsonl'):
        a=index[twin['answerable_id']];b=index[twin['unanswerable_id']]
        require(a['parent_id']==b['parent_id']==twin['parent_id'] and a['role']==b['role']==twin['role'],'twin parent')
        sa=copy.deepcopy(a['verifier']['spec']);sb=copy.deepcopy(b['verifier']['spec'])
        ta=sa['selection'].pop('target');tb=sb['selection'].pop('target')
        require(sa==sb and tb==ta+1,'unmatched twin')
        from .oracle import reachable_sums
        reachable=reachable_sums(b['verifier']['spec'])
        require(reachable and min(reachable)<tb<max(reachable) and tb not in reachable,'easy out-of-range or false UNKNOWN')
        require(reward(a,'{"answer":"UNKNOWN"}')==0 and reward(b,'{"answer":"UNKNOWN"}')==1,'abstention sentinel')
        stats['answerability_pairs']+=1
    # Manifest datasets are independent DATA VIEWS, not arbitrary executable training paths.
    used=set()
    for name,meta in m['curricula'].items():
        plan=json.loads((data/meta['path']).read_text());require(len(plan['ids'])==len(set(plan['ids']))==meta['rows'],'plan duplicate/count')
        for rid in plan['ids']:
            r=index[rid]
            if name.startswith('atomic_'):require(r['role']=='train' and r['tag']=='core' and r['skills']==[name[7:]],'atomic contamination')
            elif name.startswith('bridge_'):require(r['role']=='train' and ''.join(r['skills'])==name[7:] and name[7:] in SEEN_PAIRS,'bridge contamination')
            elif name in ('replay_train','guard_dev','guard_test','guard_challenge'):require(r['role']==name and r['tag']=='preservation','guard/replay leakage')
            else:require(r['role']+'_'+r['tag']==name,'plan role drift')
            require(rid not in used,'multiple bank ownership');used.add(rid)
    require(used==set(index),'missing bank ownership')
    warm=[r for r in index.values() if r['role']=='warmstart'];views=list(readrows(data/'sft/warmstart.messages.jsonl'));lineage=list(readrows(data/'sft/warmstart.lineage.jsonl'))
    byid={r['id']:r for r in warm};require(len(warm)==len(views)==len(lineage),'SFT length mismatch')
    for v,l in zip(views,lineage):
        r=byid.pop(l['id']);target=v['messages'][-1]['content']
        require(set(v)=={'messages'} and v['messages'][:-1]==r['prompt'] and v['messages'][-1]['role']=='assistant' and reward(r,target)==1,'SFT certificate/boundary')
        require(l['target_sha256']==hashlib.sha256(target.encode()).hexdigest(),'SFT lineage')
    require(not byid and dict(counts)==m['counts'] and stats['rows']==m['totals']['rows'],'quota or SFT leftovers')
    return {'status':'PASS_DATA_MECHANICS_ONLY','checks':dict(stats),'counts':dict(counts),'parent_clusters':len(parent_roles),
            'multiple_reference_rows':dict(multi),'cross_split_collisions':0,'v1_exact_collisions':0,'manifest_sha256':filehash(data/'manifest.json'),
            'source_sha256':source_hashes(),'native_training_runs':0,'source_correct_enrollment':'NOT_RUN','parquet':'NOT_RUN'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--out',type=Path,required=True);b.add_argument('--v1-zip',type=Path,help='v1 ZIP or the v2 included diagnostics/v1_exclusion_hashes.json inventory',required=True);b.add_argument('--scale',choices=['pilot','full'],default='full');b.add_argument('--seed',type=int,default=260922002)
    a=sub.add_parser('audit');a.add_argument('--data',type=Path,required=True);a.add_argument('--receipt',type=Path)
    x=p.parse_args()
    if x.cmd=='build':result=build(x.out,x.v1_zip,x.scale,x.seed)
    else:
        result=audit(x.data)
        if x.receipt:
            with x.receipt.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
