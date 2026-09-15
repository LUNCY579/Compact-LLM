import hashlib,json,pathlib,subprocess
p=pathlib.Path(__file__).parent
jpath=p/'architecture-beta-audit.json'
d=json.loads(jpath.read_text(encoding='utf-8'))
for r in d['branches']:
    r['executed']=r['executed'].replace('89 structural/native-random','74 structural/native-random').replace('Initial 89-test','Initial 74-test')
    for i,s in enumerate(r['source_evidence']):
        if s.get('not_found'):
            path='experiments/sink_token_mechanism_v1/interventions.py'
            content=subprocess.check_output(['git','show',r['remote_head']+':'+path],cwd=p/'RineyaBeta.git')
            r['source_evidence'][i]={'path':path,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'url':f"https://github.com/LUNCY579/RineyaBeta/blob/{r['remote_head']}/{path}"}
jpath.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
mpath=p/'architecture-beta-audit.md'
md=mpath.read_text(encoding='utf-8').replace('89 structural/native-random','74 structural/native-random').replace('Initial 89-test','Initial 74-test')
md=md.replace('`sink_mechanism_v1/interventions.py` (not found; do not cite)','[experiments/sink_token_mechanism_v1/interventions.py](https://github.com/LUNCY579/RineyaBeta/blob/5114e19543fed5f7a897fe1b36f009ba960e920d/experiments/sink_token_mechanism_v1/interventions.py)')
mpath.write_text(md,encoding='utf-8')
print(json.dumps({'branch_count':len(d['branches']),'false_ancestry':[(r['repo'],r['branch']) for r in d['branches'] if r.get('intervention_base_is_ancestor') is False],'blind_primary':d['selected_artifacts']['beta_blind_primary'],'missing_paths':[s for r in d['branches'] for s in r['source_evidence'] if s.get('not_found')]}))
