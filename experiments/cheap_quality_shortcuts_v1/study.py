import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as B

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
def answer(value): return json.dumps({'answer':value},ensure_ascii=False,separators=(',',':'))
def kind(prompt):
    if 'Answer with an integer.' in prompt:return 'integer'
    if 'Registers: ' in prompt:return 'registers'
    if 'Start with this token list: ' in prompt:return 'text'
    if 'Answer with a sorted distinct integer list.' in prompt:return 'sets'
    raise ValueError('Unrecognized public answer contract')
def zero_value(k): return 0 if k=='integer' else '' if k=='text' else [] if k=='sets' else [0,0,0]
def predict(name,prompt,modes):
    k=kind(prompt)
    nums=[int(x) for x in re.findall(r'(?<![A-Za-z])[-]?\d+',prompt)]
    if name=='selection_mode': return modes[k]
    if name in ('zero','one','minus_one','empty'):
        val={'zero':0,'one':1,'minus_one':-1,'empty':0}[name]
        return zero_value(k) if name=='empty' else val if k=='integer' else str(val) if k=='text' else [val]*3 if k=='registers' else [val]
    if name in ('first_number','last_number','min_number','max_number'):
        val=(nums[0] if name=='first_number' else nums[-1] if name=='last_number' else min(nums) if name=='min_number' else max(nums)) if nums else 0
        return val if k=='integer' else str(val) if k=='text' else [val]*3 if k=='registers' else [val]
    if name=='initial_state':
        if k=='registers':return [int(x) for x in re.findall(r'[A-Z]=(-?\d+)',prompt)][:3]
        if k=='text':
            tokens=json.loads(re.search(r'Start with this token list: (\[[^\n]+?\])\.',prompt).group(1))
            sep=json.loads(re.search(r'Join the remaining tokens with (".*?") and no spaces',prompt).group(1))
            return sep.join(tokens)
        return zero_value(k)
    if name=='first_set':
        first=re.search(r'\{([^}]*)\}',prompt)
        return sorted(set(int(x) for x in re.findall(r'-?\d+',first.group(1)))) if k=='sets' and first else zero_value(k)
    if name=='all_numbers':return sorted(set(nums)) if k=='sets' else nums[:3] if k=='registers' else ''.join(map(str,nums)) if k=='text' else sum(nums)
    raise ValueError(name)

def controls(target):
    value=B.parse_answer(target)
    wrong=answer(B.wrong_answer(value))
    return [('canonical',target,True),('pretty',json.dumps({'answer':value},indent=2),True),
        ('leading_trailing_space',' \n'+target+'\n ',True),('wrong_answer',wrong,False),
        ('null',answer(None),False),('extra_key',json.dumps({'answer':value,'note':'extra'}),False),
        ('duplicate_key','{"answer":null,"answer":'+json.dumps(value)+'}',False),
        ('fenced','```json\n'+target+'\n```',False),('trailing_prose',target+' correct',False),
        ('torn',target[:-1],False)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    config=json.loads((PACKAGE/'config.json').read_text())
    out=args.out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True); started=time.monotonic()
    frozen={'config':config,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'python':sys.version,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(B.__file__),PACKAGE/'study.py',PACKAGE/'config.json']},
        'native_model_loads':0,'protected_payloads_opened':0}
    dump(out/'FROZEN.json',frozen)
    parents,allocation=B.prepare_parents(config['role_counts'],config['seed'])
    assert len(parents)==512 and all(p.role in ('DEV','TRAIN_SELECTION') for p in parents)
    data=[]
    for p in parents:
        row=B.make_row(p,frozen['git_head'],0)
        assert B.verify_answer(p,row['target'])
        data.append((p,row))
    assert Counter((r['role'],r['family']) for _,r in data)==Counter({(role,family):64 for role in ('DEV','TRAIN_SELECTION') for family in B.FAMILIES})
    groups=defaultdict(Counter)
    for p,row in data:
        if p.role=='TRAIN_SELECTION':groups[kind(row['prompt'])][json.dumps(B.parse_answer(row['target']),sort_keys=True)]+=1
    modes={k:json.loads(sorted(counts,key=lambda s:(-counts[s],s))[0]) for k,counts in groups.items()}
    dump(out/'CALIBRATION_FROZEN.json',{'role':'TRAIN_SELECTION','modes_by_public_contract':modes,
        'selection_ids':[p.parent_id for p,_ in data if p.role=='TRAIN_SELECTION'],'rule':'most frequent, canonical lexical tie; no DEV labels used'})
    for name,view in [('public.jsonl',True),('oracle.private.jsonl',False)]:
        with (out/name).open('x',encoding='utf-8') as f:
            for p,row in data:
                obj={'id':p.parent_id,'messages':[{'role':'user','content':row['prompt']}]} if view else {'id':p.parent_id,'role':p.role,'family':row['family'],'target':row['target'],'canonical_parent_sha256':p.canonical_parent_sha256}
                f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    scores=[]; stress=[]; calls=512
    for p,row in data:
        if p.role!='DEV':continue
        for name in config['predictors']:
            output=answer(predict(name,row['prompt'],modes))
            good=B.verify_answer(p,output);calls+=1
            scores.append({'id':p.parent_id,'family':row['family'],'predictor':name,'correct':int(good),'candidate':output})
        for label,output,expected in controls(row['target']):
            accepted=B.verify_answer(p,output);calls+=1
            stress.append({'id':p.parent_id,'family':row['family'],'control':label,'expected_accept':expected,'accepted':accepted,'passed':accepted==expected})
    curves=[]
    for family in (*B.FAMILIES,'ALL'):
        for name in config['predictors']:
            panel=[r for r in scores if r['predictor']==name and (family=='ALL' or r['family']==family)]
            curves.append({'family':family,'predictor':name,'parents':len(panel),'correct':sum(r['correct'] for r in panel),'accuracy':sum(r['correct'] for r in panel)/len(panel)})
    failures=[s for s in stress if not s['passed']]
    positives=[s for s in stress if s['expected_accept']];negatives=[s for s in stress if not s['expected_accept']]
    summary={'status':'PASS_DATA_VERIFIER_DIAGNOSTIC' if not failures else 'FAIL_VERIFIER_CONTROLS',
        'evidence':'CPU_DATA_ONLY_NOT_MODEL_EFFICACY','population':{'selection':256,'DEV':256,'per_family_per_role':64},
        'allocation':allocation,'predictors_frozen':config['predictors'],'calibration_modes':modes,'shortcut_curves':curves,
        'controls':{'total':len(stress),'positive':len(positives),'negative':len(negatives),
            'false_accepts':sum(s['accepted'] for s in negatives),'false_rejects':sum(not s['accepted'] for s in positives),'failures':failures},
        'costs':{'reference_rows':512,'heuristic_candidates':len(scores),'mutation_candidates':len(stress),
            'make_row_reference_verifier_calls':1024,'additional_verifier_calls':calls,'elapsed_seconds':time.monotonic()-started,
            'model_calls':0,'model_generation_tokens':0,'optimizer_steps':0},
        'interpretation':'Per-family shortcut floors identify answer-prior/task vulnerabilities, not evidence that Rineya exploits them. Gold-derived controls are verifier tests, not deployable predictions.',
        'CHECK':'CLOSED','FINAL':'CLOSED'}
    dump(out/'report.json',summary)
    for filename,rows in [('shortcut_curves.csv',curves),('candidates.private.csv',scores),('controls.private.csv',stress)]:
        with (out/filename).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tab=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ('family','predictor','parents','correct','accuracy'))+'</tr>' for r in curves)
    (out/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>CQD-001</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>CPU shortcut and verifier audit</h1><p>'+html.escape(summary['evidence'])+'</p><p>'+html.escape(summary['interpretation'])+'</p><p>Control failures: '+str(len(failures))+'</p><table><tr><th>Family</th><th>Predictor</th><th>Parents</th><th>Correct</th><th>Accuracy</th></tr>'+tab+'</table><p>All references and generated roles are open. No model, training or protected payload was used.</p>',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('shortcut_curves','calibration_modes')},indent=2))
    return 0 if not failures else 2
if __name__=='__main__':raise SystemExit(main())
