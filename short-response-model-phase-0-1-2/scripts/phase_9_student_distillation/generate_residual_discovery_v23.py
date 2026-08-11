from __future__ import annotations

"""Generate v2.3 structurally-held-out direct reasoning discovery/eval corpus.

The v2.2 multidomain benchmark mixed answerable and abstention performance in a
way that allowed an abstention adapter to look like a broad reasoning gain.
v2.3 therefore separates answerable reasoning from epistemic restraint and uses
split-disjoint surface representations. All gold answers are deterministic and
prompt-contained; no visible chain-of-thought is stored.
"""

import argparse, hashlib, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA="2.3"
SOURCE="deterministic_phase9_residual_discovery_v23"
ABSTAIN="I don't know based on the available information."
SYSTEM=("Answer concisely using only the information in the prompt. Reason internally and return only the short final answer. "
        f'If the requested answer cannot be determined, reply exactly: "{ABSTAIN}"')
DOMAINS=["general","research","writing","communication","translation","coding","devops","systems","math","data_analysis","science","engineering","education","business","operations","finance","legal","health","consumer","travel","home","creative","media","personal","coaching","tools","agents"]
OPS=["compare_metric","filter_then_rank","aggregate_delta","capacity_fit","state_transition","dependency_order","evidence_binding","rule_exception","multi_hop_relation","diagnostic_priority","insufficient_information","conflicting_evidence"]
NAMES=["Atlas","Beacon","Cedar","Delta","Ember","Falcon","Grove","Harbor"]
SPLIT_FORMATS={
    "discovery":["prose","bullets","compact_table","key_value"],
    "dev":["email","chat","ticket"],
    "ood_dev":["json","yaml","cli","log"],
    "fresh_sealed":["xml_like","config","mixed_document"],
}


def write_jsonl(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")

def write_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()

def split_for(j:int)->str:
    if j<35:return "discovery"
    if j<50:return "dev"
    if j<65:return "ood_dev"
    return "fresh_sealed"

def difficulty_for(j:int)->str:
    return ("easy","moderate","moderate","hard","hard")[j%5]

def punct(x:str)->str:
    x=x.strip();return x if x.endswith((".","!","?")) else x+"."

def render(fmt:str,domain:str,scenario:str,facts:list[str],question:str)->str:
    title=f"{domain.replace('_',' ').title()} case {scenario}"
    if fmt=="prose":return title+": "+" ".join(facts)+" "+question
    if fmt=="bullets":return title+"\n"+"\n".join(f"- {x}" for x in facts)+f"\nQuestion: {question}"
    if fmt=="compact_table":return title+"\n| item | detail |\n|---|---|\n"+"\n".join(f"| {i+1} | {x} |" for i,x in enumerate(facts))+f"\nQuery: {question}"
    if fmt=="key_value":return title+"\n"+"\n".join(f"fact_{i+1}={x}" for i,x in enumerate(facts))+f"\nquery={question}"
    if fmt=="email":return f"Subject: {title}\nHi,\n"+" ".join(facts)+f"\nCould you answer this briefly: {question}\nThanks."
    if fmt=="chat":return f"[user] {title}\n"+"\n".join(f"[note] {x}" for x in facts)+f"\n[user] {question}"
    if fmt=="ticket":return f"TICKET {scenario}\nAREA={domain}\nDETAILS:\n"+"\n".join(f"* {x}" for x in facts)+f"\nREQUEST: {question}"
    if fmt=="json":return json.dumps({"case":scenario,"domain":domain,"facts":facts,"question":question},ensure_ascii=False,sort_keys=True)
    if fmt=="yaml":return "case: "+scenario+"\ndomain: "+domain+"\nfacts:\n"+"\n".join(f"  - {x}" for x in facts)+f"\nquestion: {question}"
    if fmt=="cli":return "$ inspect-case --id "+scenario+"\n"+"\n".join(f"FACT[{i+1}] {x}" for i,x in enumerate(facts))+f"\nQUERY> {question}"
    if fmt=="log":return f"2026-08-11 INFO case={scenario} domain={domain}\n"+"\n".join(f"2026-08-11 DATA {x}" for x in facts)+f"\n2026-08-11 QUERY {question}"
    if fmt=="xml_like":return f"<case id=\"{scenario}\" domain=\"{domain}\">\n"+"\n".join(f"  <fact>{x}</fact>" for x in facts)+f"\n  <question>{question}</question>\n</case>"
    if fmt=="config":return f"[case]\nid={scenario}\ndomain={domain}\n"+"\n".join(f"evidence_{i+1}={x}" for i,x in enumerate(facts))+f"\nrequest={question}"
    return f"DOC-A {title}\n"+"\n".join(f"A{i+1}: {x}" for i,x in enumerate(facts[:2]))+"\n---\nDOC-B\n"+"\n".join(f"B{i+1}: {x}" for i,x in enumerate(facts[2:]))+f"\nTASK: {question}"

def contract(answer:str)->dict[str,Any]:
    return {"schema_version":"4.5","open_ended":False,"acceptable_modes":[],"requirements":[{"id":"verified_short_answer","label":"explicit_required_content_missing","description":f"Reply must equal the deterministic answer: {answer.rstrip('.')}","severity":"major","weight":1.0,"match":{"any_phrases":[answer.rstrip('.')]}}],"prohibitions":[]}

def make_semantics(domain:str,op:str,j:int)->tuple[list[str],str,str,str,dict[str,Any]]:
    a,b,c=NAMES[(j+len(domain))%8],NAMES[(j*3+1)%8],NAMES[(j*5+2)%8]
    if len({a,b,c})<3:a,b,c=NAMES[j%8],NAMES[(j+2)%8],NAMES[(j+5)%8]
    base=17+(j*7+len(domain))%37
    decision="answer";v:dict[str,Any]={"operation":op,"domain":domain}
    if op=="compare_metric":
        vals={a:base+8,b:base+1,c:base+13};direction="lowest" if j%2==0 else "highest";winner=min(vals,key=vals.get) if direction=="lowest" else max(vals,key=vals.get)
        facts=[f"{k} has score {x}." for k,x in vals.items()];q=f"Which option has the {direction} score?";ans=punct(winner);v|={"values":vals,"winner":winner,"direction":direction}
    elif op=="filter_then_rank":
        floor=base+4;vals={a:(floor+4,120+j%17),b:(floor-3,78+j%19),c:(floor+8,96+j%23)};valid=[k for k,(q,_) in vals.items() if q>=floor];winner=min(valid,key=lambda k:vals[k][1])
        facts=[f"Minimum quality is {floor}."]+[f"{k}: quality {q}, cost {cost}." for k,(q,cost) in vals.items()];q="Keep only qualifying options, then choose the lowest cost.";ans=punct(winner);v|={"floor":floor,"values":vals,"winner":winner}
    elif op=="aggregate_delta":
        start=base+20;adds=[3+j%8,2+(j*2)%7];remove=1+j%6;res=start+sum(adds)-remove
        facts=[f"Initial amount is {start}.",f"Add {adds[0]}.",f"Add {adds[1]}.",f"Remove {remove}."];q="What is the final amount?";ans=f"{res}.";v|={"result":res}
    elif op=="capacity_fit":
        need=40+j%55;caps={a:need-5,b:need+4,c:need+19};winner=min((k for k in caps if caps[k]>=need),key=lambda k:caps[k])
        facts=[f"Required capacity is {need}."]+[f"{k} capacity is {x}." for k,x in caps.items()];q="Which is the smallest sufficient option?";ans=punct(winner);v|={"need":need,"caps":caps,"winner":winner}
    elif op=="state_transition":
        seq=["start","pause","resume"] if j%2==0 else ["start","finish"];state="queued";trans={("queued","start"):"running",("running","pause"):"paused",("paused","resume"):"running",("running","finish"):"done"}
        for x in seq:state=trans[(state,x)]
        facts=["queued + start -> running.","running + pause -> paused.","paused + resume -> running.","running + finish -> done.","Initial state is queued.","Actions: "+", ".join(seq)+"."];q="What is the final state?";ans=punct(state);v|={"sequence":seq,"final":state}
    elif op=="dependency_order":
        facts=[f"{a} must happen before {b}.",f"{b} must happen before {c}."];q=f"Which must happen first among {a}, {b}, and {c}?";ans=punct(a);v|={"winner":a}
    elif op=="evidence_binding":
        target=1+j%3;vals={1:a,2:b,3:c};facts=[f"record 1 = {a}.",f"record 2 = {b}.",f"record 3 = {c}."];q=f"Return the value for record {target}, not a sibling record.";ans=punct(vals[target]);v|={"target":target,"winner":vals[target]}
    elif op=="rule_exception":
        cond="priority" if j%2==0 else "weekend";applies=j%3!=0;context=cond if applies else "ordinary";winner=b if applies else a
        facts=[f"Default option is {a}.",f"If the case is {cond}, override with {b}.",f"Current case type is {context}."];q="Which option applies?";ans=punct(winner);v|={"winner":winner,"condition":cond,"context":context}
    elif op=="multi_hop_relation":
        p1,p2,p3="Ari","Bea","Chen";facts=[f"{p1} assigned the task to {p2}.",f"{p2} assigned it to {p3}."];q=f"Who received the task from the person who received it from {p1}?";ans=punct(p3);v|={"winner":p3}
    elif op=="diagnostic_priority":
        facts=["Rule: if power is absent, inspect power first.","Rule: if power is present but network is absent, inspect network first.","Power is present.","Network is absent."];q="What should be inspected first?";ans="Network.";v|={"winner":"Network"}
    elif op=="insufficient_information":
        decision="abstain";facts=[f"{a} score is {base}.",f"{b} score is {base+2}.",f"No score is provided for {c}."];q=f"Which of {a}, {b}, and {c} has the highest score?";ans=ABSTAIN;v|={"missing":c}
    else:
        decision="abstain";facts=[f"Equal-authority source A says the selected option is {a}.",f"Equal-authority source B says the selected option is {b}.","No precedence or freshness rule is provided."];q="What is the single supported selection?";ans=ABSTAIN;v|={"conflict":[a,b]}
    return facts,q,ans,decision,v

def make_row(domain:str,op:str,j:int,rng:random.Random)->dict[str,Any]:
    split=split_for(j);fmt=SPLIT_FORMATS[split][(j+len(domain)+len(op))%len(SPLIT_FORMATS[split])];facts,q,answer,decision,verification=make_semantics(domain,op,j)
    scenario=hashlib.sha1(f"{domain}|{op}|{j}|23".encode()).hexdigest()[:10];prompt=render(fmt,domain,scenario,facts,q)
    difficulty=difficulty_for(j)
    return {"id":f"v23_{domain}_{op}_{j:03d}","dataset_schema_version":SCHEMA,"source":SOURCE,"mode":"residual_reasoning","domain":domain,"reasoning_family":op,"surface_format":fmt,"difficulty":difficulty,"decision":decision,"answer_stance":"uncertain" if decision=="abstain" else "direct","answer":answer,"approved_answer":answer,"answer_aliases":[],"intended_split":split,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":answer}],"response_contract":contract(answer),"review_status":"programmatically_verified","scenario_flags":["v23_structural_holdout",domain,op,fmt,difficulty],"teacher_provenance":{"kind":"deterministic_verifier","generator":SOURCE},"verification":verification}

def stratified(rows:list[dict[str,Any]],limit:int,seed:int)->list[dict[str,Any]]:
    buckets:dict[tuple[str,str,str],list[dict[str,Any]]]=defaultdict(list)
    for r in rows:buckets[(r["domain"],r["reasoning_family"],r["decision"])].append(r)
    rng=random.Random(seed)
    for b in buckets.values():rng.shuffle(b)
    out=[];keys=sorted(buckets)
    while len(out)<limit and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out)<limit:out.append(buckets[k].pop())
    return out

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--output-root",type=Path,required=True);ap.add_argument("--seed",type=int,default=23031);ap.add_argument("--force",action="store_true");a=ap.parse_args()
    if a.output_root.exists():
        if not a.force:raise FileExistsError(f"Output exists: {a.output_root}; use --force")
        shutil.rmtree(a.output_root)
    a.output_root.mkdir(parents=True)
    rng=random.Random(a.seed);rows=[make_row(d,o,j,rng) for d in DOMAINS for o in OPS for j in range(75)]
    splits={s:[r for r in rows if r["intended_split"]==s] for s in ("discovery","dev","ood_dev","fresh_sealed")}
    write_jsonl(a.output_root/"canonical.jsonl",rows)
    for s,rs in splits.items():write_jsonl(a.output_root/"splits"/f"{s}.reviewed.jsonl",rs)
    write_jsonl(a.output_root/"probes"/"dev_probe.reviewed.jsonl",stratified(splits["dev"],720,a.seed+1));write_jsonl(a.output_root/"probes"/"ood_probe.reviewed.jsonl",stratified(splits["ood_dev"],720,a.seed+2))
    prompts=[next(m["content"] for m in r["messages"] if m["role"]=="user") for r in rows]
    counts={s:len(rs) for s,rs in splits.items()};fmt_by_split={s:sorted({r["surface_format"] for r in rs}) for s,rs in splits.items()}
    manifest={"schema_version":SCHEMA,"source":SOURCE,"seed":a.seed,"canonical_rows":len(rows),"domains":len(DOMAINS),"domain_names":DOMAINS,"operations":len(OPS),"operation_names":OPS,"split_counts":counts,"decision_counts":dict(Counter(r["decision"] for r in rows)),"difficulty_counts":dict(Counter(r["difficulty"] for r in rows)),"format_families_by_split":fmt_by_split,"unique_prompt_count":len(set(prompts)),"answerable_rows":sum(r["decision"]=="answer" for r in rows),"abstention_rows":sum(r["decision"]=="abstain" for r in rows),"design":"split-disjoint surface representations; prompt-contained deterministic reasoning; explicit answerable vs abstention metrics; no visible chain-of-thought"}
    write_json(a.output_root/"manifests"/"dataset_manifest.json",manifest)
    tracked=[a.output_root/"canonical.jsonl",*(a.output_root/"splits"/f"{s}.reviewed.jsonl" for s in splits),a.output_root/"probes"/"dev_probe.reviewed.jsonl",a.output_root/"probes"/"ood_probe.reviewed.jsonl"]
    write_json(a.output_root/"manifests"/"file_hashes.json",{str(p.relative_to(a.output_root)).replace('\\','/'):sha256(p) for p in tracked})
    (a.output_root/"SEALED_DATA_POLICY.md").write_text("# v2.3 fresh sealed policy\n\n`fresh_sealed.reviewed.jsonl` is generated now but MUST NOT be used for mining, training, dose calibration, or DEV selection. It is evaluated exactly once after the v2.3 winner is frozen. The previously consumed v2.2 sealed TEST is not reused for v2.3 tuning.\n",encoding="utf-8")
    (a.output_root/"README.md").write_text(f"# v2.3 Structural-OOD Residual Discovery Corpus\n\n- Canonical: **{len(rows):,}**\n- Domains: **{len(DOMAINS)}**\n- Operations: **{len(OPS)}**\n- Discovery / DEV / OOD-DEV / fresh sealed: **{counts['discovery']:,} / {counts['dev']:,} / {counts['ood_dev']:,} / {counts['fresh_sealed']:,}**\n- Answerable: **{manifest['answerable_rows']:,}**\n- Abstention/conflict: **{manifest['abstention_rows']:,}**\n\nSurface representation families are disjoint by split so OOD evaluation is not merely a new random seed over the same prompt template.\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__":main()
