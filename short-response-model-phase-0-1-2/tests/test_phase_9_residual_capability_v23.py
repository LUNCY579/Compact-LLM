from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.phase9_common import select_retrieved_passages
from scripts.phase_9_student_distillation.select_residual_capability_v23 import select

ROOT=Path(__file__).resolve().parents[1]
DIRECT=ROOT/'data/phase_9_student_distillation/residual_capability_v23/direct_reasoning'
RAG=ROOT/'data/phase_9_student_distillation/residual_capability_v23/procedural_rag'
CONFIG=ROOT/'configs/phase_9_residual_capability_v23.json'

def read_jsonl(p:Path):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def write(p:Path,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o),encoding='utf-8');return str(p)

def test_packaged_v23_counts_and_structural_holdout():
    dm=json.loads((DIRECT/'manifests/dataset_manifest.json').read_text());rm=json.loads((RAG/'manifests/dataset_manifest.json').read_text())
    assert dm['canonical_rows']==24300 and dm['domains']==27 and dm['operations']==12
    assert dm['split_counts']=={'discovery':11340,'dev':4860,'ood_dev':4860,'fresh_sealed':3240}
    assert dm['answerable_rows']==20250 and dm['abstention_rows']==4050
    assert rm['canonical_rows']==7200 and rm['topic_count']==1200 and rm['families']==8
    assert rm['split_counts']=={'discovery':4320,'dev':1080,'ood_dev':1080,'fresh_sealed':720}
    rows=read_jsonl(DIRECT/'canonical.jsonl');assert len({next(m['content'] for m in r['messages'] if m['role']=='user') for r in rows})==len(rows)
    formats={s:{r['surface_format'] for r in rows if r['intended_split']==s} for s in dm['split_counts']}
    names=list(formats)
    for i,a in enumerate(names):
        for b in names[i+1:]:assert not formats[a]&formats[b]

def test_v23_rag_topics_never_cross_splits_and_answerable_have_support():
    rows=read_jsonl(RAG/'canonical.jsonl');seen={}
    for r in rows:
        seen.setdefault(r['topic_id'],set()).add(r['intended_split'])
        if r['decision']=='answer':assert r['support_document_ids']
    assert all(len(v)==1 for v in seen.values())
    assert len(read_jsonl(RAG/'calibration/discovery_calibration.reviewed.jsonl'))==1200

def test_query_aware_v23_fails_closed_on_missing_explicit_record():
    q='For audit ABC123, record 5, what is the result?'
    passages=[
        {'record_id':'a','title':'audit ABC123','text':'Audit ABC123, record 1. value one','normalized_score':.9,'rank':1},
        {'record_id':'b','title':'audit ABC123','text':'Audit ABC123, record 2. value two','normalized_score':.8,'rank':2},
    ]
    # Historical policy intentionally keeps siblings when no exact record exists.
    assert select_retrieved_passages(q,passages,selected_top_k=3,kind='knowledge',policy='query_aware_v21')
    # v2.3 policy treats explicit wrong-record siblings as missing evidence.
    assert select_retrieved_passages(q,passages,selected_top_k=3,kind='knowledge',policy='query_aware_v23')==[]

def direct(ans=.1,hard=.1,macro=.1,abst=.9,false=.1,short=1.0):
    return {'evaluation_valid':True,'answerable_correctness':ans,'answerable_hard_correctness':hard,'macro_answerable_domain_correctness':macro,'abstention_correctness':abst,'false_abstention_rate':false,'short_usable_rate':short}
def rag(proc=.95,sup=.65,faith=.97,abst=.9,conf=1.0,social=1.0):
    return {'evaluation_valid':True,'conditions':{'real_rag':{'procedure_correctness':proc,'supported_answer_correctness':sup,'evidence_faithfulness':faith,'correct_abstention_rate':abst,'conflict_safe_handling_rate':conf,'social_usable_rate':social}}}
def eval_manifest(tmp:Path,name:str,d,do,orig,hp,vp,vpo):
    root=tmp/name;root.mkdir()
    vals={'v23_direct_summary':d,'v23_direct_ood_summary':do,'original_rag_summary':orig,'historical_procedural_summary':hp,'v23_procedural_summary':vp,'v23_procedural_ood_summary':vpo}
    man={}
    for k,v in vals.items():man[k]=write(root/(k+'.json'),v)
    return write(root/'manifest.json',man)

def test_v23_selector_rewards_answerable_gain_without_hiding_preservation(tmp_path:Path):
    cfg=json.loads(CONFIG.read_text())
    base={'name':'greedy_anchor','track':'base','model':'g','adapter':None,'evaluation_manifest':eval_manifest(tmp_path,'b',direct(.1,.1,.1,.9,.1),direct(.08,.08,.08,.9,.1),rag(),rag(.0),rag(.25),rag(.20))}
    good={'name':'good','track':'residual','model':'x','adapter':None,'evaluation_manifest':eval_manifest(tmp_path,'g',direct(.25,.22,.24,.9,.08),direct(.20,.18,.19,.9,.08),rag(proc=.95,sup=.66,faith=.965,abst=.91),rag(.04),rag(.32),rag(.28))}
    out=select(cfg,{'candidates':[base,good]})
    assert out['selected_candidate']=='good'

def test_train_student_supports_row_specific_anchor_kl_multiplier():
    text=(ROOT/'scripts/phase_9_student_distillation/train_student.py').read_text(encoding='utf-8')
    assert 'v23_anchor_kl_multiplier' in text
    assert 'anchor_kl_effective_weight' in text

def test_v23_runner_is_end_to_end_and_uses_fresh_sealed_only():
    text=(ROOT/'scripts/phase_9_student_distillation/run_phase_9_residual_capability_v23.ps1').read_text(encoding='utf-8')
    for token in ('Gold-verified residual','Anchored answerable-reasoning residual','Anchored procedural residual','[Dose]','Adapter interference map','Oracle routing upper bound','RunFreshSealedTest','fresh_sealed_selected','query_aware_v23'):
        assert token in text
    assert 'OriginalTest' not in text and 'V211Test' not in text

def test_training_mix_caps_scarce_residual_and_backfills_preservation():
    from scripts.phase_9_student_distillation.build_residual_training_mix_v23 import plan_mix_counts
    # Matches the live mining outcome: 126 procedure residuals cannot supply 2400 draws at repeat<=8.
    plan = plan_mix_counts(total=5000, residual_fraction=0.48, residual_pool=126, preservation_pool=1910, max_repeat=8)
    assert plan['residual_rows_sampled'] == 126 * 8
    assert plan['preservation_rows_sampled'] == 5000 - 126 * 8
    assert plan['residual_capped_by_repeat'] is True
    assert plan['total'] == 5000
