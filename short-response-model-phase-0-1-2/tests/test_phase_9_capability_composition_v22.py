from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.derive_layer_plan_v22 import derive
from scripts.phase_9_student_distillation.select_capability_composition_v22 import select

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'data/phase_9_student_distillation/multidomain_reasoning_v22'
CONFIG=ROOT/'configs/phase_9_capability_composition_v22.json'


def read_jsonl(path:Path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def write(path:Path,obj): path.write_text(json.dumps(obj),encoding='utf-8'); return str(path)

def reasoning(overall=.40,macro=.40,hard=.30,abst=.80,short=1.0,domain=True):
    d={'evaluation_valid':True,'overall_correctness':overall,'hard_correctness':hard,'abstention_correctness':abst,'short_usable_rate':short}
    if domain: d['macro_domain_correctness']=macro
    else: d['macro_family_correctness']=macro
    return d

def rag(proc=.95,supported=.63,faith=.98,abst=.81,conflict=1.0,social=1.0):
    return {'evaluation_valid':True,'conditions':{'real_rag':{'procedure_correctness':proc,'supported_answer_correctness':supported,'evidence_faithfulness':faith,'correct_abstention_rate':abst,'conflict_safe_handling_rate':conflict,'social_usable_rate':social}}}

def entry(tmp_path:Path,name:str,vals:dict,track='x'):
    root=tmp_path/name; root.mkdir()
    return {'name':name,'track':track,'model':name,'adapter':None,
      'multidomain_summary':write(root/'md.json',vals['md']),
      'v212_summary':write(root/'v212.json',vals['v212']),
      'original_rag_summary':write(root/'orig.json',vals['orig']),
      'procedural_summary':write(root/'proc.json',vals['proc'])}

def test_packaged_v22_corpus_counts_and_train_only_specialists():
    m=json.loads((CORPUS/'manifests/dataset_manifest.json').read_text())
    assert m['canonical_rows']==40500 and m['domains']==27 and m['archetypes']==15
    assert m['split_counts']=={'train':32400,'dev':4050,'test':4050}
    expected={'track_a_broad_general':15000,'track_b_anchored_mixed':15000,'specialist_general':10000,'specialist_grounding':5500,'specialist_procedure':5500,'specialist_abstention':4500,'specialist_social':4500}
    assert m['curriculum_view_counts']==expected
    train_ids={r['id'] for r in read_jsonl(CORPUS/'reviewed_splits/train.reviewed.jsonl')}
    devtest={r['id'] for s in ('dev','test') for r in read_jsonl(CORPUS/f'reviewed_splits/{s}.reviewed.jsonl')}
    assert not train_ids & devtest
    for name in expected:
        rows=read_jsonl(CORPUS/f'curriculum_views/{name}.reviewed.jsonl')
        assert all(r.get('messages') and r['messages'][-1]['role']=='assistant' for r in rows)
        # Generated v2.2 rows in any specialist view must come from TRAIN only.
        assert all(r.get('source')!='deterministic_phase9_multidomain_reasoning_v22' or r['id'] in train_ids for r in rows)
    social_source=read_jsonl(CORPUS/'curriculum_sources/social_synthetic_train.jsonl')
    assert len(social_source)==3000
    assert all(r['intended_split']=='train' and r['decision']=='social' for r in social_source)
    social_view=read_jsonl(CORPUS/'curriculum_views/specialist_social.reviewed.jsonl')
    assert sum(r.get('source')=='deterministic_phase9_social_rehearsal_v22' for r in social_view)>=2500
    anchored=read_jsonl(CORPUS/'curriculum_views/track_b_anchored_mixed.reviewed.jsonl')
    sources={r.get('source') for r in anchored}
    assert 'deterministic_phase9_multidomain_reasoning_v22' in sources
    assert 'deterministic_phase9_rag_reasoning_v3_large_v175' in sources
    assert 'deterministic_phase9_procedural_recovery_v211' in sources

def test_v22_targets_are_short_and_no_visible_cot():
    rows=read_jsonl(CORPUS/'reviewed_splits/dev.reviewed.jsonl')
    assert len({r['domain'] for r in rows})==27
    assert len({r['reasoning_family'] for r in rows})==15
    for row in rows[:500]:
        assert row['messages'][-1]['content']==row['approved_answer']
        assert len(row['approved_answer'].split())<=12
        assert 'rationale' not in row and 'chain_of_thought' not in row

def test_layer_plan_is_nonoverlapping_and_middle_biased():
    plan=derive(28,.60,['grounding','procedure','general','abstention','social'])
    all_layers=[x for v in plan.values() for x in v]
    assert len(all_layers)==len(set(all_layers))
    assert min(all_layers)>0 and max(all_layers)<27
    assert all(plan[k] for k in plan)

def test_selector_rejects_missing_original_preservation_metrics(tmp_path:Path):
    cfg=json.loads(CONFIG.read_text())
    base={'md':reasoning(.10,.10,.05,.5), 'v212':reasoning(.10,.10,.05,.5,domain=False), 'orig':rag(), 'proc':rag(proc=.02)}
    bad_orig={'evaluation_valid':True,'conditions':{'real_rag':{'procedure_correctness':.90}}}
    cand={'md':reasoning(.70,.70,.60,.9),'v212':reasoning(.60,.60,.5,.9,domain=False),'orig':bad_orig,'proc':rag(proc=.20)}
    manifest={'candidates':[entry(tmp_path,'frozen_anchor',base,'baseline'),entry(tmp_path,'candidate',cand)]}
    out=select(cfg,manifest)
    assert out['selected_candidate']=='frozen_anchor'
    packet=next(x for x in out['candidates'] if x['name']=='candidate')
    assert any(x.startswith('missing_protected_metric:') for x in packet['technical_invalidity'])

def test_selector_can_accept_broad_gain_with_preservation(tmp_path:Path):
    cfg=json.loads(CONFIG.read_text())
    base={'md':reasoning(.10,.10,.05,.5), 'v212':reasoning(.10,.10,.05,.5,domain=False), 'orig':rag(), 'proc':rag(proc=.02)}
    good={'md':reasoning(.55,.54,.45,.8),'v212':reasoning(.45,.45,.4,.8,domain=False),'orig':rag(proc=.95,supported=.64,faith=.975,abst=.80),'proc':rag(proc=.08)}
    manifest={'candidates':[entry(tmp_path,'frozen_anchor',base,'baseline'),entry(tmp_path,'good',good)]}
    out=select(cfg,manifest)
    assert out['selected_candidate']=='good'

def test_train_student_exposes_anchor_and_layer_scoped_lora_flags():
    text=(ROOT/'scripts/phase_9_student_distillation/train_student.py').read_text(encoding='utf-8')
    for flag in ('--anchor-model','--anchor-kl-weight','--anchor-temperature','--lora-layer-indices'):
        assert flag in text
    assert 'anchor_kl_loss' in text
    assert 'discover_target_modules(model, selected_layers' in text

def test_v22_runner_contains_all_four_tracks_and_sealed_gate():
    text=(ROOT/'scripts/phase_9_student_distillation/run_phase_9_capability_composition_v22.ps1').read_text(encoding='utf-8')
    for token in ('Track A','Track B','Track C','Weighted composition','Layer-separated','Greedy adapter soup','RunSealedTest','AnchorDevice','CAPABILITY_COMPOSITION_V22_SELECTION.md'):
        assert token in text


def test_greedy_probe_sampling_covers_every_v22_domain_and_family():
    from scripts.phase_9_student_distillation.build_capability_probe_v22 import sample
    rows=read_jsonl(CORPUS/'reviewed_splits/dev.reviewed.jsonl')
    probe=sample(rows,540,('domain','reasoning_family','decision'),22045)
    assert len(probe)==540
    assert len({r['domain'] for r in probe})==27
    assert len({r['reasoning_family'] for r in probe})==15
    assert {r['decision'] for r in probe}=={'answer','abstain'}


def test_weighted_composer_scales_only_lora_b_tensors(tmp_path:Path):
    import torch
    from safetensors.torch import save_file, load_file
    from scripts.phase_9_student_distillation.compose_lora_adapters_v22 import scale_adapter
    src=tmp_path/'src'; dst=tmp_path/'dst'; src.mkdir()
    save_file({
        'base_model.model.layers.1.self_attn.q_proj.lora_A.weight': torch.tensor([[2.0]]),
        'base_model.model.layers.1.self_attn.q_proj.lora_B.weight': torch.tensor([[3.0]]),
    }, str(src/'adapter_model.safetensors'))
    rec=scale_adapter(src,dst,0.25)
    tensors=load_file(str(dst/'adapter_model.safetensors'))
    assert torch.equal(tensors['base_model.model.layers.1.self_attn.q_proj.lora_A.weight'], torch.tensor([[2.0]]))
    assert torch.equal(tensors['base_model.model.layers.1.self_attn.q_proj.lora_B.weight'], torch.tensor([[0.75]]))
    assert rec['scaled_lora_B_tensors']==1
