"""Create the P0 decision from the completed, restricted train/dev audit."""
import datetime, hashlib, json, subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
WORKTREE=OUT.parents[3]
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
receipt=read('partial_audit_receipt.json');structure=read('train_dev_structure.json');shortcuts=read('train_dev_shortcuts.json');native=read('native_train_dev_profile.json');verification=read('train_dev_verification.json')
rows=read('native_train_dev_rows.json')
failures=[r for r in rows if not r['band_fit']]
producer={x['path']:x for x in receipt['explicit_input_inventory']}
decision={
 'schema':'multi_reasoning_scout.p0_acceptance/1',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'P0_STOP_NOT_ACCEPTED',
 'corpus_accepted':False,'P1_training_contract_open':False,'P2_training_authorized':False,
 'model_calls':0,'training_runs':0,
 'dataset':{'path':receipt['dataset_root'],'version':'1.1.0-full-candidate','intended_rows':6000,
   'manifest_sha256':receipt['manifest_sha256'],
   'source_spec_sha256':producer['provenance/ATTACHED_SPEC.md']['sha256'],
   'producer_run_binding_sha256':producer['reports/RUN_BINDING.json']['sha256'],
   'source_lineage':'User-confirmed delivered directory; producer tokenizer lock cites LUNCY579/ChatBubble@fbec90a0d0653aeb745d5223137ae3cfdd9c48dd experiments/model_quality_gpu_v1/source_lock.json, blob 50dd384c2eac68385634599012bdf068188f04fa. This is tokenizer source lineage, not independent attestation of corpus authorship.',
   'archive_hash':None,'archive_hash_status':'Directory supplied; no archive opened or identified'},
 'audit_source':{'worktree':str(WORKTREE),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=WORKTREE,text=True).strip(),
   'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
   'script':'audit_train_dev.py','script_sha256':receipt['source_audit_sha256'],'runtime':receipt['runtime'],'python':receipt['python'],'wall_seconds':receipt['run_seconds']},
 'access_boundary':{'payload_splits_read':['train','dev'],'rows_read':5400,'holdout_payload_reads':0,'holdout_payload_hashes_recomputed':0,
   'all_jsonl_read':False,'sealed_or_ANY_READ_payloads_read':0,'confirmation_sets_consumed':0,'aggregate_producer_metadata_read':True,
   'full_package_integrity_verified':False,'reason':'Holdout and all.jsonl deliberately excluded; full-package scripts were not run.'},
 'requirements':{
   'permitted_file_integrity':{'status':'PASS_SCOPED','all_manifest_matches':True,'files':len(receipt['integrity'])},
   'rows_and_format':{'status':'PASS_TRAIN_DEV_ONLY','train':4800,'dev':600,'unique_groups':5400,'exact_sft_projection':True,'roles':['system','user','assistant'],'maximum_answer_words':1,'holdout_rows':{'producer_reported':600,'freshly_verified':False}},
   'independent_answer_verification':{'status':'PASS_TRAIN_DEV_ONLY','rows':5400,'parsed_facts':75073,'answer_or_reason_disagreements':0,'scope':verification['independence']},
   'tokenizer_compatibility':{'status':'PASS_EXACT_FOUR_ASSETS_AND_SMOKE','transformers':'4.57.1','tokenizers':'0.22.2','runtime_class':'Qwen2TokenizerFast','enable_thinking':False,'trust_remote_code':False,'weights_read':0},
   'rendered_window_and_reserve':{'status':'PASS_TRAIN_DEV_ONLY','rows':5400,'max_prompt_tokens':1470,'max_total_tokens':1476,'prompt_over_1920':0,'total_over_2048':0,'truncations':0,'filtered_rows':0,'holdout':'UNVERIFIED'},
   'intended_context_bands':{'status':'FAIL','violations':41,'train_violations':4,'dev_violations':37,'all_violations_band':'short','required_short_max':600,'endpoint_policy':'Literal inclusive ranges; two shared-endpoint rows separately flagged, no endpoint ambiguity can explain the 41 violations.','case_ids':[r['id'] for r in failures]},
   'train_dev_isolation':{'status':'PASS_FOR_IDS_ENTITIES_WHOLE_TEXT_ONLY','overlaps':receipt['overlap']},
   'strict_template_and_structural_isolation':{'status':'NOT_ESTABLISHED','fresh_shared_name_blind_sentence_forms':392,'typed_graph_count_signature_intersection':0,'limitation':'IDs and coarse typed graph-count signatures are not proof of disjoint reduced solution structure.'},
   'answerability':{'status':'MEASURED','unknown_train':1440,'unknown_dev':180,'rate_each_split':0.3,'reasons_each_train':360,'reasons_each_dev':45,'literal_unknown':'UNKNOWN'},
   'question_only_answerability':{'status':'MEASURED_TRAIN_FIT_DEV_CHECK','method':shortcuts['question_only']['classifier'],'balanced_accuracy':receipt['question_only'],'significance_claim':False},
   'surface_shortcuts':{'status':'MEASURED_FINITE_FEATURES','train_selected_feature':shortcuts['train_selected_feature'],'balanced_accuracy':shortcuts['train_selected_feature_scores'],'feature_count':len(shortcuts['surface_thresholds']),'significance_claim':False},
   'nontrivial_abstention':{'status':'FAIL_QUALIFICATION_NOT_ESTABLISHED','cheap_selector_plus_validity_balanced_accuracy':receipt['semantic_shortcut'],'scope':'Color/location join plus explicit withdrawal/nonissuance checks; no task answer computation needed.'},
   'majority_copy_first_last_heuristics':{'status':'MEASURED_SUBSET','results':shortcuts['exact_answer_baselines'],'limitations':shortcuts['limitations']},
   'six_dimensional_balance':{'status':'MARGINALS_RECOMPUTED_TRAIN_DEV; FULL_CONDITIONAL_GATE_NOT_REEXECUTED','splits':structure['splits'],'producer_literal_within_band_max_relative_error':0.05263157894736836,'producer_status':'NOT_MET_UNDER_LITERAL_RELATIVE_INTERPRETATION','fresh_global_6000_verification':False},
   'evidence_position_and_difficulty':{'status':'PARTIAL_ONLY','span_bounds_and_unknown_evidence_contract':'RECOMPUTED_IN_VALIDATE_ROW','all_deletion_and_near_miss_difficulty_checks':'NOT_REEXECUTED; producer reports scoped passes only'},
   'external_quarantine_contamination':{'status':'UNVERIFIED','overlap_count':None,'protected_payload_reads':0,'required_action':'Authorized independent blind exclusion against protected ID/hash/structure registries, returning aggregate receipts only.'},
   'frozen_scout_nonoverlap':{'status':'UNVERIFIED','required_action':'Compare candidate registry against final frozen scout IDs/prompt hashes and structural manifest without changing the scout.'},
   'complete_sha256_inventory':{'status':'MANIFEST_PRESENT; PERMITTED_SUBSET_FRESHLY_VERIFIED','holdout':'Manifest claims retained as unverified producer metadata; no protected hash recomputation.'}},
 'minimum_repair':[
   'Publish a new immutable dataset revision with native-token-constrained rendering correcting at least the 41 observed short-band violations; do not silently truncate, relabel, filter, or change the registered training window.',
   'Replace split-shared local template/solution structures where strict structural isolation is binding; demonstrate canonical structural exclusion rather than only disjoint IDs.',
   'Repair easily separable UNKNOWN mechanisms so the registered nontrivial-abstention requirement is demonstrably met; preserve literal UNKNOWN and balanced answerability.',
   'Resolve the producer-reported within-band +/-3% failure through a prospective feasible allocation or explicitly authorized rounding convention before regeneration; do not silently weaken the requirement.',
   'Obtain aggregate, hash-bound independent external quarantine/scout exclusion and authorized holdout integrity/token/split receipts without opening protected payloads to this agent.',
   'Rerun P0 on the immutable repaired revision. Only after all binding gates pass may P1 freeze a training recipe; no training has been performed.'],
 'claims':{'software_audit_completed':True,'native_tokenizer_execution_pass':True,'native_model_execution':False,'training_input_accepted':False,'model_quality_improvement':False,'confirmation':False,'release':False},
 'artifacts':{}
}
for p in sorted(OUT.iterdir()):
    if p.is_file():decision['artifacts'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with (OUT/'DATASET_ACCEPTANCE.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(decision,f,indent=2,ensure_ascii=False);f.write('\n')
md='''# Dataset P0 acceptance: STOP — not accepted

The user-confirmed `Rineya_LCR_Full_v1_6000` candidate is not a qualified training input. **P0 fails; P1/P2 stay closed and no training ran.** Fresh checks resolve the missing-tokenizer question for train/dev and expose 41 actual token-band violations. Structural isolation, nontrivial abstention and external exclusion remain unqualified.

Audit scope was exactly 4,800 train and 600 dev rows. Holdout, `data/all.jsonl`, sealed, ANY_READ and confirmation payloads were neither read nor hashed. Producer aggregate metadata and manifests were read; their holdout claims remain unverified. No whole-package verifier or test command that could open holdout was run.

| Requirement | Fresh evidence | Decision |
|---|---|---|
| Permitted-file integrity | Train/dev data, messages-only SFT and worlds match exact manifest hashes; all explicitly read metadata/code hashes match | PASS scoped |
| Row and answer contract | 5,400 unique scenarios; exact SFT projections; system/user/assistant; one-word answers; literal UNKNOWN | PASS train/dev |
| Answer rederivation | 5,400 answers and absence reasons, 75,073 parsed facts, zero disagreements | PASS train/dev; shared author/grammar limitation |
| Native tokenizer | All four locked Supa tokenizer assets match; Transformers 4.57.1, tokenizers 0.22.2, Qwen2TokenizerFast | PASS tokenizer only |
| 2,048 window / 128 reserve | Max rendered prompt 1,470; max full chat 1,476; zero violations, filtering or truncation | PASS train/dev |
| Context length bands | **41 short-band contexts exceed 600 native tokens: 4 train, 37 dev** | **FAIL** |
| Basic split isolation | Zero train/dev ID, group, entity, template-ID, exact/NFKC/digit-blinded whole context/question overlap | PASS narrow scope |
| Strict structural/template isolation | **392 shared name-blind sentence forms**; coarse typed-graph count signatures disjoint | NOT ESTABLISHED |
| Answerability balance | UNKNOWN: 1,440/4,800 train, 180/600 dev; four absence reasons equally allocated | MEASURED |
| Question-only screen | Train-fitted unigram naive Bayes: dev balanced accuracy 47.90%; train 50.44% | No positive shortcut finding from this finite screen |
| Surface screen | Train-selected `context_words` threshold: dev 49.60%, train 52.48% balanced accuracy | Finite screen only; no significance claim |
| Nontrivial abstention | **Cheap selector plus explicit validity rule: 87.5% balanced accuracy in both splits** | Qualification not established |
| Conditional balance | Producer reports max within-band relative deviation 5.263% against literal 3%; fresh permitted marginals reproduced | Full gate not reexecuted; producer failure unresolved |
| Protected-bank and frozen-scout contamination | No qualifying blind exclusion receipt | UNVERIFIED, never zero |
| Holdout/all-row inventory and token census | Deliberately not accessed | UNVERIFIED |

The 41 band failures are enumerated in `DATASET_ACCEPTANCE.json` and `native_train_dev_rows.json`. Examples include `lcr_v1_latest_verified_870ce2ce` (611 context tokens) and `lcr_v1_latest_verified_36188298` (614), both assigned to short. The literal inclusive band convention was used; two shared-endpoint rows are explicitly flagged and do not explain these failures.

The independent solver receives only model-visible user text before comparing its result with the stored answer/reason. It is a separate supplied implementation with shared authorship and grammar, not independent human certification. First/last entity, copy-most-recent, majority and UNKNOWN baselines were recomputed. Producer logistic-regression/nearest-neighbor attacks and every detailed evidence-deletion/difficulty check were not rerun; no stronger completeness claim is made.

Publish a new immutable dataset revision that corrects the 41 native band violations without truncation, resolves the strict structure/template and easy-abstention failures, and satisfies the balance rule or a prospectively authorized rounding convention. Obtain blind, aggregate exclusion receipts against all protected Rineya banks and the frozen scout, plus authorized holdout integrity/token/isolation receipts. Then rerun P0 before freezing any training recipe. This audit does not modify the corpus or its producer receipts.

Dataset directory: `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Rineya_LCR_Full_v1_6000`.

Manifest SHA-256: `65995575f13187ef78464e4120d1768696dd9114c7dfd8a379ed3ca6e28b266d`.

Attached specification SHA-256: `041f50d4a35ecf50dc255ade4a725b5f18e23db314dfb0cbf1440c79ceecbf21`.

Audit script SHA-256: `23fb08454e4b8734f6596d1d6b9e17dc937b9793bd3570bda4453dd8f7e4d4f3`.

Execution: 46.27 seconds, zero model calls and zero optimizer/training steps. Full explicit input hashes, runtime, source branch/parent and output hashes are recorded in the JSON companion and `partial_audit_receipt.json`. Software audit completed; native tokenizer execution passed; model-quality, confirmatory and release claims remain false.
'''
with (OUT/'DATASET_ACCEPTANCE.md').open('x',encoding='utf-8',newline='\n') as f:f.write(md)
print(json.dumps({'status':decision['status'],'outputs':['DATASET_ACCEPTANCE.md','DATASET_ACCEPTANCE.json'],'branch':decision['audit_source']['branch'],'parent_commit':decision['audit_source']['parent_commit']},indent=2))
