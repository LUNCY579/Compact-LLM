"""Read-only remote/source audit. Output is create-only outside source repos."""
import datetime, hashlib, json, pathlib, subprocess

OUT = pathlib.Path(__file__).parent
REPOS = {
    "RineyaArchitecture": pathlib.Path("C:/Users/clclc/BusinessProject/RineyaArchitecture"),
    "RineyaBeta": OUT / "RineyaBeta.git",
    "RineyaRAG": OUT / "RineyaRAG.git",
}

def command(args, cwd):
    return subprocess.check_output(args, cwd=cwd).decode("utf-8").strip()

def git(repo, *args):
    return command(["git", *args], REPOS[repo])

def blob(repo, ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPOS[repo])

def datum(repo, ref, path):
    return json.loads(blob(repo, ref, path))

def entry(repo, branch, question, implemented, executed, boundary, blockers, disposition, sources, base=None, supersession=None):
    return dict(repo=repo, branch=branch, question=question, implemented=implemented,
                executed=executed, evidence_boundary=boundary, blockers=blockers,
                recommended_disposition=disposition, sources=sources,
                intervention_base=base, supersession=supersession)

A="RineyaArchitecture"; B="RineyaBeta"; R="RineyaRAG"
rows = [
entry(A,"main","Apply query refinement/composition/local-global mechanisms to exact official Supa.",
      "Exact release asset loader; native Qwen3 parity; dense query adapter/independent/feedback paths; receipt gates.",
      "74 structural/native-random CI tests in retained initial receipt; current head CI green.",
      "NATIVE_RANDOM_MECHANICS; no official-core qualification or efficacy execution located.",
      "Exact official asset/runtime qualify -> pilot chain remains absent.","Preserve reference; use successor for intervention.",
      ["docs/EXECUTION_RECORD.md","evidence/native-ci/34434540803/mechanical-receipt.json","rineya_architecture/assets.py"]),
entry(A,"research/supa-baseline-query-refinement-v1","Does observation-dependent second query beat ordinary adapter and independent second read?",
      "Four arms, rank8 layer14 adapters, exact-source native qualification and pilot gates.",
      "Initial 74-test native-random mechanics; current-head native mechanics CI green.",
      "Mechanical/native-random only; registered official protocol unexecuted.",
      "Missing passing official qualify/pilot evidence.","Contained as reference by programme and main.",
      ["protocols/query.json","docs/EXECUTION_RECORD.md","rineya_architecture/query.py"],
      supersession="Do not inherit R2 or tiny-random PASS as official Supa qualification."),
entry(A,"research/supa-cpu-baseline-readout-v1","Can exact Supa pass the unchanged baseline/readout qualification on CPU?",
      "Manual opt-in official_cpu workflow, immutable main source 2d50f33, qualify/pilot only.",
      "Current-head run 34487725498 mechanics succeeded; Official Supa CPU baseline/readout job SKIPPED.",
      "No official execution; zero optimizer updates.",
      "Exact assets, runner-local access, CPU memory and readout gates unverified.","Park as alternate execution plumbing, not a candidate.",
      ["docs/CPU_BASELINE_EXECUTION.md",".github/workflows/native.yml"],base="2d50f33cd896ade7a14dfd70741ed0e9432c82ae"),
entry(A,"research/architecture-programme-v1","Does frozen hybrid local/global attention help official Supa, and what do sinks/globals contribute?",
      "Five-arm A0/A2/A3/A8/A10 factorial; A10 zero globals; immutable programme registry and guards.",
      "148 tests in retained CI receipt (native random 1057-token cache/mask tests); current-head CI green.",
      "Native-random mechanics; official factorial and all model training unexecuted.",
      "Official qualify/pilot/factorial baseline informative gate; no verified curriculum/trained conversion.","Park topology experiment; retain framework and controls.",
      ["AGENTS.md","docs/research/EXECUTION_RECORD.md","protocols/programme.json","rineya_architecture/governance.py"],base="557fa2d57a0c475b251250485bd9e825349d679c"),
entry(A,"benchmark/001-deep-thin-bounded-implementation-v1","Can a near-596M deeper/thinner ordinary D40 outperform B28 after matched recovery?",
      "Native E40 identity expansion; activation-informed width/GQA/FFN pruning; D40 tensor lineage; paired teacher CE/KL optimizer smoke implementation. C28/C40 analytical only.",
      "117 CPU native-random tests; missing official asset preflight fails closed. Current-head native and benchmark CI fail, no official smoke.",
      "CPU_NATIVE_RANDOM_FIXTURES_ONLY, no official conversion/recovery or quality.",
      "Official G0 receipts; smoke needs >11.12GiB tensor storage +1GiB headroom before activations, rejects 8GiB GPU; broad recovery corpus absent.","Park: compression/depth joint intervention behind ordinary quality lane.",
      ["benchmarks/deep_thin_v1/README.md","benchmarks/deep_thin_v1/architecture.py","benchmarks/deep_thin_v1/mechanics.py","evidence/benchmark-001/local-benchmark-mechanical-receipt.json","evidence/benchmark-001/local-preflight-failure.json"],base="340c37731c3e2ecf51b58477b28ff61e986fbb86",
      supersession="D40 is ordinary 597,263,232 params and 25% larger analytic KV cache than B28, not implemented CED40 or a cheaper serving model."),
entry(A,"research/evidence-addressable-memory-v1","Can feedback queries recover previously unselected complete records?",
      "CPU full backing store, detached mean-key hard routing, record addressing, nine matched-parameter/dense controls; optional native-random component test.",
      "121 CPU component tests, 128-row oracle bank; native random tests NOT run locally; official calls/optimizer=0. Current-head two CI workflows pre-runner failure.",
      "Component mechanics only. Favorable synthetic selection witness is not trained routing.",
      "No native generation traversal/token-record mapper; full backing remains CPU; router learnability unqualified; old answerability bank flawed.","Park component; superseded instruments in two diagnostic siblings.",
      ["experiments/evidence_addressable_memory_v1/memory.py","experiments/evidence_addressable_memory_v1/evidence/LOCAL_VERIFICATION.json","experiments/evidence_addressable_memory_v1/evidence/VERIFICATION.md"],base="5d319e30f33e5ef68231e908029d05c316dc4fab",
      supersession="EAMI/EAM-ID expose old _ABSENT answerability cue 128/128; old bank not qualified answerability training/evaluation."),
entry(A,"research/eam-immediate-diagnostics-v1","What shortcut and gradient defects can be diagnosed before model training?",
      "72 derivative cells, old-bank classifier audit, 192 paired open fixture rows, independent finite differences.",
      "40 tests; zero official calls/updates; 18/18 singleton two-read cells dead;36/36 diverse-value cells live; 128/128 question cue.",
      "Constructed offline diagnostics; no trained efficacy.",
      "Full repo/native tests absent; historical output reanalysis not run; current-head native CI pre-runner failure.","Keep findings active in data/reader design; park model lane.",
      ["experiments/eam_immediate_v1/evidence/RESULT.md","experiments/eam_immediate_v1/evidence/paired_audit.json","experiments/eam_immediate_v1/evidence/receipt.json"],base="dd368f8292e5b3bde60e7a5365a105dad22266ac"),
entry(A,"research/eam-immediate-insight-v1","Do old shortcut cues and detached selection provide interpretable learning gradients?",
      "126-test scoped suite; 72-cell derivative/selector audit; 384-row exact paired bank; strict analyzer.",
      "126 tests, 36/36 predicted derivative-dead and36/36 live controls; max finite-difference error3.0465e-11; fake UNKNOWN-vs-oracle analyzer run.",
      "CPU_COMPONENT_DIAGNOSTIC_NOT_PRETRAINED; fake analyzer predictions explicitly synthetic; no historical model rescore.",
      "Native integration/capture and training absent. Current-head CI blocked before runner by billing/spend annotation.","Keep corrected instrument practices; park native EAM pending useful ordinary baseline.",
      ["experiments/eam_immediate_insight_v1/STATUS.json","experiments/eam_immediate_insight_v1/evidence/LOCAL_RESULT.json","experiments/eam_immediate_insight_v1/evidence/RESULT.md"],base="dd368f8292e5b3bde60e7a5365a105dad22266ac"),
entry(A,"research/looped-flow-training-framework-v1","Does iterative denoising/looped flow improve over direct/recurrent controls?",
      "Bounded numerical DIRECT/RECURRENT/FLOW/LOOPED_FLOW CPU workbench and resume trajectory checks.",
      "75 tests; engineering smoke64 total optimizer updates, 16/arm;12-row probe; no consistent looped-flow scaling benefit.",
      "Synthetic smoke, not native/pretrained or registered efficacy.",
      "Native adapter/cache path absent; stronger synthetic decision stage unexecuted.","Contained by LFD; preserve negative/inconclusive smoke.",
      ["experiments/looped_flows_v1/evidence/RESULT.md","experiments/looped_flows_v1/evidence/LOCAL_RESULTS.json"],base="2d50f33cd896ade7a14dfd70741ed0e9432c82ae"),
entry(A,"research/looped-flow-decision-screen-v1","Does looped flow earn further study against recurrent and deep-compute controls on held-out topology?",
      "Calibrate/stage-A/stage-B contracts, 5760 verified answers, four arms, source-bound permits.",
      "159 tests;25 smoke optimizer updates total;5760-answer audit; calibration and five-seed Stage A/B NOT executed.",
      "Synthetic engineering smoke only, no native quality.",
      "Learnability calibration and independent decision stages; full repo/native tests absent.","Park; do not prioritize over ordinary model-quality tuning.",
      ["experiments/looped_flow_decision_v1/evidence/RESULT.md","experiments/looped_flow_decision_v1/protocol.json"],base="543504c6adbc3c3e26c5eb5530638f7eec87808a"),
entry(A,"research/swa-bounded-replay-idea-v1","Can retained global/sink KV plus bounded local replay reconstruct useful state?",
      "Isolated proposal only.","No executable study on this idea branch.","PROPOSAL.","Requires mechanics/native/quality qualification.","Contained by mechanics and framework descendants.",
      ["docs/ideas/swa-bounded-replay-v1/README.md","docs/ideas/swa-bounded-replay-v1/status.json"],base="2d50f33cd896ade7a14dfd70741ed0e9432c82ae"),
entry(A,"research/swa-bounded-replay-mechanics-v1","Is exact prefix restoration correct, and where does bounded replay drift?",
      "Fixed random CPU LLG/LGLG replay and exact-cache reference with causal immutability checks.",
      "24 tests;720 matrix rows (704 valid,16 explicit inapplicable); exact logit error4.44e-15;R=W and2W intermediate drift exists.",
      "PASS_RANDOM_CPU_MECHANICS_ONLY; no native/pretrained/quality or optimizer.",
      "No native backend; fixture endpoint equality at2W is not universal bound.","Contained by SBR framework; preserve diagnostic evidence.",
      ["evidence/swa-bounded-replay-mechanics-v1/receipt.json","evidence/swa-bounded-replay-mechanics-v1/README.md"],base="b7855d945b71aba63e9b8c355a0227af12703eab"),
entry(A,"research/swa-bounded-replay-framework-v1","Can a bounded reusable replay workbench support trustworthy later studies?",
      "tiny_cpu_v1 backend only, exact/reduced snapshots, configurable bounded plans, verifier.",
      "88 tests;quick240 rows and edge288 rows,144 exact chunk checks each; byte-identical quick repeat. Current-head CI pre-runner billing/spend failure.",
      "Random CPU mechanics; no native model, tokenizer, training or CED.",
      "Native backend and usefulness unimplemented; current historical publication note stale.","Park as tooling; correct live status centrally.",
      ["experiments/swa_replay_lab_v1/README.md","evidence/swa-replay-lab-v1/quick_receipt.json","evidence/swa-replay-lab-v1/README.md"],base="3db45fcb4369decac91c2fa0837e692a15f5f2e0",
      supersession="Live branch154824b exists; historical evidence README still says incomplete/no branch. Publication fact only is superseded."),
entry(A,"research/model-generation-process-failure-guided-practice-v1","At matched cost, does adaptive failure-targeted practice beat fixed practice?",
      "Independent signed-balance world generator, trace diagnosis, fixed/adaptive cost plans; native-run unconditional blocker.",
      "90 stdlib component tests and scripted error demo; no learner or model calls.",
      "Offline process mechanics; default128 TRAIN+128 controller DEV use same rendering template, not confirmation.",
      "Tokenizer/trainer/optimizer/native learner unimplemented.","Keep generator/controller concepts; contained by M1 qualification successor, not MQ corpus ready.",
      ["model_development/failure_guided_practice_v1/STATUS.json","model_development/failure_guided_practice_v1/data.py","model_development/failure_guided_practice_v1/evidence/local-2026-09-13/receipt.json"],base="2d50f33cd896ade7a14dfd70741ed0e9432c82ae"),
entry(A,"research/model-generation-process-fgp-m1-frozen-qualification-v1","Can one exact frozen Supa qualify the FGP structured instrument and token-cost support?",
      "Prepare native masks/assets, one-use frozen CPU permit,132 generation cap,14 probes,shadow plans; no optimizer implementation.",
      "90 inherited+105 new scoped fixture tests; official tokenizer/model NOT run. Published implementation32d7237 with evidence child22351b4.",
      "PASS_SCOPED_COMPONENT_MECHANICS_ONLY; no adaptive-curriculum efficacy or native qualification.",
      "Registered Linux CPU3.13.5 runtime and exact assets;128/128 formatting needed at two rows/condition; potential instrument floor; no training.","Park until ordinary model format/readout improves; keep masks/verification practices.",
      ["model_development/fgp_m1_frozen_qualification_v1/STATUS.json","model_development/fgp_m1_frozen_qualification_v1/REGISTRATION.json","model_development/fgp_m1_frozen_qualification_v1/engine.py","model_development/fgp_m1_frozen_qualification_v1/evidence/PUBLICATION_2026-09-14.md"],base="9f476e0ef30f8849588182f8eadeb4df48cd1b35",
      supersession="Creation-time NOT_PUBLISHED in STATUS is explicitly superseded by PUBLICATION_2026-09-14.md; preserve registered historical bytes."),
entry(B,"main","Canonical R2 surgery/healing/cache-selection and sparse-plasticity history.",
      "474M vocab-trimmed R2; healing; cache selection; PCSM/state-sufficiency infrastructure.",
      "Historical R2 and B0 model runs; B0 correct abstentions179/202 ->202/202 in canonical history. PCSM real-model quality unrun.",
      "Historical pretrained R2 results; not exact official Supa model evidence.",
      "Main audit snapshot2026-08-30 predates five live successor research branches and major corrections.","Historical reference; later branch artifacts outrank stale main summaries.",
      ["docs/EXPERIMENT_PROGRAM_AUDIT.md","docs/MINI_MODEL_ROADMAP.md","experiments/stabilizer/peer_optimizer/PHASE_C_README.md"],
      supersession="Canonical audit 'one live branch main' obsolete: six live heads at this audit."),
entry(B,"research/cornerstone2-v1","Can whole-record retrieval/instruction preservation improve compressed R2 answers, and are old mechanism claims valid?",
      "BM25+instruction selection; capacity and structural controls; blind-label analyzer; pooled/answerable/abstention/power corrections.",
      "R2 600-row4K generation campaign; blind answer-stratum n432 primary+2.5463pp,12W/1L,p=.00341797;925 replies labels; corrected control artifacts.",
      "Pretrained R2 exploratory/bounded replication and preregistered follow-ups; not Supa, not general confirmation.",
      "One model/corpus/template; inspected development data; no true semantic-abstention gain; 16K provenance/power issues.","Preserve positive bounded answer result and negative score-selector controls; use as evaluation caution.",
      ["experiments/kv_significance/results/blind_adjudication.json","experiments/kv_significance/results/abstain_endpoint_audit.json","scripts/design/analyse_blind_adjudication.py","docs/FINDINGS_CAPACITY_FLOOR.md","docs/FINDINGS_INSTRUCTION_IDENTIFICATION.md"],base="3532c7bc8dbdd48ef8801ba6080cb80b0e90379a",
      supersession="Pooled abstention gain is canned-string artifact; intrinsic-importance narrative, equal-quality65x/129x and global equivalence claims withdrawn/bounded."),
entry(B,"research/rineya-intelligence-v1","How damaging are frozen local/global replacements, and do sinks/global placement explain effects?",
      "12-arm frozen HLGA with native/cache/source integrity and strict first-content readout.",
      "R2 CUDA math 12x72 DEV;A0=47/72,A3=18/72,3W/32L;A10=16/72;A8->A10=11W/0L;raw receipt preserved.",
      "Real historical R2 checkpoint; exploratory/open synthetic DEV. No official Supa or training/confirmation.",
      "Frozen replacements all worse; global contribution contrast has2 discordants and no equivalence margin.","Historical negative, preserve sink evidence; topology lane parked.",
      ["experiments/hybrid_attention_v1/reports/agent_a10a11_receipt.json","experiments/hybrid_attention_v1/reports/agent_a10a11_summary.json","experiments/hybrid_attention_v1/reports/AGENT_A10_A11_RESULTS.md"],base="3532c7bc8dbdd48ef8801ba6080cb80b0e90379a",
      supersession="Latest commit withdraws 'seven globals contribute essentially nothing' and 'A10 equals A3 at57% KV'; older all-zero endpoint superseded."),
entry(B,"research/sink-token-mechanism-v1","Are sink benefits position, key routing, value content, or dose effects?",
      "11 frozen arms:zero_k/zero_v/permute_v/mean_v/interior/visibility/dose controls; W512 zero globals.",
      "Scoped CPU mechanics implementation; no new model result artifact found beyond inherited HLGA.",
      "Registered frozen mechanism successor; efficacy unexecuted.",
      "Fresh exact-checkpoint/source/native qualification before frozen DEV.","Park; does not improve full-attention candidate by itself.",
      ["experiments/sink_token_mechanism_v1/README.md","experiments/sink_token_mechanism_v1/interventions.py"],base="efb05337d55491478ddd1aa92550f0db18ce5e4f"),
entry(B,"research/query-refinement-official-family-v1","Does dense observation-feedback query refinement transfer to exact official Supa/Nimbus/Halo?",
      "Pinned official-family assets, native Qwen3 wrapper and four-arm rank8 adaptation at layer14; qualify/smoke/dev.",
      "Current-head official-family mechanics CI34396032879 green; no official pretrained qualification/run artifact.",
      "Native-random fixtures only for official migration; no carried-over R2 authority.",
      "Exact local assets and source/device-bound qualification required.","Contained by Architecture official-Supa programme; retain family migration provenance.",
      ["experiments/official_family_v1/README.md","experiments/official_family_v1/profiles.json","experiments/query_refinement_v1/family_pipeline.py"],base="e5794507c4e4236068b3124ce6efe61ed8dc9e56"),
entry(B,"research/composition-qrr-official-family-v1","Can repeated query-read-update solve composition beyond one-read/matched MLP controls?",
      "Official-family loader migration; frozen diagnostic ladder; QRR logit-residual tensor prototype.",
      "Inherited65 tensor tests/768-row oracle bank; migration ancestor ffb0e4 CI failed; current5785ef1 importlib collection fix has no returned head Actions run.",
      "Tensor/native-random mechanics only; no official model/QRR integration execution.",
      "Native QRR seam/qualification and all optimization blocked; exact head CI missing.","Park; preserve controls and instrument implementation.",
      ["experiments/composition_qrr_v1/evidence/initial_cpu/RECEIPT.json","experiments/official_family_v1/README.md",".github/workflows/official-family.yml"],base="690dcf27ee48512369fc2f3ca02603dee06ddfba"),
entry(R,"main","Can external retrieval improve grounded system answers with fixed Rineya?",
      "Standalone sparse/BM25 reference, optional dense/reranker and loopback adapter, fictional corpus.",
      "99 baseline stdlib tests;10-document12-query fixture; no pretrained model.",
      "Infrastructure/synthetic retrieval only; RAG-001 draft unexecuted.",
      "Generator/encoder/reranker native qualification and proper corpus absent.","Separate system-quality lane; park until fixed generator qualifies.",
      ["README.md","evidence/bootstrap/receipt.json","experiments/RAG-001/PROTOCOL.md"]),
entry(R,"research/generator-qualification-v1","Can exact unchanged Supa satisfy token/template/loopback serving contracts?",
      "CPU frozen generator worker;7 requests max; exact assets/template/IDs/state checks; no retrieval training.",
      "157 full stdlib/fake-engine tests;actual attempt BLOCKED:EXACT_LOCAL_ASSET_PATHS_NOT_SUPPLIED;worker not started;no tokenizer/model calls.",
      "LOCAL_STDLIB_CPU_CONTRACT_MECHANICS_WITH_FAKE_ENGINES;no native import or serving qualification.",
      "Exact assets/runtime; package-host DNS failed in historical preparation;RAG-001 still draft no no-RAG/oracle/noise runner.","Park separate RAG lane; not MQ training corpus.",
      ["AGENTS.md","experiments/generator_qualification_v1/README.md","evidence/generator_qualification_v1/qualification_attempt.json","evidence/generator_qualification_v1/verification.json"],base="1be74b3bd84fec0ed4a1bfe20cad18a40d903996",
      supersession="docs/STATUS.md still next asks implementation now present in successor; official execution remains blocked."),
]

observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
remote={}; actions={}; snapshots=[]
for repo in REPOS:
    remote[repo]=dict(line.split()[::-1] for line in git(repo,"ls-remote","--heads",f"https://github.com/LUNCY579/{repo}.git").splitlines())
    api=json.loads(command(["gh","api",f"repos/LUNCY579/{repo}/actions/runs?per_page=100"],REPOS[repo]))
    actions[repo]=[{k:r.get(k) for k in ("head_branch","head_sha","status","conclusion","id","name","updated_at","html_url")} for r in api["workflow_runs"]]

for row in rows:
    repo, branch=row["repo"],row["branch"]
    row["remote_head"]=remote[repo]["refs/heads/"+branch]
    row["parents"]=git(repo,"show","-s","--format=%P",row["remote_head"]).split()
    row["tree"]=git(repo,"rev-parse",row["remote_head"]+"^{tree}")
    if row["intervention_base"]:
        row["intervention_base_is_ancestor"]=subprocess.run(["git","merge-base","--is-ancestor",row["intervention_base"],row["remote_head"]],cwd=REPOS[repo],capture_output=True).returncode==0
    row["exact_head_actions"]=[r for r in actions[repo] if r["head_sha"]==row["remote_head"]]
    row["source_evidence"]=[]
    for path in row.pop("sources"):
        try:
            data=blob(repo,row["remote_head"],path)
            row["source_evidence"].append({"path":path,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"url":f"https://github.com/LUNCY579/{repo}/blob/{row['remote_head']}/{path}"})
        except subprocess.CalledProcessError:
            row["source_evidence"].append({"path":path,"not_found":True})

artifacts={}
for label,repo,ref,path,keys in [
    ("beta_blind_primary",B,"research/cornerstone2-v1","experiments/kv_significance/results/blind_adjudication.json",["full_answer_stratum","gates","verdict"]),
    ("beta_hlga_execution",B,"research/rineya-intelligence-v1","experiments/hybrid_attention_v1/reports/agent_a10a11_receipt.json",["backend","checkpoint_id","checkpoint_unchanged","source_unchanged","code_identity","environment","stage","status","total_optimizer_steps","summary"]),
    ("eam_insight",A,"origin/research/eam-immediate-insight-v1","experiments/eam_immediate_insight_v1/evidence/LOCAL_RESULT.json",["derivative","legacy","paired_instrument","pretrained_calls","optimizer_steps","evidence_class"]),
    ("deep_thin",A,"origin/benchmark/001-deep-thin-bounded-implementation-v1","evidence/benchmark-001/local-benchmark-mechanical-receipt.json",["evidence_class","tests","failed_or_skipped","pretrained_execution","quality_improvement_established","source_sha256"]),
    ("fgp_m1",A,"origin/research/model-generation-process-fgp-m1-frozen-qualification-v1","model_development/fgp_m1_frozen_qualification_v1/evidence/local-2026-09-13/receipt.json",["schema","status","elapsed_seconds","closed","results","sha256"]),
    ("rag_qualification",R,"research/generator-qualification-v1","evidence/generator_qualification_v1/qualification_attempt.json",["status","blocker","tokenizer_loaded","pretrained_model_loaded","model_requests_attempted","worker_started","serving_qualified"]),
]:
    d=datum(repo,ref,path)
    artifacts[label]={"repo":repo,"path":path,"fields":{k:d[k] for k in keys if k in d}}

annotations={}
for check in [104056620753,103780357824]:
    d=json.loads(command(["gh","api",f"repos/LUNCY579/RineyaArchitecture/check-runs/{check}/annotations"],REPOS[A]))
    annotations[str(check)]=[{k:x.get(k) for k in ("annotation_level","message")} for x in d]

report={"observed_at_utc":observed,"audit_scope":"Read-only source/remote/API inspection; no models, tests, optimizer, branches, commits, pushes, merges or account changes executed by this audit.","remote_heads":remote,"branches":rows,"selected_artifacts":artifacts,"actions_runs":actions,"fresh_failure_annotations":annotations}
with (OUT/"architecture-beta-audit.json").open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)

md=["# RineyaArchitecture / RineyaBeta / RineyaRAG live audit", "",f"Observed UTC: {observed}","", "## Scope and conclusion", "",
    "Fresh `git ls-remote --heads`, Architecture fetch, audit-only bare clones of actual Beta/RAG, immutable Git blobs, JSON receipts, implementation source, registrations, and GitHub Actions state were inspected. No model weights were downloaded or loaded; no training, tests, branch edits, commits, or pushes occurred in this subtask. Existing dirty ChatBubble worktree named RineyaBeta-supa-fwa-v1 was left untouched.","",
    "**No new trained official-Supa model-quality candidate is established in these repositories.** Architecture, FGP, EAM, replay and looped-flow work are useful implementation/instrument research, with native/pretrained and efficacy boundaries explicitly incomplete. The strongest positive real-model result here is a bounded historical **474M R2 external record-selection** result, not an improved Supa checkpoint. This supports keeping ordinary targeted adaptation and a stronger standalone model ahead of complex structural changes.","",
    "## Material corrections and decision implications", "",
    "1. **EAM old bank leaks answerability.** Both independent sibling diagnostics find 128/128 question-only binary decisions from the `_ABSENT` convention; EAMI also finds metadata cue and no exact layout pairing. Do not use the old bank as MQ training/answerability evidence. New 192/384-row banks are open constructed instruments, not qualified training corpora. Detached hard selection plus singleton/identical selected values creates derivative dead zones; diverse-value gradients do not establish learned routing.",
    "2. **R2 abstention/compression claims need stratification.** The pooled gain contains a canned-refusal substring artifact. The committed blinded-answer artifact gives +2.5463 percentage points on correct-and-legible answers, 12 wins/1 loss over432 answerable cases (p=.00341797); it does not establish semantic abstention improvement, Supa transfer, or general model quality. BM25+instruction preservation, not accumulated-attention scoring, supplies that bounded positive result.",
    "3. **Frozen HLGA remains negative.** Historical R2 A0=47/72 versus A3=18/72, 3 wins/32 losses; A10=16/72. Sink benefit is real in that probe, but the global-layer-null and 'equal at57% KV' claims were explicitly withdrawn. Two discordant pairs cannot support equivalence. No frozen topology here beats the ordinary baseline.",
    "4. **Deep-thin is D40, not CED40.** Ordinary D40 has597,263,232 parameters versus B28's596,049,920; analytic ordinary KV/token is25% larger. Conversion code explicitly rebuilds nonpersistent RoPE after meta initialization and rejects remaining meta tensors. Only117 native-random tests ran; official missing-assets preflight failed. The paired teacher CE/KL smoke's storage lower bound exceeds an8GiB GPU. This is a structural compression/recovery programme, not evidence to prioritize quality improvement.",
    "5. **Published state and execution state differ.** SBR framework is live at154824b despite historical README saying publication incomplete. FGP M1 is live at22351b4 and its publication note explicitly supersedes creation-time NOT_PUBLISHED. Beta main's canonical2026-08-30 audit says one live branch, but six exist. These historical bytes should stay intact; a current ledger should point to superseding evidence.",
    "6. **Fresh CI blockers are now identified.** EAMI and SBR current-head jobs did not start (`runner_id=0`, empty steps); GitHub annotations say recent account payments failed **or** spending limit needs increase. This is not a source-test failure and does not authorize account changes. Architecture CPU-baseline head has green mechanics but official CPU job skipped. RAG has no Actions runs in the returned inventory.",
    "7. **Corpus contribution is practices, not a ready bank.** FGP offers deterministic world-grouped signed-balance fixtures and exact-cost plans; EAMI offers symmetric answerability/layout examples; RAG has10 fictional documents/12 mechanical queries. None is a frozen broad quality-training bank with native tokenization plus all required overlap/shortcut audits. Import design lessons with new provenance rather than relabel fixtures as training-ready.","",
    "## Live evidence matrix", "",
    "Every head and immediate parent below is exact; intervention base is separately listed where recoverable. `Recommended disposition` is this audit's interpretation, not a branch deletion/archive action. Evidence URLs point to immutable source bytes. Current Actions entries qualify only their stated mechanical scope.",""]
for row in rows:
    md += [f"### {row['repo']} — `{row['branch']}`", "",
           f"- **Remote head:** `{row['remote_head']}`; **tree:** `{row['tree']}`.",
           f"- **Immediate parent(s):** {', '.join('`'+p+'`' for p in row['parents']) or 'root commit'}."]
    if row["intervention_base"]: md.append(f"- **Intervention lineage base:** `{row['intervention_base']}`; ancestor verified: `{row['intervention_base_is_ancestor']}`.")
    for key,label in [("question","Question"),("implemented","Implemented"),("executed","Actually executed in retained evidence"),("evidence_boundary","Scientific class / official weights / registration boundary"),("blockers","Blockers"),("supersession","Supersession"),("recommended_disposition","Recommended disposition")]:
        if row.get(key): md.append(f"- **{label}:** {row[key]}")
    runs=row["exact_head_actions"]
    md.append("- **Fresh exact-head Actions:** "+("; ".join(f"[{x['name']} #{x['id']}]({x['html_url']}) `{x['status']}/{x['conclusion']}`" for x in runs) if runs else "No exact-head run in current returned repository inventory."))
    md.append("- **Underlying evidence/source:** "+"; ".join(f"[{s['path']}]({s['url']})" if not s.get("not_found") else f"`{s['path']}` (not found; do not cite)" for s in row["source_evidence"])+".")
    md.append("")
md += ["## Scope exclusions and limits", "",
       "Owner repository discovery also found RineyaWorkspace and RineyaWorkspace-bare. Fresh heads were application/offline-refactor branches, with no directly relevant model-quality research head identified; no weights or application code were inspected further. Earlier4B/8B transfer, Opallo/B28/WSD/abstention/MQ/CED evidence belongs principally to ChatBubble and SupaExperiment and is handled by the companion audit, not inferred from these repos.","",
       "Counts/metrics in this report are retained repository executions, not new runs. Source hashes, all23 live heads, immediate parents, exact-head Actions entries, selected raw-JSON fields and fresh CI annotations are in `architecture-beta-audit.json`. The audit does not authenticate raw external archives absent from Git, prove every past run, or classify a CI green as pretrained efficacy. Live remote state can change after the timestamp.",""]
with (OUT/"architecture-beta-audit.md").open("x",encoding="utf-8") as f: f.write("\n".join(md))
print(json.dumps({"branches":len(rows),"missing_evidence":[(r["repo"],r["branch"],s["path"]) for r in rows for s in r["source_evidence"] if s.get("not_found")],"files":[str(OUT/"architecture-beta-audit.md"),str(OUT/"architecture-beta-audit.json")]}))
