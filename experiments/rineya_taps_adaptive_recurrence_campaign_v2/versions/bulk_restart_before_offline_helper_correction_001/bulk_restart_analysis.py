"""Restart-only source-bound reader; original scientific aggregate preserved."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import struct

from . import bulk_bank
from .analysis import TOL, distribution, grid_headroom, known_sum, mean, number, policy, read_journal, sha, token_count
from .gpu_analysis import validate_gpu_traces
from .open_analysis import payload_digest

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
GRID = tuple(sorted((.8, .9, 1., 1.1, 1.2, 1.1473425965417514)))
GIB = 1024**3
PRACTICAL_MARGIN = .05
RESTART_LANE = 'GPU_FP32_FULL_GENERATION_RESTART'
ORIGINAL_ANALYZER_SHA = '3defe511cd8a189f75e0e0529a56b782b965f40095c139d3a0c603b2bbf0d3db'
STATE = 'b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551'
CUSTOM = 'b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d'
INTEGRITY_STAGES = (
    ('DEV_DETERMINISTIC_INTEGRITY.json', 'DEVELOPMENT_DETERMINISTIC_BEFORE_FREEZE', 1728, 1848, 192),
    ('DEV_COMPLETE_INTEGRITY.json', 'DEVELOPMENT_COMPLETE_BEFORE_VALIDATION', 2112, 2240, 192),
    ('FINAL_INTEGRITY.json', 'FINAL_MATRIX_COMPLETE', 3168, 3296, 288),
)
CORE_JOURNALS = ('outcomes.jsonl', 'trajectories.jsonl', 'forwards.jsonl', 'readout_pairs.jsonl')
OLD_ATTEMPT_GRANTS = {
    '5397a0b51cdc62891c00c0f6a2147f5e53fafe5e277759b71d28ce3b93948063',
    'da328e1942923cffe9bd84b31c70ded9335b25b7d3d785d4a2ea7593971341c0',
}


def smoke_scorer():
    path = REPO / 'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py'
    if sha(path) != '5a22e1c1ff7b0a56623931c0782f81dd2b3a9df83ccfe77ce3724d37cf320086':
        raise RuntimeError('refuse changed inherited pure scorer')
    spec = importlib.util.spec_from_file_location('bulk_pinned_smoke_scorer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.score_response


def validate_registration(registration, admission, binding, token, freeze, old_config, taps):
    """Pure, fail-closed runtime/operator/data contract; never imports a model."""
    errors = []
    def require(condition, message):
        if not condition: errors.append(message)
    require(registration.get('bank_sha256') == admission.get('bulk_bank_sha256') == token.get('bank_sha256')
        and registration.get('token_binding_sha256') == admission.get('bulk_token_binding_sha256')
        and registration.get('scorer_sha256') == admission.get('bulk_source_sha256') == token.get('generator_scorer_sha256')
        and token.get('normalizer_source_sha256') == admission.get('bulk_normalizer_source_sha256'), 'bulk registration/data/scorer/token lineage differs')
    expected = {**old_config, 'device': 'cuda:0', 'dtype': 'float32', 'max_rss_gib': 6}
    require(registration.get('old_smoke_config') == expected and registration.get('config') == {**expected, 'max_new_tokens': 8}
        and registration.get('taps') == taps and freeze.get('config') == taps
        and registration.get('native_layers') == [12, 13] and registration.get('extra_loops') == [0, 1]
        and registration.get('gold_nll_status') == 'NOT_COLLECTED_CAUSAL_GUARD', 'bulk registered original operator/caps/controller differs')
    for field in ('python', 'torch', 'transformers'):
        require(registration.get('runtime', {}).get(field) == binding.get('runtime', {}).get(field) == token.get('runtime', {}).get(field), 'bulk actual pinned runtime differs:'+field)
    require(registration.get('runtime', {}).get('tf32') is False and registration.get('runtime', {}).get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        and token.get('runtime', {}).get('qwen_sha256') == binding.get('runtime', {}).get('qwen_sha256'), 'bulk GPU math/Qwen binding differs')
    require(token.get('status') == 'BULK_TOKENIZER_BOUND_NATIVE_NOT_MEASURED' and token.get('native_model_loads') == 0
        and token.get('canonical_source') == admission.get('source') == binding.get('source_path')
        and token.get('checkpoint_files') == admission.get('checkpoint_files') == binding.get('checkpoint_files'), 'bulk canonical tokenizer/checkpoint binding differs')
    expected_resources = {'device': 'cuda:0', 'dtype': 'float32', 'attn_implementation': 'eager', 'training': False, 'native_test': False,
        'max_new_tokens': 8, 'max_bulk_input_tokens': 350, 'gpu_allocated_ceiling_bytes': 5*GIB,
        'gpu_global_free_reserve_bytes': GIB, 'gpu_initial_free_floor_bytes': 6*GIB,
        'worker_host_available_floor_bytes': 6*GIB, 'worker_rss_ceiling_bytes': 6*GIB}
    require(all(admission.get(key) == value for key, value in expected_resources.items()), 'bulk authority/shape/resource floors differ')
    counts = registration.get('count_forecast', {})
    require(counts.get('physical_evaluations') == 3296 and counts.get('generation_call_cap') == 28416
        and counts.get('maximum_shape_forward_attempts') == 18 and counts.get('cache_rejection_attempts') == 2
        and counts.get('model_API_attempt_cap') == 28436, 'bulk complete count forecast differs')
    return errors


def validate_source_manifest(admission, manifest):
    """The frozen parent excludes run artifacts, .txt and LAB data from sources()."""
    errors=[]
    roots=('experiments/rineya_taps_adaptive_recurrence_v1/', 'experiments/anchor_routed_middle_recurrence_v1/',
           'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/', PACKAGE.relative_to(REPO).as_posix()+'/')
    for relative,digest in admission.get('dispatch_hashes',{}).items():
        path=Path(relative)
        covered=relative.startswith(roots) and path.suffix in ('.py','.json','.md') and not {'runs','__pycache__','.pytest_cache'}.intersection(path.parts)
        if covered and manifest.get('sha256',{}).get(relative)!=digest:
            errors.append('bulk dispatched source manifest differs:'+relative)
    return errors


def audit_readout_pairs(readouts, bank):
    if readouts is None:
        return [], {'status':'NOT_ANALYZED','standalone_latency_seconds':None,'extra_model_calls':None,
                    'limitation':'deferred per-row null metrics are not measured comparisons; worker wall includes readout work'}
    expected={(raw['split'],raw['id'],arm,17) for raw in bank for arm in ['A0','A1',*(f'A2:{a:g}' for a in GRID),'A3']}
    expected.update((raw['split'],raw['id'],'A5',seed) for raw in bank for seed in (17,29))
    keys=[(row.get('split'),row.get('case_id'),row.get('policy'),row.get('seed',17)) for row in readouts]
    errors=[]
    if len(keys)!=3168 or set(keys)!=expected or len(set(keys))!=len(keys): errors.append('bulk paired readout coverage/provenance differs')
    fields=('logit_cosine','kl_base_to_arm','max_abs_logit_delta','top1_agreement','topk_overlap','margin_change')
    if any(row.get('physical_run') is not False or row.get('extra_forward_calls')!=0
        or any(row.get(field) is not None for field in ('gold_token','gold_rank','baseline_gold_rank'))
        or any(not number(row.get(field)) for field in fields) for row in readouts): errors.append('bulk paired readout finite/nongold/no-extra contract differs')
    summaries={arm:{field:distribution(row.get(field) for row in readouts if row.get('policy')==arm) for field in fields}
               for arm in ['A0','A1',*(f'A2:{a:g}' for a in GRID),'A3','A5']}
    return errors, {'status':'AUDITED' if not errors else 'INVALID','observed_records':len(readouts),
        'expected_records':3168,'extra_model_calls':0 if not errors else None,'standalone_latency_seconds':None,
        'by_policy':summaries,'scope':'observed first logits vs same-case A0; A5 realizations descriptive, not independent samples; worker wall includes readout work'}


def audit_causal_A5_consistency(physical, traces):
    """Full-prefix reset means identical prefix and scalar imply identical next token.

    Join the same chronological chunks as validate_traces; neither nominal seed
    nor prior random draws can change a frozen forward after its scale is fixed.
    This is a conditional operator check, not a blanket repeat-equality rule.
    """
    chunks=[]
    for event in traces:
        if event.get('forward_index')==1 and event.get('iteration')==0: chunks.append([])
        if chunks: chunks[-1].append(event)
    new_rows=[row for row in physical if row.get('arm')!='qualification_legacy']
    if len(chunks)!=len(new_rows):
        return ['bulk A5 causal audit trajectory coverage differs'], {'status':'INCOMPLETE','checked_pairs':0,'violations':[]}
    groups=defaultdict(list)
    for row,chunk in zip(new_rows,chunks):
        if row.get('extra_loops')==1 and isinstance(row.get('token_ids'),list):
            groups[(row.get('split'),row.get('case_id'))].append((row,{event.get('forward_index'):event.get('alpha') for event in chunk if event.get('iteration')==1}))
    violations=[]; checked=0
    for (split,identifier),group in groups.items():
        for i,(left,left_alpha) in enumerate(group):
            for right,right_alpha in group[i+1:]:
                if left.get('arm')!='A5' and right.get('arm')!='A5': continue
                checked+=1; a,b=left['token_ids'],right['token_ids']
                divergence=next((k for k,(x,y) in enumerate(zip(a,b)) if x!=y),None)
                reason=None
                if divergence is None:
                    if len(a)!=len(b): reason='identical shared tokens with different stopping length'
                    elif left.get('response')!=right.get('response'): reason='identical token IDs with different decoded response'
                else:
                    la,ra=left_alpha.get(divergence+1),right_alpha.get(divergence+1)
                    if number(la) and number(ra) and .8-1e-6<=la<=1.2+1e-6 and .8-1e-6<=ra<=1.2+1e-6 and struct.pack('f',la)==struct.pack('f',ra):
                        reason='same input prefix and same effective FP32 extra alpha produced different next token'
                if reason:
                    violations.append({'split':split,'case_id':identifier,'left_policy':policy(left),'left_seed':left.get('seed'),
                        'right_policy':policy(right),'right_seed':right.get('seed'),'first_differing_token_index':divergence,
                        'left_alpha':left_alpha.get(divergence+1) if divergence is not None else None,
                        'right_alpha':right_alpha.get(divergence+1) if divergence is not None else None,'reason':reason})
    return (['bulk A5 conditional deterministic operator consistency differs'] if violations else []), {
        'status':'FAIL' if violations else 'CHECKED','checked_pairs':checked,'violations':violations,
        'scope':'A5 seeds and unit/fixed/adaptive controls; exact same case, shared prior token IDs and serialized FP32 alpha; no cross-prefix state'}


def expected_cells(bank, smoke):
    deterministic = [("A0", 0), ("A1", 1)] + [(f"A2:{alpha:g}", 1) for alpha in GRID] + [("A3", 1)]
    expected = {(raw["split"], 17, raw["id"], arm, extra) for raw in bank for arm, extra in deterministic}
    expected.update((raw["split"], seed, raw["id"], "A5", 1) for raw in bank for seed in (17, 29))
    for raw in smoke:
        expected.update(("SMOKE", 17, raw["case_id"], arm, extra) for arm, extra in deterministic + [("A5", 1)])
        expected.update(("SMOKE", 17, raw["case_id"], arm, extra) for arm in ("qualification_legacy", "qualification_new") for extra in (0, 1))
        expected.update(("SMOKE", 17, raw["case_id"], arm, 1) for arm in ("qualification_reference_A3", "qualification_successor_A3"))
    return expected


def parent_category_mean(values, cases):
    parents, category = defaultdict(list), {}
    for raw in cases:
        if raw["id"] in values:
            parents[raw["parent_id"]].append(values[raw["id"]]); category[raw["parent_id"]] = raw["category"]
    categories = defaultdict(list)
    for parent, scores in parents.items(): categories[category[parent]].append(mean(scores))
    return mean(mean(scores) for scores in categories.values())


def expected_order(raw, bank):
    parents = sorted({case["parent_id"] for case in bank})
    variants = sorted(case["variant_id"] for case in bank if case["parent_id"] == raw["parent_id"])
    slot = 2*parents.index(raw["parent_id"]) + variants.index(raw["variant_id"])
    labels = ["A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3"]
    rotated = labels[slot % 9:] + labels[:slot % 9]
    return list(reversed(rotated)) if (slot//9) % 2 else rotated


def _aggregate_scientific(rows, traces, bank, smoke, tokens, freeze, native, worker, qualification, a5, forwards, resources=(), budget=(), issues=(), readouts=None):
    errors = list(issues); cases = {raw["id"]: raw for raw in bank}
    readout_errors, readout_report = audit_readout_pairs(readouts,bank); errors.extend(readout_errors)
    smoke_cases = {raw['case_id']: raw for raw in smoke}; old_score = smoke_scorer()
    parents = Counter(raw["parent_id"] for raw in bank)
    if len(cases) != 288 or len(parents) != 144 or set(parents.values()) != {2} or Counter(raw["split"] for raw in bank) != {"DEVELOPMENT": 192, "OPEN_VALIDATION": 96}:
        errors.append("bulk population/parent/split cardinality differs")
    dev_parents = {raw["parent_id"] for raw in bank if raw["split"] == "DEVELOPMENT"}
    val_parents = {raw["parent_id"] for raw in bank if raw["split"] == "OPEN_VALIDATION"}
    if dev_parents & val_parents or len(dev_parents) != 96 or len(val_parents) != 48 or len({raw["category"] for raw in bank}) != 6:
        errors.append("bulk parent-world holdout/category contract differs")
    physical = [row for row in rows if row.get("physical_run") is True]
    oracles = [row for row in rows if row.get("physical_run") is False and row.get("arm") == "A4_example"]
    aliases = [row for row in rows if row.get("physical_run") is False and row.get("arm") == "A2_frozen"]
    if len(physical)+len(oracles)+len(aliases) != len(rows): errors.append("bulk unclassified physical/reuse row")
    indexed = defaultdict(list)
    for row in physical:
        indexed[(row.get("split"), row.get("seed"), row.get("case_id"), policy(row), row.get("extra_loops"))].append(row)
        if row.get("configured_seed") != row.get("seed") or row.get("safety_stopped") is not False: errors.append("bulk seed/safety contract differs")
        if row.get("score") not in (0, 1) or row.get("format_score") not in (None, 0, 1): errors.append("bulk/SMOKE nonbinary endpoint")
        if row.get("arm") == "A2" and row.get("fixed_alpha") not in GRID: errors.append("bulk unregistered raw fixed alpha")
        cap = 24 if row.get("split") == "SMOKE" else 8
        if not isinstance(row.get("token_ids"), list) or not 1 <= token_count(row) <= cap: errors.append("bulk generation cap differs")
        if row.get("gold_nll") is not None or row.get("gold_answer") is not None: errors.append("bulk full-gold diagnostic guard violated")
        if not isinstance(row.get('response'), str) or type(row.get('invalid')) is not bool or not isinstance(row.get('termination'), str):
            errors.append('bulk required generation/validity endpoints absent')
        raw = cases.get(row.get("case_id"))
        if raw:
            grade = bulk_bank.score_response(row["response"], raw)
            if row.get("score") != int(grade["correct"]) or row.get("format_score") != int(grade["format_valid"]): errors.append("bulk strict scorer regrade differs")
            for field in ("parent_id", "category", "generator_family", "generator_variant", "variant_id", "difficulty_stratum", "pair_kind"):
                if row.get(field) != raw[field]: errors.append("bulk case/parent/variant provenance differs")
            if row["arm"] != "A5":
                order = expected_order(raw, bank)
                if row.get("deterministic_policy_order") != order or row.get("policy_order_index") != order.index(policy(row)):
                    errors.append("bulk output-blind deterministic counterbalance differs")
        elif row.get('case_id') in smoke_cases:
            meta = smoke_cases[row['case_id']]
            if row.get('score') != old_score(row['response'], meta['scorer'])['score'] or (meta.get('format_scorer') is not None and row.get('format_score') != old_score(row['response'], meta['format_scorer'])['score']):
                errors.append('bulk inherited SMOKE scorer regrade differs')
    expected = expected_cells(bank, smoke)
    missing = sorted(expected - set(indexed), key=repr); extra = sorted(set(indexed)-expected, key=repr)
    duplicates = [cell for cell, group in indexed.items() if len(group) != 1]
    if extra or duplicates: errors.append("bulk deterministic once/A5twice cells extra/duplicate")
    errors.extend(validate_gpu_traces(physical, traces, freeze.get("pool", [])))
    causal_errors,causal_report=audit_causal_A5_consistency(physical,traces); errors.extend(causal_errors)
    if any(event.get("arm") in ("A1", "qualification_new") and (not number(event.get("alpha")) or abs(event["alpha"]-1)>1e-6) for event in traces): errors.append("bulk unit trajectory scaled")
    freeze_payload = {name: value for name, value in freeze.items() if name != "freeze_sha256"}
    dev_ids = {raw["id"] for raw in bank if raw["split"] == "DEVELOPMENT"}
    if (freeze.get("status") != "FROZEN_BULK_DEVELOPMENT" or freeze.get("freeze_sha256") != payload_digest(freeze_payload)
            or set(freeze.get("selection_case_ids", [])) != dev_ids or set(freeze.get("selection_parent_ids", [])) != dev_parents
            or len(freeze.get('selection_case_ids', [])) != 192 or len(freeze.get('selection_parent_ids', [])) != 96
            or freeze.get('alpha') not in GRID or freeze.get('A5_actual_seeds') != [17, 29]
            or freeze.get("OPEN_VALIDATION_native_exposure_at_freeze") is not False or freeze.get("deterministic_seed") != 17):
        errors.append("bulk DEV source/parent/config freeze differs")
    pool = [min(1.2, max(.8, event["alpha"])) for event in traces if event.get("split") == "DEVELOPMENT" and event.get("arm") == "A3" and event.get("iteration") == 1]
    if freeze.get("pool") != pool or not pool: errors.append("bulk pool not once-only DEV generation marginal")
    first_val = next((index for index, row in enumerate(physical) if row.get("split") == "OPEN_VALIDATION"), None)
    if first_val is not None and any(row.get("split") == "DEVELOPMENT" for row in physical[first_val:]): errors.append("bulk validation before complete DEV")
    for raw in bank:
        deterministic = [policy(row) for row in physical if row.get('case_id') == raw['id'] and row.get('arm') != 'A5']
        if deterministic != expected_order(raw, bank): errors.append('bulk chronological policy order differs')
    token_ids = [row.get('case_id') for row in tokens]
    if len(token_ids) != 288 or set(token_ids) != set(cases) or len(smoke_cases) != 8: errors.append('bulk token/SMOKE metadata coverage differs')
    token_map = {row["case_id"]: row["input_tokens"] for row in tokens}; token_map.update({row["case_id"]: row["input_tokens"] for row in smoke})
    chunks = []
    for event in forwards:
        if not event.get("event", "").startswith("FORWARD_"): continue
        if event["event"] == "FORWARD_STARTED" and event.get("full_model_call") == 1: chunks.append([])
        if chunks: chunks[-1].append(event)
        else: errors.append("bulk forward start absent")
    if len(chunks) != len(physical): errors.append("bulk forward evaluation coverage differs")
    for row, chunk in zip(physical, chunks):
        calls, prompt = token_count(row), token_map.get(row.get("case_id"))
        if type(prompt) is not int: errors.append("bulk bound input count absent"); continue
        if [(event.get("event"), event.get("full_model_call")) for event in chunk] != [(name, index) for index in range(1, calls+1) for name in ("FORWARD_STARTED", "FORWARD_COMPLETED")]: errors.append("bulk forward completion sequence differs")
        if any(event.get("prefix_tokens") != prompt+event.get("full_model_call", 0)-1 for event in chunk): errors.append("bulk prefix position progression differs")
        if any(any(event.get(field) != row.get(field) for field in ('case_id','arm','seed','split','extra_loops')) for event in chunk): errors.append('bulk forward/outcome provenance differs')
        if row.get("full_model_calls_including_gold_nll") != calls or row.get("initial_block_applications") != calls or row.get("processed_token_positions") != calls*prompt+calls*(calls-1)//2 or row.get('total_token_positions_including_gold_nll') != calls*prompt+calls*(calls-1)//2 or row.get("extra_layer_applications") != 2*calls*row.get("extra_loops", 0): errors.append("bulk physical compute counters differ")
    max_start = [e for e in forwards if e.get("event") == "MAXIMUM_FORWARD_STARTED"]
    max_done = [e for e in forwards if e.get("event") == "MAXIMUM_FORWARD_COMPLETED"]
    max_failed = [e for e in forwards if e.get("event") == "MAXIMUM_FORWARD_FAILED"]
    cache = [e for e in forwards if e.get("event") == "CACHE_REJECTED"]
    cache_starts = [e for e in forwards if e.get('event') == 'CACHE_REJECTION_STARTED']
    if len(max_start) != 18 or len(max_done) != 17 or len(max_failed) != 1 or len(cache) != 2: errors.append("bulk mechanical matrix incomplete")
    if any(e.get("prefix_tokens") != 357 for e in max_start+max_done+max_failed): errors.append("bulk max357 shape differs")
    if [e.get('attempt') for e in max_start] != list(range(1,19)) or Counter(e.get('attempt') for e in max_done+max_failed) != Counter(range(1,19)): errors.append('bulk max attempt identity/coverage differs')
    if max_failed and ('EXPECTED_BULK_FORCED_EXCEPTION' not in max_failed[0].get('error', '') or max_failed[0].get('completed') is not False or max_failed[0].get('partial_compute') != 'UNKNOWN'): errors.append('bulk unexpected/dishonest max partial failure')
    if any(e.get('completed') is not True for e in max_done) or any(e.get('model_compute_started') is not False or e.get('wrapper_forward_count') != 0 or e.get('completed') is not False for e in cache): errors.append('bulk cache rejection/model completion contract differs')
    if len(cache_starts)!=2 or {(e.get('use_cache'),e.get('cache_object_present')) for e in cache_starts}!={(True,False),(False,True)} or [e['event'] for e in forwards if e.get('event') in ('CACHE_REJECTION_STARTED','CACHE_REJECTED')]!=['CACHE_REJECTION_STARTED','CACHE_REJECTED']*2:
        errors.append('bulk exact cache-rejection attempt variants/pairing differ')
    terminal = native.get("status") == "BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY" and native.get("source_unchanged") is True and native.get("checkpoint_files_unchanged") is True and native.get("trainable_parameters") == 0 and native.get("training") is False and worker.get("status") == "WORKER_TERMINAL" and worker.get("exit_code") == 0
    qualified = qualification.get("status") == "PASS_BULK_NEW_CAP_SAME_OPERATOR" and qualification.get("maximum_prefix_tokens") == 357 and qualification.get("noextra_unit_originalA3_identity") is True and qualification.get("forced_exception_restoration") is True and qualification.get("cache_rejections") == 2 and qualification.get("source_unchanged") is True and qualification.get("before_state_sha256") == qualification.get("after_state_sha256") and a5.get("status") == "PASS_FROZEN_BULK_DEV_POOL" and a5.get("actual_seed_binding") is True
    if not terminal or not qualified: errors.append("bulk terminal/mechanical/frozen-pool qualification absent")
    frozen = freeze.get("alpha")
    expected_oracle = {(raw["split"], raw["id"], 17) for raw in bank}
    if len(oracles) != 288 or {(row.get("split"), row.get("case_id"), row.get("seed")) for row in oracles} != expected_oracle: errors.append("bulk once-only oracle reuse differs")
    val_ids = {raw["id"] for raw in bank if raw["split"] == "OPEN_VALIDATION"}
    if len(aliases) != 96 or {(row.get("case_id"), row.get("seed")) for row in aliases} != {(identifier, 17) for identifier in val_ids}: errors.append("bulk frozen validation alias coverage differs")
    if not missing and not extra and not duplicates:
        for meta in smoke:
            identifier=meta['case_id']
            for extra_loops in (0,1):
                old=indexed[('SMOKE',17,identifier,'qualification_legacy',extra_loops)][0]
                new=indexed[('SMOKE',17,identifier,'qualification_new',extra_loops)][0]
                unit=indexed[('SMOKE',17,identifier,'A1' if extra_loops else 'A0',extra_loops)][0]
                if old.get('token_ids')!=new.get('token_ids') or new.get('token_ids')!=unit.get('token_ids'): errors.append('bulk raw noextra/unit value parity differs')
            reference=indexed[('SMOKE',17,identifier,'qualification_reference_A3',1)][0]
            successor=indexed[('SMOKE',17,identifier,'qualification_successor_A3',1)][0]
            actual=indexed[('SMOKE',17,identifier,'A3',1)][0]
            if reference.get('token_ids')!=successor.get('token_ids') or successor.get('token_ids')!=actual.get('token_ids'): errors.append('bulk raw original A3 parity differs')
        for row in oracles:
            candidates = [indexed[(row["split"], 17, row["case_id"], f"A2:{alpha:g}", 1)][0] for alpha in GRID]
            best = max(candidates, key=lambda value: (value["score"], -value["fixed_alpha"]))
            if any(row.get(field) != best.get(field) for field in ("score", "fixed_alpha", "token_ids")): errors.append("bulk oracle not exact fixed-grid reuse")
        for row in aliases:
            original = indexed[("OPEN_VALIDATION", 17, row["case_id"], f"A2:{frozen:g}", 1)][0]
            if any(row.get(field) != original.get(field) for field in ("score", "fixed_alpha", "token_ids")): errors.append("bulk frozen alias changed after validation")
    measurements = [row for row in physical if row.get("case_id") in cases]
    case_stats = {}
    for raw in bank:
        for arm in ["A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3", "A5"]:
            group = [row for row in measurements if row["case_id"] == raw["id"] and policy(row) == arm]
            if group:
                case_stats[(arm, raw["id"])] = {"score": mean(row["score"] for row in group), "format": mean(row["format_score"] for row in group),
                    "predictions": [row["response"] for row in group], "canonical": [bulk_bank.normalize_output(row["response"]) for row in group],
                    'literal_UNKNOWN': mean(int(row['response'].strip() == 'UNKNOWN') for row in group),
                    'canonical_UNKNOWN': mean(int(bulk_bank.normalize_output(row['response']) == 'UNKNOWN') for row in group),
                    'semantic_abstention_beyond_canonical_UNKNOWN': 'NOT_ASSESSED',
                    'invalid_rate': mean(int(row['invalid']) for row in group if type(row.get('invalid')) is bool),
                    'termination_counts': dict(Counter(row.get('termination', 'MISSING') for row in group)),
                    "false_UNKNOWN": mean(int(bulk_bank.score_response(row["response"], raw)["false_UNKNOWN"]) for row in group),
                    "unsupported": mean(int(bulk_bank.score_response(row["response"], raw)["unsupported_canonical_answer"]) for row in group if bulk_bank.score_response(row["response"], raw)["unsupported_canonical_answer"] is not None)}
    samples = [e["resources"] for e in forwards if isinstance(e.get("resources"), dict)] + list(resources)
    samples += [value for value in (native.get("after_load_resources"), native.get("final_resources"), native.get("final_resources_on_stop"), qualification.get("resources")) if isinstance(value, dict)]
    samples += [{'rss_bytes': worker.get('peak_rss_bytes')}]
    if any(number(s.get("rss_bytes")) and s["rss_bytes"] > 6*GIB or number(s.get("host_available_bytes")) and s["host_available_bytes"] < GIB or number(s.get("gpu_peak_allocated_bytes")) and s["gpu_peak_allocated_bytes"] > 5*GIB or number(s.get('gpu_allocated_bytes')) and s['gpu_allocated_bytes'] > 5*GIB or number(s.get("gpu_free_bytes")) and s["gpu_free_bytes"] < GIB or number(s.get('gpu_global_inventory', {}).get('free_bytes')) and s['gpu_global_inventory']['free_bytes'] < GIB for s in samples): errors.append("bulk recorded resource contract breached")
    if not any(number(s.get('gpu_peak_allocated_bytes')) and number(s.get('gpu_free_bytes')) and number(s.get('host_available_bytes')) for s in samples): errors.append('bulk measured GPU/host resource evidence absent')
    complete = not missing and not errors; reports = {}
    policies = ["A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3", "A5"]
    for split in ("DEVELOPMENT", "OPEN_VALIDATION"):
        subset = [raw for raw in bank if raw["split"] == split]
        def values(arm, field="score"): return {raw["id"]: case_stats[(arm, raw["id"])][field] for raw in subset if (arm, raw["id"]) in case_stats}
        curve = {alpha: parent_category_mean(values(f"A2:{alpha:g}"), subset) for alpha in GRID}
        envelope = max(GRID, key=lambda alpha: (curve[alpha], -alpha)) if all(number(x) for x in curve.values()) else None
        if split == "DEVELOPMENT" and envelope != frozen: errors.append("bulk DEV winner does not match six-constant freeze"); complete = False
        oracle_values = {row["case_id"]: row["score"] for row in oracles if row["split"] == split}
        oracle_score = parent_category_mean(oracle_values, subset) if complete else None
        best_score = curve[envelope] if complete else None
        adaptive = parent_category_mean(values("A3"), subset)
        contrasts = {}
        if complete:
            for label, comparator in (("frozen_fixed", f"A2:{frozen:g}"), ("matched_A5", "A5"), ("intact", "A0"), ("unit", "A1")):
                variant_rows = []
                for raw in subset:
                    a, b = case_stats[("A3", raw["id"])], case_stats[(comparator, raw["id"])]
                    variant_rows.append({"case_id": raw["id"], "parent_id": raw["parent_id"], "category": raw["category"], "difficulty": raw["difficulty_stratum"],
                        "variant_id": raw["variant_id"], "source_status": raw["source_status"], "treatment_score": a["score"], "comparator_score": b["score"], "delta": a["score"]-b["score"],
                        "format_delta": a["format"]-b["format"], "false_UNKNOWN_delta": a["false_UNKNOWN"]-b["false_UNKNOWN"],
                        "unsupported_delta": a["unsupported"]-b["unsupported"] if number(a["unsupported"]) and number(b["unsupported"]) else None})
                parent_rows = {parent: mean(row["delta"] for row in variant_rows if row["parent_id"] == parent) for parent in {raw["parent_id"] for raw in subset}}
                contrasts[label] = {"comparator": comparator, "score_delta": parent_category_mean({row["case_id"]: row["delta"] for row in variant_rows}, subset),
                    'practical_margin': PRACTICAL_MARGIN, 'meets_margin': parent_category_mean({row['case_id']:row['delta'] for row in variant_rows},subset)>=PRACTICAL_MARGIN-TOL,
                    "parent_units": len(parent_rows), "parent_deltas": parent_rows, "variant_wins": [row["case_id"] for row in variant_rows if row["delta"] > TOL],
                    "variant_losses": [row["case_id"] for row in variant_rows if row["delta"] < -TOL], "correct_case_retention_losses": [row["case_id"] for row in variant_rows if row["comparator_score"] == 1 and row["treatment_score"] < 1],
                    "observed_preservation_screen": all(row["delta"] >= -TOL and row["format_delta"] >= -TOL and row["false_UNKNOWN_delta"] <= TOL and (row["unsupported_delta"] is None or row["unsupported_delta"] <= TOL) for row in variant_rows),
                    "per_variant": variant_rows, "noninferiority": "NOT_ESTABLISHED", "confidence_interval": None, "confirmatory_p_value": None}
        pair_tables = {}
        for arm in policies:
            pairs = []
            for parent in {raw["parent_id"] for raw in subset}:
                variants = sorted((raw for raw in subset if raw["parent_id"] == parent), key=lambda value: value["variant_id"])
                if not all((arm, raw["id"]) in case_stats for raw in variants): continue
                entries = [{"id": raw["id"], "variant_id": raw["variant_id"], "gold": raw["gold"], "answerability": raw["answerability"], "source_status": raw["source_status"], **case_stats[(arm, raw["id"]) ]} for raw in variants]
                relation = variants[0]["pair_kind"]
                changed = Counter(entries[0]['canonical']) != Counter(entries[1]['canonical'])
                pairs.append({"parent_id": parent, "pair_kind": relation, "variants": entries,
                              "both_correct": all(entry["score"] == 1 for entry in entries),
                              "response_changed": changed, "expected_response_change": relation == "EVIDENCE_TRANSFER",
                              "appropriate_evidence_response": changed and all(entry["score"] == 1 for entry in entries) if relation == "EVIDENCE_TRANSFER" else None,
                              "same_gold_response_consistency": not changed if relation == "SAME_GOLD_PROMPT_VARIANTS" else None})
            pair_tables[arm] = pairs
        strata = {}
        for category in sorted({raw["category"] for raw in subset}):
            for difficulty in ("easy", "hard"):
                selected = [raw for raw in subset if raw["category"] == category and raw["difficulty_stratum"] == difficulty]
                strata[category+":"+difficulty] = {"cases": len(selected), "parents": len({raw["parent_id"] for raw in selected}),
                    "policy_accuracy": {arm: parent_category_mean(values(arm), selected) if complete else None for arm in policies}}
        def stratified(fields):
            table = {}
            for key in sorted({tuple(str(raw[field]) for field in fields) for raw in subset}):
                selected = [raw for raw in subset if tuple(str(raw[field]) for field in fields) == key]
                table[':'.join(key)] = {'cases': len(selected), 'parents': len({raw['parent_id'] for raw in selected}),
                    'policy_endpoints': {arm: {field: parent_category_mean(values(arm, field), selected) if complete else None
                        for field in ('score','format','literal_UNKNOWN','canonical_UNKNOWN','false_UNKNOWN','unsupported','invalid_rate')} for arm in policies},
                    'paired_deltas': {label: parent_category_mean({row['case_id']: row['delta'] for row in contrast['per_variant']}, selected) for label, contrast in contrasts.items()}}
            return table
        floors = {}
        for raw in subset:
            for name, response in bulk_bank.blind_predictions(raw['prompt']).items():
                floors.setdefault(name, {})[raw['id']] = float(bulk_bank.score_response(response, raw)['correct'])
        floor_tables = {'overall': {name: {'correct_cases': sum(v.values()), 'cases': len(subset), 'parent_category_accuracy': parent_category_mean(v, subset)} for name,v in floors.items()}}
        for fields in (('category','difficulty_stratum'), ('category',), ('source_status',), ('variant_id',)):
            floor_tables[':'.join(fields)] = {':'.join(key): {name: parent_category_mean(v, [raw for raw in subset if tuple(str(raw[field]) for field in fields)==key]) for name,v in floors.items()}
                for key in sorted({tuple(str(raw[field]) for field in fields) for raw in subset})}
        feature_tables = {}
        selected_events = [event for event in traces if event.get('split') == split and event.get('arm') == 'A3' and event.get('iteration') == 1 and event.get('case_id') in cases]
        for event in selected_events:
            raw = cases[event['case_id']]; length = token_map[raw['id']] + event.get('forward_index',1)-1
            key = raw['category']+':'+raw['difficulty_stratum']+':'+('1-64' if length<=64 else '65-128' if length<=128 else '129-256' if length<=256 else '257-357')
            feature_tables.setdefault(key, []).append(event)
        feature_tables = {key: {'prefix_observations': len(events), 'alpha': distribution(e.get('alpha') for e in events),
            'bound_saturation_fraction': mean(int(abs(e['alpha']-.8)<=1e-6 or abs(e['alpha']-1.2)<=1e-6) for e in events if number(e.get('alpha'))),
            'features': {field: distribution(e.get(field) for e in events) for field in ('delta_norm','delta_cosine','previous_ema_cosine','ema_cosine','progress','fluctuation','progress_fluctuation_ratio','balance')},
            'weighting': 'generation prefix observations; not independent cases'} for key, events in feature_tables.items()}
        reports[split] = {"case_count": len(subset), "parent_units": len({raw["parent_id"] for raw in subset}), "category_weight": 1/6,
            "tables": [{"policy": arm, "accuracy": parent_category_mean(values(arm), subset) if complete else None, "observed_partial_accuracy": parent_category_mean(values(arm), subset)} for arm in policies],
            "fixed_curve": [{"alpha": alpha, "accuracy": score if complete else None} for alpha, score in curve.items()],
            "frozen_fixed_primary": {"alpha": frozen, "accuracy": curve.get(frozen) if complete else None}, "retrospective_envelope": {"alpha": envelope, "accuracy": best_score, "diagnostic_only": True},
            "constant_oracle": {"accuracy": oracle_score, **grid_headroom(adaptive, best_score, oracle_score, complete), "scope": "per-case six-constant diagnostic, not schedule bound"},
            "contrasts": contrasts, "category_difficulty": strata, "parent_variant_pairs": pair_tables,
            'category_difficulty_endpoints': stratified(('category','difficulty_stratum')), 'source_status': stratified(('source_status',)),
            'query_role': stratified(('query_role',)), 'family': stratified(('generator_family',)),
            'own_bank_blind_floors': floor_tables, 'alpha_features_by_category_difficulty_prefix': feature_tables,
            "generator_dependence": "shared authored categories/templates; parent-world holdout only"}
    if not complete:
        for report in reports.values():
            report["contrasts"] = {}; report["constant_oracle"].update(accuracy=None, **grid_headroom(None, None, None, False)); report["frozen_fixed_primary"]["accuracy"] = report["retrospective_envelope"]["accuracy"] = None
            for row in report["tables"]+report["fixed_curve"]: row["accuracy"] = None
    completed = [e for e in forwards if e.get("event") in ("FORWARD_COMPLETED", "MAXIMUM_FORWARD_COMPLETED")]
    events = [event for event in budget if event.get("event") == "native_completed"]
    cumulative = sum(event["native_worker_seconds"] for event in events) if events and all(number(event.get("native_worker_seconds")) for event in events) else None
    return {"lane": "BULK_GPU_FP32_INTERVENTION", "status": "COMPLETE_EXPLORATORY_BULK" if complete else "INCOMPLETE_NO_PRIMARY_COMPARISON", "complete": complete,
        "errors": errors, "missing_cells": missing, "extra_cells": extra, "duplicate_cells": duplicates, "physical_rows": len(physical), "expected_physical_rows": len(expected),
        "oracle_nonphysical_rows": len(oracles), "frozen_alias_nonphysical_rows": len(aliases), "deterministic_repetitions": 1, "A5_repetitions": 2,
        "primary_validation_parent_units": 48, "reports": reports, "no_new_controller_revision": True,
        'practical_margin': PRACTICAL_MARGIN, 'margin_scope': 'registered exploratory strict accuracy difference; preservation and uncertainty separate',
        'paired_first_logit_diagnostics':readout_report,
        'A5_causal_consistency':causal_report,
        "costs": {"physical_only": {field: known_sum(physical, field) for field in ("wall_seconds", "controller_seconds", "controller_logging_seconds", "forward_logging_seconds", "full_model_calls_including_gold_nll", "processed_token_positions", "extra_layer_applications")},
            "completed_forward_count": len(completed), "completed_prefix_positions": known_sum(completed, "prefix_tokens"), "partial_compute_positions": None,
            'measured_forward_seconds': known_sum(completed+max_failed+cache+[e for e in forwards if e.get('event')=='FORWARD_FAILED'], 'forward_seconds'),
            'ordinary_attempts': sum(e.get('event')=='FORWARD_STARTED' for e in forwards),
            'failed_ordinary_attempts': sum(e.get('event')=='FORWARD_FAILED' for e in forwards),
            'generated_tokens': sum(token_count(row) for row in physical) if physical else None,
            "max_attempts": len(max_start), "cache_rejections": len(cache), "oracle_alias_extra_generation": 0, "worker_wall_seconds": worker.get("native_worker_seconds"),
            "cumulative_native_worker_seconds": cumulative, "peak_sampled_worker_rss_bytes": worker.get("peak_rss_bytes"),
            "peak_gpu_allocated_bytes": max((s["gpu_peak_allocated_bytes"] for s in samples if number(s.get("gpu_peak_allocated_bytes"))), default=None),
            "scope": "instrumented synchronized shared-device path; no production-speed or timing-repeat claim"},
        "independent_source_confirmation": False, "broad_intelligence_claim": False, "protected_TEST": "CLOSED"}


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.utcoffset() == timezone.utc.utcoffset(parsed) else None
    except (TypeError, ValueError):
        return None


def read_human_scope(admission, worker, inputs):
    """Validate actual grant bytes with the child-safe helper after a run.

    Parent anti-replay is deliberately not called: a legitimate completed run
    has consumed its one-attempt grant. The actual file/scope and terminal join
    remain mandatory, and its bytes are included in report manifest inputs.
    """
    from .full_generation_restart_authority import validate_human_scope
    try:
        grant = validate_human_scope(admission)
        path = Path(admission['human_extension_path']).resolve()
        digest = sha(path)
        if digest in OLD_ATTEMPT_GRANTS or worker.get('human_extension_sha256') != digest:
            raise PermissionError('actual full-generation grant missing/wrong parent terminal binding')
        inputs.append(path)
        return {'child_safe_scope_verified':True,'path':str(path),'sha256':digest,'payload':grant}, []
    except (OSError, ValueError, KeyError, TypeError, PermissionError) as error:
        return {}, ['restart actual human scope unavailable/different: '+type(error).__name__+': '+str(error)]


def validate_restart_integrity(context, bank, freeze, native, qualification):
    """Require recorded epoch, boundary state and actual committed journal prefixes.

    The IO reader supplies raw proof/journal bytes and actual file hashes. Counts
    or matching absent fields never substitute for those recorded commitments.
    """
    errors = []
    context = context if isinstance(context, dict) else {}
    admission, registration = context.get('admission', {}), context.get('registration', {})
    proof_bytes, journal_bytes = context.get('proof_bytes', {}), context.get('journal_bytes', {})
    events = context.get('integrity_journal', [])
    dispatch, assets = admission.get('dispatch_hashes', {}), admission.get('checkpoint_files', {})
    def require(condition, message):
        if not condition: errors.append('restart integrity: '+message)
    require(bool(dispatch) and bool(assets) and len(assets) == 10, 'complete admitted source/assets absent')
    require(admission.get('status') == 'READY_FOR_BULK_RESTART_FP32_DISPATCH'
        and admission.get('lane') == registration.get('lane') == RESTART_LANE,
        'restart epoch absent/different')
    require(admission.get('hard_deadline_seconds') == 16000
        and admission.get('native_finalization_reserve_seconds') == 1800, 'deadline/reserve differs')
    require(dispatch.get((PACKAGE/'bulk_analysis.py').relative_to(REPO).as_posix()) == ORIGINAL_ANALYZER_SHA,
        'original scientific analyzer lineage missing/different')
    runner_hash = dispatch.get((PACKAGE/'bulk_restart_native.py').relative_to(REPO).as_posix())
    require(isinstance(runner_hash, str) and len(runner_hash) == 64 and freeze.get('runner_sha256') == runner_hash,
        'restart runner/freeze source lineage absent/different')
    human = context.get('human_scope', {})
    digest = human.get('sha256')
    expected_grant = {'status':'EXPLICIT_ADDITIONAL_FULL_GENERATION_RESTART_AUTHORIZED',
        'id':'FULL-GENERATION-RESTART-001','lane':RESTART_LANE,'attempts':1,
        'max_native_worker_seconds':16000,'model_API_attempts':28436,'state_adjoints':0,
        'cases':288,'physical_evaluations':3296,'maximum_prefix_tokens':357,
        'scientific_generation_cap':8,'SMOKE_generation_cap':24,'no_further_retry':True,
        'parent_run':'native-bulk-001','training':False,'full_bulk_matrix_retry':True,'worker_sha256':runner_hash}
    require(human.get('child_safe_scope_verified') is True and admission.get('requires_additional_human_authorization') is True
        and isinstance(digest, str) and len(digest) == 64 and digest not in OLD_ATTEMPT_GRANTS
        and digest == admission.get('human_extension_sha256') == context.get('worker', {}).get('human_extension_sha256')
        and human.get('path') == admission.get('human_extension_path')
        and Path(human.get('path','__missing__')).name == 'HUMAN_FULL_GENERATION_RESTART_ALLOWANCE_001.json'
        and all(human.get('payload', {}).get(key) == value for key,value in expected_grant.items()),
        'actual full-generation grant IO/path/sha/scope/terminal binding absent/different')
    require(native.get('loaded_state_sha256') == native.get('before_state_sha256') == native.get('after_state_sha256') == STATE
        and qualification.get('before_state_sha256') == qualification.get('after_state_sha256') == STATE,
        'exact loaded/qualified/final state absent/different')
    require(native.get('loaded_custom_code_sha256') == CUSTOM and native.get('parameter_gradients') == 'NONE'
        and native.get('requires_grad_false') is True, 'native custom-head/frozen-gradient proof absent')
    require(freeze.get('OPEN_VALIDATION_native_exposure_scope') == 'THIS_RUN_ONLY'
        and freeze.get('historical_validation_exposure') == {'prior_run':'native-bulk-001',
            'attempted_or_exposed_cases':59, 'completed_deterministic_cases':58, 'independent_confirmation':False},
        'historical/this-run validation exposure scope differs')
    checks = [e for e in events if e.get('event', '').startswith('INTEGRITY_CHECK_')]
    require([(e.get('event'), e.get('sequence')) for e in checks] ==
        [(name, sequence) for index in range(3) for name, sequence in
         (('INTEGRITY_CHECK_STARTED', 2*index+1), ('INTEGRITY_CHECK_PASSED', 2*index+2))],
        'exact successful START/PASS callback sequence absent/different')
    require(all(number(e.get('parent_elapsed_seconds')) and 0 <= e['parent_elapsed_seconds'] < 16000
        and _utc(e.get('timestamp_utc')) is not None for e in events), 'callback clocks missing/nonfinite/expired')
    require(all(a.get('parent_elapsed_seconds', 1e99) <= b.get('parent_elapsed_seconds', -1)
        for a, b in zip(events, events[1:])), 'callback parent clock moves backward')
    cases = {r['id']:r for r in bank}; dev = {r['id'] for r in bank if r['split'] == 'DEVELOPMENT'}
    summaries = []; decoded = {}
    for index, (name, phase, science_count, physical_count, case_count) in enumerate(INTEGRITY_STAGES, 1):
        blob = proof_bytes.get(name)
        if not isinstance(blob, bytes):
            errors.append('restart integrity: missing proof bytes:'+name); continue
        try:
            proof = json.loads(blob)
        except (ValueError, TypeError):
            errors.append('restart integrity: malformed proof:'+name); continue
        decoded[name] = proof; digest = hashlib.sha256(blob).hexdigest()
        require(proof.get('status') == 'PASS_MODEL_ASSETS_SOURCE_INTEGRITY' and proof.get('phase') == phase
            and proof.get('checkpoint_index') == index and proof.get('start_event_sequence') == 2*index-1
            and proof.get('terminal_event_sequence') == 2*index, 'proof phase/status/sequence differs:'+name)
        require(proof.get('before_state_sha256') == proof.get('after_state_sha256') == STATE
            and proof.get('loaded_custom_code_sha256') == CUSTOM and proof.get('checkpoint_files') == assets
            and proof.get('source_sha256') == dispatch, 'state/custom-code/assets/source differs:'+name)
        require(all(proof.get(k) is True for k in ('eval_mode','requires_grad_false','parameter_gradients_none'))
            and proof.get('parameter_gradients') == 'NONE' and proof.get('trainable_parameters') == 0
            and proof.get('model_forward_calls_added') == 0, 'frozen/no-added-forward proof differs:'+name)
        require(proof.get('counts') == {'scientific_physical_rows':science_count,'case_count':case_count,
            'recorded_physical_evaluations':physical_count,'model_forward_calls_added_by_checkpoint':0},
            'exact structural counts differ:'+name)
        require(number(proof.get('elapsed_seconds')) and proof['elapsed_seconds'] >= 0
            and number(proof.get('parent_elapsed_seconds')) and 0 <= proof['parent_elapsed_seconds'] < 16000
            and _utc(proof.get('ended_utc')) is not None, 'proof cost/clock missing:'+name)
        pair = [e for e in checks if e.get('checkpoint_index') == index]
        if len(pair) != 2:
            errors.append('restart integrity: callback pair missing:'+name)
        else:
            require(pair[0].get('phase') == pair[1].get('phase') == phase
                and pair[1].get('proof_file') == name and pair[1].get('proof_sha256') == digest,
                'callback file/phase/hash differs:'+name)
            require(number(proof.get('parent_elapsed_seconds')) and number(pair[0].get('parent_elapsed_seconds'))
                and number(pair[1].get('parent_elapsed_seconds'))
                and pair[0]['parent_elapsed_seconds'] <= proof['parent_elapsed_seconds'] <= pair[1]['parent_elapsed_seconds'],
                'proof not between START/PASS clocks:'+name)
        commitments = proof.get('journal_commitments', {})
        expected_files = set(CORE_JOURNALS) | (set() if index == 1 else {'frozen_development.json','A5_qualification.json'})
        require(set(commitments) == expected_files, 'journal commitment set differs:'+name)
        prefixes = {}
        for filename in expected_files:
            record, raw = commitments.get(filename, {}), journal_bytes.get(filename)
            length, lines = record.get('bytes'), record.get('lines')
            if (not isinstance(raw, bytes) or type(length) is not int or not 0 < length <= len(raw)
                    or type(lines) is not int or lines <= 0):
                errors.append('restart integrity: missing/invalid byte/line boundary:'+name+'/'+filename); continue
            prefix = raw[:length]
            require(prefix.endswith(b'\n') and len(prefix.splitlines()) == lines
                and hashlib.sha256(prefix).hexdigest() == record.get('sha256'), 'raw prefix commitment differs:'+name+'/'+filename)
            if filename.endswith('.json'):
                require(length == len(raw), 'frozen JSON changed after boundary:'+name+'/'+filename)
                continue
            try:
                prefixes[filename] = [json.loads(line) for line in prefix.splitlines()]
            except (TypeError, ValueError):
                errors.append('restart integrity: malformed committed journal:'+name+'/'+filename)
        physical = [r for r in prefixes.get('outcomes.jsonl', []) if r.get('physical_run') is True]
        scientific = [r for r in physical if r.get('case_id') in cases]
        require(len(physical) == physical_count and len(scientific) == science_count,
            'committed physical/scientific rows differ:'+name)
        subset = bank if index == 3 else [r for r in bank if r['id'] in dev]
        expected = {(r['id'], label, 17, extra) for r in subset for label, extra in
            [('A0',0),('A1',1),*((f'A2:{a:g}',1) for a in GRID),('A3',1)]}
        if index != 1: expected.update((r['id'],'A5',seed,1) for r in subset for seed in (17,29))
        cells = [(r.get('case_id'),policy(r),r.get('seed'),r.get('extra_loops')) for r in scientific]
        require(len(cells) == len(set(cells)) and set(cells) == expected,
            'committed exact scientific cells differ:'+name)
        require(index == 3 or not any(r.get('split') == 'OPEN_VALIDATION' for r in prefixes.get('outcomes.jsonl', [])),
            'validation outcome already present at DEV boundary:'+name)
        require(index != 1 or not any(r.get('arm') == 'A5' for r in physical), 'A5 before deterministic freeze boundary')
        stage_forwards = prefixes.get('forwards.jsonl', [])
        starts = [e for e in stage_forwards if e.get('event') == 'FORWARD_STARTED']
        require(index == 3 or not any(e.get('split') == 'OPEN_VALIDATION' for e in starts),
            'validation forward already present at DEV boundary:'+name)
        clocks = [e.get('parent_elapsed_seconds') for e in stage_forwards if e.get('event') in ('FORWARD_STARTED','FORWARD_COMPLETED')]
        require(bool(clocks) and all(number(t) and 0 <= t < 16000 for t in clocks), 'forward clocks missing:'+name)
        if clocks and all(number(t) for t in clocks) and pair:
            require(max(clocks) <= pair[0].get('parent_elapsed_seconds', -1), 'checkpoint starts before committed forwards finish:'+name)
        sample = proof.get('resources', {})
        require(all(number(sample.get(k)) for k in ('rss_bytes','host_available_bytes','gpu_allocated_bytes','gpu_free_bytes'))
            and 0 <= sample.get('rss_bytes', 1e99) <= 6*GIB and sample.get('host_available_bytes', -1) >= GIB
            and 0 <= sample.get('gpu_allocated_bytes', 1e99) <= 5*GIB and sample.get('gpu_free_bytes', -1) >= GIB,
            'boundary resource proof absent/violated:'+name)
        summaries.append({'phase':phase,'proof_sha256':digest,'committed_physical_rows':len(physical),
            'committed_scientific_rows':len(scientific),'elapsed_seconds':proof.get('elapsed_seconds')})
    keys = ('DEV_deterministic_integrity_sha256','DEV_complete_integrity_sha256','FINAL_integrity_sha256')
    hashes = {name:hashlib.sha256(blob).hexdigest() for name, blob in proof_bytes.items() if isinstance(blob, bytes)}
    for key, (name, *_rest) in zip(keys, INTEGRITY_STAGES):
        require(bool(hashes.get(name)) and native.get(key) == hashes.get(name), 'terminal proof link missing/different:'+key)
    require(bool(hashes.get(INTEGRITY_STAGES[0][0])) and freeze.get(keys[0]) == hashes.get(INTEGRITY_STAGES[0][0]),
        'fixed/pool freeze not bound to deterministic PASS')
    freeze_elapsed = freeze.get('created_parent_elapsed_seconds')
    require(number(freeze_elapsed) and _utc(freeze.get('created_timestamp_utc')) is not None, 'freeze creation clock missing')
    markers = [e for e in events if e.get('event') == 'FREEZE_COMMITTED']
    raw_freeze = journal_bytes.get('frozen_development.json')
    require(len(events) == 7 and len(markers) == 1 and isinstance(raw_freeze, bytes), 'freeze callback missing/extra')
    if len(markers) == 1 and isinstance(raw_freeze, bytes):
        marker = markers[0]
        require(marker.get('freeze_file') == 'frozen_development.json'
            and marker.get('freeze_sha256') == hashlib.sha256(raw_freeze).hexdigest()
            and marker.get('freeze_payload_sha256') == freeze.get('freeze_sha256'), 'freeze callback hash differs')
        require(events.index(marker) == 2 and number(freeze_elapsed)
            and freeze_elapsed <= marker.get('parent_elapsed_seconds', -1), 'freeze callback order/clock differs')
    all_forwards = []
    try:
        all_forwards = [json.loads(line) for line in journal_bytes.get('forwards.jsonl', b'').splitlines()]
    except (TypeError, ValueError):
        errors.append('restart integrity: full forward journal malformed')
    a5_start = [e.get('parent_elapsed_seconds') for e in all_forwards if e.get('event') == 'FORWARD_STARTED' and e.get('arm') == 'A5']
    val_start = [e.get('parent_elapsed_seconds') for e in all_forwards if e.get('event') == 'FORWARD_STARTED' and e.get('split') == 'OPEN_VALIDATION']
    if len(checks) == 6 and number(freeze_elapsed):
        require(checks[1].get('parent_elapsed_seconds', 1e99) <= freeze_elapsed
            and a5_start and all(number(t) and freeze_elapsed <= t for t in a5_start), 'freeze precedes PASS or follows A5 start')
        require(val_start and all(number(t) and checks[3].get('parent_elapsed_seconds', 1e99) <= t for t in val_start),
            'preVAL checkpoint occurs after validation started')
    return errors, {'status':'PASS_RECORDED_RESTART_INTEGRITY' if not errors else 'INCOMPLETE_RESTART_PROOFS',
        'boundaries':summaries,'historical_validation_exposed':True,'independent_confirmation':False,
        'source_original_analyzer_sha256':ORIGINAL_ANALYZER_SHA,'model_forward_calls_added':0 if not errors else None,
        'wall_time_scope':'Measured in worker elapsed; no zero-time assumption or new model forwards.'}


def aggregate(rows, traces, bank, smoke, tokens, freeze, native, worker, qualification, a5, forwards,
              resources=(), budget=(), issues=(), readouts=None, restart=None):
    integrity_errors, integrity = validate_restart_integrity(restart, bank, freeze, native, qualification)
    if native.get('lane') != RESTART_LANE or worker.get('lane') != RESTART_LANE:
        integrity_errors.append('restart integrity: native/parent lane absent/different')
    report = _aggregate_scientific(rows, traces, bank, smoke, tokens, freeze, native, worker, qualification, a5,
        forwards, resources, budget, [*issues,*integrity_errors], readouts)
    report['lane'] = RESTART_LANE
    report['restart_integrity'] = integrity
    report['historical_validation_exposure'] = {'prior_run':'native-bulk-001','attempted_or_exposed_cases':59,
        'completed_deterministic_cases':58,'confirmation':False}
    return report


def run(run_root, out):
    run_root, out = Path(run_root).resolve(strict=True), Path(out).resolve()
    if PACKAGE not in run_root.parents or PACKAGE not in out.parents or run_root in out.parents or out in run_root.parents or run_root == out: raise PermissionError("bulk analysis needs separate campaign paths")
    out.mkdir(parents=True, exist_ok=False); receipt = {"status": "FAIL", "native_model_loads": 0}; inputs, errors = [], []
    try:
        def read(path):
            path = Path(path)
            if not path.is_file(): errors.append("missing bulk artifact:"+path.name); return {}
            inputs.append(path); return json.loads(path.read_text(encoding="utf-8"))
        result = run_root / "result"; admission, review, manifest = read(run_root/"admission.json"), read(run_root/"review.json"), read(run_root/"source_manifest.json")
        def require(condition, message):
            if not condition: errors.append(message)
        require(admission.get("status") == "READY_FOR_BULK_RESTART_FP32_DISPATCH" and review.get("status") == "PASS_BULK_RESTART_FP32_DISPATCH" and review.get("admission_sha256") == sha(run_root/"admission.json"), "restart admission/review binding differs")
        bank_path, token_path = Path(admission["bulk_bank_path"]).resolve(), Path(admission["bulk_token_binding_path"]).resolve()
        if PACKAGE not in bank_path.parents or PACKAGE not in token_path.parents: raise PermissionError("bulk packet paths unscoped")
        bank, token = read(bank_path), read(token_path)
        for key, path in (("bulk_bank_sha256", bank_path), ("bulk_token_binding_sha256", token_path), ("bulk_manifest_sha256", bank_path.parent/"manifest.json"), ("bulk_audit_sha256", bank_path.parent/"audit.json"), ("bulk_positive_controls_sha256", bank_path.parent/"positive_controls.json")):
            inputs.append(path); require(path.is_file() and sha(path) == admission.get(key), "bulk source packet hash differs:"+key)
        require(sha(PACKAGE/"bulk_bank.py") == admission.get("bulk_source_sha256") and sha(PACKAGE/"open_bank.py") == admission.get("bulk_normalizer_source_sha256"), "bulk strict scorer/normalizer differs")
        if sha(PACKAGE/"bulk_bank.py") != admission.get("bulk_source_sha256"): raise RuntimeError("unbound bulk scorer cannot regrade")
        require(bool(admission.get('dispatch_hashes')), 'bulk dispatched closure absent')
        for name in ('bulk_analysis.py','bulk_restart_analysis.py','BULK_RESTART_ANALYSIS_PLAN_001.md',
                     'analysis.py','gpu_analysis.py','open_analysis.py','bulk_bank.py','open_bank.py',
                     'full_generation_restart_authority.py'):
            path = PACKAGE/name; inputs.append(path)
            require(path.is_file() and admission.get('dispatch_hashes', {}).get(path.relative_to(REPO).as_posix()) == sha(path),
                'restart required reader/helper/plan binding missing/different:'+name)
        errors.extend(validate_source_manifest(admission,manifest))
        for relative, digest in admission.get("dispatch_hashes", {}).items():
            path = (REPO/relative).resolve()
            allowed = PACKAGE in path.parents or any(path.is_relative_to((REPO/folder).resolve()) for folder in
                ('experiments/rineya_taps_adaptive_recurrence_v1','experiments/anchor_routed_middle_recurrence_v1','Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src')) or path == (REPO/'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl').resolve()
            if not allowed: raise PermissionError('unscoped bulk dispatch reference')
            require(path.is_file() and sha(path) == digest, 'bulk actual source/metadata differs:'+relative)
            if path.is_file(): inputs.append(path)
        model_path = PACKAGE/"runs/binding-001/MODEL_DATA_BINDING.json"; smoke_path = PACKAGE/"runs/binding-001/PROMPT_AUDIT.json"
        for path in (model_path, smoke_path): require(admission.get("dispatch_hashes", {}).get(path.relative_to(REPO).as_posix()) == sha(path), "bulk original model/SMOKE metadata differs")
        binding = read(model_path); smoke = [row for row in read(smoke_path) if row["split"] == "SMOKE"]
        native, worker, qualifier, a5, freeze, registration = [read(path) for path in (result/"receipt.json", run_root/"terminal_receipt.json", result/"qualification.json", result/"A5_qualification.json", result/"frozen_development.json", result/"registration.json")]
        require(registration.get("lane") == RESTART_LANE and registration.get("admission_sha256") == sha(run_root/"admission.json") and registration.get("grid") == list(GRID)
            and registration.get("deterministic_seeds") == [17] and registration.get("A5_seeds") == [17,29], "bulk once-only/control registration differs")
        taps_path = REPO/'experiments/rineya_taps_adaptive_recurrence_v1/configs/inference_v1.json'
        history_path = REPO/'experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json'
        errors.extend(validate_registration(registration, admission, binding, token, freeze, read(history_path)['config'], read(taps_path)['controller']))
        require(native.get('lane') == worker.get('lane') == RESTART_LANE
            and Path(worker.get('run','__missing__')).resolve() == run_root, 'bulk bound parent/worker lane/path differs')
        for name in ('BULK_NATIVE_REGISTRATION_001.md','BULK_ANALYSIS_PLAN_001.md'):
            path = PACKAGE/name; inputs.append(path)
            require(admission.get('dispatch_hashes',{}).get(path.relative_to(REPO).as_posix()) == sha(path), 'bulk prospective endpoints/protocol differs:'+name)
        token_rows = token.get('rows',[]); token_by_id = {row.get('case_id'): row for row in token_rows}
        require(len(token_rows)==288 and set(token_by_id)=={raw['id'] for raw in bank}, 'bulk token packet incomplete/duplicate')
        for raw in bank:
            bound=token_by_id.get(raw['id'],{}); ids=bound.get('input_token_ids',[])
            require(bound.get('raw_prompt')==raw['prompt'] and 1<=len(ids)<=350 and bound.get('input_tokens')==len(ids)
                and bound.get('max_new_tokens')==8 and bound.get('max_forward_prefix_tokens')==len(ids)+7
                and all(bound.get(field)==raw[field] for field in ('parent_id','split','category','variant_id','difficulty_stratum','generator_family')), 'bulk source-rendered token/parent/shape metadata differs')
        packet_manifest, audit, positive = read(bank_path.parent/'manifest.json'), read(bank_path.parent/'audit.json'), read(bank_path.parent/'positive_controls.json')
        require(packet_manifest.get('source_sha256')==admission.get('bulk_source_sha256') and packet_manifest.get('normalizer_source_sha256')==admission.get('bulk_normalizer_source_sha256')
            and packet_manifest.get('status')=='PASS_PREPARATION_ONLY' and audit.get('status')=='PASS_PREPARATION_ONLY'
            and audit.get('failures')==[] and audit.get('positive_control_correct')==288
            and len(positive)==288 and {row['id'] for row in positive}=={raw['id'] for raw in bank} and all(row.get('correct') is True for row in positive), 'bulk exact generator/positive-control apparatus differs')
        forecast = read(result/'matrix_forecast.json')
        require(forecast.get('admitted') is True and forecast.get('forecast_factor')==1.5
            and forecast.get('parent_deadline_seconds')==admission.get('hard_deadline_seconds')
            and number(forecast.get('max_observed_warm_forward_seconds')) and forecast['max_observed_warm_forward_seconds']>0
            and number(forecast.get('remaining_forecast_seconds')) and number(forecast.get('elapsed_seconds'))
            and forecast.get('remaining_forecast_seconds',0)+forecast.get('elapsed_seconds',0)+forecast.get('finalization_reserve_seconds',0)<=admission.get('hard_deadline_seconds',0), 'bulk fresh complete matrix forecast absent/refused')
        require(freeze.get("bank_sha256") == sha(bank_path) and freeze.get("token_binding_sha256") == sha(token_path) and freeze.get("runner_sha256") == admission.get("dispatch_hashes", {}).get((PACKAGE/"bulk_restart_native.py").relative_to(REPO).as_posix()), "restart calibration source/data freeze differs")
        require(native.get("loaded_state_sha256") == binding.get("historical_expected_loaded_fp32_state_sha256") and qualifier.get("before_state_sha256") == qualifier.get("after_state_sha256") == native.get("loaded_state_sha256"), "bulk actual model state identity differs")
        code = Path(native.get("loaded_custom_code_path", "__missing__")).resolve()
        require((result/"hf_cache").resolve() in code.parents and code.is_file(), "bulk actual custom code path missing")
        if (result/"hf_cache").resolve() in code.parents and code.is_file(): inputs.append(code); require(sha(code) == native.get("loaded_custom_code_sha256") == binding["checkpoint_files"]["modeling_supa_a11.py"], "bulk actual custom code differs")
        journals = []
        for path in (result/"outcomes.jsonl", result/"trajectories.jsonl", result/"forwards.jsonl", run_root/"resources.jsonl", PACKAGE/"BUDGET_LEDGER.jsonl",result/'readout_pairs.jsonl'):
            data, failure = read_journal(path); journals.append(data); errors.extend(failure)
            if path.is_file(): inputs.append(path)
        rows, traces, forwards, resources, budget, readouts = journals
        prefix = bytearray()
        if (result/"trajectories.jsonl").is_file():
            for line in (result/"trajectories.jsonl").read_bytes().splitlines(keepends=True):
                try: event = json.loads(line)
                except ValueError: errors.append("bulk partial trajectory freeze line"); break
                if event.get("arm") == "A5": break
                prefix.extend(line)
        require(freeze.get("generation_trajectory_sha256_at_freeze") == hashlib.sha256(prefix).hexdigest(), "bulk generation-only DEV pool snapshot differs")
        human_scope, human_errors = read_human_scope(admission, worker, inputs); errors.extend(human_errors)
        restart = {'admission':admission,'registration':registration,'worker':worker,'human_scope':human_scope,
                   'proof_bytes':{},'journal_bytes':{},'integrity_journal':[]}
        for name, *_rest in INTEGRITY_STAGES:
            path = result/name
            if path.is_file(): inputs.append(path); restart['proof_bytes'][name] = path.read_bytes()
            else: errors.append('missing restart integrity artifact:'+name)
        for name in (*CORE_JOURNALS,'frozen_development.json','A5_qualification.json'):
            path = result/name
            if path.is_file(): restart['journal_bytes'][name] = path.read_bytes()
        path = result/'integrity_journal.jsonl'
        restart['integrity_journal'], failure = read_journal(path); errors.extend(failure)
        if path.is_file(): inputs.append(path)
        require(sha(PACKAGE/'bulk_analysis.py') == ORIGINAL_ANALYZER_SHA, 'old scientific analyzer changed')
        report = aggregate(rows, traces, bank, smoke, token["rows"], freeze, native, worker, qualifier, a5, forwards, resources, budget, errors, readouts, restart)
        evidence = {"analyzer_sha256": sha(__file__), "helpers": {name: sha(PACKAGE/name) for name in ("analysis.py","gpu_analysis.py","open_analysis.py","bulk_bank.py","open_bank.py")}, "inputs": {str(path): sha(path) for path in inputs if path.is_file()}, "native_model_loads": 0,
                    "git_head": subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()}
        for name, value in (("REPORT.json",report),("MANIFEST.json",evidence)):
            with (out/name).open("x",encoding="utf-8") as stream: json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write("\n")
        (out/"REPORT.md").write_text("# Bulk parent-world analysis\n\n"+report["status"]+"\n\nValidation primary:48 parent units, frozen DEV comparator, equal six-category weights.\nNo source-confirmation, broad intelligence or production-speed claim.\n",encoding="utf-8")
        receipt.update(status="ANALYZED",matrix_complete=report["complete"],report_sha256=sha(out/"REPORT.json"))
    except Exception as error: receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out/"receipt.json").open("x",encoding="utf-8") as stream: json.dump(receipt,stream,indent=2,sort_keys=True); stream.write("\n")
    return 0 if receipt["status"] == "ANALYZED" else 2


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); raise SystemExit(run(args.run_root,args.out))
