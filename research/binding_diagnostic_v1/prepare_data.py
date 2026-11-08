"""Prospective contact rules, create-only diagnostic preparation, no model work.

Run `rules` only after all generators/tests are frozen; then `build` with the
printed rules SHA under the pinned RAB Python. Failed candidates are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

BASE = Path('C:/Users/clclc/BusinessProject')
RECORDS = BASE / 'rineya-dataset-retention-records'
ROOT = RECORDS / 'binding-diagnostic001-20260920'
HERE = Path(__file__).resolve().parent
PILOT = BASE / 'ChatBubble-retention-doselora-resume-20260920'
PYTHON = BASE / '.venv-rab001-cu126/Scripts/python.exe'
SOURCE = BASE / 'Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze'
MANIFEST = RECORDS / 'lcr-exclusion-manifest/LCR_EXCLUSION_MANIFEST.json'
MANIFEST_SHA = '54b02b35c37b1cc842dbf413aac56980d3d3ade203f3ea5092bb332416914858'
TEXT_CONTACT = PILOT / 'experiments/reasoning_posttraining_v1/text_contact.py'
TEXT_CONTACT_SHA = 'f42ce00d9764043282a4dfeb9810e1938cf85b4212bccf851c770dc8c39a895e'
BANKS = {
    'dose': (RECORDS / 'process-transition-probe-001/CANDIDATE_VIEWS.jsonl', '45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e', 384),
    'select': (RECORDS / 'select-views-001/SELECT_VIEWS.jsonl', '7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3', 384),
    'arithmetic_panel': (PILOT / 'experiments/architecture_benchmark_001/data/monitor_probe.jsonl', '5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266', 150),
    'binding_v2_train': (RECORDS / 'binding-signal-prep-20260920/candidate-v2/TRAIN.jsonl', 'c50d21b5e47fd819f1bee8171922feff44e0f195515f001e7bf9b829ea59a8ea', 768),
    'binding_v2_dev': (RECORDS / 'binding-signal-prep-20260920/candidate-v2/DEV.jsonl', '062c13ad3872d8d62e4f51f935095e46de3188a23d4c6a03d4d3fc19fa968cf0', 384),
}
SIDECARS = {
    'config.json': '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
    'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
}
FROZEN_MODELS = {
    'source': '962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b',
    'lora': 'e025d88a1c18b1127bb58069e478ce2f3ee9609382fdcdcc10864018f9c4bae4',
    'mixed': '894190b696498aa29b19285c86a7335070b051385c9135dc46d937cb0ba55f54',
}
LITERAL_FIELDS = ('forbidden_proper_nouns', 'forbidden_titles', 'forbidden_domains', 'forbidden_task_type_names')
ID_FIELDS = {'id', 'row_id', 'parent_id', 'group_id', 'source_id', 'base_world_id'}
ID_LIST_FIELDS = {'row_ids', 'parent_ids', 'group_ids', 'source_ids', 'base_world_ids'}
FORBIDDEN_DOMAINS = {'arcane_academy', 'coastal_council', 'detective_bureau', 'expedition', 'family_enterprise', 'museum', 'orbital_station', 'research_institute'}
CODE_NAMES = ('binding.py', 'arithmetic.py', 'prepare_data.py')
TEST_NAMES = ('test_binding.py', 'test_arithmetic.py', 'test_prepare_data.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe input path')
    require(path.is_file() and path.name != 'model.safetensors' and path.suffix not in ('.bin', '.pt', '.safetensors'), 'missing input or prohibited weight read')
    return path


def sha(path):
    with regular(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(regular(path).read_text(encoding='utf-8-sig'))


def write_new(path, value, *, jsonl=False):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe output path')
    with path.open('xb') as stream:
        if jsonl:
            for row in value:
                stream.write(canonical(row) + b'\n')
        else:
            stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def no_model_runtime():
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'flax')), 'forbidden model runtime imported')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, regular(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def approved_manifest():
    require(sha(MANIFEST) == MANIFEST_SHA, 'approved LC metadata changed')
    manifest = read(MANIFEST)
    require(all(manifest[k] is False for k in ('contains_questions', 'contains_answers', 'contains_answer_key')), 'exclusion aid contains protected content')
    require(set(manifest['forbidden_domains']) == FORBIDDEN_DOMAINS, 'semantic domain review inventory changed')
    return manifest


def rules_record():
    # Only approved metadata is examined here, never generated candidate text.
    approved_manifest()
    return {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_PROSPECTIVE_CONTACT_RULES', 'experiment': 'BINDING-DIAGNOSTIC-001',
            'seed': 31, 'candidate': 'candidate-v1', 'max_new_tokens': 24, 'max_sequence': 2048,
            'mandatory_zero': ['approved_LC_user_literal_terms', 'exact_cross_field_ids', 'full_NFKC_casefold_user_prompt', 'full_digit_blinded_user_template'],
            'disclosure_only': ['native_user_8gram', 'native_digit_blinded_short_stem'],
            'filter_drop_or_resample': False, 'code_sha256': {n: sha(HERE / n) for n in CODE_NAMES},
            'tests_sha256': {n: sha(HERE / n) for n in TEST_NAMES},
            'approved_manifest': {'path': str(MANIFEST), 'sha256': MANIFEST_SHA},
            'native_contact_helper': {'path': str(TEXT_CONTACT), 'sha256': TEXT_CONTACT_SHA},
            'banks': {k: {'path': str(p), 'sha256': h, 'rows': n} for k, (p, h, n) in BANKS.items()},
            'semantic_domain_review': {'status': 'PASS_NAMED_METADATA_SCOPE', 'forbidden_domains': sorted(FORBIDDEN_DOMAINS),
                'review_kind': 'automated prospective metadata review; not human review or universal semantic certificate',
                'glider_hangar': 'Stationary glider-hangar task assignments; no expedition, orbital operation, research institution, council, academy, detective, museum, or family-ownership narrative.',
                'orchard_press': 'Fruit-processing workplace assignments; no family ownership/succession narrative, expedition, institution, council, academy, detective, museum, or orbital setting.'},
            'unknown_coverage': {'new_native_LC_contact': 'UNVERIFIED_NO_NEW_RAW_LC_READ', 'legacy_LC_stopword_fingerprints': 'UNVERIFIED', 'reserved_bank_coverage': 'UNVERIFIED', 'unknown_overlap_counts': None},
            'tokenizer_python': str(PYTHON), 'tokenizer_versions': {'transformers': '4.51.3', 'tokenizers': '0.21.4'},
            'source_sidecars_sha256': SIDECARS, 'training_permission': False, 'model_execution_permission': False,
            'frozen_models_weight_sha256': FROZEN_MODELS,
            'independent_confirmation': False, 'release_permitted': False}


def freeze_rules(root=ROOT):
    root = Path(root).absolute()
    require(root.is_dir(), 'diagnostic record root must already exist')
    require(not (root / 'candidate-v1').exists(), 'candidate already exists before prospective rules')
    path = root / 'CONTACT_RULES.json'
    write_new(path, rules_record())
    return sha(path)


def user_turns(row):
    messages = row.get('prompt', row.get('messages'))
    require(isinstance(messages, list), 'chat messages required')
    turns = [m['content'] for m in messages if m['role'] == 'user']
    require(turns and all(isinstance(t, str) and t.strip() for t in turns), 'nonempty user text required')
    return turns


def normal(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def literal_normal(text):
    return ' '.join(re.sub(r'[\W_]+', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def identities(row):
    result = set()
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ID_FIELDS:
                    require(isinstance(child, str) and child.strip(), 'invalid identity field')
                    result.add(child)
                elif key in ID_LIST_FIELDS:
                    require(isinstance(child, list) and all(isinstance(x, str) for x in child), 'invalid identity list')
                    result.update(child)
                elif isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(row)
    return result


def prompt_key(row, *, blind=False):
    turns = [normal(t) for t in user_turns(row)]
    return digest([re.sub(r'\d+', '#', t) for t in turns] if blind else turns)


def compare_bank(rows, bank, guard):
    id_index, exact, blind = defaultdict(set), defaultdict(set), defaultdict(set)
    shingle_index = defaultdict(set)
    for r in bank:
        for key in identities(r):
            id_index[key].add(r['id'])
        exact[prompt_key(r)].add(r['id'])
        blind[prompt_key(r, blind=True)].add(r['id'])
        for turn in user_turns(r):
            words = re.sub(r'[^a-z0-9 ]+', ' ', turn.lower()).split()
            for index in range(max(0, len(words) - 7)):
                shingle_index[' '.join(words[index:index + 8])].add(r['id'])
    # Retain the original native disclosure policy and discard its filtered output.
    before = digest(rows)
    _, native = guard.audit_and_filter(rows, bank)
    require(digest(rows) == before, 'native helper changed candidate')
    disclosures = {r['id']: r for r in native['dropped']}
    contacts = []
    for row in rows:
        found = {'exact_cross_field_ids': sorted({v for key in identities(row) for v in id_index.get(key, ())}),
                 'full_NFKC_casefold_user_prompt': sorted(exact.get(prompt_key(row), ())),
                 'full_digit_blinded_user_template': sorted(blind.get(prompt_key(row, blind=True), ()))}
        mandatory = {k: v for k, v in found.items() if v}
        n = disclosures.get(row['id'], {})
        disclosed = [x for x in n.get('reasons', []) if x in ('user_8gram', 'digit_blinded_panel_stem')]
        if mandatory or disclosed:
            matching_grams = {}
            for turn in user_turns(row):
                words = re.sub(r'[^a-z0-9 ]+', ' ', turn.lower()).split()
                for index in range(max(0, len(words) - 7)):
                    gram = ' '.join(words[index:index + 8])
                    if gram in shingle_index:
                        matching_grams[hashlib.sha256(gram.encode()).hexdigest()] = sorted(shingle_index[gram])
            contacts.append({'candidate_id': row['id'], 'parent_id': row['parent_id'], 'mandatory_matches': mandatory,
                             'disclosure_reasons': disclosed, 'native_8gram_hash_matches': matching_grams,
                             'native_short_stem_bank_ids': n.get('matched_panel_stem_ids', [])})
    return {'candidate_rows': len(rows), 'bank_rows': len(bank), 'mandatory_contact_rows': sum(bool(r['mandatory_matches']) for r in contacts),
            'disclosure_contact_rows': sum(bool(r['disclosure_reasons']) for r in contacts), 'contacts': contacts}


def literal_audit(rows, manifest):
    texts = [(r['id'], [' ' + literal_normal(t) + ' ' for t in user_turns(r)]) for r in rows]
    contacts = []
    for field in LITERAL_FIELDS:
        for term in manifest[field]:
            token = literal_normal(term)
            require(bool(token), 'empty approved exclusion term')
            hits = [key for key, turns in texts if any(' ' + token + ' ' in text for text in turns)]
            if hits:
                contacts.append({'field': field, 'term': term, 'candidate_ids': hits})
    return {'mandatory_contact_rows': len({key for r in contacts for key in r['candidate_ids']}), 'contacts': contacts}


def validate_bundles(binding, arithmetic, binding_module, arithmetic_module):
    binding_module.validate_rows(binding)
    arithmetic_module.validate_rows(arithmetic)
    rows, gold = [], []
    for label, bundle, count, parents, views in [('binding', binding, 512, 32, 16), ('arithmetic', arithmetic, 384, 96, 4)]:
        policy, targets = bundle['policy_inputs'], bundle['gold']
        require(len(policy) == len(targets) == count and len(bundle['metadata']) == parents, 'section geometry differs')
        require(Counter(Counter(r['parent_id'] for r in policy).values()) == {views: parents}, 'parent derivatives differ')
        require(len({m['parent_id'] for m in bundle['metadata']}) == parents and
                {m['parent_id'] for m in bundle['metadata']} == {r['parent_id'] for r in policy}, 'parent metadata coverage differs')
        for row, target in zip(policy, targets):
            require(set(row) == {'id', 'parent_id', 'prompt'} and row['id'] == target['id'] and row['parent_id'] == target['parent_id'], 'gold/input join differs')
            require([m['role'] for m in row['prompt']] == ['system', 'user'] and all(set(m) == {'role', 'content'} for m in row['prompt']), 'gold or extra fields in policy')
            parsed = binding_module.parse_policy_input(row['prompt']) if label == 'binding' else arithmetic_module.parse_policy_input(row['prompt'])
            if label == 'binding':
                require(parsed['reference'] == target['reference'] and parsed['state'] == target['state'], 'independent binding oracle differs')
            else:
                require(arithmetic_module.reference_text(parsed) == target['reference'] and arithmetic_module.typed_targets(parsed) == target['targets'], 'independent arithmetic oracle/typed targets differ')
        rows.extend(policy)
        # The section tag is an offline join discriminator. Preserve the original
        # arithmetic module's contract and add it only to copied combined gold.
        gold.extend({**target, 'section': label} for target in targets)
    require(len(rows) == len({r['id'] for r in rows}) == 896 and len({r['parent_id'] for r in rows}) == 128, 'combined identities overlap')
    expected = set(itertools.product(binding_module.DOMAINS, (0, 1), ('missing', 'vacant'), (False, True)))
    actual = defaultdict(set)
    for g in binding['gold']:
        actual[g['parent_id']].add((g['domain'], g['template'], g['pair'], g['answerable']))
    require(all(v == expected for v in actual.values()) and sum(g['answerable'] for g in binding['gold']) == 256, 'binding factors/balance differ')
    return rows, gold, {'binding': binding['metadata'], 'arithmetic': arithmetic['metadata']}


def compile_inputs(rows, gold, tokenizer):
    compiled, lengths, answers = [], [], []
    for row, target in zip(rows, gold):
        ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        answer = tokenizer.encode(target['reference'], add_special_tokens=False)
        require(isinstance(ids, list) and ids and all(type(t) is int and 0 <= t < 151936 for t in ids), 'invalid prompt token IDs')
        require(len(ids) + 24 <= 2048 and len(answer) + 1 <= 24, 'prompt/answer exceeds frozen budget; truncation prohibited')
        compiled.append({**row, 'prompt_ids': ids})
        lengths.append(len(ids)); answers.append(len(answer))
    require(all(set(row) == {'id', 'parent_id', 'prompt', 'prompt_ids'} for row in compiled), 'gold leaked to compiled policy inputs')
    return compiled, {'rows': len(rows), 'parents': len({r['parent_id'] for r in rows}), 'prompt_tokens': sum(lengths),
                     'max_prompt_tokens': max(lengths), 'max_answer_content_tokens': max(answers), 'answer_content_tokens': sum(answers),
                     'max_new_tokens': 24, 'max_sequence': 2048, 'reserved_generated_tokens_per_model': 24 * len(rows),
                     'reserved_forward_calls_per_model': 24 * len(rows), 'reserved_processed_input_tokens_per_model': sum(lengths) + 23 * len(rows),
                     'truncated_rows': 0}


def prepare(rules_sha, root=ROOT):
    root = Path(root).absolute(); rules_path = root / 'CONTACT_RULES.json'
    require(sha(rules_path) == rules_sha, 'prospective rules pin differs')
    rules = read(rules_path)
    require(rules == rules_record(), 'rules/code/pins changed after freeze')
    require(Path(sys.executable).resolve() == PYTHON.resolve(), 'pinned RAB Python required')
    no_model_runtime()
    out = root / rules['candidate']
    require(not out.exists() and not out.is_symlink(), 'candidate outputs must be fresh')
    out.mkdir()
    stage = 'generation'
    try:
        write_new(out / 'PROSPECTIVE_BINDING.json', {'contact_rules_path': str(rules_path), 'contact_rules_sha256': rules_sha, 'code_sha256': rules['code_sha256']})
        binding_module = load_module('diagnostic_frozen_binding', HERE / 'binding.py')
        arithmetic_module = load_module('diagnostic_frozen_arithmetic', HERE / 'arithmetic.py')
        binding, arithmetic = binding_module.build_rows(seed=31), arithmetic_module.build_rows(seed=31)
        require(set(binding_module.DOMAINS) == {'glider_hangar', 'orchard_press'}, 'reviewed domain inventory differs')
        rows, gold, metadata = validate_bundles(binding, arithmetic, binding_module, arithmetic_module)
        for name, value, is_lines in [('RAW_INPUTS.jsonl', rows, True), ('GOLD.jsonl', gold, True), ('METADATA.json', metadata, False)]:
            write_new(out / name, value, jsonl=is_lines)
        raw_hashes = {n: sha(out / n) for n in ('RAW_INPUTS.jsonl', 'GOLD.jsonl', 'METADATA.json')}
        write_new(out / 'BUILD.json', {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_BUILD', 'status': 'PASS_INTERNAL_STRUCTURE_ONLY',
            'seed': 31, 'rows': {'binding': 512, 'arithmetic': 384}, 'parents': {'binding': 32, 'arithmetic': 96},
            'binding_answerable': 256, 'independent_policy_oracles_verified': True, 'complete_paired_factor_inventories': True,
            'files_sha256': raw_hashes, 'code_sha256': rules['code_sha256'], 'contact_rules_sha256': rules_sha,
            'training_permission': False, 'model_execution_permission': False})
        stage = 'contact_audit'
        manifest = approved_manifest()
        require(sha(TEXT_CONTACT) == TEXT_CONTACT_SHA, 'native contact helper differs')
        guard = load_module('diagnostic_native_contact', TEXT_CONTACT)
        require(guard.POLICY == 'user_text_contact_v1_id_8gram_digit_blinded_stem', 'native contact policy differs')
        bank_records, comparisons = {}, {}
        for label, (path, pin, count) in BANKS.items():
            require(sha(path) == pin, 'named open bank changed: ' + label)
            bank = [decode(line) for line in regular(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            require(len(bank) == len({r['id'] for r in bank}) == count, 'named bank inventory differs')
            bank_records[str(path)] = pin
            comparisons[label] = compare_bank(rows, bank, guard)
        literal = literal_audit(rows, manifest)
        passed = not literal['mandatory_contact_rows'] and not any(v['mandatory_contact_rows'] for v in comparisons.values())
        contact = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_CONTACT_AUDIT', 'status': 'PASS_MANDATORY_CONTACT_RULES' if passed else 'BLOCKED_MANDATORY_CONTACT',
                   'contact_rules_sha256': rules_sha, 'candidate_files_sha256': raw_hashes, 'approved_manifest_sha256': MANIFEST_SHA,
                   'bank_files_sha256': bank_records, 'native_helper_sha256': TEXT_CONTACT_SHA, 'literal': literal, 'comparisons': comparisons,
                   'short_native_contacts_disclosure_only': True, 'candidate_rows_modified': False, 'filtering_performed': False,
                   'unknown_coverage': rules['unknown_coverage'], 'raw_LC_or_ANY_READ_files_opened': False, 'model_calls': 0}
        write_new(out / 'CONTACT_AUDIT.json', contact)
        require(passed, 'mandatory contact gate failed; preserve entire candidate')
        stage = 'tokenizer_compilation'
        os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TORCH='0', USE_TF='0', USE_FLAX='0')
        assets = {n: sha(SOURCE / n) for n in SIDECARS}
        require(assets == SIDECARS, 'local tokenizer/config assets differ')
        versions = {n: importlib.metadata.version(n) for n in rules['tokenizer_versions']}
        require(versions == rules['tokenizer_versions'], 'pinned tokenizer library versions differ')
        from transformers import AutoTokenizer
        no_model_runtime()
        tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tokenizer.is_fast and tokenizer.eos_token_id == 151645, 'fast tokenizer/EOS differs')
        compiled, token_counts = compile_inputs(rows, gold, tokenizer)
        no_model_runtime()
        write_new(out / 'DIAGNOSTIC_INPUTS.jsonl', compiled, jsonl=True)
        write_new(out / 'TOKEN_AUDIT.json', {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_TOKEN_AUDIT', 'status': 'PASS_TOKENIZER_ONLY',
            'totals': token_counts, 'runtime': versions, 'python_path': str(PYTHON), 'source_sidecars_sha256': assets,
            'input_files_sha256': raw_hashes, 'compiled_input_sha256': sha(out / 'DIAGNOSTIC_INPUTS.jsonl'),
            'gold_sha256': sha(out / 'GOLD.jsonl'), 'policy_fields': ['id', 'parent_id', 'prompt', 'prompt_ids'],
            'gold_separate': True, 'model_runtime_imported': False, 'model_weights_read': False, 'model_calls': 0})
        stage = 'final_custody'
        require(sha(rules_path) == rules_sha and rules_record() == rules, 'frozen rules/code changed during preparation')
        require(all(sha(out / n) == pin for n, pin in raw_hashes.items()), 'candidate changed during preparation')
        require(all(sha(p) == pin for p, pin in bank_records.items()) and {n: sha(SOURCE / n) for n in SIDECARS} == assets, 'named inputs/assets changed during preparation')
        no_model_runtime()
        evidence = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        admission = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_SCOPED_DATA_ADMISSION', 'status': 'ADMITTED_FOR_NAMED_EXPLORATORY_DIAGNOSTIC',
            'scope': 'Frozen fresh binding/arithmetic diagnostics on source, high-dose LoRA and mixed models; no training, model selection, confirmation or LC/reserved-bank claim.',
            'evidence_sha256': evidence, 'code_sha256': rules['code_sha256'], 'contact_rules_sha256': rules_sha,
            'frozen_models_weight_sha256': rules['frozen_models_weight_sha256'],
            'rows_per_model': 896, 'binding_parent_worlds': 32, 'arithmetic_parent_worlds': 96, 'policy_input_path': str(out / 'DIAGNOSTIC_INPUTS.jsonl'),
            'policy_input_sha256': sha(out / 'DIAGNOSTIC_INPUTS.jsonl'), 'gold_path': str(out / 'GOLD.jsonl'), 'gold_sha256': sha(out / 'GOLD.jsonl'),
            'mandatory_checks_passed': True, 'short_contact_disclosures_retained': True, 'external_exclusions_complete': False,
            **rules['unknown_coverage'], 'semantic_contamination_certificate': False, 'independent_confirmation': False,
            'human_data_review': 'NOT_PERFORMED', 'training_permission': False, 'model_execution_permission': False, 'release_permitted': False,
            'required_before_model_work': 'Separate prospective analysis/measurement registration, exact plan and approved numeric grant, frozen models and supervised unchanged guards.'}
        write_new(out / 'DATA_ADMISSION.json', admission)
        return {'status': admission['status'], 'candidate': str(out), 'admission_sha256': sha(out / 'DATA_ADMISSION.json'), 'totals': token_counts}
    except Exception as exc:
        write_new(out / 'FAILURE.json', {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_PREPARATION_FAILURE', 'status': 'FAILED_OR_BLOCKED',
            'stage': stage, 'error_type': type(exc).__name__, 'message': str(exc), 'contact_rules_sha256': rules_sha,
            'partial_outputs_preserved': True, 'training_permission': False, 'model_execution_permission': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('rules', 'build'))
    parser.add_argument('--contact-rules-sha256')
    args = parser.parse_args()
    if args.phase == 'rules':
        require(args.contact_rules_sha256 is None, 'rules phase does not accept a prior pin')
        result = {'status': 'PROSPECTIVE_RULES_FROZEN', 'path': str(ROOT / 'CONTACT_RULES.json'), 'sha256': freeze_rules()}
    else:
        require(isinstance(args.contact_rules_sha256, str) and re.fullmatch('[0-9a-f]{64}', args.contact_rules_sha256), 'build requires exact prospective rules SHA256')
        result = prepare(args.contact_rules_sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
