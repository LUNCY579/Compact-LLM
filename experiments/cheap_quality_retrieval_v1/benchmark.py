"""Prospectively fixed synthetic retrieval benchmark. No model loading or answer scoring."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SCOPE = ('Open synthetic evidence-retention construction diagnosis only. No answer correctness, '
         'model gains, native total-envelope qualification, or Lost in the Middle replication.')
PUBLIC_KEYS = {'case_id', 'question', 'sentences'}
SENTENCE_KEYS = {'sentence_id', 'text'}
METHODS = ('lead_prefix', 'seeded_random', 'bm25_one_hop', 'bm25_two_pass')
FORBIDDEN_PACKAGES = ('torch', 'transformers', 'tensorflow', 'flax', 'jax', 'numpy', 'safetensors')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':')).encode('utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def cpu_import_audit(event, args):
    """Refuse model/tensor packages before import; not an adversarial sandbox."""
    if event == 'import' and args and isinstance(args[0], str):
        if args[0].split('.')[0] in FORBIDDEN_PACKAGES:
            raise RuntimeError('forbidden CPU-only package import: '+args[0])


def assert_cpu_imports_clean():
    present = sorted(name for name in sys.modules
                     if name.split('.')[0] in FORBIDDEN_PACKAGES)
    require(not present, 'CPU-only worker already imported forbidden packages: '+repr(present))


def install_cpu_import_guard():
    assert_cpu_imports_clean()
    sys.addaudithook(cpu_import_audit)
    return {'status': 'INSTALLED_BEFORE_TOKENIZER_IMPORT',
            'forbidden_packages': list(FORBIDDEN_PACKAGES),
            'scope': 'ordinary runtime import containment; not an adversarial security sandbox'}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_jsonl(path, values):
    with Path(path).open('wb') as handle:
        for value in values:
            handle.write(canonical(value)+b'\n')


def public_view(case):
    """Gold/evaluator metadata is rejected, never silently filtered for a retriever."""
    require(set(case) == PUBLIC_KEYS, 'public case contains missing/extra fields or evaluator metadata')
    require(isinstance(case['question'], str) and isinstance(case['case_id'], str), 'invalid public IDs/question')
    require(isinstance(case['sentences'], list) and case['sentences'], 'public factual sentences missing')
    seen = set()
    for sentence in case['sentences']:
        require(set(sentence) == SENTENCE_KEYS, 'public sentence contains evaluator metadata')
        require(isinstance(sentence['sentence_id'], str) and isinstance(sentence['text'], str),
                'invalid public sentence')
        require(sentence['sentence_id'] not in seen, 'duplicate public sentence ID')
        seen.add(sentence['sentence_id'])
    return case['question'], case['sentences']


def construct_worlds(config):
    """Return public cases and separate gold records; constructor does not run algorithms."""
    require(config['single_hop_worlds'] == config['two_hop_worlds'] == 32, 'world inventory changed')
    require(config['irrelevant_sentences_per_world'] == 64, 'irrelevant fact inventory changed')
    publics, evaluator = [], []
    for world_index in range(64):
        rng = random.Random(config['construction_seed'] + world_index * 1009)
        entity_codes = rng.sample(range(1000, 10000), 99)
        couriers, parcels, lockers = [entity_codes[index:index+33] for index in (0, 33, 66)]
        facts = []
        for index in range(32):
            facts.extend([f'Courier C{couriers[index]} carries parcel P{parcels[index]}.',
                          f'Parcel P{parcels[index]} is stored in locker L{lockers[index]}.'])
        rng.shuffle(facts)
        two_hop = world_index >= 32
        supports = ([f'Courier C{couriers[32]} carries parcel P{parcels[32]}.'] if two_hop else [])
        supports.append(f'Parcel P{parcels[32]} is stored in locker L{lockers[32]}.')
        question = (f'Which locker stores the parcel carried by courier C{couriers[32]}?' if two_hop
                    else f'Which locker stores parcel P{parcels[32]}?')
        # ID assignment is shuffled independently of the support/distractor distinction.
        names = [f's{value:03d}' for value in rng.sample(range(100, 999), len(facts)+len(supports))]
        fact_records = [{'sentence_id': names[index], 'text': text} for index, text in enumerate(facts)]
        support_records = [{'sentence_id': names[len(facts)+index], 'text': text}
                           for index, text in enumerate(supports)]
        for position in config['positions']:
            insert = {'early': 0, 'middle': len(facts)//2, 'late': len(facts)}[position]
            sentences = fact_records[:insert]+support_records+fact_records[insert:]
            case_id = f'case-{world_index:02d}-{position}'
            publics.append({'case_id': case_id, 'question': question, 'sentences': sentences})
            evaluator.append({'case_id': case_id, 'world_id': f'world-{world_index:02d}',
                              'world_type': 'two_hop' if two_hop else 'single_hop',
                              'position': position, 'support_sentence_ids': [s['sentence_id'] for s in support_records],
                              'support_source_indices': list(range(insert, insert+len(support_records))),
                              'gold_answer': f'L{lockers[32]}', 'irrelevant_sentence_count': 64,
                              'split': 'explicitly_openDEV_synthetic'})
    return publics, evaluator


def lexical_terms(text):
    return re.findall(r'[A-Za-z0-9]+', text.casefold())


def bm25_rank(question, sentences, k1=1.2, b=0.75):
    docs = [Counter(lexical_terms(sentence['text'])) for sentence in sentences]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths)/len(lengths)
    df = Counter(term for doc in docs for term in doc)
    query = set(lexical_terms(question))
    scores = []
    for index, doc in enumerate(docs):
        score = 0.0
        for term in sorted(query):
            if term not in doc:
                continue
            idf = math.log(1+(len(docs)-df[term]+0.5)/(df[term]+0.5))
            frequency = doc[term]
            score += idf * frequency*(k1+1)/(frequency+k1*(1-b+b*lengths[index]/average))
        scores.append(score)
    return sorted(range(len(sentences)), key=lambda index: (-scores[index], index))


def rank_public(question, sentences, method, base_seed):
    """Only public question + factual sentence records cross this algorithm boundary."""
    require(method in METHODS, 'unknown retriever')
    public_view({'case_id': 'algorithm-schema-check', 'question': question, 'sentences': sentences})
    if method == 'lead_prefix':
        return list(range(len(sentences)))
    if method == 'seeded_random':
        inventory = sorted(sentences, key=lambda sentence: sentence['sentence_id'])
        seed_hash = hashlib.sha256(canonical({'seed': base_seed, 'question': question,
                                              'sentences': inventory})).digest()
        names = [sentence['sentence_id'] for sentence in inventory]
        random.Random(int.from_bytes(seed_hash, 'big')).shuffle(names)
        source_index = {sentence['sentence_id']: index for index, sentence in enumerate(sentences)}
        return [source_index[name] for name in names]
    initial = bm25_rank(question, sentences)
    if method == 'bm25_one_hop':
        return initial
    seed_index = initial[0]
    expanded = question+' '+sentences[seed_index]['text']
    second = bm25_rank(expanded, sentences)
    return [seed_index]+[index for index in second if index != seed_index]


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def pad_exact(text, budget, tokenizer, suffix_candidates):
    """Evaluator-independent padding. Every step must add exactly one actual raw token."""
    original_tokens = token_count(tokenizer, text)
    require(original_tokens <= budget, 'selected context already exceeds token budget')
    padded, count, used = text, original_tokens, []
    while count < budget:
        for suffix in suffix_candidates:
            candidate = padded+suffix
            if token_count(tokenizer, candidate) == count+1:
                padded, count = candidate, count+1
                used.append(suffix)
                break
        else:
            raise ValueError('fixed filler suffixes cannot add one token; preserve failed attempt')
    require(token_count(tokenizer, padded) == budget, 'padding tokenizer equality failed')
    return {'context_text': padded, 'selected_context_text': text, 'padding_text': ''.join(used),
            'padding_suffix_steps': used, 'consumed_tokens': budget,
            'selected_context_tokens': original_tokens, 'net_padding_tokens': budget-original_tokens}


def pack_context(sentences, ranking, budget, tokenizer, config):
    require(sorted(ranking) == list(range(len(sentences))), 'ranking is not an exact sentence permutation')
    selected = []
    for index in ranking:
        candidate = sorted(selected+[index])
        text = config['sentence_separator'].join(sentences[item]['text'] for item in candidate)
        if token_count(tokenizer, text) <= budget:
            selected.append(index)
        else:
            break
    selected.sort()
    text = config['sentence_separator'].join(sentences[index]['text'] for index in selected)
    padded = pad_exact(text, budget, tokenizer, config['padding_suffix_candidates'])
    return {**padded, 'selected_source_indices': selected,
            'selected_sentence_ids': [sentences[index]['sentence_id'] for index in selected],
            'selected_sentence_count': len(selected)}


def evidence_metrics(selected_ids, support_ids):
    selected, support = set(selected_ids), set(support_ids)
    require(len(support) in (1, 2), 'expected one or two distinct support sentences')
    hits = len(selected & support)
    return {'full_evidence_recall': int(support <= selected), 'any_evidence_recall': int(hits > 0),
            'support_precision': hits/len(selected) if selected else 0.0,
            'selected_support_count': hits, 'required_support_count': len(support)}


def validate_frozen_cases(publics, evaluator):
    require(len(publics) == len(evaluator) == 192, 'frozen case inventory changed')
    gold = {record['case_id']: record for record in evaluator}
    require(len(gold) == 192 and len({case['case_id'] for case in publics}) == 192,
            'duplicate frozen case IDs')
    require(set(gold) == {case['case_id'] for case in publics}, 'public/evaluator case ID mismatch')
    require(Counter(record['world_type'] for record in evaluator) == {'single_hop': 96, 'two_hop': 96},
            'world type inventory changed')
    require(Counter(record['position'] for record in evaluator) == {'early': 64, 'middle': 64, 'late': 64},
            'placement inventory changed')
    worlds = defaultdict(list)
    for case in publics:
        _, sentences = public_view(case)
        label = gold[case['case_id']]
        support = label['support_sentence_ids']
        require(len(set(support)) == (2 if label['world_type'] == 'two_hop' else 1), 'support cardinality changed')
        require(len(sentences)-len(support) == label['irrelevant_sentence_count'] == 64,
                'irrelevant fact inventory changed')
        require([sentences[index]['sentence_id'] for index in label['support_source_indices']] == support,
                'evaluator source-index binding changed')
        worlds[label['world_id']].append((case, label))
    require(len(worlds) == 64 and all(len(cases) == 3 for cases in worlds.values()), 'paired world inventory changed')
    for cases in worlds.values():
        first_public, first_gold = cases[0]
        first_facts = {sentence['sentence_id']: sentence['text'] for sentence in first_public['sentences']}
        for public, label in cases:
            require(public['question'] == first_public['question']
                    and {s['sentence_id']: s['text'] for s in public['sentences']} == first_facts
                    and label['support_sentence_ids'] == first_gold['support_sentence_ids']
                    and label['gold_answer'] == first_gold['gold_answer'], 'paired worlds differ beyond placement')
    return gold


NUMERIC_METRICS = ('full_evidence_recall', 'any_evidence_recall', 'support_precision',
                   'selected_sentence_count', 'consumed_tokens', 'selected_context_tokens',
                   'net_padding_tokens', 'retrieval_seconds', 'packing_seconds')


def summarize(records):
    return {'case_count': len(records), 'world_count': len({row['world_id'] for row in records}),
            **{name: sum(row[name] for row in records)/len(records) for name in NUMERIC_METRICS}}


def aggregate(records):
    groups, worlds = defaultdict(list), defaultdict(list)
    for row in records:
        for stratum, value in (('all', 'all'), ('world_type', row['world_type']),
                               ('position', row['position']), ('type_position', row['world_type']+':'+row['position'])):
            groups[(row['method'], row['budget'], stratum, value)].append(row)
        worlds[(row['method'], row['budget'], row['world_id'])].append(row)
    clusters = [{'method': key[0], 'budget': key[1], 'world_id': key[2], **summarize(values)}
                for key, values in sorted(worlds.items())]
    aggregates = []
    for key, values in sorted(groups.items()):
        cluster_values = defaultdict(list)
        for row in values:
            cluster_values[row['world_id']].append(row)
        means = [summarize(v) for v in cluster_values.values()]
        aggregates.append({'method': key[0], 'budget': key[1], 'stratum': key[2], 'stratum_value': key[3],
                           **summarize(values), 'world_cluster_full_recall_mean': sum(
                               row['full_evidence_recall'] for row in means)/len(means),
                           'world_cluster_full_recall_min': min(row['full_evidence_recall'] for row in means),
                           'world_cluster_full_recall_max': max(row['full_evidence_recall'] for row in means)})
    return aggregates, clusters


def memory_snapshot():
    if os.name == 'nt':
        from ctypes import wintypes
        class ProcessMemory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile',
                'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
        kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        process, status = ProcessMemory(), MemoryStatus()
        process.cb, status.dwLength = ctypes.sizeof(process), ctypes.sizeof(status)
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb),
                'cannot read process RSS')
        require(kernel.GlobalMemoryStatusEx(ctypes.byref(status)), 'cannot read host memory')
        return {'rss_bytes': process.WorkingSetSize, 'peak_rss_bytes': process.PeakWorkingSetSize,
                'free_physical_bytes': status.ullAvailPhys, 'total_physical_bytes': status.ullTotalPhys,
                'system_memory_load_percent': status.dwMemoryLoad,
                'commit_used_bytes': status.ullTotalPageFile-status.ullAvailPageFile,
                'commit_limit_bytes': status.ullTotalPageFile}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    return {'peak_rss_bytes': peak, 'rss_bytes': None, 'free_physical_bytes': None}


class Budget:
    def __init__(self, out, spec):
        self.out, self.spec, self.started, self.samples = out, spec, time.monotonic(), []

    def check(self, phase):
        sample = {'phase': phase, 'elapsed_seconds': time.monotonic()-self.started, **memory_snapshot(),
                  'output_bytes': sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())}
        self.samples.append(sample)
        with (self.out/'resources.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(sample, sort_keys=True)+'\n')
        require(sample['peak_rss_bytes'] <= self.spec['max_tree_rss_bytes'], 'RSS budget exceeded')
        require(sample['elapsed_seconds'] <= self.spec['max_wall_seconds'], 'wall budget exceeded')
        require(sample['output_bytes'] <= self.spec['max_output_bytes'], 'output budget exceeded')
        return sample


def csv_write(path, records):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def report_html(report):
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    content = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic evidence retention</title>'
               '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px}'
               'table{border-collapse:collapse;margin:16px 0;font-size:14px}td,th{border:1px solid #ccc;'
               'padding:8px;text-align:left}th{background:#edf2f5}pre{white-space:pre-wrap}</style>'
               '<h1>Retrieval-only evidence retention</h1><p>'+html.escape(SCOPE)+'</p>'
               '<p>64 fresh openDEV synthetic worlds, 32 single-hop and 32 two-hop; three paired '
               'early/middle/late placements give 192 contexts. Two-hop full recall requires both '
               'courier→parcel and parcel→locker supports.</p><p>All methods consume exactly '
               '128/256/512 raw native context tokens, including recorded evaluator-independent '
               'padding. Query/chat overhead is excluded. Selected sentence counts differ. '
               'Public ranking never receives support IDs, world type or answers.</p>')
    all_rows = [row for row in report['aggregates'] if row['stratum'] == 'all']
    content += '<h2>All worlds</h2>'+table(
        ['Method', 'Tokens', 'Full recall', 'Any recall', 'Precision', 'Selected facts', 'Padding tokens', 'Rank ms'],
        [[r['method'], r['budget'], f"{r['full_evidence_recall']:.3f}", f"{r['any_evidence_recall']:.3f}",
          f"{r['support_precision']:.3f}", f"{r['selected_sentence_count']:.1f}",
          f"{r['net_padding_tokens']:.1f}", f"{1000*r['retrieval_seconds']:.3f}"] for r in all_rows])
    for stratum in ('world_type', 'position'):
        content += '<h2>'+html.escape(stratum.replace('_', ' ').title())+'</h2>'+table(
            ['Method', 'Tokens', 'Stratum', 'Worlds', 'Cases', 'Full recall', 'Any recall'],
            [[r['method'], r['budget'], r['stratum_value'], r['world_count'], r['case_count'],
              f"{r['full_evidence_recall']:.3f}", f"{r['any_evidence_recall']:.3f}"]
             for r in report['aggregates'] if r['stratum'] == stratum])
    content += '<h2>World-cluster descriptive results</h2><p>Each world is reused in three positions. '
    content += 'The table describes the 64 world means, with their range; it supplies no population inference.</p>'+table(
        ['Method', 'Tokens', 'Mean world full recall', 'Minimum world mean', 'Maximum world mean'],
        [[r['method'], r['budget'], f"{r['world_cluster_full_recall_mean']:.3f}",
          f"{r['world_cluster_full_recall_min']:.3f}", f"{r['world_cluster_full_recall_max']:.3f}"] for r in all_rows])
    content += '<h2>Timing and provenance</h2><p>Ranking is timed once per case/method and reused '
    content += ('across three budgets. Packing and raw native encoding are timed separately. '
                'cases.csv retains selected/support IDs, source indices, raw packed text and padding. '
                'The generated public/evaluator byte hashes were frozen before retrieval scoring.</p><pre>'
                +html.escape(json.dumps(report['frozen_inputs'], indent=2))+'</pre></html>')
    return content


def run(out):
    config, protocol = read_json(PACKAGE/'config.json'), read_json(PACKAGE/'protocol.json')
    out = Path(out).resolve()
    require((PACKAGE/'runs') in out.parents, 'output must be a new directory under package runs/')
    require(not out.exists(), 'output already exists; preserve prior attempts')
    out.mkdir(parents=True)
    budget = Budget(out, config['resource_budget'])
    implementation = {str(p.relative_to(PACKAGE)): digest(p) for p in PACKAGE.rglob('*')
                      if p.is_file() and 'runs' not in p.relative_to(PACKAGE).parts and '__pycache__' not in p.parts}
    bindings = {'implementation_sha256': implementation, 'config_sha256': digest(PACKAGE/'config.json'),
                'protocol_sha256': digest(PACKAGE/'protocol.json'), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(), 'pid': os.getpid()}
    write_json(out/'RUN.json', {'status': 'STARTED', 'bindings': bindings, 'budget': config['resource_budget'], 'scope': SCOPE})
    try:
        import_guard = install_cpu_import_guard()
        budget.check('start')
        tokenizer_path = Path(config['tokenizer']['path'])
        require(digest(tokenizer_path) == config['tokenizer']['sha256'], 'native tokenizer hash changed')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import tokenizers
        require(tokenizers.__version__ == config['tokenizer']['tokenizers_version'], 'tokenizers version changed')
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        assert_cpu_imports_clean()
        source = {'tokenizer_path': str(tokenizer_path), 'bytes': tokenizer_path.stat().st_size,
                  'sha256': digest(tokenizer_path), 'tokenizers_version': tokenizers.__version__,
                  'api': 'Tokenizer.from_file + encode(add_special_tokens=False).ids',
                  'weights_read': False, 'model_loaded': False, 'protected_data_read': False,
                  'tokenizers_parallelism': os.environ['TOKENIZERS_PARALLELISM'],
                  'cpu_import_guard': import_guard,
                  'forbidden_packages_absent_after_tokenizer_load': True}
        write_json(out/'SOURCE.json', source)
        budget.check('native_raw_tokenizer_loaded')
        publics, evaluator = construct_worlds(config)
        validate_frozen_cases(publics, evaluator)
        write_jsonl(out/'public_contexts.jsonl', publics)
        write_jsonl(out/'evaluator_only.jsonl', evaluator)
        frozen = {'schema': 1, 'status': 'FROZEN_BEFORE_ANY_RETRIEVAL_SCORE', 'bindings': bindings,
                  'source': source, 'case_count': 192, 'world_count': 64,
                  'files': {name: {'bytes': (out/name).stat().st_size, 'sha256': digest(out/name)}
                            for name in ('public_contexts.jsonl', 'evaluator_only.jsonl')}}
        write_json(out/'FROZEN_INPUTS.json', frozen)
        budget.check('public_and_evaluator_bytes_frozen_before_scores')
        for name, declared in frozen['files'].items():
            require(digest(out/name) == declared['sha256'] and (out/name).stat().st_size == declared['bytes'],
                    'frozen generated bytes changed')
        # Analyze only re-read frozen inputs, never unsaved constructor state.
        publics = [json.loads(line) for line in (out/'public_contexts.jsonl').read_text(encoding='utf-8').splitlines()]
        evaluator = [json.loads(line) for line in (out/'evaluator_only.jsonl').read_text(encoding='utf-8').splitlines()]
        gold = validate_frozen_cases(publics, evaluator)
        records = []
        for case_index, public in enumerate(publics):
            question, sentences = public_view(public)
            source_tokens = token_count(tokenizer, config['sentence_separator'].join(s['text'] for s in sentences))
            require(source_tokens > max(config['context_token_budgets']), 'synthetic context fits largest budget')
            for method in config['methods']:
                before_rank = time.perf_counter()
                ranking = rank_public(question, sentences, method, config['random_retriever_seed'])
                retrieval_seconds = time.perf_counter()-before_rank
                for native_budget in config['context_token_budgets']:
                    before_pack = time.perf_counter()
                    packed = pack_context(sentences, ranking, native_budget, tokenizer, config)
                    packing_seconds = time.perf_counter()-before_pack
                    label = gold[public['case_id']]
                    records.append({'case_id': public['case_id'], 'world_id': label['world_id'],
                                    'world_type': label['world_type'], 'position': label['position'],
                                    'method': method, 'budget': native_budget, 'source_context_tokens': source_tokens,
                                    'retrieval_seconds': retrieval_seconds, 'packing_seconds': packing_seconds,
                                    **packed, 'required_support_ids': label['support_sentence_ids'],
                                    **evidence_metrics(packed['selected_sentence_ids'], label['support_sentence_ids'])})
            if case_index % 24 == 23:
                budget.check(f'processed_cases_{case_index+1}')
        require(len(records) == 192*4*3, 'result case/method/budget grid incomplete')
        require(all(row['consumed_tokens'] == row['budget'] for row in records), 'methods have unmatched token costs')
        aggregates, clusters = aggregate(records)
        budget.check('retrieval_scoring_complete')
        report = {'schema': 1, 'status': 'COMPLETE_SYNTHETIC_RETRIEVAL_DIAGNOSIS', 'scope': SCOPE,
                  'licenses_registered_training': False, 'licenses_efficacy_claim': False, 'opens_gate': False,
                  'native_total_envelope_qualified': False, 'answer_correctness_measured': False,
                  'frozen_inputs': frozen, 'aggregates': aggregates, 'world_clusters': clusters,
                  'case_results': records, 'resource_samples': budget.samples}
        write_json(out/'analysis.json', report)
        csv_write(out/'cases.csv', records)
        csv_write(out/'world_clusters.csv', clusters)
        csv_write(out/'aggregates.csv', aggregates)
        (out/'report.html').write_text(report_html(report), encoding='utf-8')
        final = budget.check('reports_written')
        write_json(out/'COMPLETED.json', {'status': report['status'], 'resources': final,
                   'artifact_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'COMPLETED.json'}})
        print(json.dumps({'status': report['status'], 'out': str(out), 'wall_seconds': final['elapsed_seconds'],
                          'peak_rss_bytes': final['peak_rss_bytes']}, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(out/'FAILED.json', {'status': 'FAILED', 'error_type': type(exc).__name__, 'reason': str(exc),
                                     'bindings': bindings, 'resources': budget.samples})
        print(f'FAILED: {exc}; preserved {out}', file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New directory within this experiment runs/')
    args = parser.parse_args()
    return run(args.out)
