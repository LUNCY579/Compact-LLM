"""Hash-bound cached-answer study. Python standard library only.

The existing pure retention parser remains authoritative. Historical correctness fields
are only cross-checks; raw output text and EOS termination determine every result.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import html
import importlib.util
import itertools
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
ORACLE_SCOPE = ('Finite-panel oracle headroom only. Uses known targets to pick correct '
                'outputs; not a deployable selector, sampling gain, merged weights, '
                'population inference, or current runtime qualification.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':')).encode('utf-8')


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    # Reject duplicate JSON keys, including duplicate output IDs.
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'duplicate JSON key: {key}')
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact_ids(mapping, expected, label):
    require(isinstance(mapping, dict), f'{label}: outputs must be an object')
    require(set(mapping) == set(expected),
            f'{label}: ID mismatch; missing={sorted(set(expected)-set(mapping))}; '
            f'extra={sorted(set(mapping)-set(expected))}')
    for item_id, output in mapping.items():
        require(isinstance(output, dict) and isinstance(output.get('text'), str)
                and type(output.get('terminated')) is bool,
                f'{label}: invalid raw text/termination for {item_id}')


def memory_snapshot():
    """Current + process peak RSS and system free memory, without heavyweight libraries."""
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        class Status(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    'ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                    'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                             wintypes.DWORD]
        counters = Counters(cb=ctypes.sizeof(Counters))
        status = Status(dwLength=ctypes.sizeof(Status))
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), 'GlobalMemoryStatusEx failed')
        return {'rss_bytes': counters.WorkingSetSize,
                'peak_rss_bytes': counters.PeakWorkingSetSize,
                'free_physical_bytes': status.ullAvailPhys,
                'total_physical_bytes': status.ullTotalPhys,
                'system_memory_load_percent': status.dwMemoryLoad,
                'commit_used_bytes': status.ullTotalPageFile - status.ullAvailPageFile,
                'commit_limit_bytes': status.ullTotalPageFile,
                'measurement': 'Windows process working set + GlobalMemoryStatusEx'}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == 'darwin' else 1024
    return {'rss_bytes': None, 'peak_rss_bytes': peak,
            'free_physical_bytes': None, 'measurement': 'POSIX ru_maxrss; peak RSS'}


class Budget:
    """Checks phase boundaries; external supervisor enforces continuous admission/timeout."""
    def __init__(self, spec, out):
        self.spec, self.out, self.start = spec, Path(out), time.monotonic()
        self.samples = []

    def check(self, phase):
        sample = {'phase': phase, 'elapsed_seconds': time.monotonic() - self.start,
                  **memory_snapshot()}
        sample['output_bytes'] = sum(p.stat().st_size for p in self.out.rglob('*')
                                     if p.is_file())
        self.samples.append(sample)
        with (self.out / 'resources.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(sample, sort_keys=True) + '\n')
        require(sample['peak_rss_bytes'] <= self.spec['max_process_rss_bytes'],
                'process RSS budget exceeded')
        require(sample['elapsed_seconds'] <= self.spec['max_wall_seconds'],
                'wall-time budget exceeded')
        require(sample['output_bytes'] <= self.spec['max_output_bytes'],
                'output-size budget exceeded')
        return sample


def load_scorer(path):
    spec = importlib.util.spec_from_file_location('frozen_retention_parser', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_probe(probe, audit, spec):
    ids = [row['id'] for row in probe]
    require(len(ids) == len(set(ids)) == spec['row_count'], 'duplicate/missing probe IDs')
    require(object_hash(sorted(ids)) == spec['sorted_ids_sha256'], 'probe ID digest changed')
    require(len({row['group_id'] for row in probe}) == spec['unique_family_count'],
            'family inventory changed')
    require(dict(Counter(row['reply_class'] for row in probe)) == spec['class_counts'],
            'class inventory changed')
    audits = {row['id']: row for row in audit['rows']}
    require(len(audits) == len(audit['rows']) == len(probe), 'audit IDs/count changed')
    require(set(audits) == set(ids), 'audit/probe ID mismatch')
    for line, row in enumerate(probe, 1):
        label = audits[row['id']]
        require(label['source_line_1based'] == line, 'audit line binding changed')
        require(label['source_row_sha256'] == object_hash(row), 'audit row binding changed')
        require(label['messages_sha256'] == object_hash(row['messages']),
                'audit full-thread message binding changed')
        index = label['operative_user_message_index_0based']
        require(row['messages'][index]['role'] == 'user', 'operative message is not user')
        prompt = row['messages'][index]['content']
        require(label['operative_prompt'] == prompt
                and label['operative_prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(),
                'audit operative prompt binding changed')
        require(label['original_answer'] == row['answer']
                and label['family_id'] == row['group_id']
                and label['reply_class'] == row['reply_class'], 'audit label metadata changed')
    return audits


def verify_provenance(documents, protocol, ids):
    banks = [documents[key] for key in ('last20', 'fullb28')]
    source_sha = protocol['provenance']['source_weight_sha256']
    for key, bank in zip(('last20', 'fullb28'), banks):
        declared = protocol['bank_metadata'][key]
        require(bank['completed'] is True and bank['aborted_reason'] is None,
                f'{key}: historical run incomplete')
        for field, want in declared.items():
            require(bank.get(field) == want, f'{key}: changed {field}')
        require(bank['source_weights_sha256'] == source_sha
                and bank['source_assets_sha256']['model.safetensors'] == source_sha,
                f'{key}: source weight metadata mismatch')
        require(bank['data_sha256']['monitor_probe'] == protocol['inputs']['probe']['sha256']
                and bank['data_sha256']['probe_label_audit'] == protocol['inputs']['audit']['sha256'],
                f'{key}: historical data binding mismatch')
        require(bank['checkpoint']['dtype'] == 'float32', f'{key}: precision changed')
        require(bank['checkpoint']['sha256'] == protocol['provenance'][key+'_checkpoint_sha256'],
                f'{key}: historical checkpoint metadata changed')
        require(len(bank['checkpoints']) == 2, f'{key}: expected initial and final captures')
        initial, final = bank['checkpoints']
        require(initial['tokens'] == initial['steps'] == 0, f'{key}: not initial capture')
        require(final['tokens'] == bank['supervised_target_tokens'] == 5233
                and final['steps'] == bank['optimizer_steps'] == 41,
                f'{key}: final dose binding mismatch')
        for index, checkpoint in enumerate(bank['checkpoints']):
            exact_ids(checkpoint['outputs'], ids, f'{key}:checkpoint{index}')
    require(banks[0]['checkpoints'][0]['outputs'] == banks[1]['checkpoints'][0]['outputs'],
            'Supa initial baselines differ between banks')
    require(banks[0]['train_row_ids_sha256'] == banks[1]['train_row_ids_sha256']
            and banks[0]['train_row_ids_consumed'] == banks[1]['train_row_ids_consumed'],
            'matched training draw binding differs')
    panel, core, files = (documents[key] for key in ('opallo', 'opallo_core', 'opallo_files'))
    exact_ids(panel['predictions'], ids, 'published Opallo')
    require(panel['probe']['rows'] == 150 and panel['probe']['device'] == 'cpu'
            and panel['probe']['dtype_at_inference'] == 'float32'
            and panel['probe']['decoding'] == 'greedy', 'Opallo inference metadata changed')
    require(core['source']['base']['weight_sha256'] == source_sha
            and core['source']['checkpoint']['trainable_sha256'] == banks[0]['checkpoint']['sha256']
            and core['source']['checkpoint']['tag'] == banks[0]['canonical_tag'],
            'Opallo ancestry metadata does not match last20 capture')
    require(core['checks']['dtype_counts'] == {'bfloat16': 310}
            and core['checks']['parameter_count_serialized'] == 596049920,
            'Opallo serialized precision/shape metadata changed')
    require(core['checks']['weight_sha256'] == files['model.safetensors']['sha256']
            == protocol['provenance']['opallo_weight_sha256'], 'Opallo weight metadata mismatch')
    require(panel['cast_parity']['float32_tag'] == banks[0]['canonical_tag'],
            'Opallo panel cast parent mismatch')
    return {'supa_initial': banks[0]['checkpoints'][0]['outputs'],
            'last20_fp32': banks[0]['checkpoints'][1]['outputs'],
            'fullb28_fp32': banks[1]['checkpoints'][1]['outputs'],
            'opallo_bf16_storage_fp32_compute': panel['predictions']}


def pair_stats(left, right, ids):
    universe, a, b = set(ids), set(left), set(right)
    require(a <= universe and b <= universe, 'correct ID sets exceed panel')
    return {'both_correct': len(a & b), 'left_only_correct': len(a - b),
            'right_only_correct': len(b - a), 'neither_correct': len(universe - (a | b)),
            'oracle_union': len(a | b), 'best_single': max(len(a), len(b)),
            'oracle_headroom_over_best_single': len(a | b) - max(len(a), len(b)),
            'left_only_ids': sorted(a-b), 'right_only_ids': sorted(b-a),
            'neither_ids': sorted(universe-(a|b)), 'scope': ORACLE_SCOPE}


def subset_oracles(correct, ids):
    models, universe = list(correct), set(ids)
    records = []
    for size in range(1, len(models)+1):
        for subset in itertools.combinations(models, size):
            union = set().union(*(correct[name] for name in subset))
            best = max(len(correct[name]) for name in subset)
            records.append({'models': list(subset), 'oracle_union': len(union),
                            'best_single': best, 'oracle_headroom_over_best_single': len(union)-best,
                            'joint_failure_count': len(universe-union),
                            'joint_failure_ids': sorted(universe-union), 'scope': ORACLE_SCOPE})
    return records


def failure_kind(record):
    if record['correct']:
        return 'correct'
    if not record['terminated']:
        return 'nontermination'
    if record['parsed'] is None:
        return 'format_or_type_rejection'
    return 'parsed_value_mismatch'


def comparison_kind(left, right, scorer, target):
    """Typed-value comparison ignores EOS only for this named mechanism diagnostic."""
    if left == right:
        return 'identical_raw_text_and_termination'
    if left['terminated'] != right['terminated']:
        return 'termination_differs'
    if scorer.normalise(left['text']) == scorer.normalise(right['text']):
        return 'whitespace_or_case_only'
    values = []
    for output in (left, right):
        try:
            values.append(scorer.parse_prediction(output['text'], target))
        except scorer.ParseError:
            values.append(None)
    if None in values:
        return 'one_or_both_format_or_type_rejected'
    return 'same_typed_value_different_text' if values[0] == values[1] else 'different_typed_values'


def analyze(probe, audits, outputs, scorer):
    ids, rows = [row['id'] for row in probe], {row['id']: row for row in probe}
    results, row_records = {}, []
    for policy in ('strict', 'permissive'):
        targets = {key: scorer.target_from_audit(audits[key], policy) for key in ids}
        scores = {model: {key: scorer.score_prediction(raw[key]['text'], raw[key]['terminated'],
                                                     targets[key]) for key in ids}
                  for model, raw in outputs.items()}
        correct = {model: {key for key in ids if score[key]['correct']}
                   for model, score in scores.items()}
        source = correct['supa_initial']
        models = {}
        for model, records in scores.items():
            totals = {'rows': len(ids), 'correct': len(correct[model]),
                      'terminated': sum(record['terminated'] for record in records.values()),
                      'source_retention': scorer.monitor_counts(source, correct[model]),
                      'failure_categories': dict(Counter(failure_kind(v) for v in records.values()))}
            totals['by_class'] = {}
            for cls in sorted({row['reply_class'] for row in probe}):
                class_ids = {key for key in ids if rows[key]['reply_class'] == cls}
                totals['by_class'][cls] = {
                    'rows': len(class_ids), 'correct': len(correct[model] & class_ids),
                    'source_retention': scorer.monitor_counts(source & class_ids,
                                                             correct[model] & class_ids)}
            models[model] = totals
            for key in ids:
                row_records.append({'policy': policy, 'model': model, 'id': key,
                                    'family_id': rows[key]['group_id'],
                                    'class': rows[key]['reply_class'],
                                    'template': audits[key]['independent_recomputation']['template'],
                                    'typed_target': targets[key], **records[key],
                                    'failure_category': failure_kind(records[key]),
                                    'source_correct': key in source,
                                    'gained_vs_source': key in correct[model]-source,
                                    'lost_vs_source': key in source-correct[model]})
        pairs = []
        for left, right in itertools.combinations(outputs, 2):
            typed_differences = {key: comparison_kind(outputs[left][key], outputs[right][key],
                                                      scorer, targets[key]) for key in ids}
            pairs.append({'left': left, 'right': right,
                          **pair_stats(correct[left], correct[right], ids),
                          'text_or_typed_value_difference_counts': dict(Counter(typed_differences.values())),
                          'text_or_typed_value_differences': typed_differences})
        results[policy] = {'models': models, 'pairs': pairs,
                           'subset_oracles': subset_oracles(correct, ids),
                           'correct_ids': {model: sorted(keys) for model, keys in correct.items()},
                           'all_model_joint_failure_ids': sorted(set(ids)-set().union(*correct.values()))}
    sensitivity = {}
    for model in outputs:
        strict = set(results['strict']['correct_ids'][model])
        permissive = set(results['permissive']['correct_ids'][model])
        require(strict <= permissive, 'strict policy unexpectedly accepts a permissive rejection')
        sensitivity[model] = {'permissive_only_ids': sorted(permissive-strict),
                              'permissive_only_count': len(permissive-strict)}
    return results, row_records, sensitivity


def historical_crosscheck(results, documents):
    """Stored scores cannot influence scoring; a disagreement invalidates comparison."""
    model_capture = {
        'supa_initial': documents['last20']['checkpoints'][0]['retention'],
        'last20_fp32': documents['last20']['checkpoints'][1]['retention'],
        'fullb28_fp32': documents['fullb28']['checkpoints'][1]['retention'],
        'opallo_bf16_storage_fp32_compute': documents['opallo']['by_policy']}
    checks = []
    for policy, result in results.items():
        for model, totals in result['models'].items():
            stored = model_capture[model][policy]
            for name, got in [('correct_total', totals['correct']),
                              ('terminated_total', totals['terminated']),
                              ('lost_total', totals['source_retention']['lost']),
                              ('gained_total', totals['source_retention']['gained']),
                              ('retained_total', totals['source_retention']['retained'])]:
                expected = stored[name]
                checks.append({'policy': policy, 'model': model, 'metric': name,
                               'stored': expected, 'recomputed': got, 'equal': got == expected})
                require(got == expected, f'historical {policy}/{model}/{name} mismatch: '
                        f'{got} recomputed versus {expected} stored')
            for cls, counts in totals['by_class'].items():
                require(counts['correct'] == stored['per_class'][cls]['arm_correct'],
                        f'historical {policy}/{model}/{cls} correct IDs disagree')
            if 'correct_ids' in stored:
                require(set(stored['correct_ids']) == set(result['correct_ids'][model]),
                        f'historical {policy}/{model} correct-ID set mismatch')
    # Independently bind the second bank's step-zero scoring as well as raw baseline equality.
    for policy in results:
        require(documents['fullb28']['checkpoints'][0]['retention'][policy]['correct_total']
                == results[policy]['models']['supa_initial']['correct'],
                'second bank historical initial score differs')
    for key, model in (('last20', 'last20_fp32'), ('fullb28', 'fullb28_fp32')):
        for index, name in ((0, 'supa_initial'), (1, model)):
            per_row = documents[key]['checkpoints'][index]['retention']['per_row']
            recomputed = set(results['permissive']['correct_ids'][name])
            require(set(per_row) == set(documents[key]['checkpoints'][index]['outputs'])
                    and {item_id for item_id, value in per_row.items() if value} == recomputed,
                    f'{key}: historical permissive per-row correctness mismatch')
    cast = documents['opallo']['cast_parity']
    last = documents['last20']['checkpoints'][1]['outputs']
    published = documents['opallo']['predictions']
    require(set(cast['differing_ids']) == {key for key in last
            if last[key]['text'] != published[key]['text']}, 'Opallo cast text differences changed')
    require(cast['identical_termination_rows'] == sum(last[key]['terminated'] == published[key]['terminated']
            for key in last), 'Opallo cast termination comparison changed')
    left = set(results['permissive']['correct_ids']['last20_fp32'])
    right = set(results['permissive']['correct_ids']['opallo_bf16_storage_fp32_compute'])
    require(set(cast['lost_vs_float32']) == left-right
            and set(cast['gained_vs_float32']) == right-left,
            'Opallo cast correctness comparison changed')
    return checks


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: (json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                               if isinstance(row.get(key), (dict, list)) else row.get(key))
                         for key in fields} for row in rows)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(v))+'</th>' for v in headers) + (
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'
        for v in row)+'</tr>' for row in rows)+'</tbody></table>')


def write_reports(out, report, row_records):
    (out / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n',
                                     encoding='utf-8')
    write_csv(out / 'rows.csv', row_records,
              ['policy', 'model', 'id', 'family_id', 'class', 'template', 'raw', 'terminated',
               'kind', 'typed_target', 'target', 'parsed', 'reason', 'correct', 'failure_category',
               'source_correct', 'gained_vs_source', 'lost_vs_source'])
    pairs = [{'policy': policy, **pair} for policy, result in report['by_policy'].items()
             for pair in result['pairs']]
    write_csv(out / 'pairs.csv', pairs,
              ['policy', 'left', 'right', 'both_correct', 'left_only_correct', 'right_only_correct',
               'neither_correct', 'oracle_union', 'best_single', 'oracle_headroom_over_best_single',
               'text_or_typed_value_difference_counts', 'scope'])
    subsets = [{'policy': policy, **subset} for policy, result in report['by_policy'].items()
               for subset in result['subset_oracles']]
    write_csv(out / 'subset_oracles.csv', subsets,
              ['policy', 'models', 'oracle_union', 'best_single', 'oracle_headroom_over_best_single',
               'joint_failure_count', 'joint_failure_ids', 'scope'])
    summary = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Cached model '
               'complementarity</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;'
               'padding:0 16px;color:#17212b}table{border-collapse:collapse;margin:16px 0;'
               'font-size:14px}th,td{padding:8px;border:1px solid #cdd5dc;text-align:left}'
               'th{background:#edf2f5}code{overflow-wrap:anywhere}details{margin:20px 0}</style>'
               '<h1>Cached arithmetic model complementarity</h1><p>'+html.escape(ORACLE_SCOPE)+'</p>'
               '<p>Historical exposed development panel: 150 rows, 150 distinct families. '
               'Label overlay and bare-value policy human review remain pending. '
               'No weights were loaded. No probabilities/logits are available.</p>'
               '<p>Supa initial and trained captures use FP32 inference. Published Opallo stores '
               'BF16 weights and computes in FP32. These are different artifacts of the same lineage.</p>')
    for policy, result in report['by_policy'].items():
        summary += '<h2>'+html.escape(policy.title())+' typed scoring</h2>'
        summary += table(['Model', 'Correct / 150', 'Source retained', 'Source lost', 'Source gained'],
                         [[model, v['correct'], v['source_retention']['retained'],
                           v['source_retention']['lost'], v['source_retention']['gained']]
                          for model, v in result['models'].items()])
        summary += table(['Left', 'Right', 'Both', 'Left only', 'Right only', 'Neither',
                          'Oracle union', 'Headroom over best'],
                         [[p['left'], p['right'], p['both_correct'], p['left_only_correct'],
                           p['right_only_correct'], p['neither_correct'], p['oracle_union'],
                           p['oracle_headroom_over_best_single']] for p in result['pairs']])
        summary += '<h3>All subset oracle ceilings</h3>'+table(
            ['Subset', 'Union', 'Best single', 'Headroom', 'Joint failures'],
            [[', '.join(s['models']), s['oracle_union'], s['best_single'],
              s['oracle_headroom_over_best_single'], s['joint_failure_count']]
             for s in result['subset_oracles']])
        summary += '<h3>Class scores</h3>'+table(['Model', 'Class', 'Correct', 'Rows', 'Source lost'],
                         [[model, cls, counts['correct'], counts['rows'], counts['source_retention']['lost']]
                          for model, totals in result['models'].items()
                          for cls, counts in totals['by_class'].items()])
        summary += '<h3>Answer failure categories</h3>'+table(
            ['Model', 'Correct', 'Format/type rejected', 'Parsed value wrong', 'No EOS'],
            [[model, totals['failure_categories'].get('correct', 0),
              totals['failure_categories'].get('format_or_type_rejection', 0),
              totals['failure_categories'].get('parsed_value_mismatch', 0),
              totals['failure_categories'].get('nontermination', 0)]
             for model, totals in result['models'].items()])
        summary += '<details><summary>Pairwise formatting and typed-value differences</summary>'+table(
            ['Left', 'Right', 'Category', 'Rows'],
            [[pair['left'], pair['right'], kind, count] for pair in result['pairs']
             for kind, count in sorted(pair['text_or_typed_value_difference_counts'].items())])+'</details>'
        joint = result['all_model_joint_failure_ids']
        summary += '<details><summary>All-model joint failures ('+str(len(joint))+')</summary><ol>' + (
            ''.join('<li><code>'+html.escape(key)+'</code></li>' for key in joint)+'</ol></details>')
    summary += '<h2>Bare-unit sensitivity</h2>'+table(['Model', 'Permissive-only correct rows'],
        [[model, v['permissive_only_count']] for model, v in report['bare_policy_sensitivity'].items()])
    summary += '<h2>Formatting / numeric differences</h2><p>Format or type rejection, parsed '
    summary += ('value mismatch, nontermination, and bare-unit sensitivity are recorded separately '
                'in analysis.json and rows.csv. A format/type rejection is not proof of a wrong '
                'numeric belief. Pairwise typed values are compared only when both outputs parse '
                'under the fixed grammar; the mechanism diagnostic ignores EOS but correctness '
                'always requires EOS.</p><h2>Run bindings</h2><pre>'+html.escape(json.dumps(
                report['bindings'], indent=2))+'</pre></html>')
    (out / 'report.html').write_text(summary, encoding='utf-8')


def run(out):
    protocol = load_json(PACKAGE / 'protocol.json')
    out = Path(out).resolve()
    require((PACKAGE / 'runs') in out.parents, 'outputs must be under this package runs/')
    require(not out.exists(), 'output directory already exists; preserve historical attempts')
    out.mkdir(parents=True)
    budget = Budget(protocol['resource_budget'], out)
    bindings = {'protocol_sha256': file_hash(PACKAGE / 'protocol.json'),
                'package_files_sha256': {str(p.relative_to(PACKAGE)): file_hash(p)
                                         for p in PACKAGE.rglob('*')
                                         if p.is_file() and 'runs' not in p.relative_to(PACKAGE).parts
                                         and '__pycache__' not in p.parts},
                'input_hashes': protocol['inputs'], 'python': sys.version,
                'platform': platform.platform(), 'pid': os.getpid(),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'RUN.json').write_text(json.dumps({'status': 'STARTED', 'bindings': bindings,
        'budget': protocol['resource_budget'], 'scope': ORACLE_SCOPE}, indent=2)+'\n', encoding='utf-8')
    try:
        budget.check('start')
        paths = {}
        for key, declared in protocol['inputs'].items():
            path = (ROOT / declared['path']).resolve()
            require(ROOT in path.parents, f'{key}: input path outside worktree')
            require(path.stat().st_size == declared['bytes']
                    and file_hash(path) == declared['sha256'], f'{key}: frozen input hash changed')
            paths[key] = path
        budget.check('input_hashes_verified')
        documents = {key: load_json(path) for key, path in paths.items()
                     if key not in ('probe', 'scorer', 'scoring_contract')}
        probe = [json.loads(line) for line in paths['probe'].read_text(encoding='utf-8').splitlines()
                 if line.strip()]
        audit = documents['audit']
        require(audit['source_files'][0]['sha256'] == protocol['inputs']['probe']['sha256'],
                'audit declared probe hash differs')
        audits = bind_probe(probe, audit, protocol['panel'])
        outputs = verify_provenance(documents, protocol, [row['id'] for row in probe])
        budget.check('row_and_provenance_bindings_verified')
        scorer = load_scorer(paths['scorer'])
        results, records, sensitivity = analyze(probe, audits, outputs, scorer)
        checks = historical_crosscheck(results, documents)
        budget.check('raw_answers_rescored')
        report = {'schema': 1, 'status': 'COMPLETE_OFFLINE_ANALYSIS',
                  'evidence_kind': 'retrospective_cached_arithmetic_complementarity',
                  'scope': ORACLE_SCOPE, 'licenses_registered_training': False,
                  'licenses_efficacy_claim': False, 'opens_gate': False,
                  'weights_loaded': False, 'protected_data_accessed': False,
                  'probabilities_available': False, 'bindings': bindings,
                  'label_human_review_status': audit['human_review_status'],
                  'unit_or_entity_label_mismatch_count': audit['unit_or_entity_mismatch_count'],
                  'precision_provenance': protocol['models'], 'by_policy': results,
                  'bare_policy_sensitivity': sensitivity, 'historical_score_crosschecks': checks,
                  'resources_at_analysis_completion': budget.samples[-1]}
        write_reports(out, report, records)
        final = budget.check('reports_written')
        (out / 'COMPLETED.json').write_text(json.dumps({'status': report['status'],
            'resources': final, 'artifact_sha256': {p.name: file_hash(p) for p in out.iterdir()
            if p.is_file() and p.name != 'COMPLETED.json'}}, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'out': str(out),
                          'wall_seconds': final['elapsed_seconds'],
                          'peak_rss_bytes': final['peak_rss_bytes']}, sort_keys=True))
        return 0
    except Exception as exc:
        (out / 'FAILED.json').write_text(json.dumps({'status': 'FAILED',
            'error_type': type(exc).__name__, 'reason': str(exc), 'bindings': bindings,
            'resource_samples': budget.samples}, indent=2)+'\n', encoding='utf-8')
        print(f'FAILED: {exc}; preserved {out}', file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New output directory; existing paths refused')
    args = parser.parse_args()
    return run(args.out)
