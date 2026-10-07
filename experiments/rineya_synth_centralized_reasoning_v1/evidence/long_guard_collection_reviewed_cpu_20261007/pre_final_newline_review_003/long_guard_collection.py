"""Decoded long2304 guard producer and independent saved-raw CPU audit.

Capture uses a frozen cap-homogeneous permutation; the unscored consumer
scatters original raw receipts back to canonical world/length/variant order.
Native invocation needs a separately approved, source-bound owned asset loader.
This staged source grants no launch, scoring, training or qualification.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from datetime import datetime
import ast, hashlib, json, math, os, stat, sys

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')
EXPECTED = PACKAGE / 'long_guard_collection_v1'
SCOPE = 'REGISTERED_LONG2304_M0_DECODED_RAW_COLLECTION_V1'
ENVELOPES = {992: 1024, 2016: 2048, 4064: 4096}
FAMILIES = ('three_fact_keyed_join', 'latest_authorized_record_selection')
PLACEMENTS = ('early', 'middle', 'late', 'distributed')
VARIANTS = {'answerable_neutral': (True, False), 'answerable_adversarial': (True, True),
            'unanswerable_neutral': (False, False), 'unanswerable_adversarial': (False, True)}
INPUT_FIELDS = {'row_id', 'prompt_sha256', 'prompt_ids', 'prompt_tokens', 'attention_mask'}
META_FIELDS = {'row_id', 'component_id', 'family', 'placement', 'missing_index',
               'total_sequence_envelope', 'variant', 'answerable', 'adversarial'}
REF_FIELDS = {'path', 'bytes', 'sha256', 'value_sha256'}

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()

def seal(value): return {**value, 'sha256': digest(value)}

def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def _selected_bytes(ref, *, json_value=True):
    fields = REF_FIELDS if json_value else {'path','bytes','sha256'}
    require(type(ref) is dict and set(ref) == fields and type(ref['bytes']) is int
            and 0 < ref['bytes'] <= 64 * 1024**2 and _hash(ref['sha256']), 'bounded selected long byte descriptor')
    path = Path(ref['path'])
    require(path.is_absolute(), 'selected absolute long path')
    for part in (path,*path.parents):
        if part.exists():
            value=part.lstat();require(not part.is_symlink() and not (getattr(value,'st_file_attributes',0)&0x400), 'selected long path reparse/symlink forbidden')
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno());require(stat.S_ISREG(opened.st_mode) and opened.st_size==ref['bytes'], 'selected regular bounded long file')
        raw=stream.read(ref['bytes']+1);after=os.fstat(stream.fileno())
    require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            and len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'], 'selected long byte binding')
    final=path.stat();require((final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'selected long path stable after read')
    return raw

def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate selected long JSON key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite long JSON')))

def read_ref(ref):
    require(type(ref) is dict and set(ref) == REF_FIELDS and type(ref['bytes']) is int
            and 0 < ref['bytes'] <= 64 * 1024**2, 'bounded selected long JSON descriptor')
    raw = _selected_bytes(ref);value=_json(raw)
    require(digest(value) == ref['value_sha256'], 'selected long value binding')
    return value

def _helpers():
    """Import only the pure original component; no native worker/API import."""
    path = PACKAGE / 'long_inference_component.py'
    module = ModuleType('source_bound_long_inference_component'); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module

def _cap_factory():
    token = object()
    def get(): return token
    def check(value): require(value is token, 'private authored CPU long capability')
    return get, check

_CPU_capability, _check_capability = _cap_factory()

def population(source, validate_row):
    """Validate all original quotas and canonical world -> length -> variant order."""
    require(type(source) is dict and set(source) == {'rows', 'metadata', 'declared_domains'}, 'whole long source fields')
    rows, metadata, domains = source['rows'], source['metadata'], source['declared_domains']
    require(type(rows) is type(metadata) is list and len(rows) == len(metadata) == 2304, 'whole original2304 long rows')
    ids = [r['row_id'] for r in rows]
    require(len(set(ids)) == 2304 and ids == [m['row_id'] for m in metadata], 'unique canonical long input/metadata order')
    for row, meta in zip(rows, metadata):
        require(type(row) is dict and set(row) == INPUT_FIELDS and type(meta) is dict
                and set(meta) == META_FIELDS, 'long prompt-only rows and unlabelled metadata')
        length = row['prompt_tokens']; require(length in ENVELOPES, 'exact original long prompt lengths')
        validate_row(row, context_cap=ENVELOPES[length], generation_cap=32)
        require(meta['total_sequence_envelope'] == ENVELOPES[length] and meta['variant'] in VARIANTS
                and type(meta['answerable']) is type(meta['adversarial']) is bool
                and (meta['answerable'], meta['adversarial']) == VARIANTS[meta['variant']], 'original variant flags and envelope')
    worlds = []; counts = Counter(); missing_counts = Counter()
    for offset in range(0, 2304, 12):
        group = metadata[offset:offset + 12]; component = group[0]['component_id']
        require(type(component) is str and component and all(m['component_id'] == component for m in group), 'canonical consecutive twelve-derivative world')
        require(len({(m['family'], m['placement'], m['missing_index']) for m in group}) == 1, 'constant world strata')
        family, placement, missing = group[0]['family'], group[0]['placement'], group[0]['missing_index']
        require(family in FAMILIES and placement in PLACEMENTS and type(missing) is int and missing in (0, 1, 2), 'original family/placement/missing strata')
        require([(rows[offset + i]['prompt_tokens'], m['variant']) for i, m in enumerate(group)]
                == [(length, variant) for length in ENVELOPES for variant in VARIANTS], 'canonical world then length then four variants')
        worlds.append(component); counts[family, placement] += 1; missing_counts[family, placement, missing] += 1
    require(len(set(worlds)) == 192 and counts == {(f, p): 24 for f in FAMILIES for p in PLACEMENTS}
            and missing_counts == {(f, p, m): 8 for f in FAMILIES for p in PLACEMENTS for m in (0, 1, 2)}, 'original192 world quota inventory')
    require(type(domains) is list and domains, 'externally declared nonempty long domain roster')
    joined = {}; component_domains = {}; names = set()
    for domain in domains:
        require(type(domain) is dict and set(domain) == {'domain_id', 'class', 'row_ids', 'component_ids'}
                and type(domain['domain_id']) is str and domain['domain_id'].strip()
                and domain['domain_id'] != 'synth_task_retention001' and domain['domain_id'] not in names
                and domain['class'] == 'long_context' and type(domain['row_ids']) is type(domain['component_ids']) is list
                and 0 < len(domain['row_ids']) == len(domain['component_ids']), 'exact externally frozen long domain')
        names.add(domain['domain_id'])
        for row_id, component in zip(domain['row_ids'], domain['component_ids']):
            require(row_id not in joined and (component not in component_domains or component_domains[component] == domain['domain_id']), 'row/domain unique and whole world domain membership')
            joined[row_id] = component; component_domains[component] = domain['domain_id']
    require(joined == {m['row_id']: m['component_id'] for m in metadata}, 'every original row/component in frozen domains')
    return source

def freeze_plan(source, validate_row):
    source = population(source, validate_row); rows = source['rows']; batches = []; permutation = []
    for length, cap in ENVELOPES.items():
        indices = [i for i, row in enumerate(rows) if row['prompt_tokens'] == length]
        require(len(indices) == 768, 'twelve cap-homogeneous64 batches for each original envelope')
        for offset in range(0, 768, 64):
            chosen = indices[offset:offset + 64]; selected = [rows[i] for i in chosen]
            batches.append({'ordinal': len(batches) + 1, 'context_cap': cap, 'source_indices': chosen,
                'source_packet_sha256': digest(selected), 'maximum_attempted_forwards': 2048,
                'maximum_input_positions': sum(32 * row['prompt_tokens'] + 496 for row in selected),
                'maximum_generated_tokens': 32})
            permutation.extend(chosen)
    inverse = [None] * 2304
    for capture_index, source_index in enumerate(permutation): inverse[source_index] = capture_index
    return seal({'kind': 'LONG_GUARD_FROZEN_CAP_BATCH_PLAN_V1', 'source_sha256': digest(source),
        'row_order_sha256': digest([r['row_id'] for r in rows]), 'batches': batches,
        'capture_to_source': permutation, 'source_to_capture': inverse, 'source_rows': 2304,
        'world_components': 192, 'scoring_authorized': False, 'native_launch_authorized': False})

def check_plan(source, plan, validate_row):
    require(plan == freeze_plan(source, validate_row), 'exact frozen cap permutation/inverse/forward-position dose')

def _durable_type(component):
    class DurableLongCollection(component.InferenceComponent):
        """Only add short collector's synchronous token sink to original math."""
        def __init__(self, *args, source_packet_sha256, **kwargs):
            self.packet_sha = source_packet_sha256; super().__init__(*args, **kwargs)
        def _trace(self, row, operation, trace, step_preserve=None):
            def preserve_token(step):
                receipt = self._receipt(row, self.packet_sha, trace, 'RAW_TOKEN_PARTIAL_UNSCORED')
                if receipt['finish_reason'] != 'eos' and len(trace) < self.generation_cap:
                    receipt['finish_reason'] = 'in_progress'; receipt['truncated'] = False
                receipt['row_ordinal'] = self.row_cursor; receipt['step_ordinal'] = len(trace)
                self._persist('raw_row%04d_token%02d' % (self.row_cursor, len(trace)), receipt)
            return super()._trace(row, operation, trace, preserve_token)
    return DurableLongCollection

def collect_loaded(source, plan, ordinal, *, torch, model, decoder, journal, guard, device, component):
    """One64 packet using already-loaded assets; caller owns lifecycle/base hashes."""
    require(type(ordinal) is int and 1 <= ordinal <= 36, 'one explicit long64 batch ordinal')
    check_plan(source, plan, component.validate_row); batch = plan['batches'][ordinal - 1]
    rows = [source['rows'][i] for i in batch['source_indices']]
    require(callable(journal.call) and callable(journal.evidence) and callable(guard), 'original journal/guard mandatory')
    collector = _durable_type(component)(torch, model, decoder, journal.call, guard, journal.evidence,
        source_packet_sha256=batch['source_packet_sha256'], context_cap=batch['context_cap'], generation_cap=32,
        max_attempted_forwards=batch['maximum_attempted_forwards'], max_input_positions=batch['maximum_input_positions'], device=device)
    result = collector.collect(rows, source_packet_sha256=batch['source_packet_sha256'])
    require(len(result['rows']) == 64 and [r['row_id'] for r in result['rows']] == [r['row_id'] for r in rows], 'whole fixed long64 result order')
    envelope = seal({'kind': 'LONG_GUARD_DECODED_RAW_BATCH_V1', 'scope': SCOPE, 'ordinal': ordinal,
        'plan_sha256': plan['sha256'], 'source_indices': batch['source_indices'], 'collection': result,
        'parameter_preservation': collector.verify_unchanged(), 'fixture_only': device == 'cpu',
        'native_PASS': False, 'scoring_authorized': False, 'retention_certified': False})
    journal.evidence('long_guard_batch', envelope)
    return envelope

def _check_raw(row, raw, batch, *, fixture_only):
    require(raw['row_id'] == row['row_id'] and raw['prompt_sha256'] == row['prompt_sha256']
            and raw['prompt_ids_sha256'] == digest(row['prompt_ids']) and raw['input_row_sha256'] == digest(row)
            and raw['source_packet_sha256'] == batch['source_packet_sha256'], 'saved raw exact source/prompt/packet provenance')
    require(raw['status'] == 'RAW_RESPONSE_COLLECTED_UNSCORED' and raw['scope'] == ('CPU_FAKE_COMPONENT' if fixture_only else 'UNQUALIFIED_NATIVE_IMPLEMENTATION')
            and raw['oracle_inputs_read'] is False and raw['capability_correctness_scored'] is False
            and raw['prompt_truncated'] is False and raw['padding'] is False and raw['packing'] is False
            and raw['decoding'] == 'greedy_no_cache_no_grad', 'complete unscored original long raw mechanics')
    ids, lp, trace = raw['raw_generated_ids'], raw['selected_logprobs'], raw['trace']
    require(type(ids) is type(lp) is type(trace) is list and 1 <= len(ids) == len(lp) == len(trace) <= 32
            and all(type(t) is int and 0 <= t < 151936 for t in ids) and not any(t in (151643,151645) for t in ids[:-1]), 'typed first-EOS bounded raw trace')
    for i, step in enumerate(trace):
        require(type(step) is dict and set(step) == {'token','logprob','prefix_length'} and type(step['token']) is int and step['token'] == ids[i]
                and type(lp[i]) in (int,float) and math.isfinite(lp[i]) and lp[i] <= 0
                and type(step['logprob']) in (int,float) and math.isfinite(step['logprob']) and step['logprob'] <= 0 and step['logprob'] == lp[i]
                and type(step['prefix_length']) is int and step['prefix_length'] == row['prompt_tokens'] + i, 'saved raw token/logprob/prefix exact lineage')
    eos = ids[-1] in (151643,151645)
    require(raw['finish_reason'] == ('eos' if eos else 'generation_cap') and raw['truncated'] is (not eos)
            and ((type(raw['eos_token_id']) is int and raw['eos_token_id']==ids[-1]) if eos else raw['eos_token_id'] is None) and (eos or len(ids) == 32)
            and type(raw['context_cap']) is int and raw['context_cap'] == batch['context_cap'] and type(raw['maximum_generated_tokens']) is int and raw['maximum_generated_tokens'] == 32
            and type(raw['decoded_response']) is str, 'original EOS/cap/decode/envelope semantics')
    require(raw['parameter_preservation']['objects_pointers_dtypes_flags_gradients_preserved'] is True
            and raw['parameter_preservation']['mutation_counters_preserved'] is True, 'original parameter-preservation evidence')
    require(raw['native_launch_authorized'] is False and raw['native_qualification_established'] is False and raw['confirmation_eligible'] is False, 'original raw closed qualification flags')
    def typed_counts(value):
        if type(value) is dict:
            for item in value.values():typed_counts(item)
        else:require(type(value) is int and value>=0, 'strict integer cost counters')
    typed_counts(raw['costs'])

def audit_saved_batch(source, plan, ordinal, saved, *, component, fixture_only):
    """Re-read bytes and every partial/final; no in-memory producer trust or grader."""
    require(fixture_only is True, 'native long saved audit requires absent qualified owned asset/decoder loader')
    require(type(ordinal) is int and 1<=ordinal<=36, 'strict original long batch ordinal')
    check_plan(source, plan, component.validate_row)
    return _audit_saved_batch(source,plan,ordinal,saved,fixture_only=True)

def _audit_saved_batch(source,plan,ordinal,saved,*,fixture_only):
    require(type(ordinal) is int and 1<=ordinal<=36, 'strict original long batch ordinal')
    batch=plan['batches'][ordinal-1]
    require(type(saved) is dict and set(saved) == {'batch','rows','tokens','events'}, 'complete saved raw descriptor roster')
    envelope = read_ref(saved['batch'])
    require(envelope['sha256'] == digest({k:v for k,v in envelope.items() if k != 'sha256'})
            and envelope['kind'] == 'LONG_GUARD_DECODED_RAW_BATCH_V1' and envelope['scope'] == SCOPE
            and type(envelope['ordinal']) is int and envelope['ordinal'] == ordinal and envelope['plan_sha256'] == plan['sha256']
            and envelope['source_indices'] == batch['source_indices'] and envelope['fixture_only'] is fixture_only
            and envelope['native_PASS'] is False and envelope['scoring_authorized'] is False, 'saved exact batch/plan/scope binding')
    require(envelope['collection']['scope']=='CPU_FAKE_COMPONENT' and envelope['collection']['native_launch_authorized'] is False
            and envelope['collection']['native_qualification_established'] is False and envelope['collection']['confirmation_eligible'] is False
            and envelope['retention_certified'] is False and envelope['parameter_preservation']['objects_pointers_dtypes_flags_gradients_preserved'] is True
            and envelope['parameter_preservation']['mutation_counters_preserved'] is True, 'saved closed collection scope and preservation')
    outputs = envelope['collection']['rows']; require(len(outputs) == len(saved['rows']) == len(saved['tokens']) == 64, 'whole saved64 partial/final population')
    cumulative_forwards = cumulative_positions = 0; joined = []
    for index, (source_index, raw, row_ref, token_refs) in enumerate(zip(batch['source_indices'], outputs, saved['rows'], saved['tokens']), 1):
        row = source['rows'][source_index]; _check_raw(row, raw, batch, fixture_only=fixture_only)
        require(read_ref(row_ref) == raw and len(token_refs) == len(raw['trace']), 'terminal/final bytes and complete token partial roster')
        for step, ref in enumerate(token_refs, 1):
            partial = read_ref(ref)
            expected = {k:deepcopy(v) for k,v in raw.items() if k not in ('decoded_response','parameter_preservation')}
            expected.update({'status':'RAW_TOKEN_PARTIAL_UNSCORED','raw_generated_ids':raw['raw_generated_ids'][:step],
                'selected_logprobs':raw['selected_logprobs'][:step], 'trace':raw['trace'][:step], 'row_ordinal':index,'step_ordinal':step})
            token_eos = expected['raw_generated_ids'][-1] in (151643,151645)
            expected.update({'finish_reason':'eos' if token_eos else ('generation_cap' if step == 32 else 'in_progress'),
                'truncated':step == 32 and not token_eos,'eos_token_id':expected['raw_generated_ids'][-1] if token_eos else None})
            attempted = cumulative_forwards + step; positions = cumulative_positions + sum(row['prompt_tokens'] + j for j in range(step))
            expected['costs'] = {'operations':{'inference_forward':{'attempted':attempted,'returned':attempted,'failed':0,'pending':0}},
                'input_positions':{'attempted':positions,'returned':positions,'failed':0,'pending':0},
                'attempted_forwards':attempted,'maximum_attempted_forwards':2048,'maximum_input_positions':batch['maximum_input_positions']}
            require(partial == expected and digest(partial)==digest(expected), 'exact typed saved per-token partial trace/finish/cost progression')
        cumulative_forwards += len(raw['trace']); cumulative_positions += sum(row['prompt_tokens'] + j for j in range(len(raw['trace'])))
        require(raw['costs'] == expected['costs'], 'final raw cumulative cost exact progression')
        joined.append({'source_index':source_index,'capture_batch_ordinal':ordinal,'capture_row_ordinal':index,
            'row_id':row['row_id'],'input':deepcopy(row),'guard_metadata':deepcopy(source['metadata'][source_index]),'raw':deepcopy(raw)})
    require(envelope['collection']['source_packet_sha256'] == batch['source_packet_sha256'] and envelope['collection']['costs'] == outputs[-1]['costs'], 'batch terminal cost/source join')
    events_path = Path(saved['events']['path']);raw_events=_selected_bytes(saved['events'],json_value=False)
    events = [_json(line) for line in raw_events.splitlines()]
    expected_events=[];call_id=0
    def measurement(ref,name):
        require(Path(ref['path'])==events_path.parent/name, 'selected exact original journal measurement path')
        expected_events.append({'event':'measurement_preserved','path':name,'sha256':ref['sha256'],'qualification_established':False})
    for row_ordinal,(raw,row_ref,token_refs) in enumerate(zip(outputs,saved['rows'],saved['tokens']),1):
        for step,ref in enumerate(token_refs,1):
            call_id+=1;expected_events.extend([{'event':'attempt','id':call_id,'operation':'inference_forward'},{'event':'returned','id':call_id,'operation':'inference_forward'}])
            measurement(ref,'stage_raw_row%04d_token%02d.json'%(row_ordinal,step))
        measurement(row_ref,'stage_inference_row_%04d.json'%row_ordinal)
    measurement(saved['batch'],'stage_long_guard_batch.json')
    require(len(events)==len(expected_events), 'complete exact saved journal causal event roster')
    for event,expected in zip(events,expected_events):
        require(type(event) is dict and set(event)==set(expected)|{'utc'} and type(event['utc']) is str, 'exact original journal event fields')
        datetime.fromisoformat(event['utc'])
        for key,value in expected.items():require(type(event[key]) is type(value) and event[key]==value, 'exact original journal causal order and selected measurement join')
    _recheck_saved(saved)
    return {'rows':joined,'batch_reference':deepcopy(saved['batch']),'attempted_forwards':cumulative_forwards,'input_positions':cumulative_positions}

def _recheck_saved(saved):
    for ref in [saved['batch'],*saved['rows'],*(r for refs in saved['tokens'] for r in refs)]:_selected_bytes(ref)
    _selected_bytes(saved['events'],json_value=False)

def audit_saved_population(source, plan, saved_batches, *, component, fixture_only):
    require(fixture_only is True, 'native long saved audit requires absent qualified owned asset/decoder loader')
    require(type(saved_batches) is list and len(saved_batches) == 36, 'whole original36 cap batches required')
    check_plan(source,plan,component.validate_row);original_source=digest(source);original_plan=digest(plan)
    canonical = [None] * 2304; refs = []
    for ordinal, saved in enumerate(saved_batches, 1):
        checked = _audit_saved_batch(source, plan, ordinal, saved, fixture_only=True); refs.append(checked['batch_reference'])
        for row in checked['rows']:
            require(canonical[row['source_index']] is None, 'unique inverse raw join'); canonical[row['source_index']] = row
    require(all(r is not None for r in canonical) and digest([r['row_id'] for r in canonical]) == plan['row_order_sha256'], 'whole canonical2304 inverse order')
    for saved in saved_batches:_recheck_saved(saved)
    require(digest(source)==original_source and digest(plan)==original_plan, 'source and frozen plan unchanged after all saved reads')
    return seal({'kind':'LONG_GUARD_CANONICAL_UNSCORED_IMPORT_V1','scope':SCOPE,'rows':canonical,
        'declared_domains':deepcopy(source['declared_domains']),'plan_sha256':plan['sha256'],
        'batch_references':refs,'fixture_only':fixture_only,'scoring_authorized':False,'retention_certified':False,'native_PASS':False})

def _CPU_collect(source, plan, ordinal, *, capability, **boundaries):
    _check_capability(capability)
    require(boundaries.get('device') == 'cpu' and getattr(boundaries.get('torch'),'_synth_fake_component',False) is True, 'authored fake CPU collection only')
    return collect_loaded(source, plan, ordinal, **boundaries)

def run_registered(*args, **kwargs):
    # No loader or real asset calls precede source installation and registration.
    require(HERE == EXPECTED, 'staged long collector has no native authority')
    raise ValueError('native long DEV owned loaded-asset qualification/dispatch registration absent; TRAIN NoDecode qualification cannot substitute')
