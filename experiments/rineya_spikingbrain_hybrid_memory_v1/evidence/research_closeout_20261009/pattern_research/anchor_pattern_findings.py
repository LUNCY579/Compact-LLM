"""Deterministic grouped findings from the preserved anchor pattern receipt."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_RESULTS = '0806a6a6a5704e09ba4aca12865157b71bbacd794e4a773df2c1c7bfc316eba5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def findings():
    path = ROOT/'anchor-patterns.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    data = receipt['results']
    assert digest(data) == receipt['results_sha256'] == EXPECTED_RESULTS
    assert sha(ROOT/'anchor_patterns.py') == receipt['analyzer_sha256']
    for reference in receipt['input_files'].values():
        assert sha(reference['path']) == reference['sha256']
    update = data['update_rows']
    grouped = defaultdict(list)
    for row in update:
        grouped[row['world_id']].append(row)
    per_world = []
    for world, rows in sorted(grouped.items()):
        count = [sum(r['code_lexeme_payload_match'] for r in rows if r['latest_position'] == p) for p in range(3)]
        assert all(sum(r['latest_position'] == p for r in rows) == 8 for p in range(3))
        per_world.append({'world_id': world, 'rows_per_latest_position': 8,
            'latest_payload_code_lexeme_matches_by_first_middle_last': count,
            'middle_minus_first': count[1]-count[0], 'middle_minus_last': count[1]-count[2]})
    missing = data['missing_rows']
    absent = [r for r in missing if not r['answerable']]
    revision = [r for r in update if r['selection_class'] == 'visible_revision_numeral']
    observations = {
        'middle_vs_first': {'worlds': 16,
            'middle_better_worlds': sum(w['middle_minus_first'] > 0 for w in per_world),
            'middle_tied_worlds': sum(w['middle_minus_first'] == 0 for w in per_world),
            'middle_worse_worlds': sum(w['middle_minus_first'] < 0 for w in per_world),
            'per_world': per_world,
            'latest_lexeme_matches_first_middle_last': [33,105,60],
            'rows_per_latest_position': 128,
            'interpretation':'Changing only record order changes the selected visible value on this fixed renderer; middle placement helps in every dependent world.'},
        'middle_vs_last': {'middle_better_worlds': sum(w['middle_minus_last'] > 0 for w in per_world),
            'middle_tied_worlds': sum(w['middle_minus_last'] == 0 for w in per_world),
            'middle_worse_worlds': sum(w['middle_minus_last'] < 0 for w in per_world)},
        'field_binding_on_absence': {'absent_rows':32, 'absent_worlds':16,
            'same_subject_wrong_field_rows':sum(r['same_subject_wrong_field_weight'] for r in absent),
            'same_subject_wrong_field_worlds':len({r['world_id'] for r in absent if r['same_subject_wrong_field_weight']}),
            'wrong_subject_code_rows':sum(r['wrong_subject_code'] for r in absent),
            'donor_code_rows':sum(r['donor_code_match'] for r in absent),
            'other_decoy_code_rows':sum(r['other_decoy_code_match'] for r in absent),
            'literal_null_rows':sum(r['literal_null'] for r in absent),
            'unsupported_literal_rows':sum(r['unobserved_literal'] for r in absent),
            'present_requested_code_lexeme_matches':31,'present_rows':32,
            'interpretation':'In absent cases all outputs copy visible facts, chiefly the correct subject with the wrong field; presence of a subject is being substituted for presence of the requested field.'},
        'revision_key_vs_payload': {'conflict_rows':192,'latest_payload_lexeme_selected':66,
            'greatest_code_payload_selected':85,
            'greatest_code_wrong_selections_worlds':len({r['world_id'] for r in update if r['selection_class']=='greatest_code_only'}),
            'revision_numeral_outputs':len(revision),
            'revision_numeral_distant_rows':sum(r['role']=='distant_heldout' for r in revision),
            'revision_numeral_near_rows':sum(r['role']=='near_control' for r in revision),
            'greatest_revision_numeral_outputs':sum(r['matches_greatest_revision_numeral'] for r in revision),
            'interpretation':'Some errors compare payload magnitude instead of revision; others emit a revision numeral rather than reading the selected record payload.'},
        'representation': {'update_rows':384,'valid_json_integer_outputs':384,
            'original_strict_update_correct':0,'original_typed_update_correct':0,
            'type_exact_payload_matches':0,'canonical_code_lexeme_payload_matches':198,
            'interpretation':'Wrapping numbers alone is insufficient: the requested quoted codes also require string types. Digit-selection diagnostics are not declared correctness.'},
        'consistency_not_correctness':data['six_order_bundle_summary'],
        'matched_distance':data['update_near_distant_summary'],
    }
    assert observations['middle_vs_first']['middle_better_worlds'] == 16
    assert observations['middle_vs_last']['middle_better_worlds'] == 15
    assert observations['field_binding_on_absence']['same_subject_wrong_field_worlds'] == 15
    hypotheses = [
        {'priority':1,'hypothesis':'Explicit binding of both requested subject and field can reduce copying of same-subject wrong-field facts when the requested field is absent.',
         'evidence':'25/32 absent rows copy weight for the requested subject; spread across 15/16 worlds. Present code selection is 31/32 by the separately declared lexeme diagnostic.',
         'minimal_discriminating_control':'A future separately authorized wording control can keep every visible fact and its order fixed, changing only the query to require both subject and field before reading a value, otherwise null. Compare against the original terse query on matched present and absent cases.',
         'model_quality_followup_if_control_supports_it':'Only then consider teaching select-then-read field binding while keeping present/absent exposure balanced and testing transfer without the explicit wording. This is a hypothesis, not an implemented successor.',
         'alternatives':['The terse query may be the primary presentation problem rather than a learnable-model deficit.','Extra query wording may alter attention or token distance.','Historical adaptation already showed an all-null shortcut; more absent examples alone is not justified.']},
        {'priority':2,'hypothesis':'Balanced record ordering can reduce dependence on text position, but revision comparison and payload return must be distinguished from largest-payload copying.',
         'evidence':'Middle latest placement wins over first in every one of 16 worlds; only 8/64 bundles choose latest digits across all six orders. In conflict cases largest payload is chosen 85 times versus latest 66.',
         'minimal_discriminating_control':'The already registered balanced-order candidate versus repeated-order control is the relevant prospective contrast; do not infer its result from this anchor. Evaluate the predeclared typed metrics and all-six correctness; describe existing conflict rows separately without changing registered gates.',
         'alternatives':['The renderer may encourage comparison of the code number.','A largest-code heuristic can be stable across orders and still be wrong.','Small LoRA adaptation can learn a different positional shortcut rather than the revision rule.']},
        {'priority':3,'hypothesis':'Selection instructions should specify the comparison key and separately identify the returned field; quoted-code output needs both JSON wrapper and string type.',
         'evidence':'43 outputs are visible revision numerals, 33 of which are the greatest revision; 41/43 occur distant. All 384 update outputs are integer scalars.',
         'minimal_discriminating_control':'A future separately authorized source-only wording contrast can say compare revision numbers, then return the corresponding quoted code as a JSON string. Preserve facts, all orders, counterfactuals and near/distant pairing.',
         'alternatives':['Formatting/type repair may raise contract scores without improving record selection.','Returning the comparison key can reflect query ambiguity or retrieval/field composition.','A prompt-only improvement is presentation sensitivity, not intrinsic model learning or permission to change serving policy.']},
    ]
    return {'schema':'RINEYA-ANCHOR-OFFLINE-PATTERN-FINDINGS/v1',
        'status':'REPRODUCIBLE_DIAGNOSTIC_PATTERNS_NO_MEASURED_GAIN',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'input_receipt':{'path':str(path),'sha256':sha(path),'results_sha256':EXPECTED_RESULTS},
        'input_analyzer':{'path':str(ROOT/'anchor_patterns.py'),'sha256':sha(ROOT/'anchor_patterns.py')},
        'findings_analyzer_sha256':sha(__file__), 'frozen_bindings':receipt['frozen_bindings'],
        'observations':observations,'hypotheses_not_gain_claims':hypotheses,
        'no_quality_gain_measured':True,'native_retry_performed':False,'training_performed':False,
        'frozen_corpus_or_metrics_modified':False,'serving_policy_modified':False,
        'limitations':['One source and fixed synthetic DEV renderer; dependent variants grouped by 16 worlds.',
            'Canonical code-lexeme equality deliberately erases output type for diagnosis; raw strict and typed remain unchanged.',
            'No proposed intervention was executed and no transfer or broad retention gain was established.',
            'The prior native pilot stopped at the unchanged RAM guard; no automatic rescheduling or expanded training budget is authorized.']}


if __name__=='__main__':
    output=ROOT/'anchor-pattern-findings.json'
    value=findings()
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':value['status'],'output':str(output),'observations_sha256':digest(value['observations'])},sort_keys=True))
