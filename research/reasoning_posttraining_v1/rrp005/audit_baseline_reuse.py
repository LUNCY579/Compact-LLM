"""Verify reusable completed source measurements by exact input/source/protocol identity.

Selections depend only on identity, never on correctness. Historical measurements are
attributed as reused; changed prompts cannot inherit their old outputs.
"""
import collections
import json
from pathlib import Path
import sys

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
sys.path.insert(0, str(ROOT))
from experiments.reasoning_posttraining_v1.repair_preservation import tokenizer_only
from experiments.reasoning_posttraining_v1.serialization import compile_row, compile_dialogue
from experiments.reasoning_posttraining_v1.sources import stable_hash
from experiments.model_quality_gpu_v1.integrity import write_new

records = ROOT.parent / 'rineya-dataset-retention-records'
old_run = ROOT.parent / 'rineya-reasoning-posttraining-records/rrp002-baseline-a3-journal/runs/baseline-pres-002'
old_file = old_run / 'baseline.json'
old = json.loads(old_file.read_text())
supervisor = json.loads((old_run / 'supervisor.json').read_text())
if supervisor['status'] != 'EXITED' or supervisor['returncode'] != 0:
    raise ValueError('historical source run did not finish successfully')
if old['source_state_before'] != old['source_state_after'] or old['optimizer_steps'] != 0:
    raise ValueError('historical source state integrity absent')
if old['max_sequence'] != 2048 or old['generation_allowance_reserved'] != old['rows'] * 64:
    raise ValueError('historical context/generation limits differ')
old_bank_file = ROOT / 'research/reasoning_posttraining_v1/evidence/inspection-20260915/RRP2_PRES_DEV_BANK.jsonl'
if stable_hash(old_bank_file) != old['bank_sha256']:
    raise ValueError('historical bank bytes differ')
def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
old_bank = {r['id']: r for r in rows(old_bank_file)}
old_predictions = {r['id']: r for r in old['records']}
corrected = records / 'preservation-correction-002'
new = rows(corrected / 'RRP5_BASELINE_INPUTS.jsonl')
stream = records / 'preservation-baseline-a2-journal/runs/baseline-002/prediction-stream'
context = json.loads((stream / 'START.json').read_text())['context']
if (context['binding']['source_state_sha256'] != old['source_state_before'] or
    context['binding']['source'] != old['binding']['source'] or
    context['binding']['runtime'] != old['binding']['runtime']):
    raise ValueError('source weights/state or recorded runtime differ')
fresh = [json.loads(p.read_text())['record'] for p in sorted((stream / 'records').glob('*.json'))]
tokenizer = tokenizer_only(ROOT / 'Supa')
reuse, acquire, reasons = [], [], collections.Counter()
for row in new:
    previous = old_bank.get(row['id'])
    if previous is None:
        acquire.append(row); reasons['new_id'] += 1
        continue
    if previous['prompt'] != row['prompt'] or previous['reference_text'] != row['reference_text']:
        acquire.append(row); reasons['changed_prompt_or_target'] += 1
        continue
    old_compiled = compile_row(previous['prompt'], previous['reference_text'], tokenizer, max_sequence=2048)
    new_compiled = compile_dialogue(row['prompt'], row['reference_text'], tokenizer, max_sequence=2048)
    for key in ('ids', 'labels', 'prompt_ids', 'answer_start', 'answer_end'):
        if old_compiled[key] != new_compiled[key]:
            raise ValueError('text equality did not reproduce token equality')
    prediction = old_predictions[row['id']]
    if (prediction['reference'] != row['reference_text'] or
        prediction['prefill_tokens'] != len(new_compiled['prompt_ids']) or
        prediction['fixed_ce_input_tokens'] != len(new_compiled['ids'])):
        raise ValueError('historical measurement does not match the compiled input')
    reuse.append(row['id'])
reuse_set = set(reuse)
checked = 0
for prediction in fresh:
    if prediction['id'] not in reuse_set:
        continue
    previous = old_predictions[prediction['id']]
    for key in ('prediction', 'generated_ids_including_terminal', 'terminated', 'reference',
                'prefill_tokens', 'fixed_ce_input_tokens'):
        if prediction[key] != previous[key]:
            raise ValueError('fresh overlap disagrees with historical source output')
    checked += 1
out = records / 'baseline-reuse-audit-001'
out.mkdir(exist_ok=False)
with (out / 'CHANGED_OR_NEW_INPUTS.jsonl').open('x', encoding='utf8', newline='\n') as f:
    for row in acquire:
        f.write(json.dumps(row, sort_keys=True) + '\n')
report = {'record': 'RRP005_COMPLETED_SOURCE_REUSE_AUDIT',
    'historical_baseline': str(old_file), 'historical_baseline_sha256': stable_hash(old_file),
    'historical_bank_sha256': stable_hash(old_bank_file),
    'source_state_sha256': old['source_state_after'], 'source_and_runtime_identical': True,
    'native_token_and_mask_identity_checked': len(reuse),
    'fresh_overlap_checked': checked, 'fresh_overlap_differences': 0,
    'reused_ids': reuse, 'new_acquisition_ids': [r['id'] for r in acquire],
    'new_acquisition_sha256': stable_hash(out / 'CHANGED_OR_NEW_INPUTS.jsonl'),
    'acquisition_reasons': dict(reasons),
    'execution_difference': 'Old run retained unused CUDA allocator blocks; A2 releases them between independent cases. Inference arithmetic/policy is unchanged and all available fresh overlap matches token-for-token. No latency/cost equivalence is claimed.',
    'selection_rule': 'exact prompt, target, tokens, masks, source state and protocol identity only; no correctness filtering',
    'reused_evidence_class': 'PREVIOUSLY_COMPLETED_NATIVE_SOURCE_MEASUREMENT_NEWLY_VERIFIED_IDENTITY',
    'model_calls_in_this_audit': 0, 'quality_improvement_established': False}
write_new(out / 'AUDIT.json', report)
print(json.dumps({'reusable_completed_predictions': len(reuse), 'new_inputs_required': len(acquire),
                  'fresh_overlap_token_parity': checked, 'source_state_identical': True,
                  'new_input_reasons': dict(reasons)}, indent=2))
