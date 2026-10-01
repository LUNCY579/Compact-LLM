"""S0 enrollment and churn runs, stage S0-E (COMP-RS-001 Amendment 1, section 16; sections 1, 8.1, 8.3, 8.5, 12).

python -m experiments.maxrl_comp002_pretraining_v1.s0_enroll --run 1 --runtime-freeze RECORDS\\S0_RUNTIME_FREEZE.json
python -m experiments.maxrl_comp002_pretraining_v1.s0_enroll --run 2 --runtime-freeze RECORDS\\S0_RUNTIME_FREEZE.json
    [--out RECORDS\\S0_ENROLL_RUN<n>.json] [--lease-timeout-s 900]
(theta0': --runtime-freeze RECORDS\\S0_RUNTIME_FREEZE_WARM.json; the default out is RECORDS\\S0_ENROLL_RUN<n>_WARM.json)

Each run greedy-decodes the source with batch size 1 (section 8.1): arm B (Supa streamed into FP32), or theta0' =
Supa + delta_warm (section 18) when the runtime freeze carries an anchor, loaded, bound and named as in
s0_competence. The order is all 1,200 guard_dev rows in bank order, then the 450 CALIBRATION core R, S and P rows
(R, S, P, each sorted by id). Run 2 decodes the same rows in a separate process; s0_churn compares the two runs'
token IDs. Nothing here trains, saves weights, or opens DEV, TEST, CHALLENGE or guard_test.

- Rows. guard_dev comes from scoped_scorer.load_role (manifest sha, file sha, row count). CALIBRATION comes from
  s0_competence.load_calibration, with every skill asserted against evidence/S0PRE_CALIBRATION_SCOPE.json. Policy
  messages must equal the canonical prompts and prompt_hash must equal contracts.digest(prompt). Registered mode
  requires the full plan: every guard_dev row in bank order, then every core R/S/P row in order.
- Decoding. s0_generate.greedy_decode: do_sample False, batch 1, max_new_tokens 128, EOS (151645, 151643), PAD
  151643, thinking disabled. Every greedy token is replayed against the raw-logit argmax. Each journal line holds
  the token IDs, the response text (s0_competence.RESPONSE_RULE), the stop reason, the generated tokens, the raw
  step logprobs, the decoder record, the wall and the process.
- Journal (section 12). <out>.jsonl through s0_generate.ResumableJsonl (row hash, fsync). The bind fields are
  the anchor, the arm and weights digests, decoding_sha256, the generation contract, the plan digest, the
  tokenizer receipt, the code bundle and the run number. A launch resumes at the first missing row.
  <out>.attempts.jsonl records every launch. hostbudget.guard runs before every row.
- After completion:
  - guard_dev. behaviour.enroll(data, 'guard_dev', responses, run_meta) runs unchanged on the full bank. The
    responses are exactly {id, prompt_hash, response, generated_tokens}, in bank order. run_meta:
    - model_sha256 = the asset receipt's model.safetensors sha (the loader checks the bytes against it) for plain
      Supa, the anchor arm digest (s0_runtime.ANCHOR_DIGEST_RULE) for theta0' (s0_competence.run_meta);
    - tokenizer_sha256 = the tokenizer receipt's file sha (s0_competence.TOKENIZER_SHA_RULE);
    - decoding_sha256 = s0_runtime.decoding_sha256 of the section 8.2 preimage.
    The same responses are also scored through scoped_scorer with canary.score_with_canary (legacy section), and
    its per-row strict scores must equal behaviour.enroll's row_scores (the section 8.0 fixture, at run time).
    A partial guard_dev plan (test mode only) gets no behaviour.enroll; behaviour.enroll refuses partial sets.
  - CALIBRATION. The 450 rows go through canary.score_with_canary, bound to the frozen scope receipt and to the
    expected scorer code bundle. That bundle is the runtime freeze's scoring_code_bundle_sha256 when a freeze is
    given (always, in registered mode), and otherwise the live bundle (test mode).
  - The receipt <out> is create-only, stage S0_ENROLL_RUN<n>. Run 1 is the enrollment of record (section 16:
    enrollment = churn run 1). Run 2's scores are recorded for the churn cross-check.

Registered mode takes no size or path overrides: cuda, the full plan, the pinned model, asset and tokenizer
receipts, the clean checkout holding this code, the pinned lease and host budget, and the runtime freeze (its
host floor is the launch floor; its preimage must reproduce the live decoding_sha256). --test-mode admits a tiny
model, a --host-floor-gib, a missing freeze, row subsets (by replacing guard_dev_plan and
s0_competence.calibration_plan) and injected lease/host modules. Exit status: 0 for a complete run, 3 for an
incomplete read (resumable).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from experiments.maxrl_composable_skills_v2 import behaviour, corpus

from . import canary, content_scoring, scoped_scorer
from . import s0_competence as comp
from . import s0_runtime as s0
from .preflight import digest, require

SCHEMA = 'COMP-RS-001-S0-ENROLL-1'
REGISTRATION = 'COMP_RS_001_AMENDMENT_1.md section 16, enrollment and churn (S0-E)'
RUNS = (1, 2)
GUARD_DEV_ROWS = 1200
PLAN_RULE = ('all guard_dev rows in bank (file) order, then the CALIBRATION core rows of R, S and P, each sorted '
             'by id')
CODE_FILES = comp.code_files(comp.PACKAGE + 's0_enroll.py')
_IMPORT_CODE = comp.code_sha256(CODE_FILES)
require(Path(behaviour.__file__).resolve() in {p.resolve() for p in CODE_FILES.values()},
        'behaviour is not loaded from its hashed file')


def stage_name(run: int) -> str:
    require(run in RUNS, 'run must be 1 or 2')
    return 'S0_ENROLL_RUN%d' % run


def guard_dev_plan(bank) -> list:
    """The registered guard_dev order: the whole bank in file order (tests replace this with a subset)."""
    return list(bank)


def _row_group(row: dict) -> str:
    family = content_scoring.row_family(row)
    return row['skills'][0] if family is None else family


def _enroll_row(tokenizer, max_new_tokens: int):
    def decode_one(g, model, item, device) -> dict:
        result, wall = comp.timed(device, lambda: g.greedy_decode(model, item['prompt_ids'],
                                                                  max_new_tokens=max_new_tokens))
        text = comp.response_text(tokenizer, result['completion_ids'], result['stop_reason'])
        row = item['row']
        return {'id': item['id'], 'index': item['index'], 'bank': item['bank'], 'group': _row_group(row),
                'parent_id': row['parent_id'], 'prompt_hash': row['prompt_hash'],
                'prompt_tokens': item['prompt_tokens'], 'prompt_ids_sha256': item['prompt_ids_sha256'],
                'render_sha256': item['render_sha256'], 'completion_ids': result['completion_ids'],
                'stop_reason': result['stop_reason'], 'generated_tokens': result['generated_tokens'],
                'raw_step_logprobs': result['raw_step_logprobs'], 'response': text['response'],
                'body_special_ids': text['body_special_ids'], 'body_unknown_ids': text['body_unknown_ids'],
                'lenient_response': text['lenient_response'], 'decoder': result['decoder'], 'wall_s': wall}
    return decode_one


def scorer_item(row: dict) -> dict:
    """The scoped scorer's response schema."""
    return {'id': row['id'], 'prompt_hash': row['prompt_hash'], 'response': row['response'],
            'generated_tokens': row['generated_tokens'], 'stop_reason': row['stop_reason']}


def enrollment_responses(bank, rows: list) -> list:
    """behaviour.enroll's response list: exactly {id, prompt_hash, response, generated_tokens}, in bank order."""
    by_id = {row['id']: row for row in rows}
    require(set(by_id) == {row['id'] for row in bank} and len(by_id) == len(rows),
            'behaviour.enroll needs the full guard_dev bank')
    return [{'id': row['id'], 'prompt_hash': by_id[row['id']]['prompt_hash'], 'response': by_id[row['id']]['response'],
             'generated_tokens': by_id[row['id']]['generated_tokens']} for row in bank]


def _scoring_record(result: dict) -> dict:
    return {'canary_overall': result['canary']['overall'], 'canary_receipt_sha256': result['canary_receipt_sha256'],
            'scoring_receipt_sha256': result['scoring_receipt_sha256'],
            'code_bundle_sha256': result['code_bundle_sha256'], 'scoring': result['scoring']}


def _lenient(rows: list, bank_rows: dict, strict: dict) -> dict:
    """Rows whose pilot-rule (lenient) text differs, and how many of those it would score differently."""
    differing = [row for row in rows if row['lenient_response'] is not None]
    flips = [row['id'] for row in differing
             if int(corpus.reward(bank_rows[row['id']], row['lenient_response'])) != strict[row['id']]]
    return {'lenient_text_differs': len(differing), 'lenient_reward_differs': len(flips), 'lenient_flip_ids': flips}


def _walls(rows: list) -> dict:
    walls = [row['wall_s'] for row in rows]
    tokens = sum(row['generated_tokens'] for row in rows)
    return {'rows': len(rows), 'wall_s': math.fsum(walls), 'wall_per_completion_s': comp._mean(walls),
            'wall_max_s': max(walls) if walls else None, 'generated_tokens': tokens,
            'wall_per_token_s': math.fsum(walls) / tokens if tokens else None,
            'prompt_tokens': sum(row['prompt_tokens'] for row in rows),
            'stop_reason': {key: sum(row['stop_reason'] == key for row in rows) for key in ('eos', 'length')}}


def score_enrollment(data, cal: dict, guard_bank, prompts: list, rows: list, meta: dict, bundle: str, *,
                     full_guard: bool, sections=None) -> dict:
    """Score a complete journal: CALIBRATION and guard_dev through score_with_canary; guard_dev enrollment."""
    planned = [item['id'] for item in prompts]
    require([row['id'] for row in rows] == planned, 'journal rows differ from the plan')
    guard_rows = [row for row in rows if row['bank'] == 'guard_dev']
    cal_rows = [row for row in rows if row['bank'] == 'calibration']
    require(rows == guard_rows + cal_rows and guard_rows and cal_rows, 'the plan is guard_dev rows, then CALIBRATION')
    sections = comp.canary_sections(data, cal['bank'], guard_bank) if sections is None else sections
    receipts = {'calibration': cal['receipt']}
    signatures = list(comp.CANARY_SIGNATURES)
    cal_scope = scoped_scorer.declare_scope(cal['bank'], signatures=list(comp.SKILLS),
                                            ids=[row['id'] for row in cal_rows], tag='core')
    calibration = canary.score_with_canary(sections, cal['bank'], cal_scope, [scorer_item(r) for r in cal_rows],
                                           meta, signatures=signatures, expected_code_bundle=bundle,
                                           scope_receipts=receipts)
    guard_scope = scoped_scorer.declare_scope(guard_bank, tag='preservation', ids=[row['id'] for row in guard_rows])
    guard = canary.score_with_canary(sections, guard_bank, guard_scope, [scorer_item(r) for r in guard_rows], meta,
                                     signatures=signatures, expected_code_bundle=bundle, scope_receipts=receipts)
    guard_strict = {record['id']: record['strict'] for record in guard['scoring']['rows']}
    cal_strict = {record['id']: record['strict'] for record in calibration['scoring']['rows']}
    if full_guard:
        enrollment = behaviour.enroll(Path(data), 'guard_dev', enrollment_responses(guard_bank, guard_rows), meta)
        require(enrollment['bank_sha256'] == guard_bank.bank_sha256, 'behaviour.enroll read another guard_dev bank')
        require(enrollment['row_scores'] == guard_strict,
                'behaviour.enroll and the scoped scorer disagree on guard_dev')
        status = 'ENROLLED_FULL_BANK'
    else:
        enrollment, status = None, 'NOT_ENROLLED_PARTIAL_BANK (test mode: behaviour.enroll needs the full bank)'
    families = {}
    for row in guard_rows:
        entry = families.setdefault(row['group'], {'rows': 0, 'source_correct': 0})
        entry['rows'] += 1
        entry['source_correct'] += guard_strict[row['id']]
    skills = {}
    for row in cal_rows:
        entry = skills.setdefault(row['group'], {'rows': 0, 'strict_correct': 0, 'unknown_answer': 0})
        entry['rows'] += 1
        entry['strict_correct'] += cal_strict[row['id']]
        entry['unknown_answer'] += comp.unknown_answer(row['response'])
    bank_rows = {row['id']: row for row in list(guard_bank) + list(cal['bank'])}
    return {
        'guard_dev_enrollment': enrollment, 'guard_dev_enrollment_status': status,
        'guard_dev_scoring': _scoring_record(guard), 'calibration_scoring': _scoring_record(calibration),
        'canary_receipt': calibration['canary'],
        'canary_receipts_identical': calibration['canary_receipt_sha256'] == guard['canary_receipt_sha256'],
        'summary': {
            'guard_dev': {'rows': len(guard_rows), 'source_correct': sum(guard_strict.values()),
                          'source_correct_ids_sha256': digest(sorted(i for i, v in guard_strict.items() if v)),
                          'by_family': dict(sorted(families.items())),
                          'counts': guard['scoring']['counts']['overall'], 'walls': _walls(guard_rows),
                          'lenient': _lenient(guard_rows, bank_rows, guard_strict)},
            'calibration': {'rows': len(cal_rows), 'strict_correct': sum(cal_strict.values()),
                            'strict_correct_ids_sha256': digest(sorted(i for i, v in cal_strict.items() if v)),
                            'by_skill': {s: skills[s] for s in comp.SKILLS if s in skills},
                            'counts': calibration['scoring']['counts'], 'walls': _walls(cal_rows),
                            'lenient': _lenient(cal_rows, bank_rows, cal_strict)},
            'walls': _walls(rows)}}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_enroll', description='COMP-RS-001 S0-E enrollment / churn run.')
    ap.add_argument('--run', type=int, choices=RUNS, required=True)
    comp.add_common_arguments(ap)
    return ap


def _config(args, lease_module, host_module) -> dict:
    stage = stage_name(args.run)
    cfg = comp.common_config(args, lease_module, host_module, stage=stage,
                             label='comp-rs-001-s0-enroll-run%d' % args.run, default_name=stage + '.json')
    cfg['code'] = _IMPORT_CODE
    return cfg


def main(argv=None, *, lease_module=None, host_module=None) -> int:
    t0 = time.perf_counter()
    args = _parser().parse_args(argv)
    cfg = _config(args, lease_module, host_module)
    host = host_module if host_module is not None else s0.hostbudget()
    state = comp.new_state(argv, t0, comp.prepare_launch(args, cfg, host))
    guard_bank = scoped_scorer.load_role(args.data, 'guard_dev')
    guard_rows = guard_dev_plan(guard_bank)
    cal = comp.load_calibration(args.data)
    cal_rows = comp.calibration_plan(cal['rows_by_skill'])
    require(guard_rows and all(row['role'] == 'guard_dev' and row['kind'] == 'legacy' and row['tag'] == 'preservation'
                               for row in guard_rows), 'the guard_dev plan holds only guard_dev legacy rows')
    require(cal_rows and all(row['role'] == 'calibration' and row['tag'] == 'core' and row['kind'] == 'composition'
                             and row['skills'] in (['R'], ['S'], ['P']) for row in cal_rows),
            'the CALIBRATION plan holds only core R/S/P rows')
    full_guard = ([row['id'] for row in guard_rows] == [row['id'] for row in guard_bank]
                  and len(guard_rows) == GUARD_DEV_ROWS)
    full_cal = comp.is_full_plan(cal_rows, cal['rows_by_skill'])
    require((full_guard and full_cal) or cfg['test_mode'],
            'the registered plan is every guard_dev row, then every core R/S/P CALIBRATION row')
    policy = {'guard_dev': comp.check_policy(args.data, 'guard_dev', guard_rows),
              'calibration': comp.check_policy(args.data, 'calibration', cal_rows)}
    tokenizer = s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)
    prompts = (comp.render_prompts(tokenizer, guard_rows, 'guard_dev')
               + comp.render_prompts(tokenizer, cal_rows, 'calibration'))
    for index, item in enumerate(prompts):
        item['index'] = index
    planned = [item['id'] for item in prompts]
    require(len(set(planned)) == len(planned), 'planned ids repeat')
    state['inputs'] = dict(cal['record'], guard_dev_bank_sha256=guard_bank.bank_sha256, policy=policy,
                           plan_rule=PLAN_RULE, planned=len(planned), plan_ids_sha256=digest(planned),
                           plan_by_bank={'guard_dev': len(guard_rows), 'calibration': len(cal_rows)},
                           plan_by_group=dict(sorted(Counter(_row_group(i['row']) for i in prompts).items())),
                           full_guard_dev=full_guard, full_calibration=full_cal)
    def finish() -> dict:
        g = comp.generate_module()
        rows = g.ResumableJsonl(cfg['journal'], key='id', bind=state['bind']).rows()
        scored = score_enrollment(args.data, cal, guard_bank, prompts, rows, state['run_meta'],
                                  state['bundle']['expected'], full_guard=full_guard)
        receipt = dict(comp.base_receipt(args, cfg, state, schema=SCHEMA, registration=REGISTRATION),
                       run=args.run, enrollment_of_record=args.run == 1, inputs=state['inputs'],
                       max_new_tokens=cfg['max_new_tokens'],
                       decoder=comp.distinct_decoders(rows, 'decoder', cfg['max_new_tokens']),
                       completed=True, code_sha256=_IMPORT_CODE, native_outputs_scored=True, **scored,
                       note='Greedy batch-1 reads of the anchor; run 1 is the guard_dev enrollment of record, run 2 the '
                            'churn replicate (s0_churn).')
        return {'rows': len(rows), 'summary': scored['summary'], 'status': scored['guard_dev_enrollment_status'],
                'sha256': comp.write_stage_receipt(cfg, receipt, args.repo, CODE_FILES, _IMPORT_CODE)}

    comp.run_read(args, cfg, host, state, prompts, bind_extra={'run': args.run},
                  decode_one=_enroll_row(tokenizer, cfg['max_new_tokens']), lease_module=lease_module, finish=finish)
    if not state['complete']:
        return comp.incomplete_exit(cfg, state)
    done = state['finished']
    print(json.dumps({'stage': cfg['stage'], 'test_mode': cfg['test_mode'], 'rows': done['rows'],
                      'guard_dev_source_correct': done['summary']['guard_dev']['source_correct'],
                      'calibration_strict_correct': done['summary']['calibration']['strict_correct'],
                      'enrollment': done['status'], 'receipt': str(cfg['out']), 'sha256': done['sha256']}),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
