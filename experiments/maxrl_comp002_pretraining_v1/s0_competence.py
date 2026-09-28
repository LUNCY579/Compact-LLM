"""S0 source competence, stage S0-C (COMP-RS-001 Amendment 1, section 16; sections 1, 8.1, 8.5 and 12).

python -m experiments.maxrl_comp002_pretraining_v1.s0_competence --runtime-freeze RECORDS\\S0_RUNTIME_FREEZE.json \
    [--out RECORDS\\S0_COMPETENCE.json] [--lease-timeout-s 900]

The source (arm B: Supa streamed into FP32) samples G = 8 completions for every CALIBRATION core R, S and P row
(150 each). Every attempt is kept, with no refill. Nothing here trains, saves weights, or opens DEV, TEST,
CHALLENGE or guard_test.

- Rows. canonical/calibration.jsonl comes through scoped_scorer.load_role (manifest sha, file sha, row count).
  For each of R, S and P, the core rows' sorted ids and parents must reproduce the ids_sha256, parents_sha256,
  rows and parents of evidence/S0PRE_CALIBRATION_SCOPE.json (file sha pinned). The whole R/S/P core scope must
  also bind 'full' under scoped_scorer.bind_scope_receipt. The plan order is R, S, P, each sorted by id
  (calibration_plan). Every planned row's policy messages (policy/calibration.jsonl, manifest-checked) must equal
  its canonical prompt, and contracts.digest(prompt) must equal its prompt_hash. Prompts are rendered with
  s0_runtime.render_prompt (thinking disabled).
- Sampling. s0_generate.sample_group: one unpadded batch of G identical prompts under the Q2 sampler (T 1.0,
  top-p 1.0, top-k 0, repetition penalty 1.0), max_new_tokens 128, EOS (151645, 151643), PAD 151643. Seed per
  prompt: int(SHA256('COMP-RS-001-S0-COMP|' + id).hexdigest()[:16], 16) mod 2^63 (comp_seed).
- Response rule (RESPONSE_RULE, shared with S0-E). The response is the strict body: the completion minus its
  terminal EOS (by position), decoded with skip_special_tokens=False. A special token inside the body stays as
  its text. An ID the tokenizer cannot render becomes a visible placeholder. So malformed extra content is never
  stripped into a success (QUALIFICATION.md). The pilot's lenient text (skip_special_tokens=True) is recorded
  only where it differs, together with its strict reward (lenient_reward). It is diagnostic only.
- Reward. scoped_scorer.score_one, the scorer's per-row strict path: strict = the unchanged corpus.reward, plus
  bounded content, the normalization, format_only_failure and the failure_taxonomy subtype.
- Journal (section 12 reads). One line per prompt group is appended to <out>.jsonl through
  s0_generate.ResumableJsonl (row hash, fsync). Every line carries the bind fields: arm and weights digests,
  decoding_sha256, the generation contract, the plan digest, the tokenizer receipt, the code bundle and the
  group size. A launch resumes at the first missing row and refuses a journal that is not a prefix of the plan
  or that has a different bind. <out>.attempts.jsonl logs a start event and an end event per launch (outcome
  complete, aborted or error), so an interruption stays documented. hostbudget.guard runs before every prompt.
- Final scoring (section 8.5). Once the journal is complete, each sample index k = 0..G-1 is one batch of the
  plan's rows, scored through canary.score_with_canary. That runs the canary on CALIBRATION plus guard_dev,
  bound to the frozen scope receipt and the expected scorer code bundle, which is the runtime freeze's
  scoring_code_bundle_sha256. Every per-completion strict/content/subtype record must equal the journal's.
- Summary per skill: pass@1 (mean over the G samples), pass@G (the share of prompts with at least one success,
  maxrl.pass_at_k with n = k = G), all-fail/mixed/all-success counts, the success histogram, UNKNOWN answers
  (content-contract candidate 'UNKNOWN'), format diagnostics (format-only failures, normalizations, subtypes),
  cutoffs (stop_reason 'length', LENGTH_CAP) and token costs (generated, decode steps, prompt and prefill
  tokens, wall).
- Rule (no_signal_rule, frozen in section 16). m_min = 0.10. STOP_NO_SIGNAL if R's or S's mixed-group fraction
  is below 0.10 (fewer than 15 of 150). P's fraction is recorded, not gated (D2). Delta_acq = 13 rows is
  recorded for the section 3 H1 gate. The section 12 S1 order is also recorded: the lower mixed fraction trains
  first, and a tie goes to R.

Registered mode takes no size or path overrides: cuda, G 8, the full 450-row plan, the pinned model, asset and
tokenizer receipts, the clean checkout holding this code, the pinned lease and host budget, and the runtime
freeze. The freeze's decoding preimage must reproduce the live decoding_sha256, its scorer bundle must equal the
live one, and its host floor is the launch floor. --test-mode (recorded everywhere; CPU only, refused with --device
cuda) admits a tiny model, a smaller G, a --host-floor-gib, a missing freeze (the live scorer bundle is then used)
and injected lease/host modules.
Exit status: 0 for SIGNAL, 2 for STOP_NO_SIGNAL, 3 for an incomplete read (resumable).
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

from experiments.maxrl_composable_skills_v2 import contracts, corpus
from experiments.maxrl_composable_skills_v2.contracts import strict_json
from experiments.maxrl_verifiable_data_v1 import maxrl

from . import canary, content_scoring, preflight, scoped_scorer
from . import s0_runtime as s0
from .preflight import MANIFEST, digest, require

SCHEMA = 'COMP-RS-001-S0-COMPETENCE-2'
STAGE = 'S0_COMPETENCE'
LABEL = 'comp-rs-001-s0-competence'
REGISTRATION = 'COMP_RS_001_AMENDMENT_1.md section 16, source competence (S0-C)'
REPO = s0.HERE.parents[1]
PACKAGE = 'experiments/maxrl_comp002_pretraining_v1/'
SKILLS = ('R', 'S', 'P')
GATED = ('R', 'S')                       # section 16: m_min gates R and S; P is recorded, not gated (D2)
M_MIN = Fraction(1, 10)
DELTA_ACQ_ROWS = 13                      # section 16: Delta_acq for the section 3 H1 gate, fixed before S0-C
SEED_SALT = 'COMP-RS-001-S0-COMP|'
SEED_LIMIT = 2 ** 63
SEED_RULE = "int(sha256('COMP-RS-001-S0-COMP|' + id).hexdigest()[:16], 16) mod 2**63"
SAMPLE_SEED_RULE = ("C13: the G samples of a prompt are G sequential single-sample calls; sample k is seeded with "
                    "int(sha256('COMP-RS-001-S0-COMP|' + id + '|' + k).hexdigest()[:16], 16) mod 2**63")
PLAN_RULE = 'CALIBRATION core rows of R, then S, then P; within a skill sorted by id'
REGISTERED = {'device': 'cuda:0', 'group_size': 8, 'max_new_tokens': s0.MAX_NEW_TOKENS, 'rows_per_skill': 150}
CALIBRATION_SCOPE = s0.HERE / 'evidence' / 'S0PRE_CALIBRATION_SCOPE.json'
CALIBRATION_SCOPE_SHA = '1acd416dea7bcf61cd439fa54665975ffc34c047acc0dfde2a37c1fac370140f'
CANARY_SIGNATURES = ('R', 'S', 'P')      # CALIBRATION has no RS (the scope receipt's absent_signatures)
GUARD_GIB = 1.0                          # hostbudget.guard floor before every unit (the S0-Q1 guard)
MAX_RECEIPT = 1 << 26
HEX = frozenset('0123456789abcdef')
FREEZE_STAGE = 'S0_RUNTIME_FREEZE'
FREEZE_FIELDS = ('decoding_preimage', 'decoding_sha256', 'host_floor_gib', 'scoring_code_bundle_sha256',
                 'inventory')
UNRENDERABLE = '<|unrenderable:%d|>'
RESPONSE_RULE = (
    "Strict body. For stop_reason 'eos' the completion must hold exactly one EOS (151645 or 151643), as its last "
    "token, and the body is the completion without it; for 'length' the completion holds no EOS and the body is "
    'the whole completion. response = tokenizer.decode(body, skip_special_tokens=False, '
    'clean_up_tokenization_spaces=False): a special token inside the body keeps its text (e.g. <think>), so '
    "corpus.reward's strict_json fails on it. An ID the tokenizer cannot render (>= len(tokenizer)), which decode "
    "would drop silently, is rendered as '<|unrenderable:ID|>'. The body's runs of renderable IDs are decoded "
    'separately. Malformed extra content is therefore never stripped into a success. lenient_response = '
    'decode(completion, skip_special_tokens=True), the lenient rule superseded by C11, is recorded only where it differs from '
    'response; in S0-C, lenient_reward = corpus.reward(row, lenient_response) is recorded with it. Both are '
    'diagnostic only.')
TOKENIZER_SHA_RULE = ('tokenizer_sha256 = sha256 of the bytes of evidence/Q1_TOKENIZER_PROMPT_LENGTHS.json '
                      '(s0_runtime.TOKENIZER_RECEIPT_SHA). That receipt binds the chat-template sha, the tokenizer '
                      'class, max length, the special and EOS/PAD IDs, and asset_receipt_sha256, which binds the six '
                      'tokenizer asset file hashes. s0_runtime.load_tokenizer(asset_receipt=...) checks all of them.')
PASS_AT_K_RULE = ('pass@1 = successes / (prompts x G), the mean over all samples; pass@G = mean over prompts of '
                  'maxrl.pass_at_k(G, c, G), i.e. the share of prompts with at least one success among their own '
                  'G samples (n = k = G, not an extrapolation to other budgets)')
COMMON_TEST_OPTIONS = ('model_dir', 'asset_receipt', 'host_floor_gib')


# ------------------------------------------------------------------------------------------------ code
def code_files(*own: str) -> dict:
    """Repo-relative name -> path of every source file on the read and scoring path (own: the stage module)."""
    names = {PACKAGE + name for name in ('s0_competence.py', 's0_runtime.py', 's0_generate.py', 'mechanics.py',
                                         'canary.py', 'preflight.py')}
    names |= set(own) | set(scoped_scorer.CODE_FILES) | {'experiments/maxrl_verifiable_data_v1/maxrl.py',
                                                         'experiments/maxrl_verifiable_data_v1/specs.py'}
    return {name: REPO / name for name in sorted(names)}


def code_sha256(files: dict) -> dict:
    return {name: preflight.sha(path) for name, path in files.items()}


CODE_FILES = code_files()
_IMPORT_CODE = code_sha256(CODE_FILES)
for _module in (s0, canary, scoped_scorer, maxrl):
    require(Path(_module.__file__).resolve() in {p.resolve() for p in CODE_FILES.values()},
            'module not loaded from its hashed file: ' + _module.__name__)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex64(value) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - HEX


def _same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def json_file(path, what: str) -> tuple:
    """(strict JSON object, sha256 of the bytes) of a small receipt file."""
    raw = Path(path).read_bytes()
    require(len(raw) <= MAX_RECEIPT, what + ' too large')
    value = strict_json(raw.decode('utf-8'), MAX_RECEIPT)
    require(type(value) is dict, what + ' is not a JSON object')
    return value, _sha256(raw)


def generate_module():
    """s0_generate, imported on first use (it imports torch); it must be the file hashed at import."""
    from . import s0_generate
    require(Path(s0_generate.__file__).resolve() == (REPO / (PACKAGE + 's0_generate.py')).resolve(),
            's0_generate is not the hashed file')
    return s0_generate


def process_identity(session: str) -> dict:
    """This launch (a fresh uuid per main() call) and its OS process (pid + creation time)."""
    import psutil
    require(type(session) is str and len(session) == 32 and not set(session) - HEX, 'session id')
    return {'session': session, 'pid': os.getpid(), 'process_created': psutil.Process().create_time()}


# ------------------------------------------------------------------------------------------ plan
def sample_seed(row_id: str, k: int) -> int:
    """C13: the seed of sample k of a competence prompt (one single-sample call per sample)."""
    require(type(row_id) is str and row_id, 'row id')
    require(type(k) is int and k >= 0, 'sample index')
    return int(hashlib.sha256((SEED_SALT + row_id + '|' + str(k)).encode('utf-8')).hexdigest()[:16], 16) % SEED_LIMIT


def comp_seed(row_id: str) -> int:
    require(type(row_id) is str and row_id, 'row id')
    return int(hashlib.sha256((SEED_SALT + row_id).encode('utf-8')).hexdigest()[:16], 16) % SEED_LIMIT


def _core_rows(bank, skill: str) -> list:
    return sorted((row for row in bank if row['kind'] == 'composition' and row['tag'] == 'core'
                   and row['skills'] == [skill]), key=lambda row: row['id'])


def load_calibration(data, scope_path=CALIBRATION_SCOPE, expected_file_sha256=CALIBRATION_SCOPE_SHA) -> dict:
    """The CALIBRATION bank and its core R/S/P rows, each skill asserted against the frozen scope receipt."""
    bank = scoped_scorer.load_role(data, 'calibration')
    receipt = scoped_scorer.load_scope_receipt(scope_path, expected_file_sha256=expected_file_sha256)
    require(receipt.get('stage') == 'S0PRE_CALIBRATION_SCOPE' and receipt.get('role') == 'calibration',
            'not the CALIBRATION scope receipt')
    require(receipt.get('dataset_manifest_sha256') == MANIFEST, 'scope receipt binds another dataset manifest')
    require(receipt.get('canonical_calibration_sha256') == bank.bank_sha256,
            'scope receipt binds another CALIBRATION bank')
    signatures = receipt.get('signatures')
    require(type(signatures) is dict, 'scope receipt has no signatures')
    rows_by_skill, asserted = {}, {}
    for skill in SKILLS:
        rows = _core_rows(bank, skill)
        entry = signatures.get(skill)
        require(type(entry) is dict, 'scope receipt lacks skill ' + skill)
        ids = [row['id'] for row in rows]
        parents = sorted({row['parent_id'] for row in rows})
        require(digest(ids) == entry.get('ids_sha256'), 'core ' + skill + ' ids differ from the scope receipt')
        require(digest(parents) == entry.get('parents_sha256'),
                'core ' + skill + ' parents differ from the scope receipt')
        require(len(ids) == entry.get('rows') == REGISTERED['rows_per_skill'] and len(parents) == entry.get('parents'),
                'core ' + skill + ' row/parent counts differ from the scope receipt')
        rows_by_skill[skill] = rows
        asserted[skill] = {key: entry[key] for key in ('ids_sha256', 'parents_sha256', 'rows', 'parents')}
    scope = scoped_scorer.declare_scope(bank, signatures=list(SKILLS), tag='core')
    binding = scoped_scorer.bind_scope_receipt(bank, scope, receipt)
    require(all(binding['signatures'][skill]['binding'] == 'full' for skill in SKILLS),
            'the core R/S/P scope does not bind full')
    return {'bank': bank, 'receipt': receipt, 'rows_by_skill': rows_by_skill,
            'record': {'scope_receipt': str(Path(scope_path).resolve()),
                       'scope_receipt_file_sha256': preflight.sha(Path(scope_path)),
                       'scope_receipt_digest': digest(receipt), 'canonical_calibration_sha256': bank.bank_sha256,
                       'dataset_manifest_sha256': MANIFEST, 'asserted': asserted, 'binding': binding}}


def calibration_plan(rows_by_skill: dict) -> list:
    """The registered read order: R, S, P, each sorted by id (tests replace this with a subset)."""
    return [row for skill in SKILLS for row in rows_by_skill[skill]]


def is_full_plan(rows: list, rows_by_skill: dict) -> bool:
    """True iff the plan is exactly the 450 core R/S/P rows in the registered order."""
    expected = [row['id'] for skill in SKILLS for row in rows_by_skill[skill]]
    return [row['id'] for row in rows] == expected and len(expected) == len(SKILLS) * REGISTERED['rows_per_skill']


def check_policy(data, role: str, rows: list) -> dict:
    """Stream policy/<role>.jsonl (manifest-checked) and require each planned row's messages == its prompt."""
    root = Path(data).resolve(strict=True)
    manifest_raw = (root / 'manifest.json').read_bytes()
    require(_sha256(manifest_raw) == MANIFEST, 'dataset manifest drift')
    relative = 'policy/' + role + '.jsonl'
    entry = json.loads(manifest_raw.decode('utf-8'))['files'].get(relative)
    require(type(entry) is dict, 'data file not in the manifest: ' + relative)
    wanted = {row['id']: row for row in rows}
    require(len(wanted) == len(rows), 'duplicate planned id')
    hasher, size, seen = hashlib.sha256(), 0, set()
    with open(root / relative, 'rb') as stream:
        for line in stream:
            hasher.update(line)
            size += len(line)
            policy = strict_json(line.decode('utf-8'), scoped_scorer.MAX_LINE)
            row = wanted.get(policy.get('id')) if type(policy) is dict else None
            if row is None:
                continue
            require(policy['id'] not in seen, 'duplicate policy id: ' + policy['id'])
            seen.add(policy['id'])
            require(set(policy) == {'id', 'messages'} and policy['messages'] == row['prompt'],
                    'policy messages differ from the canonical prompt: ' + row['id'])
    require(size == entry['bytes'] and hasher.hexdigest() == entry['sha256'], relative + ' drift')
    require(seen == set(wanted), 'planned ids missing from ' + relative)
    for row in rows:
        require(contracts.digest(row['prompt']) == row['prompt_hash'], 'prompt_hash mismatch: ' + row['id'])
    return {'file': relative, 'sha256': entry['sha256'], 'rows_checked': len(rows)}


def render_prompts(tokenizer, rows: list, bank: str) -> list:
    """Planned prompts: rendered token IDs plus the digests every journal line carries."""
    prompts = []
    for index, row in enumerate(rows):
        text, ids = s0.render_prompt(tokenizer, row['prompt'])
        prompts.append({'index': index, 'id': row['id'], 'bank': bank, 'row': row, 'prompt_ids': ids,
                        'prompt_tokens': len(ids), 'prompt_ids_sha256': digest(ids),
                        'render_sha256': _sha256(text.encode('utf-8'))})
    return prompts


def canary_sections(data, calibration_bank, guard_bank=None) -> list:
    """The canary's sections (section 8.5): CALIBRATION plus its paired guard_dev, each with its oracles."""
    guard = scoped_scorer.load_role(data, 'guard_dev') if guard_bank is None else guard_bank
    require(calibration_bank.role == 'calibration' and guard.role == 'guard_dev', 'canary section roles')
    sections = []
    for bank in (calibration_bank, guard):
        oracles, oracle_sha = canary.load_oracles(data, bank.role)
        sections.append((bank, oracles, oracle_sha))
    return sections


# ------------------------------------------------------------------------------------------ responses
def special_ids(tokenizer) -> frozenset:
    """Every ID decode(skip_special_tokens=True) removes: the special tokens and the special added tokens."""
    marked = {i for i, token in tokenizer.added_tokens_decoder.items() if token.special}
    return frozenset(set(tokenizer.all_special_ids) | marked)


def _decode(tokenizer, ids: list) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) if ids else ''


def response_text(tokenizer, completion_ids, stop_reason) -> dict:
    """The scorer's response string under RESPONSE_RULE, plus the body diagnostics."""
    ids = list(completion_ids)
    require(ids and all(type(t) is int and t >= 0 for t in ids), 'completion IDs')
    eos = [j for j, t in enumerate(ids) if t in s0.EOS_IDS]
    if stop_reason == 'eos':
        require(eos == [len(ids) - 1], "stop_reason 'eos' needs exactly one EOS, the last token")
        body = ids[:-1]
    else:
        require(stop_reason == 'length' and not eos, "stop_reason 'length' must carry no EOS")
        body = ids
    size = len(tokenizer)
    unknown = [t for t in body if t >= size]
    if unknown:
        parts, run = [], []
        for t in body:
            if t >= size:
                parts.append(_decode(tokenizer, run))
                parts.append(UNRENDERABLE % t)
                run = []
            else:
                run.append(t)
        parts.append(_decode(tokenizer, run))
        response = ''.join(parts)
    else:
        response = _decode(tokenizer, body)
    specials = special_ids(tokenizer)
    lenient = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {'response': response, 'body_special_ids': [t for t in body if t in specials],
            'body_unknown_ids': unknown, 'lenient_response': None if lenient == response else lenient}


def unknown_answer(response: str, family=None) -> bool:
    """True iff the content-contract candidate is the string 'UNKNOWN' (any normalization)."""
    try:
        answer, _, multiple = content_scoring.parse_candidate(response, family)
    except content_scoring.PARSE_ERRORS:
        return False
    return not multiple and type(answer) is str and answer == 'UNKNOWN'


def score_completion(row: dict, completion: dict, text: dict) -> dict:
    """The scorer's per-row strict path (scoped_scorer.score_one) on one completion."""
    item = {'id': row['id'], 'prompt_hash': row['prompt_hash'], 'response': text['response'],
            'generated_tokens': completion['generated_tokens'], 'stop_reason': completion['stop_reason']}
    scored = scoped_scorer.score_one(row, item)
    lenient = text['lenient_response']
    return {'reward': scored['strict'], 'content_pass': scored['content_pass'],
            'normalization': scored['normalization'], 'format_only_failure': scored['format_only_failure'],
            'subtype': scored['subtype'], 'unknown_answer': unknown_answer(text['response']),
            'lenient_reward': None if lenient is None else int(corpus.reward(row, lenient))}


# ------------------------------------------------------------------------------------------ freeze
def live_bundle() -> str:
    return digest(scoped_scorer.code_sha256())


def read_freeze(path, *, registered: bool) -> dict:
    """The S0-F runtime freeze (FREEZE_FIELDS): the preimage must reproduce its decoding_sha256."""
    value, file_sha = json_file(path, 'runtime freeze')
    require(value.get('stage') == FREEZE_STAGE, 'not an ' + FREEZE_STAGE + ' receipt')
    missing = [key for key in FREEZE_FIELDS if key not in value]
    require(not missing, 'runtime freeze lacks: ' + ', '.join(missing))
    preimage = value['decoding_preimage']
    require(type(preimage) is dict, 'runtime freeze decoding_preimage is not an object')
    decoding = s0.decoding_sha256(preimage)
    require(value['decoding_sha256'] == decoding, 'runtime freeze decoding_sha256 is not the digest of its preimage')
    floor = value['host_floor_gib']
    require(type(floor) in (int, float) and math.isfinite(floor) and floor > 0, 'runtime freeze host floor')
    require(_hex64(value['scoring_code_bundle_sha256']), 'runtime freeze scorer code bundle')
    if registered:
        require(value.get('test_mode', False) is False and value.get('registered_run', True) is True,
                'the runtime freeze is not a registered run')
        require(preimage.get('device') == REGISTERED['device'], 'the registered runtime freeze is not for cuda:0')
    return {'path': str(Path(path).resolve()), 'sha256': file_sha, 'decoding_sha256': decoding,
            'host_floor_gib': float(floor), 'scoring_code_bundle_sha256': value['scoring_code_bundle_sha256'],
            'inventory': value['inventory']}


# ------------------------------------------------------------------------------------------ rules
def group_outcome(rewards) -> str:
    require(type(rewards) is list and rewards and all(type(r) is int and r in (0, 1) for r in rewards),
            'binary rewards')
    return 'all_fail' if not any(rewards) else 'all_success' if all(rewards) else 'mixed'


def no_signal_rule(counts: dict) -> dict:
    """Section 16: STOP_NO_SIGNAL iff R's or S's mixed-group fraction < m_min = 0.10; P is recorded only."""
    require(type(counts) is dict and all(skill in counts for skill in GATED), 'mixed counts for R and S')
    fractions, below, table = {}, [], {}
    for skill in sorted(counts, key=lambda s: SKILLS.index(s) if s in SKILLS else len(SKILLS)):
        require(skill in SKILLS, 'unknown skill: ' + repr(skill))
        entry = counts[skill]
        mixed, prompts = entry.get('mixed'), entry.get('prompts')
        require(type(mixed) is int and type(prompts) is int and prompts > 0 and 0 <= mixed <= prompts,
                'mixed/prompt counts for ' + skill)
        fraction = Fraction(mixed, prompts)
        fractions[skill] = fraction
        table[skill] = {'mixed': mixed, 'prompts': prompts, 'mixed_fraction': float(fraction),
                        'gated': skill in GATED, 'below_m_min': fraction < M_MIN}
        if skill in GATED and fraction < M_MIN:
            below.append(skill)
    s1_first = 'R' if fractions['R'] <= fractions['S'] else 'S'
    return {'rule': 'STOP_NO_SIGNAL iff the mixed-group fraction of R or S is < m_min (P recorded, not gated)',
            'm_min': float(M_MIN), 'gated': list(GATED), 'recorded_only': [s for s in SKILLS if s not in GATED],
            'skills': table, 'below_m_min': below, 'outcome': 'STOP_NO_SIGNAL' if below else 'SIGNAL',
            'delta_acq_rows': DELTA_ACQ_ROWS,
            's1_first_skill': s1_first,
            's1_order_rule': 'section 12: the skill with the lower mixed-group fraction trains first (tie: R)'}


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def summarise(rows: list, group_size: int) -> dict:
    """Per-skill competence summary over complete journal rows (section 16 S0-C 'Reported')."""
    require(type(group_size) is int and group_size >= 2, 'group size')
    out = {}
    for skill in SKILLS:
        mine = [row for row in rows if row['skill'] == skill]
        if not mine:
            continue
        completions = [c for row in mine for c in row['completions']]
        require(all(len(row['completions']) == group_size == len(row['rewards']) for row in mine),
                'incomplete group in ' + skill)
        successes = [sum(row['rewards']) for row in mine]
        outcomes = Counter(group_outcome(row['rewards']) for row in mine)
        pass1 = Fraction(sum(successes), len(completions))
        passk = [maxrl.pass_at_k(group_size, c, 1) for c in successes]
        require(abs(_mean(passk) - float(pass1)) < 1e-12, 'pass@1 estimators disagree')
        generated = [c['generated_tokens'] for c in completions]
        out[skill] = {
            'prompts': len(mine), 'samples': len(completions), 'successes': sum(successes),
            'pass_at_1': float(pass1), 'pass_at_G': _mean(maxrl.pass_at_k(group_size, c, group_size)
                                                          for c in successes),
            'G': group_size,
            'groups': {key: outcomes.get(key, 0) for key in ('all_fail', 'mixed', 'all_success')},
            'success_histogram': {str(c): successes.count(c) for c in range(group_size + 1)},
            'diagnostics': {
                'unknown_answer': sum(c['unknown_answer'] for c in completions),
                'content_pass': sum(c['content_pass'] for c in completions),
                'format_only_failures': sum(c['format_only_failure'] for c in completions),
                'normalization': dict(sorted(Counter(c['normalization'] for c in completions).items())),
                'subtype': dict(sorted(Counter(c['subtype'] for c in completions if c['subtype']).items())),
                'stop_reason': {key: sum(c['stop_reason'] == key for c in completions) for key in ('eos', 'length')},
                'cutoff_length_cap': sum(c['subtype'] == 'LENGTH_CAP' for c in completions),
                'body_special': sum(bool(c['body_special_ids']) for c in completions),
                'body_unknown': sum(bool(c['body_unknown_ids']) for c in completions),
                'lenient_text_differs': sum(c['lenient_response'] is not None for c in completions),
                'lenient_reward_differs': sum(c['lenient_reward'] is not None and c['lenient_reward'] != c['reward']
                                              for c in completions)},
            'tokens': {'generated': sum(generated), 'generated_mean': _mean(generated),
                       'generated_max': max(generated), 'decode_steps': sum(row['decode_steps'] for row in mine),
                       'prompt_tokens': sum(row['prompt_tokens'] for row in mine),
                       'prefill_tokens': sum(row['prompt_tokens'] * group_size for row in mine)},
            'wall_s': math.fsum(row['wall_s'] for row in mine)}
    return out


# ------------------------------------------------------------------------------------------ stage machinery
def add_common_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--repo', type=Path, default=REPO)
    ap.add_argument('--data', type=Path, default=s0.DATA)
    ap.add_argument('--runtime-freeze', type=Path, help='S0_RUNTIME_FREEZE.json (required when registered)')
    ap.add_argument('--lease-timeout-s', type=float, default=900.0)
    ap.add_argument('--test-mode', action='store_true')
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--asset-receipt', type=Path)
    ap.add_argument('--host-floor-gib', type=float)


def common_config(args, lease_module, host_module, *, stage: str, label: str, default_name: str,
                  test_options=COMMON_TEST_OPTIONS) -> dict:
    """Validate the command line (refusing before any heavy work) and return the effective settings."""
    if not args.test_mode:
        given = [name for name in test_options if getattr(args, name) is not None]
        if given:
            raise SystemExit('test-mode only option(s): ' + ', '.join('--' + g.replace('_', '-') for g in given))
        if args.device != 'cuda':
            raise SystemExit('the registered stage runs on cuda')
        if lease_module is not None or host_module is not None:
            raise SystemExit('injected lease/host modules are test-mode only')
        if args.runtime_freeze is None:
            raise SystemExit('the registered stage asserts the S0 runtime freeze (--runtime-freeze)')
        if not _same_path(args.repo, REPO):
            raise SystemExit('the registered stage binds the checkout that holds this code: ' + str(REPO))
        out = args.out if args.out is not None else s0.RECORDS / default_name
        if not _same_path(Path(out).resolve().parent, s0.RECORDS):
            raise SystemExit('registered records are written to ' + str(s0.RECORDS))
        model_dir, asset_receipt = s0.MODEL_DIR, s0.ASSET_RECEIPT
    else:
        if args.device != 'cpu':
            raise SystemExit('--test-mode is refused with --device cuda: a GPU stage is always a registered run')
        if args.out is None:
            raise SystemExit('--test-mode needs --out')
        if args.host_floor_gib is None and args.runtime_freeze is None:
            raise SystemExit('--test-mode needs --host-floor-gib or --runtime-freeze')
        out = args.out
        model_dir = s0.MODEL_DIR if args.model_dir is None else args.model_dir
        asset_receipt = s0.ASSET_RECEIPT if args.asset_receipt is None else args.asset_receipt
    if not (math.isfinite(args.lease_timeout_s) and args.lease_timeout_s > 0):
        raise SystemExit('lease timeout must be > 0')
    if args.host_floor_gib is not None and not (math.isfinite(args.host_floor_gib) and args.host_floor_gib > 0):
        raise SystemExit('--host-floor-gib must be > 0')
    target = s0.assert_outside_checkout(out, args.repo)
    if target.suffix != '.json':
        raise SystemExit('the stage receipt is a .json file: ' + str(target))
    if target.exists():
        raise SystemExit('receipt exists (create-only): ' + str(target))
    return {'out': target, 'journal': target.with_suffix('.jsonl'),
            'attempts': target.with_name(target.stem + '.attempts.jsonl'), 'model_dir': Path(model_dir),
            'asset_receipt': Path(asset_receipt), 'device': s0.DEVICES[args.device], 'stage': stage, 'label': label,
            'test_mode': bool(args.test_mode), 'max_new_tokens': s0.MAX_NEW_TOKENS}


def launch_snapshot(host) -> dict:
    """Free physical RAM (hostbudget.snapshot), this worker's RSS, and whether torch is imported yet."""
    import psutil
    snapshot = host.snapshot()
    require(type(snapshot) is dict, 'host snapshot')
    return {'free_phys_gib': snapshot.get('free_phys_gib'), 'host': snapshot,
            'process_rss_gib': psutil.Process().memory_info().rss / s0.GIB,
            'torch_imported': 'torch' in sys.modules, 'utc': _now()}


def prepare_launch(args, cfg: dict, host) -> dict:
    """Launch floor (hostbudget.require), runtime and checkout checks, receipt pins, the freeze and the bundle."""
    launch = launch_snapshot(host)
    freeze = read_freeze(args.runtime_freeze, registered=not args.test_mode) if args.runtime_freeze else None
    if args.host_floor_gib is not None:
        floor = {'floor_gib': float(args.host_floor_gib), 'source': '--host-floor-gib (test mode)'}
    else:
        floor = {'floor_gib': freeze['host_floor_gib'], 'source': FREEZE_STAGE + ' host_floor_gib',
                 'freeze_sha256': freeze['sha256']}
    launch['floor'] = floor
    launch['require'] = s0.host_require(floor['floor_gib'], cfg['label'], module=host)
    if cfg['test_mode']:
        runtime = {'python_executable': sys.executable, 'python': platform.python_version(),
                   'packages': s0.installed_versions(sorted(s0.PINS['packages'])), 'asserted': False}
        checkout = {'checked': False, 'reason': 'test mode'}
    else:
        runtime = dict(s0.assert_runtime(), asserted=True)
        checkout = dict(s0.assert_clean_checkout(args.repo), checked=True)
        require(preflight.sha(s0.ASSET_RECEIPT) == s0.ASSET_RECEIPT_SHA, 'asset receipt drift')
    require(preflight.sha(s0.TOKENIZER_RECEIPT) == s0.TOKENIZER_RECEIPT_SHA, 'tokenizer receipt drift')
    live = live_bundle()
    if freeze is not None:
        require(freeze['scoring_code_bundle_sha256'] == live, 'scorer code bundle differs from the runtime freeze')
        bundle = {'expected': freeze['scoring_code_bundle_sha256'], 'source': FREEZE_STAGE, 'live': live}
    else:
        bundle = {'expected': live, 'source': 'live scoped_scorer.code_sha256() (test mode, no runtime freeze)',
                  'live': live}
    return {'launch': launch, 'freeze': freeze, 'runtime': runtime, 'checkout': checkout, 'bundle': bundle}


def run_meta(identity: dict, decoding_sha256: str) -> dict:
    """behaviour.validate_meta's three run hashes (tokenizer: TOKENIZER_SHA_RULE)."""
    meta = {'model_sha256': identity['weights_sha256'], 'tokenizer_sha256': s0.TOKENIZER_RECEIPT_SHA,
            'decoding_sha256': decoding_sha256}
    require(all(_hex64(value) for value in meta.values()), 'run metadata hashes')
    return meta


def _sync(device) -> None:
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def timed(device, call):
    """(result, wall seconds) of call(), synchronised on CUDA so the wall covers the kernels."""
    _sync(device)
    start = time.perf_counter()
    result = call()
    _sync(device)
    return result, time.perf_counter() - start


def _release(device) -> None:
    gc.collect()
    if device is not None and device.type == 'cuda':
        import torch
        torch.cuda.empty_cache()


def run_read(args, cfg: dict, host, state: dict, prompts: list, *, bind_extra: dict, decode_one, lease_module,
             finish) -> None:
    """Load the source and decode every missing planned row into the journal; once the journal is complete,
    state['finished'] = finish() (scoring and the receipt). All of it runs under the GPU lease (section 16: every
    GPU stage holds the lease for its whole run)."""
    def body():
        _read(cfg, host, state, prompts, bind_extra, decode_one)
        if state['complete']:
            state['finished'] = finish()
    use_lease = cfg['device'] != 'cpu' or lease_module is not None
    if use_lease:
        with s0.lease(cfg['label'], args.lease_timeout_s, module=lease_module) as token:
            state['lease'] = {'label': cfg['label'], 'held': bool(token),
                              'module': 'injected' if lease_module is not None else 'pinned gpu_lease.py'}
            body()
    else:
        state['lease'] = {'label': cfg['label'], 'held': False, 'module': None, 'reason': 'cpu, no GPU used'}
        body()


def _read(cfg: dict, host, state: dict, prompts: list, bind_extra: dict, decode_one) -> None:
    g = generate_module()
    state['settings'] = s0.configure(cfg['device'])
    preimage = s0.decoding_preimage(cfg['device'])
    decoding = s0.decoding_sha256(preimage)
    state['decoding_preimage'], state['decoding_sha256'] = preimage, decoding
    freeze = state['freeze']
    if freeze is not None:
        require(freeze['decoding_sha256'] == decoding, 'decoding_sha256 differs from the runtime freeze')
    contract = g.generation_contract(cfg['max_new_tokens'])
    state['generation_contract'] = contract
    model, identity = s0.load_model_fp32(cfg['model_dir'], cfg['device'], cfg['asset_receipt'])
    device = next(model.parameters()).device
    try:
        require(identity['attn_implementation'] == preimage['attn_implementation'], 'attention implementation drift')
        state['identity'] = {k: v for k, v in identity.items()
                             if k not in ('stored_tensor_sha256', 'parameter_inventory')}
        state['run_meta'] = run_meta(identity, decoding)
        planned = [item['id'] for item in prompts]
        bind = dict({'stage': cfg['stage'], 'test_mode': cfg['test_mode'], 'arm_sha256': identity['arm_sha256'],
                     'weights_sha256': identity['weights_sha256'], 'decoding_sha256': decoding,
                     'generation_contract_sha256': contract['contract_sha256'], 'plan_sha256': digest(planned),
                     'tokenizer_sha256': s0.TOKENIZER_RECEIPT_SHA, 'code_sha256': digest(cfg['code']),
                     'max_new_tokens': cfg['max_new_tokens']}, **bind_extra)
        state['bind'] = bind
        journal = g.ResumableJsonl(cfg['journal'], key='id', bind=bind)
        attempts = g.ResumableJsonl(cfg['attempts'], key='event', bind=bind)
        start = journal.next_index(planned)
        process = process_identity(state['session'])
        state['resume_index'] = start
        state['resources'] = {'after_load': s0.resources(cfg['device'], module=host)}
        attempts.append({'event': state['session'] + ':start', 'kind': 'start', 'process': process, 'utc': _now(),
                         'resume_index': start, 'planned': len(planned), 'argv': state['argv'],
                         'recovered': journal.recovered})
        appended, outcome = 0, 'error'
        try:
            low_memory = host.LowMemoryAbort
            for item in prompts[start:]:
                try:
                    s0.host_guard(GUARD_GIB, cfg['label'] + ' ' + item['id'], module=host)
                except low_memory as error:
                    state['abort'] = {'type': type(error).__name__, 'message': str(error)[:2000],
                                      'before': item['id'], 'index': item['index']}
                    break
                row = decode_one(g, model, item, device)
                journal.append(dict(row, process=process, utc=_now()))
                appended += 1
            after = s0.state_fingerprint(model)
            require(after == identity['state_fingerprint'], 'the model state changed during the read')
            state['fingerprint_unchanged'] = True
            outcome = 'complete' if len(journal) == len(planned) else 'aborted'
        except BaseException as error:
            state['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            raise
        finally:  # the read's own error, if any, is the one that propagates
            state['appended'] = appended
            try:
                state['resources']['end'] = s0.resources(cfg['device'], module=host)
            except Exception as error:
                state['resources']['end'] = {'error': type(error).__name__ + ': ' + str(error)[:500]}
            try:
                attempts.append({'event': state['session'] + ':end', 'kind': 'end', 'process': process,
                                 'utc': _now(), 'outcome': outcome, 'rows_done': len(journal), 'appended': appended,
                                 'abort': state.get('abort'), 'error': state.get('error')})
            except Exception as error:
                state['attempts_error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
        state['complete'] = outcome == 'complete'
        state['journal_rows'] = len(journal)
    finally:
        del model
        _release(device)


def journal_record(cfg: dict, state: dict) -> dict:
    path = cfg['journal']
    return {'path': str(path), 'sha256': preflight.sha(path), 'bytes': path.stat().st_size,
            'rows': state['journal_rows'], 'bind': state['bind'], 'bind_sha256': digest(state['bind'])}


def attempts_record(cfg: dict, state: dict) -> dict:
    """The launches that built the journal: start/end events per session; interrupted = no 'complete' end."""
    g = generate_module()
    rows = g.ResumableJsonl(cfg['attempts'], key='event', bind=state['bind']).rows()
    sessions = {}
    for row in rows:
        entry = sessions.setdefault(row['process']['session'], {'process': row['process'], 'start': None,
                                                                'end': None})
        entry[row['kind']] = {k: row[k] for k in row if k not in ('process', 'bind', 'row_sha256')
                              and k not in state['bind']}
    interrupted = sorted(s for s, entry in sessions.items()
                         if entry['end'] is None or entry['end'].get('outcome') != 'complete')
    return {'path': str(cfg['attempts']), 'sha256': preflight.sha(cfg['attempts']), 'events': len(rows),
            'sessions': sessions, 'launches': len(sessions), 'interrupted_sessions': interrupted,
            'resumed': len(sessions) > 1}


def decoder_key(record: dict, prompt_tokens: int, max_new_tokens: int) -> dict:
    """A decoder/sampler record without its one prompt-dependent field, effective max_length (checked here)."""
    config = record['effective_config']
    require(config['max_length'] == prompt_tokens + max_new_tokens and config['max_new_tokens'] == max_new_tokens,
            'effective max_length is not prompt tokens + max_new_tokens')
    return dict(record, effective_config={k: v for k, v in config.items() if k != 'max_length'})


def distinct_decoders(rows: list, key: str, max_new_tokens: int) -> dict:
    """The decoder/sampler records the rows carry (max_length aside), by digest; a read holds exactly one."""
    keys = [decoder_key(row[key], row['prompt_tokens'], max_new_tokens) for row in rows]
    seen = Counter(digest(value) for value in keys)
    require(len(seen) == 1, 'the read used more than one decoder configuration')
    ((sha, count),) = seen.items()
    return {'sha256': sha, 'rows': count, 'record': keys[0],
            'rule': 'identical in every row except effective_config.max_length = prompt tokens + max_new_tokens'}


def write_stage_receipt(cfg: dict, receipt: dict, repo, files: dict, imported: dict) -> str:
    require(code_sha256(files) == imported, cfg['stage'] + ' source changed during the run')
    return s0.write_receipt(cfg['out'], receipt, repo=repo)


def base_receipt(args, cfg: dict, state: dict, *, schema: str, registration: str) -> dict:
    return {'experiment': 'COMP-RS-001', 'stage': cfg['stage'], 'schema': schema, 'registration': registration,
            'test_mode': cfg['test_mode'], 'registered_run': not cfg['test_mode'], 'argv': state['argv'],
            'started_utc': state['started_utc'], 'finished_utc': _now(),
            'wall_s': time.perf_counter() - state['t0'], 'session': state['session'],
            'runtime': state['runtime'], 'settings': state.get('settings'), 'checkout': state['checkout'],
            'lease': state.get('lease'), 'launch': state['launch'], 'runtime_freeze': state['freeze'],
            'scorer_bundle': state['bundle'], 'decoding_preimage': state.get('decoding_preimage'),
            'decoding_sha256': state.get('decoding_sha256'), 'generation_contract': state.get('generation_contract'),
            'identity': state.get('identity'), 'run_meta': state.get('run_meta'),
            'tokenizer_sha256_rule': TOKENIZER_SHA_RULE, 'response_rule': RESPONSE_RULE,
            'fingerprint_unchanged': state.get('fingerprint_unchanged'), 'resources': state.get('resources'),
            'journal': journal_record(cfg, state), 'attempts': attempts_record(cfg, state),
            'this_launch': {'resume_index': state.get('resume_index'), 'appended': state.get('appended')},
            'model_asset': {'model_dir': str(cfg['model_dir']), 'asset_receipt': str(cfg['asset_receipt']),
                            'asset_receipt_sha256': preflight.sha(cfg['asset_receipt'])},
            'weights_saved': False, 'training_run': False}


def new_state(argv, t0: float, prepared: dict) -> dict:
    return {'t0': t0, 'started_utc': _now(), 'session': uuid.uuid4().hex,
            'argv': [str(a) for a in (sys.argv[1:] if argv is None else argv)], 'abort': None, 'complete': False,
            **prepared}


def incomplete_exit(cfg: dict, state: dict) -> int:
    print(json.dumps({'stage': cfg['stage'], 'status': 'INCOMPLETE', 'rows_done': state.get('journal_rows'),
                      'abort': state.get('abort'), 'journal': str(cfg['journal']),
                      'resume': 'relaunch the same command; the read resumes at the first missing row'}), flush=True)
    return 3


# ------------------------------------------------------------------------------------------ S0-C
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_competence', description='COMP-RS-001 S0-C source competence.')
    add_common_arguments(ap)
    ap.add_argument('--group-size', type=int)
    return ap


def _competence_row(tokenizer, group_size: int, max_new_tokens: int):
    def decode_one(g, model, item, device) -> dict:
        seeds = [sample_seed(item['id'], k) for k in range(group_size)]

        def sequential():
            calls = []
            for seed in seeds:                          # C13: one sample per call (memory), i.i.d. samples
                one = g.sample_group(model, item['prompt_ids'], 1, seed, max_new_tokens=max_new_tokens)
                require(one['seed'] == seed and one['group_size'] == 1 and len(one['completions']) == 1,
                        'single-sample call shape')
                calls.append(one)
            return calls
        calls, wall = timed(device, sequential)
        sampler = {k: v for k, v in calls[0]['sampler'].items() if k != 'seed_path'}
        require(all({k: v for k, v in c['sampler'].items() if k != 'seed_path'} == sampler for c in calls),
                'the sampler record differs between the sequential sample calls')
        result = {'completions': [c['completions'][0] for c in calls],
                  'decode_steps': sum(c['decode_steps'] for c in calls), 'sampler': calls[0]['sampler']}
        completions = []
        for k, completion in enumerate(result['completions']):
            text = response_text(tokenizer, completion['completion_ids'], completion['stop_reason'])
            completions.append(dict({'k': k, 'completion_ids': completion['completion_ids'],
                                     'stop_reason': completion['stop_reason'],
                                     'generated_tokens': completion['generated_tokens'],
                                     'raw_step_logprobs': completion['raw_step_logprobs'],
                                     'response': text['response'], 'body_special_ids': text['body_special_ids'],
                                     'body_unknown_ids': text['body_unknown_ids'],
                                     'lenient_response': text['lenient_response']},
                                    **score_completion(item['row'], completion, text)))
        rewards = [c['reward'] for c in completions]
        row = item['row']
        return {'id': item['id'], 'index': item['index'], 'skill': row['skills'][0], 'parent_id': row['parent_id'],
                'prompt_hash': row['prompt_hash'], 'prompt_tokens': item['prompt_tokens'],
                'prompt_ids_sha256': item['prompt_ids_sha256'], 'render_sha256': item['render_sha256'],
                'seed': item['seed'], 'sample_seeds': seeds, 'sampling': 'C13 sequential single-sample calls',
                'group_size': group_size, 'decode_steps': result['decode_steps'],
                'completions': completions, 'rewards': rewards, 'outcome': group_outcome(rewards),
                'sampler': {k: v for k, v in result['sampler'].items() if k != 'seed_path'}, 'wall_s': wall}
    return decode_one


def score_competence(data, cal: dict, prompts: list, rows: list, meta: dict, bundle: str, group_size: int,
                     sections=None) -> dict:
    """Section 8.5: each sample index is one batch through canary.score_with_canary; it must equal the journal."""
    planned = [item['id'] for item in prompts]
    require([row['id'] for row in rows] == planned, 'journal rows differ from the plan')
    bank = cal['bank']
    sections = canary_sections(data, bank) if sections is None else sections
    scope = scoped_scorer.declare_scope(bank, signatures=list(SKILLS), ids=planned, tag='core')
    batches, first = [], None
    for k in range(group_size):
        responses = [{'id': row['id'], 'prompt_hash': row['prompt_hash'],
                      'response': row['completions'][k]['response'],
                      'generated_tokens': row['completions'][k]['generated_tokens'],
                      'stop_reason': row['completions'][k]['stop_reason']} for row in rows]
        result = canary.score_with_canary(sections, bank, scope, responses, meta,
                                          signatures=list(CANARY_SIGNATURES), expected_code_bundle=bundle,
                                          scope_receipts={'calibration': cal['receipt']})
        scoring = result['scoring']
        by_id = {record['id']: record for record in scoring['rows']}
        for row in rows:
            mine, theirs = row['completions'][k], by_id[row['id']]
            require(theirs['strict'] == mine['reward'] and theirs['content_pass'] == mine['content_pass']
                    and theirs['normalization'] == mine['normalization']
                    and theirs['format_only_failure'] == mine['format_only_failure']
                    and theirs['subtype'] == mine['subtype'],
                    'canary-bound scoring differs from the journal: %s sample %d' % (row['id'], k))
        batches.append({'k': k, 'canary_overall': result['canary']['overall'],
                        'canary_receipt_sha256': result['canary_receipt_sha256'],
                        'scoring_receipt_sha256': result['scoring_receipt_sha256'],
                        'code_bundle_sha256': result['code_bundle_sha256'],
                        'responses_sha256': scoring['responses_sha256'], 'scope_binding': scoring['scope_binding'],
                        'strict_correct': len(scoring['strict_correct_ids']), 'counts': scoring['counts']})
        if first is None:
            first = result['canary']
    return {'rule': 'one batch per sample index k through canary.score_with_canary (section 8.5); every '
                    'per-completion strict/content/normalization/subtype record equals the journal',
            'batches': batches, 'agrees_with_journal': True,
            'canary_receipts_identical': len({b['canary_receipt_sha256'] for b in batches}) == 1,
            'code_bundles': sorted({b['code_bundle_sha256'] for b in batches}), 'canary_receipt': first,
            'scope': {k: v for k, v in scope.items() if k != 'ids'}}


def _competence_config(args, lease_module, host_module) -> dict:
    options = COMMON_TEST_OPTIONS + ('group_size',)
    cfg = common_config(args, lease_module, host_module, stage=STAGE, label=LABEL,
                        default_name=STAGE + '.json', test_options=options)
    group_size = REGISTERED['group_size'] if args.group_size is None else args.group_size
    if type(group_size) is not int or group_size < 2:
        raise SystemExit('group size must be an int >= 2')
    cfg['group_size'] = group_size
    cfg['code'] = _IMPORT_CODE
    return cfg


def main(argv=None, *, lease_module=None, host_module=None) -> int:
    t0 = time.perf_counter()
    args = _parser().parse_args(argv)
    cfg = _competence_config(args, lease_module, host_module)
    host = host_module if host_module is not None else s0.hostbudget()
    state = new_state(argv, t0, prepare_launch(args, cfg, host))
    cal = load_calibration(args.data)
    rows = calibration_plan(cal['rows_by_skill'])
    require(rows and all(row['role'] == 'calibration' and row['tag'] == 'core' and row['kind'] == 'composition'
                         and row['skills'] in (['R'], ['S'], ['P']) for row in rows),
            'the plan holds only core R/S/P CALIBRATION rows')
    full = is_full_plan(rows, cal['rows_by_skill'])
    require(full or cfg['test_mode'], 'the registered plan is every core R, S and P CALIBRATION row in order')
    policy = check_policy(args.data, 'calibration', rows)
    tokenizer = s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)
    prompts = render_prompts(tokenizer, rows, 'calibration')
    for item in prompts:
        item['seed'] = comp_seed(item['id'])
    state['inputs'] = dict(cal['record'], policy=policy, plan_rule=PLAN_RULE, seed_rule=SEED_RULE, sample_seed_rule=SAMPLE_SEED_RULE,
                           planned=len(prompts), plan_ids_sha256=digest([p['id'] for p in prompts]),
                           plan_by_skill={s: sum(p['row']['skills'] == [s] for p in prompts) for s in SKILLS},
                           full_plan=full)
    def finish() -> dict:
        g = generate_module()
        journal_rows = g.ResumableJsonl(cfg['journal'], key='id', bind=state['bind']).rows()
        scoring = score_competence(args.data, cal, prompts, journal_rows, state['run_meta'],
                                   state['bundle']['expected'], cfg['group_size'])
        summary = summarise(journal_rows, cfg['group_size'])
        rule = no_signal_rule({skill: {'mixed': entry['groups']['mixed'], 'prompts': entry['prompts']}
                               for skill, entry in summary.items()})
        receipt = dict(base_receipt(args, cfg, state, schema=SCHEMA, registration=REGISTRATION),
                       inputs=state['inputs'], group_size=cfg['group_size'], max_new_tokens=cfg['max_new_tokens'],
                       sampler=distinct_decoders(journal_rows, 'sampler', cfg['max_new_tokens']), seed_rule=SEED_RULE,
                       pass_at_k_rule=PASS_AT_K_RULE, scoring=scoring, summary=summary, rule=rule,
                       outcome=rule['outcome'], m_min=float(M_MIN), delta_acq_rows=DELTA_ACQ_ROWS,
                       completed=True, code_sha256=_IMPORT_CODE, native_outputs_scored=True,
                       note='Source competence: every sampled attempt kept, no refill; rewards via the strict path.')
        return {'rule': rule, 'sha256': write_stage_receipt(cfg, receipt, args.repo, CODE_FILES, _IMPORT_CODE)}

    run_read(args, cfg, host, state, prompts, bind_extra={'group_size': cfg['group_size']},
             decode_one=_competence_row(tokenizer, cfg['group_size'], cfg['max_new_tokens']),
             lease_module=lease_module, finish=finish)
    if not state['complete']:
        return incomplete_exit(cfg, state)
    rule = state['finished']['rule']
    print(json.dumps({'stage': STAGE, 'test_mode': cfg['test_mode'], 'outcome': rule['outcome'],
                      'mixed_fraction': {s: rule['skills'][s]['mixed_fraction'] for s in rule['skills']},
                      'receipt': str(cfg['out']), 'sha256': state['finished']['sha256']}), flush=True)
    return 0 if rule['outcome'] == 'SIGNAL' else 2


if __name__ == '__main__':
    raise SystemExit(main())
