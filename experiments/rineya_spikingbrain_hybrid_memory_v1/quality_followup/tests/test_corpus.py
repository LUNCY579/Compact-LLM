"""CPU synthetic mechanics fixtures, never native Rineya capability evidence."""
import copy
import json
from pathlib import Path
import re
import tempfile

import pytest

from quality_followup import corpus
from quality_followup.audit import audit_packet, raw_oracle
from quality_followup.utils import Blocked, RECORDS_ROOT, digest, sha


class FixtureTokenizer:
    is_fast = True
    eos_token_id = 42

    def apply_chat_template(self, messages, **kwargs):
        return '<user>\n' + messages[0]['content'] + '\n<assistant>\n'

    def __call__(self, text, **kwargs):
        pieces = list(re.finditer(r'[A-Za-z_]+|\d|[^\w\s]', text))
        return {'input_ids': [int(digest(m.group())[:7], 16) for m in pieces],
                'offset_mapping': [m.span() for m in pieces]}

    def encode(self, text, **kwargs):
        return self(text)['input_ids']


@pytest.fixture(scope='module')
def packet():
    # Even fixture outputs stay outside tracked source and .codex.
    scratch = RECORDS_ROOT / '20261005-implementation-01a10d69' / 'fixture-corpus'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        tokenizer_source = root / 'fixture-tokenizer'
        tokenizer_source.mkdir()
        for name in corpus.TOKENIZER_FILES:
            (tokenizer_source / name).write_text('fixture only; no pretrained source', encoding='utf-8')
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(corpus, 'load_tokenizer', lambda _: FixtureTokenizer())
            output = root / 'packet'
            summary = corpus.build_packet(output, tokenizer_source)
        yield output, FixtureTokenizer(), summary


def _rewrite_packet(source, target, condition, mutate):
    target.mkdir()
    for file in source.iterdir():
        (target / file.name).write_bytes(file.read_bytes())
    name = corpus.FILES[condition]
    rows = [json.loads(line) for line in (target / name).read_text().splitlines()]
    design = json.loads((target / 'design.json').read_text())
    mutate(rows, design)
    (target / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    (target / 'design.json').write_text(json.dumps(design), encoding='utf-8')
    manifest = json.loads((target / 'manifest.json').read_text())
    for key in (condition, 'design'):
        file = target / corpus.FILES[key]
        manifest['files'][key].update(sha256=sha(file), bytes=file.stat().st_size)
    (target / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_full_fixture_audit_and_budget(packet):
    path, tokenizer, summary = packet
    result = audit_packet(path, tokenizer)
    assert result['status'] == 'PASS_OFFLINE_AUDIT'
    assert result['full_tokenizer_replay'] is True
    assert result['counts'] == {'control': 512, 'candidate': 512, 'dev': 672}
    assert summary['reviewed'] is False
    assert audit_packet(path)['full_tokenizer_replay'] is False


def test_exact_exposure_invariants_and_orders(packet):
    data = corpus.load_packet(packet[0])
    assert [r['id'] for r in data['control']] == [r['id'] for r in data['candidate']]
    assert len({entry['slot_id'] for entry in data['ledger']}) == 512
    assert all(e['labels_sha256']['control'] == e['labels_sha256']['candidate'] for e in data['ledger'])
    selected = [m for m in data['design']['candidate'].values() if m['category'] == 'update']
    assert {m['permutation'] for m in selected} == set(range(6))
    assert {m['latest_position'] for m in selected} == {0, 1, 2}
    for condition in ('control', 'candidate', 'dev'):
        for row in data[condition]:
            assert json.loads(row['target']) == raw_oracle(row['prompt'], data['design'][condition][row['id']])


def test_tied_revision_rejected(packet):
    data = corpus.load_packet(packet[0])
    row = next(r for r in data['dev'] if data['design']['dev'][r['id']]['category'] == 'update')
    meta = copy.deepcopy(data['design']['dev'][row['id']])
    old, replacement = meta['events'][0]['version'], meta['events'][1]['version']
    prompt = row['prompt'].replace(f'revision {old} |', f'revision {replacement} |')
    meta['events'][0]['version'] = replacement
    meta['evidence_block_text'] = meta['evidence_block_text'].replace(f'revision {old} |', f'revision {replacement} |')
    with pytest.raises(Blocked, match='distinct numeric revisions'):
        raw_oracle(prompt, meta)


def test_missing_required_field_rejected(packet):
    data = corpus.load_packet(packet[0])
    row = next(r for r in data['dev'] if data['design']['dev'][r['id']]['category'] == 'lookup')
    meta = copy.deepcopy(data['design']['dev'][row['id']])
    factual = next(e for e in meta['events'] if e['subject'] == meta['query']['subject'])
    old = f'{factual["subject"]} | code'
    prompt = row['prompt'].replace(old, f'{factual["subject"]} | weight')
    meta['evidence_block_text'] = meta['evidence_block_text'].replace(old, f'{factual["subject"]} | weight')
    factual['field'] = 'weight'
    with pytest.raises(Blocked, match='required field missing'):
        raw_oracle(prompt, meta)


def test_receipt_hash_binding(packet):
    source = packet[0]
    with tempfile.TemporaryDirectory(dir=source.parent) as directory:
        target = Path(directory) / 'tampered'
        target.mkdir()
        for file in source.iterdir():
            (target / file.name).write_bytes(file.read_bytes())
        with (target / 'candidate_train.jsonl').open('a') as stream:
            stream.write('{}\n')
        with pytest.raises(Blocked, match='packet file changed'):
            corpus.load_packet(target)


def test_malicious_rehashed_oracle_target_rejected(packet):
    with tempfile.TemporaryDirectory(dir=packet[0].parent) as directory:
        target = Path(directory) / 'wrong-target'
        def mutate(rows, design):
            row = next(r for r in rows if design['candidate'][r['id']]['category'] == 'update')
            row['target'] = '{"v":"consistently wrong"}'
            design['candidate'][row['id']]['row_sha256'] = digest(row)
        _rewrite_packet(packet[0], target, 'candidate', mutate)
        with pytest.raises(Blocked, match='oracle or compact target'):
            audit_packet(target)


def test_malicious_permutation_sidecar_rejected(packet):
    with tempfile.TemporaryDirectory(dir=packet[0].parent) as directory:
        target = Path(directory) / 'wrong-order'
        def mutate(rows, design):
            row = next(r for r in rows if design['candidate'][r['id']]['category'] == 'update')
            meta = design['candidate'][row['id']]
            meta['permutation'] = (meta['permutation'] + 1) % 6
        _rewrite_packet(packet[0], target, 'candidate', mutate)
        with pytest.raises(Blocked, match='permutation sidecar'):
            audit_packet(target)


def test_split_templates_are_separate(packet):
    data = corpus.load_packet(packet[0])
    train = {m['template_id'] for m in data['design']['control'].values()}
    dev = {m['template_id'] for m in data['design']['dev'].values()}
    assert not train.intersection(dev)


def test_malformed_prompt_and_retention_fields_rejected(packet):
    data = corpus.load_packet(packet[0])
    row = next(r for r in data['dev'] if data['design']['dev'][r['id']]['category'] == 'update')
    with pytest.raises(Blocked, match='block structure'):
        raw_oracle(row['prompt'].replace('\n---\n', '\n', 1), data['design']['dev'][row['id']])
    row = next(r for r in data['dev'] if data['design']['dev'][r['id']]['category'] == 'arithmetic')
    prompt = re.sub(r'"c":\d+', '"unknown":3', row['prompt'])
    with pytest.raises(Blocked, match='retention fields'):
        raw_oracle(prompt, data['design']['dev'][row['id']])


def test_token_replay_detects_encoder_drift(packet):
    class WrongTokenizer(FixtureTokenizer):
        def encode(self, text, **kwargs):
            return [v + 1 for v in super().encode(text, **kwargs)]
    with pytest.raises(Blocked, match='runtime token/label hash'):
        audit_packet(packet[0], WrongTokenizer())


def test_retention_instruction_contradiction_rejected(packet):
    data = corpus.load_packet(packet[0])
    row = next(r for r in data['dev'] if data['design']['dev'][r['id']].get('operation') == 'add')
    with pytest.raises(Blocked, match='instruction contradicts'):
        raw_oracle(row['prompt'].replace('Compute a+b.', 'Compute a-b.'), data['design']['dev'][row['id']])
