"""Q2b: Titan-Sparse's training data, verified by hash chain and by the recorded step trace."""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TITAN_DIR, dump, jl, sha  # noqa: E402

CORE = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
            'Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/CORE.json')
core = json.loads(CORE.read_text(encoding='utf-8'))
us = core['source']['update_source']
mat = json.loads((TITAN_DIR / 'control-hf' / 'MATERIALIZATION.json').read_text(encoding='utf-8'))
run = TITAN_DIR / 'control-journal' / 'runs' / 'train-001'
res_ = json.loads((run / 'result.json').read_text(encoding='utf-8'))
dc_path = TITAN_DIR / 'control-prepared' / 'DATA_CONTRACT.json'
dc = json.loads(dc_path.read_text(encoding='utf-8'))
corpus = TITAN_DIR / 'control-prepared' / 'microfit-train.jsonl'
chain = {
    'CORE.update_source.checkpoint endswith checkpoint-00028': us['checkpoint'].replace('\\', '/').endswith('train-001/microfit/checkpoint-00028'),
    'CORE.checkpoint_manifest_sha == MATERIALIZATION.checkpoint_manifest_sha': us['checkpoint_manifest_sha256'] == mat['checkpoint_manifest_sha256'],
    'MATERIALIZATION.checkpoint_progress.updates == 28': mat['checkpoint_progress']['updates'] == 28,
    'result.json binding.data_contract_sha == sha(DATA_CONTRACT.json)': res_['binding']['data_contract_sha256'] == sha(dc_path),
    'DATA_CONTRACT.files[microfit-train.jsonl] == sha(corpus)': dc['files']['microfit-train.jsonl'] == sha(corpus),
    'checkpoint-00028 directory present on disk': (run / 'microfit' / 'checkpoint-00028').exists(),
}
for k, v in chain.items():
    print(f'  {k}: {v}')
rows = jl(corpus)
steps = jl(run / 'microfit' / 'steps.jsonl')
trained = [i for s in steps for i in s['row_ids']]
print(f'steps.jsonl: {len(steps)} updates, {len(trained)} row presentations, {len(set(trained))} distinct; '
      f'== corpus ids: {set(trained) == {r["id"] for r in rows}}; supervised tokens {sum(s["supervised_tokens"] for s in steps)}')
print('corpus families', dict(collections.Counter(r['family'] for r in rows)), '| source_release',
      dict(collections.Counter(r['source_release'] for r in rows)), '| source_split', dict(collections.Counter(r['source_split'] for r in rows)))
oce = [s['objective_ce'] for s in steps]
print(f'Titan objective_ce (response+EOS CE on each update group at FIRST exposure): updates 1-3 mean '
      f'{statistics.mean(oce[:3]):.4f}, 26-28 mean {statistics.mean(oce[-3:]):.4f}')
# Example transition row
r = rows[0]
print('example:', r['family'], '| USER', r['prompt'][1]['content'][:160], '| ANSWER', r['reference_text'])
print('wrote q2b_titan_lineage.json', dump('q2b_titan_lineage.json', {
    'chain': chain, 'core_sha': sha(CORE), 'corpus_sha': sha(corpus), 'updates': len(steps),
    'presentations': len(trained), 'distinct': len(set(trained)), 'objective_ce': oce}))
