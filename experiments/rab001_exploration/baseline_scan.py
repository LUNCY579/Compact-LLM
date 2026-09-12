"""Which corpus classes can the FROZEN SOURCE actually do?

A retention probe only measures retention if the capability exists to begin with. The
arithmetic_chain class looked like the obvious verifiable task, but the source scores near
zero on it, so using it would measure acquisition of something the model never had — and
would report "0/N before and after" for every arm, which is uninformative.

This scans verifiable classes and reports the frozen source's baseline numeric-match rate on
each, so the retention probe can be chosen from classes where the source is meaningfully
above zero. Choosing the eval on the FROZEN SOURCE's ability is legitimate: it is a property
of the teacher, measured before any arm is trained, and it is not tuned on any student
outcome.

Exploratory. Outside the hashed root; no receipt, no gate.

usage: baseline_scan.py [rows_per_class]
"""
import json, random, re, sys
from pathlib import Path
import torch

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _load_tokenizer

CORPUS = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class')
E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EOS = {151643, 151645}

# Classes whose answers are checkable without a judge.
CLASSES = ['arithmetic_chain', 'percentage_reasoning', 'ratio_proportion', 'counting_sets',
           'unit_conversion', 'time_duration', 'comparison_ranking', 'sequence_pattern',
           'schedule_ordering', 'estimation_sanity']

tok = _load_tokenizer(SRC)
model, _, _ = _load(SRC, 'cpu'); model.eval()
rng = random.Random(17)
_int = lambda s: (re.search(r'-?\d+', s).group(0) if re.search(r'-?\d+', s) else None)

results = {}
for cls in CLASSES:
    f = CORPUS / f'{cls}.jsonl'
    if not f.is_file():
        continue
    rows = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
    # one row per family, so the sample is independent
    byfam = {}
    for r in rows:
        byfam.setdefault(r['metadata']['template_family'], []).append(r)
    fams = sorted(byfam); rng.shuffle(fams)
    sample = [rng.choice(byfam[fa]) for fa in fams[:N]]

    numeric = strict = numeric_answers = 0
    outs = []
    with torch.inference_mode():
        for r in sample:
            msgs = r['messages']
            if msgs[-1]['role'] != 'assistant':
                continue
            want = msgs[-1]['content'].strip()
            numeric_answers += _int(want) is not None
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             enable_thinking=False, add_generation_prompt=True)
            cur = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
            cache, got = None, []
            for _ in range(24):
                step = model(cur, past=cache, use_cache=True, last_position_only=True)
                cache = step.cache
                t = int(step.logits[0, -1].argmax())
                if t in EOS:
                    break
                got.append(t); cur = torch.tensor([[t]], dtype=torch.long)
            text = tok.decode(got, skip_special_tokens=True).strip()
            ok = _int(want) is not None and _int(text) == _int(want)
            numeric += ok; strict += (text == want)
            if len(outs) < 3:
                # Console encoding on this host is cp1252; keep printed samples ASCII-safe so
                # a stray arrow or dash cannot kill the run before the artifact is written.
                safe = lambda x: x[:40].encode('ascii', 'replace').decode('ascii')
                outs.append({'expected': safe(want), 'got': safe(text), 'ok': bool(ok)})
    n = len(sample)
    results[cls] = {'rows': n, 'numeric_answerable': numeric_answers,
                    'numeric_hits': numeric, 'numeric_rate': numeric / n if n else 0.0,
                    'strict_hits': strict, 'samples': outs}
    print(f"{cls:24} {numeric:3d}/{n:<3} ({numeric/n if n else 0:6.1%})  "
          f"numeric-answerable {numeric_answers}/{n}", flush=True)
    for o in outs[:2]:
        print(f"      expected={o['expected']!r:26} got={o['got']!r:34} ok={o['ok']}", flush=True)

out = E / 'outputs/rigorous'
out.mkdir(parents=True, exist_ok=True)
(out / 'SOURCE_BASELINE_BY_CLASS.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_baseline_capability_scan_only',
    'note': 'Frozen source only, no training. Establishes which classes the teacher can '
            'actually do, so a retention probe is not chosen on a capability it lacks. '
            'Measured before any arm is trained and not tuned on any student outcome. '
            'Outside the experiment package; no receipt, no gate.',
    'rows_per_class': N, 'sampling': 'one row per template family, seed 17',
    'classes': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / 'SOURCE_BASELINE_BY_CLASS.json')
