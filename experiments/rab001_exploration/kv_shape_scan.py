"""Which published cores admit a *free, exact* KV-cache reduction, and by how much.

A layer caches `2 * n_kv * head_dim` values per token. Those values are produced by
`W_KV = [W_K ; W_V]`, of shape `(2 * n_kv * head_dim, hidden)`, whose rank is at most
`min(2 * n_kv * head_dim, hidden)`. So a latent of that rank reproduces K and V *exactly*
and the cache shrinks by

    free_exact_reduction = 2 * n_kv * head_dim / min(2 * n_kv * head_dim, hidden)

This is arithmetic about the weight shapes, not a measurement, and it is only above 1.0 when
`2 * n_kv * head_dim > hidden` -- which happens when a model keeps a large `head_dim` and a
GQA head count while shrinking `hidden`. It is also not novel: at rank = hidden the latent is
an invertible map of the layer input, i.e. caching the layer input instead of K and V. It is
recorded because it tells you, per core, how much reduction needs no experiment at all, and
therefore what the *measured* frontier below that point has to beat to be worth anything.

Reads config.json only. Loads no weights, touches no sealed data, opens no gate.
"""
import json, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CORES = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores')
OUT = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
           r'\experiments\architecture_benchmark_001\outputs\latent')


def shape_facts(cfg: dict) -> dict | None:
    hidden = cfg.get('hidden_size')
    kv_heads = cfg.get('num_key_value_heads')
    q_heads = cfg.get('num_attention_heads')
    head_dim = cfg.get('head_dim') or (hidden // q_heads if hidden and q_heads else None)
    if not (hidden and kv_heads and head_dim):
        return None
    cached = 2 * kv_heads * head_dim
    max_rank = min(cached, hidden)
    return {'hidden_size': hidden, 'num_attention_heads': q_heads,
            'num_key_value_heads': kv_heads, 'head_dim': head_dim,
            'layers': cfg.get('num_hidden_layers'),
            'cached_values_per_token_per_layer': cached,
            'max_useful_latent_rank': max_rank,
            'free_exact_reduction': cached / max_rank,
            'free_reduction_available': cached > hidden}


rows, unreadable = {}, {}
for cfg_path in sorted(CORES.glob('*/config.json')):
    name = cfg_path.parent.name
    try:
        facts = shape_facts(json.loads(cfg_path.read_text(encoding='utf-8')))
    except Exception as exc:
        unreadable[name] = repr(exc)
        continue
    if facts is None:
        unreadable[name] = 'config lacks hidden_size / num_key_value_heads / head_dim'
    else:
        rows[name] = facts

print(f"{'core':40}{'hidden':>8}{'kv_h':>6}{'head_dim':>10}{'L':>5}{'KV/layer':>10}"
      f"{'max rank':>10}{'free x':>8}")
for name, f in rows.items():
    print(f"{name:40}{f['hidden_size']:>8}{f['num_key_value_heads']:>6}{f['head_dim']:>10}"
          f"{str(f['layers']):>5}{f['cached_values_per_token_per_layer']:>10}"
          f"{f['max_useful_latent_rank']:>10}{f['free_exact_reduction']:>8.2f}")
for name, why in unreadable.items():
    print(f'  SKIPPED {name}: {why}')

factors = sorted({f['free_exact_reduction'] for f in rows.values()})
print(f'\n{len(rows)} cores read, {len(unreadable)} skipped | distinct free reduction factors: '
      f'{", ".join(f"{x:.2f}x" for x in factors)}')

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'KV_SHAPE_SCAN.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'derived_from_config_shapes_only',
    'note': 'Arithmetic on weight shapes, not a measurement. free_exact_reduction is the '
            'cache reduction obtainable with zero training and zero quality loss, because '
            'rank(W_KV) <= min(2*n_kv*head_dim, hidden). At rank = hidden this is equivalent '
            'to caching the layer input instead of K and V, which is a known trick, not a '
            'finding. No weights loaded, no sealed data, no gate opened.',
    'formula': 'free_exact_reduction = 2*n_kv*head_dim / min(2*n_kv*head_dim, hidden)',
    'cores': rows, 'skipped': unreadable,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('wrote', OUT / 'KV_SHAPE_SCAN.json')
