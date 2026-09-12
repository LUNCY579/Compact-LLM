"""Per-layer low-rank KV compression (MLA-style), fitted to the existing weights. No training.

The displaced-layer law measured in this programme says damage scales with the number of
layers attending to a K/V bank they did not produce, and that *every removed bank displaces
exactly one layer by construction*. So every sharing topology sits on one frontier priced at
~0.5 nats per bank.

This design removes no bank. Each layer keeps its own K and V; only the *storage* changes.
Per layer, `k_proj` and `v_proj` are stacked into

    W_KV = [W_K ; W_V]   with shape (2 * n_kv * head_dim, hidden)

and factorised into a down-projection (rank r) and an up-projection. The cache then holds the
latent `c = W_D n` (r values per token per layer) instead of K and V together
(2 * n_kv * head_dim values). Displaced layers: **zero**. The law predicts the sharing price
should not apply. Whether anything else breaks instead is the question this measures.

Two honest caveats stated up front, because they bound what any result here can mean:

1. **r = hidden is not a discovery.** `W_KV` here is 2048x1024, so its rank is at most 1024
   and a latent of r = 1024 is an *exact* re-parameterisation that halves the cache. But at
   r = hidden the down-projection is an invertible square map, so this is just the standard
   observation that when `2 * n_kv * head_dim > hidden` you may cache the layer input instead
   of K and V. It is free, it is exact, and it is well known. The scientific content is
   entirely at **r < hidden**.

2. **RoPE is not decoupled.** DeepSeek's MLA splits off a small RoPE-carrying head so `W_UK`
   can be absorbed into `W_Q`, which is what buys the *speed*. Here RoPE and `k_norm` are
   applied after up-projection exactly as in the plain model, so the computation is
   numerically the same and only the cache shrinks. This measures the quality-versus-cache
   frontier, not latency.

Two fits are provided, and they answer different questions:

- `plain` truncates the SVD of `W_KV`, minimising weight-space error ||W - W_r||_F. It is
  data-free and it is the wrong objective: it spends rank on directions the model never
  activates.
- `whitened` minimises *activation*-space error E||W n - W_r n||^2 under the layer's real
  input covariance C, via the SVD of W S where C = S S^T. Calibration inputs come from
  **TRAIN only** — fitting on DEV would contaminate the DEV measurement it is scored on.

Exploratory. Outside the hashed experiment root; produces no receipt and opens no gate.
"""
from __future__ import annotations
import sys
import torch
from torch import Tensor, nn

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.model import dense_attention, rope

RIDGE = 1e-6   # relative to mean eigenvalue; only to make the Cholesky well-posed


class LatentKV(nn.Module):
    """A down/up projection pair standing in for one layer's `k_proj` and `v_proj`."""

    def __init__(self, hidden: int, kv: int, rank: int) -> None:
        super().__init__()
        self.kv, self.rank = kv, rank
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, 2 * kv, bias=False)

    @torch.no_grad()
    def _store(self, u: Tensor, s: Tensor, right: Tensor) -> None:
        """Split the retained spectrum symmetrically between the two factors."""
        r = self.rank
        root = torch.sqrt(s[:r])
        self.down.weight.copy_((root[:, None] * right[:r]).to(self.down.weight.dtype))
        self.up.weight.copy_((u[:, :r] * root[None, :]).to(self.up.weight.dtype))

    @torch.no_grad()
    def fit(self, w_k: Tensor, w_v: Tensor, cov: Tensor | None = None) -> dict:
        """Factorise [W_K ; W_V] at the configured rank.

        `cov` None  -> plain SVD, minimises weight-space error.
        `cov` given -> whitened SVD, minimises activation-space error under that covariance.

        The reported energy is spectral energy retained in whichever norm was optimised. It
        is a property of the factorisation, not a prediction about downstream quality.
        """
        w = torch.cat([w_k, w_v], dim=0).to(torch.float64)      # (2*kv, hidden)
        r = self.rank
        if cov is None:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            right = vh
        else:
            c = cov.to(torch.float64)
            c = c + torch.eye(c.shape[0], dtype=c.dtype) * (RIDGE * torch.diagonal(c).mean())
            chol = torch.linalg.cholesky(c)                     # c = chol @ chol.T
            u, s, vh = torch.linalg.svd(w @ chol, full_matrices=False)
            # A_r = U_r S_r Vh_r inv(chol); fold inv(chol) into the right factor.
            # `left=False` solves X @ chol = vh, i.e. X = vh @ inv(chol). Passing chol.T
            # here instead gives vh @ inv(chol).T, which is a different matrix and breaks
            # the r = full_rank exactness control -- as it did, before the control caught it.
            right = torch.linalg.solve_triangular(chol, vh, upper=False, left=False)
        total = float((s ** 2).sum())
        kept = float((s[:r] ** 2).sum())
        self._store(u, s, right)
        return {'rank': r, 'full_rank': int(s.numel()),
                'spectral_energy_retained': kept / total if total else 1.0,
                'smallest_kept_singular_value': float(s[r - 1]),
                'largest_dropped_singular_value': float(s[r]) if r < s.numel() else 0.0}

    def forward(self, n: Tensor) -> tuple[Tensor, Tensor]:
        both = self.up(self.down(n))
        return both[..., :self.kv], both[..., self.kv:]


class LatentKVModel(nn.Module):
    """The donor model with every layer's K/V storage replaced by a rank-r latent.

    The donor's blocks are *referenced*, not copied, so this costs only the latents. All
    other weights are bitwise the donor's, which is what makes the r = hidden exactness
    control meaningful.
    """

    def __init__(self, source, rank: int) -> None:
        super().__init__()
        c = source.config
        if c.architecture != 'plain':
            raise ValueError('latent KV wraps a plain donor')
        self.kv = c.num_key_value_heads * c.head_dim
        self.max_rank = min(2 * self.kv, c.hidden_size)
        if not 0 < rank <= self.max_rank:
            raise ValueError(f'rank must be in 1..{self.max_rank}')
        self.config, self.rank, self.donor = c, rank, source
        self.latents = nn.ModuleList(LatentKV(c.hidden_size, self.kv, rank)
                                     for _ in range(c.encoder_layers))

    @torch.no_grad()
    def fit(self, covs: list[Tensor] | None = None) -> dict:
        fits = []
        for i, (blk, lat) in enumerate(zip(self.donor.encoder, self.latents)):
            fits.append(lat.fit(blk.k_proj.weight, blk.v_proj.weight,
                                None if covs is None else covs[i]))
        return {'operation': 'per_layer_low_rank_kv_compression',
                'fit': 'plain_svd' if covs is None else 'activation_whitened_svd',
                'rank': self.rank, 'uncompressed_kv_per_layer': 2 * self.kv,
                'kv_ratio': self.kv_ratio(),
                'exact_reparameterisation': self.rank >= self.max_rank,
                'displaced_layers': 0,
                'min_spectral_energy_retained': min(f['spectral_energy_retained'] for f in fits),
                'mean_spectral_energy_retained': sum(f['spectral_energy_retained'] for f in fits) / len(fits),
                'per_layer': fits}

    def kv_ratio(self) -> float:
        return self.rank / (2 * self.kv)

    def forward(self, input_ids: Tensor, *, past: list | None = None,
                use_cache: bool = False, compute_logits: bool = True):
        """Dense forward, with an optional incremental cache for greedy decoding.

        The cache stores the *up-projected* K and V rather than the latent. That is a speed
        choice and not a quiet weakening of the claim: up-projection is a deterministic
        function of each token's own latent, so expanding at write time and expanding the
        whole cache at read time produce bit-identical values. The reported cache cost is
        the latent size (r per token per layer), which is what an implementation would hold.
        """
        b, t = input_ids.shape
        device = input_ids.device
        past_len = 0 if past is None else past[0][0].shape[2]
        total = past_len + t
        pos = torch.arange(past_len, total, device=device).expand(b, t)
        kpos = torch.arange(total, device=device).expand(b, total)
        qvalid = torch.ones((b, t), dtype=torch.bool, device=device)
        kvalid = torch.ones((b, total), dtype=torch.bool, device=device)
        qseg = torch.zeros((b, t), dtype=torch.long, device=device)
        kseg = torch.zeros((b, total), dtype=torch.long, device=device)
        d, c = self.donor, self.config
        h = d.embed_tokens(input_ids)
        present = [] if use_cache else None
        for i, (blk, lat) in enumerate(zip(d.encoder, self.latents)):
            n = blk.input_norm(h)
            shape = lambda z, heads: z.view(b, t, heads, c.head_dim).transpose(1, 2)
            q = rope(blk.q_norm(shape(blk.q_proj(n), c.num_attention_heads)), pos, c.rope_theta)
            # The only change from SelfBlock.forward: K and V come from the latent.
            k_flat, v_flat = lat(n)
            k = rope(blk.k_norm(shape(k_flat, c.num_key_value_heads)), pos, c.rope_theta)
            v = shape(v_flat, c.num_key_value_heads)
            if past is not None:
                k = torch.cat((past[i][0], k), 2)
                v = torch.cat((past[i][1], v), 2)
            if use_cache:
                present.append((k, v))
            a = dense_attention(q, k, v, pos, kpos, qvalid, kvalid, qseg, kseg)
            h = h + blk.o_proj(a.transpose(1, 2).reshape(b, t, -1))
            h = h + blk.mlp(blk.post_norm(h))
        normalized = d.final_norm(h)
        out = (torch.nn.functional.linear(normalized, d.embed_tokens.weight)
               if compute_logits else normalized)
        return (out, present) if use_cache else out


@torch.no_grad()
def collect_input_covariance(source, batches: list[Tensor]) -> tuple[list[Tensor], int]:
    """E[n n^T] per layer, where n is the post-`input_norm` input to k_proj/v_proj.

    Accumulated in float64. Calibration rows must come from TRAIN; passing DEV rows here
    would fit the compression to the split it is later scored on.
    """
    c = source.config
    covs = [torch.zeros(c.hidden_size, c.hidden_size, dtype=torch.float64)
            for _ in range(c.encoder_layers)]
    tokens = 0
    for ids in batches:
        b, t = ids.shape
        valid = torch.ones((b, t), dtype=torch.bool)
        seg = torch.zeros((b, t), dtype=torch.long)
        pos = torch.arange(t).expand(b, t)
        h = source.embed_tokens(ids)
        for i, blk in enumerate(source.encoder):
            n = blk.input_norm(h)
            flat = n.reshape(-1, c.hidden_size).to(torch.float64)
            covs[i] += flat.T @ flat
            shape = lambda z, heads: z.view(b, t, heads, c.head_dim).transpose(1, 2)
            q = rope(blk.q_norm(shape(blk.q_proj(n), c.num_attention_heads)), pos, c.rope_theta)
            k = rope(blk.k_norm(shape(blk.k_proj(n), c.num_key_value_heads)), pos, c.rope_theta)
            v = shape(blk.v_proj(n), c.num_key_value_heads)
            a = dense_attention(q, k, v, pos, pos, valid, valid, seg, seg)
            h = h + blk.o_proj(a.transpose(1, 2).reshape(b, t, -1))
            h = h + blk.mlp(blk.post_norm(h))
        tokens += b * t
    for cov in covs:
        cov /= tokens
    return covs, tokens


def covariance_health(covs: list[Tensor], tokens: int, hidden: int) -> dict:
    """Whitening is only meaningful if the covariance is well determined.

    With fewer calibration tokens than `hidden`, C is rank-deficient by construction and the
    whitened fit would report near-perfect retained energy for a rank it has no evidence
    for. This reports the conditioning so that failure is visible instead of flattering.
    """
    eigs = [torch.linalg.eigvalsh(c) for c in covs]
    eff = [float((e.clamp_min(0).sum() ** 2) / (e.clamp_min(0) ** 2).sum()) for e in eigs]
    return {'calibration_tokens': tokens, 'hidden': hidden,
            'tokens_per_dimension': tokens / hidden,
            'rank_deficient_by_construction': tokens < hidden,
            'min_effective_rank': min(eff), 'mean_effective_rank': sum(eff) / len(eff),
            'worst_condition_number': max(float(e.max() / e.clamp_min(0).min().clamp_min(1e-30))
                                          for e in eigs)}
