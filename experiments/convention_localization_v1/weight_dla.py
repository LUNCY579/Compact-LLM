"""CONVENTION-LOC-001, step 0 (exploratory, weight-only): where could dOpallo WRITE the trailing period?

Background. ADD-100 = Supa + dSparse + dOpallo ends 178/192 development answers with a period that
neither SPARSE nor AVG emits. The period arrives as the merged token '}.' after a JSON answer
(114/178) or as '.' after a bare number (64/178). The two decisions are therefore
    D1: '}.' vs '}'            (JSON families)
    D2: '.'  vs '<|im_end|>'   (number family)

This step asks, WITHOUT a forward pass, which residual writers (o_proj, down_proj) of which block
have dOpallo aligned with the unembedding contrast of D1/D2, i.e. a direct-logit-attribution
CAPACITY. For a writer delta dW (d_model x d_in) and a contrast direction v = g * (e_a - e_b)
(g = Supa's final RMSNorm gain; embeddings are tied),

    r(v) = d_model * ||dW^T v_hat||^2 / ||dW||_F^2

is 1 for an isotropic dW. Because embeddings are anisotropic, the reference is NOT 1 but the
distribution of r over 512 random token-pair contrasts (seed 0). Reported: r for D1/D2 and its
percentile in that null. dSparse, which emits no period, is run through the identical pipeline as
a CONTROL.

Limits, stated up front: direct path only (ignores everything downstream of the writer and the
final-norm scale of the actual residual); magnitude only (no activations, so no sign); a
screening prior for the GPU localization screen, NOT a mechanism claim.

Read-only: reads three FP32 arms from titan_opallo_skill_vector_v1/arms. Writes WEIGHT_DLA.json here.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "titan_model_soup_mini_v1" / "local-runs"))
from verify_independent import ST  # noqa: E402

ARMS = EXP / "titan_opallo_skill_vector_v1" / "arms"
TOK = ARMS / "BASE"
NULL_PAIRS = 512
WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def full(st, k):
    shape = st.info(k)[1]
    return st.rows(k, 0, shape[0]).reshape(shape).astype(np.float64)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK), local_files_only=True)
    ids = {"}.": tok.convert_tokens_to_ids("}."), "}": tok.convert_tokens_to_ids("}"),
           ".": tok.convert_tokens_to_ids("."), "<|im_end|>": tok.convert_tokens_to_ids("<|im_end|>")}
    assert all(isinstance(v, int) and v >= 0 for v in ids.values()), ids
    assert tok.tokenize("}.") == ["}."], tok.tokenize("}.")

    base, opl, spa = (ST(ARMS / a / "model.safetensors") for a in ("BASE", "OPALLO", "SPARSE"))
    assert "lm_head.weight" not in base.keys(), "expected tied embeddings"
    g = full(base, "model.norm.weight")
    emb_row = lambda i: base.rows("model.embed_tokens.weight", i, i + 1)[0].astype(np.float64)

    def contrast(a, b):
        v = g * (emb_row(a) - emb_row(b))
        return v / np.linalg.norm(v)

    D = {"D1_}._vs_}": contrast(ids["}."], ids["}"]), "D2_._vs_im_end": contrast(ids["."], ids["<|im_end|>"])}
    rng = np.random.default_rng(0)
    vocab = 151643                                  # ordinary tokens only
    null = np.stack([contrast(*rng.choice(vocab, 2, replace=False)) for _ in range(NULL_PAIRS)])

    out = {"experiment": "CONVENTION-LOC-001 step 0 (weight-only, exploratory)", "token_ids": ids,
           "null_pairs": NULL_PAIRS, "blocks": {}}
    for L in range(28):
        for w in WRITERS:
            k = f"model.layers.{L}.{w}"
            W0 = full(base, k)
            for name, st in (("dOpallo", opl), ("dSparse", spa)):
                dW = full(st, k) - W0
                fro2 = float((dW ** 2).sum())
                if fro2 == 0.0:
                    continue
                dm = dW.shape[0]
                r = lambda v: dm * float(((dW.T @ v) ** 2).sum()) / fro2
                rn = np.array([dm * float(((dW.T @ v) ** 2).sum()) / fro2 for v in null])
                rec = {"fro": fro2 ** 0.5, "null_median": float(np.median(rn)), "null_p95": float(np.percentile(rn, 95))}
                for dn, v in D.items():
                    rv = r(v)
                    rec[dn] = {"r": rv, "percentile_in_null": float((rn < rv).mean()),
                               "ratio_to_null_median": rv / float(np.median(rn)),
                               "abs_write": float(np.linalg.norm(dW.T @ v))}
                out["blocks"].setdefault(str(L), {}).setdefault(w, {})[name] = rec
        print(f"  block {L} done", flush=True)
    for s in (base, opl, spa):
        s.close()
    (HERE / "WEIGHT_DLA.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())

    print("\nblock writer        | dOpallo D1 pct  ratio | D2 pct  ratio || dSparse D1 pct ratio | D2 pct ratio")
    for L, ws in out["blocks"].items():
        for w, rec in ws.items():
            o, s = rec.get("dOpallo"), rec.get("dSparse")
            fmt = lambda x, d: f"{x[d]['percentile_in_null']:.2f} {x[d]['ratio_to_null_median']:5.2f}" if x else "   -     - "
            print(f"{L:>5} {w.split('.')[1]:12s} | {fmt(o, 'D1_}._vs_}')} | {fmt(o, 'D2_._vs_im_end')} || "
                  f"{fmt(s, 'D1_}._vs_}')} | {fmt(s, 'D2_._vs_im_end')}")


if __name__ == "__main__":
    main()
