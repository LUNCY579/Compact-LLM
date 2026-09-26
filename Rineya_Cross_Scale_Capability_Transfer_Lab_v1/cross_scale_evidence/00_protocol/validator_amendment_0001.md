# Functional Validator Amendment 0001

Visible evidence before amendment: Supa and official Qwen3-0.6B generated
`READY.` followed by EOS; projected P1-D2-W2 4B generated a deterministic
two-token multilingual loop and no EOS. No projected donor benchmark or merge
result was run.

The initial validator checked shape/key integrity, finite logits, deterministic
generation, valid EOS configuration, and active embedding tying, but did not
require the requested smoke response to be coherent or EOS to be emitted. It
therefore mislabeled a collapsed projected donor `GO`.

The corrected functional gate additionally requires:

- normalized response exactly `READY` for the exact-response smoke prompt;
- EOS emission within eight new tokens;
- absence of a repeated period-1 through period-4 token cycle.

All earlier validation packets are retained under attempt directories. Supa,
official Qwen3-0.6B, and every projected control are revalidated. No checkpoint
may enter MERGEDEV without the corrected `GO`.
