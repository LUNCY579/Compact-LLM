# Rineya MaxRL Verifiable Data v1

**MAXRL-DATA-001 — generated and CPU-audited; no pretrained training.**

Dedicated branch: `research/maxrl-verifiable-data-v1` in `LUNCY579/ChatBubble`.
Code commit: `fbb0480e45d5b43a192fe456f729b0f0609ad2f6`.
Original registration: `63f73080c009cb0fabece6a84b47f141292c2fb9`.
Base: `main@a6ee824584a9e28881826ab92a11db48e124d90c`.

The target is likelihood of **any verifier-accepted completion**, not imitation
of one reference string. This is an original synthetic short-answer method port,
not a reproduction of the paper's benchmark numbers or unrestricted CoT recipe.
See `EXPERIMENT.md` for the isolated GRPO-versus-MaxRL study and remaining gates.

## Delivered corpus

32,800 unique canonical problem instances: 24,000 TRAIN; 2,400 DEV;
2,400 exposed TEST; 2,400 harder CHALLENGE; 1,600 separate WARMSTART.
Eight families, each with 3,000 TRAIN examples over three algorithmic difficulty
levels. All evaluation files are exposed and **not sealed**. WARMSTART is disjoint
from the online-RL pool and every evaluation split.

The complete generated JSONL corpus is delivered in the conversation ZIP. To
avoid committing 130 MB of redundant rendered views, Git stores the generator,
source-bound receipts, corpus manifest and verified examples. Rebuild exact data
with the command below. The downloadable ZIP includes all rendered files already;
generation is not required to use it. It also includes the full source package.

## Run from the repository or extracted ZIP root

```bash
python -m pytest -q experiments/maxrl_verifiable_data_v1/tests
python -m experiments.maxrl_verifiable_data_v1.corpus build --out ./maxrl_data
python -m experiments.maxrl_verifiable_data_v1.corpus audit --data ./maxrl_data --receipt ./maxrl_audit.json
```

Outputs must be new paths. The default seed is `260202710`; the full build is
reproducible on the tested Python 3.13.5 environment. A `--scale pilot` build has
528 instances and is for debugging, not an independent replication; it can
share instances with the full build. Generation and verification use only the
standard library. The tests used pytest 9.0.2.

For the already-generated ZIP use `--data ./data`, not a rebuild over that path.

## Data views and authority boundaries

`canonical/*.jsonl` binds IDs, split, difficulty, exact public prompt, verifier
version/specification, provenance, and hashes. `policy/*.jsonl` contains IDs and
system/user messages only; feed only `messages` into the actor.

`verl_jsonl/*.jsonl` contains `data_source`, `prompt`, `ability`, `reward_model`,
and `extra_info`. Only `prompt` is policy input. `ground_truth` is a serialized
verifier specification, NOT a preferred answer trace. Wire `verifier.compute_score`
as the custom reward callback, with package importability established on the host.
The callback has the verl signature `(data_source, solution_str, ground_truth,
extra_info=None)`. This adapter has data-level tests, not native trainer qualification.

`oracles/*.jsonl` contains independently recovered correct certificates and a
live incorrect control per instance. These are not model-generated rollouts and
must not be fed to the actor. Reference search returns up to three certificates;
the verifier accepts any valid solution, including unlisted alternatives.

`sft/warmstart.messages.jsonl` contains only `messages` at the top level and follows
system -> user -> assistant, with concise final answers and no visible CoT.
Its IDs/provenance live in a separate line-aligned sidecar. All 1,600 targets
have at most 21 regex proxy tokens under `\w+|[^\w\s]`; this is NOT a native-token
or complete legacy-pipeline qualification. Warm-start both comparison arms
identically or neither; do not silently train on evaluation oracle files.

## Optional formats and tokenizer check

```bash
python -m experiments.maxrl_verifiable_data_v1.export parquet --data ./data --out ./maxrl_parquet
python -m experiments.maxrl_verifiable_data_v1.export tokenizer-audit --data ./data --tokenizer /absolute/local/tokenizer --out ./tokenizer_receipt.json --context 2048 --response-budget 256
```

**No Parquet files were produced here.** PyArrow is unavailable and network
installation failed. The exporter stops with exit 2 rather than substituting a
mislabelled file; its successful PyArrow roundtrip remains unexecuted. The
canonical and verl-shaped JSONL files are complete. A native verl run normally
requires the Parquet conversion and runtime binding first.

The tokenizer command is local-only, disables remote code, downloads no model,
requests a non-thinking chat template, and never truncates. It checks prompt
lengths only; actual generation/EOS, completion masking, native logprobs, and SFT
response token lengths still need qualification. No tokenizer assets were loaded
in this session.

## Validation

177/177 CPU tests passed. Every one of the 32,800 persisted visible problems was
independently re-solved: 66,822 reference certificates accepted; 360,800 negative
checks rejected. Exact canonical-instance/prompt/ID overlap: zero. The tests also
exercise altered specifications, multiple accepted answers, suboptimal-but-valid
paths, corrupt metadata, parser attacks, group identities and the exact small-
domain MaxRL estimator identity. See `evidence/` and `DATA_CARD.md`.

Source code blobs published at the code commit were compared with the locally
tested file hashes. No existing branch was advanced, no repository history was
rewritten, and no pretrained assets, old experiment outcomes, or sealed data were
modified. No GitHub Actions execution is claimed.

## Sources

Paper v3: https://arxiv.org/html/2602.02710v3
Official project: https://zanette-labs.github.io/MaxRL/
Official data interface: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
Official reward interface: https://verl.readthedocs.io/en/latest/preparation/reward_function.html

No paper benchmark examples or third-party corpus rows were copied into this data.
No blanket third-party license claim is made; preserve the repository's licensing
and provenance policies when redistributing.
