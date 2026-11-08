# TRVC-001 executed results

**2026-10-07 | CPU software/reference evidence only | Native Rineya efficacy unknown**

## Validation

The scoped suite passed **129 tests**, with zero failures, errors or skips and no
reported warnings in the final run. `compileall` passed. Runtime: Python 3.13.5,
PyTorch 2.10.0+cpu, pytest 9.0.2. This was the working container, not TheFakeGod.
The full ChatBubble suite and remote GitHub Actions were not run.

Tests exercise exact transitions, role/parent/world isolation, hostile action
strings, depth-bound joins, stale rescoring, graph capacity, deduplication,
provenance, prefix/suffix composition, coverage counters, budget overrun accounting,
causal masking/EOS, soft gradients, probability-mass checks, mock-LM backward paths,
reference detachment, replay-loss requirements, paired metrics and fail-closed
native intake. Test counts establish software checks, not model intelligence.

The scripted mechanism demo recorded zero complete root successes but one verified
solution after a successful frontier continuation. Its first origin is explicitly
`restart_completion`, with supporting attempts `failed-root` and
`successful-frontier`. Before observing that suffix, composition produced no
solution. Root-only negative controls cannot create an accepting terminal from
wholly unsuccessful root traces. A separate test constructs a genuinely new graph
path after an earlier root success supplies a valid ending.

The mathematical gradient fixture also demonstrates a shared-parameter update
that raises the best correct path while reducing total correct probability.
This is a local counterexample, not a replication of any published benchmark.

## Tiny-policy pilot

The pilot used a **230-parameter symbolic actor**, eight TRAIN worlds, eight DEV
worlds and eight toy GUARD worlds. Three seeds (7, 19, 31), seven treatments and
eight optimizer steps per treatment gave **21 treatment conditions / 168 updates**.
The entire pilot was repeated once and its complete JSON compared equal on this
host. The repeat is a reproducibility check, not three additional independent seeds.
No LLM token was generated: the actor samples categorical environment actions.

| Seed | Problems with a verified path | Verified paths | Paths sampled from root | Problems with multiple paths | Baseline DEV pass@1 |
|---|---:|---:|---:|---:|---:|
| 7 | 6/8 | 8 | 6 | 2/8 | 0.09375 |
| 19 | 3/8 | 3 | 3 | 0/8 | 0.06250 |
| 31 | 6/8 | 12 | 6 | 3/8 | 0.06250 |

Each DEV measurement below uses four independent draws on each of eight worlds.
These are exposed, small-sample engineering measurements, not Rineya results.

| Treatment | Seed 7 DEV pass@1 | Seed 19 | Seed 31 |
|---|---:|---:|---:|
| Root-only-data uniform | 0.21875 | 0.06250 | 0.09375 |
| Full-bank uniform | 0.21875 | 0.06250 | 0.06250 |
| Hard maximum | 0.21875 | 0.06250 | 0.09375 |
| Coverage, sequence-normalized | 0.21875 | 0.06250 | 0.09375 |
| Soft T=0.25 | 0.21875 | 0.06250 | 0.06250 |
| Soft T=1 | 0.21875 | 0.06250 | 0.06250 |
| Secondary per-token imitation | 0.21875 | 0.15625 | 0.06250 |

### Interpretation, not promotion

All three banks fail the proposed requirement that at least half of enrolled
problems have multiple verified candidates. Seed 19 has exactly one candidate on
each eligible problem: the primary sequence objectives have identical gradients
there. A secondary length-normalized treatment can still differ because it
reweights problems by response length. That is not evidence about hard versus soft
aggregation. No objective is selected as a Rineya winner from this pilot.

The full archive supplies paths absent from complete root sampling in seeds 7 and
31. This is useful mechanism evidence on a tiny controlled domain. It does not
show that extra data always improves the trained actor, nor that an archive is a
substitute for actual model retention.

In the unprotected acquisition runs, seed 7 loses one of two fixed-first-draw
source-correct toy guard examples. Seed 19 loses two of two for the primary
sequence treatments. Seed 31 has no initially correct first-draw guard examples,
so its retention denominator is zero and retention is reported as null, not 100%.
The guard is not a qualified legacy-skill instrument. These observations reinforce
the need for explicit retention controls; they do not estimate real Rineya damage.

Root-only data has a different candidate inventory. Shared-bank primary arms have
the same candidate-forward inventory and update cap, but different effective
path-gradient allocations. No matched wall-clock, total compute or native token
budget claim is made. Every recorded acquisition/evaluation cost remains in the
pilot receipt; losses and scoring calls are not silently treated as free.

## Native status and next executable milestone

Native intake returns `BLOCKED_NATIVE_UNQUALIFIED` and exit code 2. Zero Rineya
checkpoints were loaded, zero user-GPU jobs were launched and zero Codex tasks were
created. No existing experiment, schedule, source checkpoint or base branch was
modified. The object bridge was exercised with a mock causal LM, not a 0.6B model.

The next native milestone is an exact-source, zero-update prompt/tokenizer/scoring
and resource qualification, followed by a frozen-policy collector screen. It must
produce a candidate-diverse bank before an objective ranking is interpreted. A
native preservation study additionally needs qualified acquired/legacy reservoirs,
its sequential scheduler and fresh evaluation instruments. See RESEARCH_PLAN.md.

## Saved evidence

- `evidence/tests.txt` and `tests.junit.xml.gz`: final scoped pytest output and all cases.
- `evidence/mechanism.json`: complete scripted archive/provenance and cost receipt.
- `evidence/toy_pilot.json.gz`: complete pilot object, including task specs, bank
  paths/origins, per-update losses, raw evaluation actions/outcomes and counters.
- `evidence/native_preflight.json`: actual blocked intake result.
- `evidence/validation.json`: runtime, executed scope, repeat check and evidence hashes.
- `source_manifest.json`: exact source/config bytes used for the recorded checks.

The gzip files are lossless, deterministic-mtime standard-library compression.
No native checkpoint, private user training data, generated corpus from another
experiment or author benchmark data is included.
