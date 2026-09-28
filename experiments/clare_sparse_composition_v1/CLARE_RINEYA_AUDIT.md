# CLARE-RINEYA audit — 2026-09-27

## Exact lineage and scope

Repository: LUNCY579/ChatBubble. The remote parent head was read before creating
`research/titan-clare-sparse-composable-skills-v1` at exactly
`816a1e8f4d1ab1c9d1ab847697d962a141521bac` (tree
`e6913d5e622c0323d05557a24b83a7978683fa5d`). The parent branch is
`research/titan-model-soup-mini-v1`, NOT main. Only the new experiment directory
is added. No existing study, checkpoint, branch, guard or dataset is rewritten.

The container cannot resolve github.com for shell git transport. Remote reads,
branch creation, Git objects, commits and non-force ref updates use the connected
GitHub API. New source files are tested locally; inherited native modules are
inspected through GitHub but are not a complete local clone. Consequently no
whole-repository regression pass or native integration pass is claimed. Evidence
binds the tested new source bytes to Git blob hashes and the actual source commit.

## Inspected parent components and reuse decisions

| Source at the exact parent | Inspection and decision |
|---|---|
| `experiments/titan_model_soup_mini_v1/soup.py` | Read checkpoint pins, hashing, architecture/inventory and interpolation interfaces. `native.py` imports its BASE_SHA, ASSETS, checked_file, read_json, canonical_config, inventory and check_tied_head. Its full-checkpoint interpolation is not duplicated. |
| `experiments/titan_model_soup_mini_v1/collect_native.py` | Read admission, suite and provenance boundaries. Its SOUP-MINI-001 authorization is unchanged; the new rollback smoke is a different instrument and does not satisfy that study's gates. |
| `experiments/titan_model_soup_mini_v1/score.py` | Read paired-report, source-correct and parent-ledger contracts. Preserve their semantics. New matrix/logit/cluster-bootstrap helpers are needed for successive skill snapshots; they do not replace the existing retention suite or certify an external scorer. |
| `experiments/fresh_dual_skill_instrument_v1/gpu_lease.py` | Read entire lease. Reuse its context manager, lock location, process checks and token-checked release, with immediate refusal rather than a long wait. |
| `experiments/module_disjoint_composition_v1/run_p6.py` | Read construction/scoring lease scope; native hash/inventory/load/step work stays within the lease. |
| `experiments/fresh_dual_skill_instrument_v1/score_arms.py` | Read existing fixed-template/native evaluation integration. Existing banks are retained; collecting matched generation records for this new study is still a host task. |
| Parent experiment tree | Listed delta geometry, functional additivity, interference, module-disjoint and Titan/Opallo directories. A listing is not a line-by-line audit of every experiment. No claim of superseding those implementations. |

Small streaming SHA/strict-JSON helpers in the new package are deliberately
local, to keep the offline core independent of experiment-specific top-level
imports such as `score.py`'s `from soup import ...`. No dataset generator, native
model class, full-weight merger, or replacement GPU lease has been copied.

The remote head's OSIDE-001 commit records a NULL module-route-versus-scalar
comparison. This is context for a falsifiable study, not evidence that CLARE works.
The supplied earlier cosine (~0.0022), support (28.5%), rank and norm observations
are motivation only, not remeasured here. In particular, neither 0.8035 nor 1.846
is used to fit or predict retention. Geometry is recomputed from each actual delta.

## Existing corpus: verified source, unavailable native bytes

The parent does not contain MAXRL-COMP-002. The accessible source was inspected
in this same repository at `a97df24c2c7dd062985942ead8c3375bf20c66b9`, under
`experiments/maxrl_composable_skills_v2/`, from the separate
`research/maxrl-composable-skills-data-v2` lineage. Read README, corpus build/audit,
atomic manifest writer and canonical schema/hash definitions. The rendered data
is an external delivered corpus, not Git blobs; it was not downloaded or copied.

Exact published full-data manifest SHA-256:
`86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44`.
`data.py` opens only this manifest, `manifests/atomic_<skill>.json` and
`canonical/train.jsonl`, checking bound file sizes/hashes, membership and prompt
hashes. It never opens `oracles/` or evaluation-answer files. Synthetic fixture
schema tests do NOT constitute an audit of the unavailable full corpus.

Source supports R/U/S/P atomic training, separate RU/US/SP bridges, and held-out
RS/RP/UP/all triples/RUSP. Supplied evaluation banks are EXPOSED, not sealed.
General-retention enrollment should reuse existing guard banks and recorded base
predictions, not invent a new arbitrary suite or assume all base answers correct.

## CLARE adaptation, not a vision reproduction

Primary reference: Fu, Lou, Liao and Yu, *Scalable Class-Incremental Continual
Learning via a Sparsity-Based Framework*, arXiv:2609.17026 (2026).
https://arxiv.org/abs/2609.17026
https://arxiv.org/html/2609.17026v1

Adapt: sparse movement discovery on unused coordinates, selection, reset to the
pre-discovery shared state, ordinary-loss retraining and frozen allocation.
Do not import its classification-head/task assumptions. V1 uses explicitly bound
**direct FP32 parameter coordinates**, not an unqualified claim that disjoint
LoRA factors imply disjoint effective weight updates. Only a bounded eligible pool
is trainable. The smoke ceiling is not a declaration of the final scientific pool.

Deliberate difference: exact top-k with canonical deterministic tie breaking.
A threshold including all tied values can exceed the intended quota. Our
half-up rounding plus clipping/minimum-one and exact k are specified and tested;
D and E get identical coordinate counts. Continuous 1-rho^4 capacity percentages
are approximations, not exact finite-pool accounting.

## Hardware and remaining gaps

Actual inherited pinned base is a 28-layer, width-1024, tied-head Qwen3-formatted
Supa export with 596,049,920 coordinates, not an assumed 12-layer checkpoint.
Native code reuses the parent model/tokenizer hashes and verifies that inventory.
No native model, tokenizer or corpus has been loaded in this container.

Additional conservative guards: two entry RAM checks >=7.4 GiB, worker RAM
>=6.0 GiB, bounded coordinate pool, conservative VRAM estimate, and the existing
exclusive Windows lease. These are new admission restrictions, NOT a claim that
those RAM constants were defined in the parent lease. No inherited threshold is
lowered. The final native loss, module pool, response-token/EOS masks, optimizer
memory peak, real runtime and actual retention collector remain unqualified.
