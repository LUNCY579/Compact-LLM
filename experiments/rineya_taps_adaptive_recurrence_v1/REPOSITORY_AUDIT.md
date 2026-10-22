# Pre-implementation repository audit

Remote: LUNCY579/ChatBubble. Audited 2026-10-01 through the authorized GitHub
connector. Shell `git fetch origin` was attempted in a fresh isolated staging
repository and failed DNS resolution. No existing working tree was modified.
All 36 returned branch heads were inspected; the requested branch did not exist.
It was created from integrated `post-training-Mimo` at
`03db218ceebc61a2bc04f2482cf7e094045e01c1` (merge of PR #27), rather than older main
`b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2` or an unrelated experimental tip.
Root tree: `6a1e3b9b16789f7ce211e443654912859dd37ed0`.
Publication uses Git-data commits and non-force branch updates, not a claimed
successful shell fetch/push. Only this new experiment directory may change.

## Read before implementation

Paths below are in ChatBubble at the integrated base unless a tip is stated.

| Work/source | Actual evidence and boundary | Reuse / distinction |
|---|---|---|
| `experiments/anchor_routed_middle_recurrence_v1/README.md`, `AGENTS.md`, `CPU_BUDGET_V2_REPORT.md`, `evidence/00_audit/report.md` | L12 twice: 12/16, same as intact; L12-13 block twice: 13/16, one paired correction/no loss on previously inspected DEV. All 48 DEV generations completed after a preregistered deadline-only change; 19 fidelity checks. NOT independent held-out efficacy. Three L12 applications unlicensed; L27 screen had a preserved RMS failure. | Smallest concrete frozen recurrence host; preserve L0-2, custom Supa head, positions, masks, no KV and native first pass. |
| ARMR `src/armr/hooks.py`, `recurrence.py`, `metrics.py`, `scripts/run_experiment.py`, `run_fixed_recurrence.py`, test inventory and `tests/test_mechanics.py` | Scoped tensor-returning decoder wrappers, exact restoration, cache rejection even with use_cache=False, hidden/logit diagnostics, source/state hashes and unsealed scoring. | Integrate a separate scaling wrapper; never patch inherited code or grant deeper loops from smoke. |
| `research/rineya-compute-depth-elasticity-v1@7a7f7b60e38c18dbeda7e182daf037e993ca8afd`, package README | Offline depth/width, same-prefix, oracle and DEV-policy apparatus; native collector BLOCKED, no measured quality improvement. | Keep how MANY loops separate from how LARGE each update is. No silent sibling merge. |
| `research/ced-looped-flow-paired-host-v1@627253edde9a0e67334034d252e5d9dd9daead9a`, commit/README/host rules | B28 versus source-initialized E24/D4 collector; historical 15 host and 90 companion fixture tests; native capture unmeasured. Exact loader/source, explicit permits, 10 GiB host-RAM capture guard. | Reader/architecture experiment is complementary, not a qualified TAPS host; its permits do not transfer. |
| `experiments/loopspec_v1/README.md` | Exact speculative scheduling apparatus; ordinary 28-layer Qwen cannot self-certify as recurrent; native prefix/cache-depth qualification remains BLOCKED. | No LoopSpec speed claim or cache changes in this experiment. |
| `research/reasoning_posttraining_v1/RESULTS.md` | TRAIN microfit is engineering evidence, not reasoning improvement. Held-out DEV baselines include strong negative results, shortcut-floor corrections and non-independent clusters; protected residual bank belongs elsewhere. | Reuse scorer/split discipline, not its spent training grant or protected data. |
| ARMR Phase-0 pointer to sibling SupaExperiment normalized-gate/GRU and recurrence-atlas work | Historical source audit says these are separate FP16 protocols; the atlas was CPU toy-only. External current code/results were NOT re-executed or exhaustively re-audited here. | Do not turn remembered gate-collapse numbers into newly verified results. Existing gates multiply raw updates; scale gains cannot be claimed to replace those gates. |

## Scientific distinction

A null recurrence outcome does not decide whether the scale was wrong. Conversely,
alpha=1 losing to a different constant is not evidence that adaptation is needed.
Hold operator, weights, first pass, task/scorer and budget fixed. Compare global
fixed alpha, per-example fixed-alpha oracle and bounded per-step oracle before
crediting a causal scheduler. Report null/negative results and controller cost.

A one-extra-pass host can still supply two observations: the ordinary full block
update seeds the EMA, without changing its output; the repeated block is the
first controllable update. For a two-layer block the seed is measured from the
entry of its FIRST layer, not merely the last layer's residual. No extra loop is
silently granted to obtain controller history. More than one extra application
needs fresh host qualification; old failed depth gates remain failed.

## Reusable versus missing

Reusable: established ARMR DEV allowlist/scorer, immutable run conventions,
exact custom checkpoint identity, cache-free model mechanics, scalar diagnostics,
pytest conventions, and baseline restoration checks. Missing in this execution
host: canonical weights, Transformers, adequate RAM, native qualification and
protected-evaluation authorization. CPU apparatus tests cannot fill those gaps.
