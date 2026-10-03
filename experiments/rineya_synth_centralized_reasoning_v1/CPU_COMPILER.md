# CPU compiler milestone — 2026-10-03

**State: CPU_MECHANICS_IMPLEMENTED_NATIVE_UNQUALIFIED.** This implements the
owner's bounded next step for SYNTH-CRS-001. It does not answer whether condition
diversity teaches transferable skills. That requires the registered M0/M1/M2
student comparison and fresh-family evaluation with prior-correct preservation.
The founding manifest, receipt, CRS-001, and other experiment records are retained.

## Implemented boundary

| File | Responsibility |
| --- | --- |
| [rs_contract.py](rs_contract.py) | Strict public schema, bounded parsing, four surface renderings, and final-answer format |
| [rs_solver.py](rs_solver.py) | Direct R chain traversal and cardinality/sum dynamic programming with every S witness |
| [rs_verifier.py](rs_verifier.py) | Independent R terminal closure and exhaustive S combinations from exact student-visible messages |
| [task_compiler.py](task_compiler.py) | Hashed source registry, parent components, deterministic interventions, record validation, allowlisted transient exports |
| [audit_cpu.py](audit_cpu.py) | Agreement, answer-dependence/invariance, deliberate negatives, arm allocation, tests, and content-hash receipt |
| [tests](tests) | Registration, semantic truth tables, malformed inputs, lineage, and export-boundary checks |

The reference paths share the public parser and schema, but share no solving
code or candidate answers. The verifier does not import the compiler or primary
solver. Independent bit-mask truth-table tests cover 5,645 finite S tasks, rather
than using either solver's preferred answer as ground truth. Common parser bugs
remain a possible correlated failure; hand-specified schema/rendering tests
address that risk without claiming universal verification correctness.

## Public semantics and bounds

Semantics follow the read-only public
[MAXRL v2 contract](../maxrl_composable_skills_v2/contracts.py), version `2.0.0`,
inspected at source HEAD `23cebe93a8684302eda3a114337482b1c95a26ba`.
Its SHA-256 is `6f04b61a31e07c23b4621cfb850cbf4711d8cc95e2bcf4bad28a328bf4d587e9`.
No existing data bank is imported.

- R returns **all distinct item indices** reached by every requested alias.
  Duplicate requests and convergent chains collapse. Unrequested broken links
  and cycles are irrelevant. A broken requested chain invalidates the whole
  request; partial results are incorrect.
- S chooses exactly `k` distinct indices over concrete initial values. Equal
  values at different indices are distinct items. Every satisfying subset and
  ordering is valid. There is no privileged teacher-selected subset.
- Public bounds: 6–15 items, values from −100 to 100, 1–80 links, 1–n requests,
  alias names matching `[a-z][a-z0-9_]{0,19}`, `k` from 1 to min(5,n), and targets
  from −50000 to 50000. R contains no selection; S contains no lookup. Updates
  and precedence are rejected. RS is supported only for evaluation fixtures.
- The historical output is `{"answer":[indices]}` or `{"answer":"UNKNOWN"}`.
  The audit distinguishes `insufficient_evidence` for unresolved requested R
  references from `impossible` for exhaustively disproved S selection. Both have
  no valid list and use the same historical sentinel. Ambiguous duplicate JSON
  definitions are malformed input, rather than a new branching-reference rule.
- JSON duplicate keys, extra fields, booleans in integer positions, floats,
  nonfinite values, altered rules, overlength prompts, and invalid certificates
  are rejected. The parser has a 65,536-byte ceiling; this is not a model window.

Only solved examples enter transient training exports. Unsolved outcomes and
incorrect candidates are audit controls. Intermediate supervision is unsupported:
extra `steps` fields are rejected, not interpreted as verified reasoning traces.
The deterministic generator selects the first lexicographic valid subset as its
canonical target; the certificate verifier accepts all valid alternatives.

## Fixture intervention and split policy

The default audit constructs 48 local parents from seed 1003, evenly divided
between R and S. This fixture generator uses 6–12 items with distinct sampled
values, R with two requests and depth 1–4, and S with `k` 1–4 and a constructed
feasible target. These are engineering fixtures, not a sampled training corpus
or a completed curriculum. Full public bounds are exercised by solver tests.

Each parent owns a content-hashed base world, provenance, attribution, and a
license/provenance sidecar identifying locally constructed repository fixtures.
Parent worlds must match at least one declared source's content. All shared
source IDs, shared source-content hashes even under different IDs, declared
families, and conservative structural near-duplicate fingerprints connect
parents transitively. The frozen planned intervention footprints are checked
before allocating whole components to TRAIN/DEV/eval_fixture (60/20/20 hash
buckets). Accepted rows are constructed only after that catalog is frozen.
Changing the catalog can change assignments; freeze its hash before reuse.

R fingerprints ignore aliases, duplicate requests, irrelevant numeric values,
and unrequested distractors. S fingerprints ignore item permutations. This is a
conservative screen with possible over-grouping, not a proof that all related
worlds have been discovered. Generated variants are checked again for collisions
across splits; any collision rejects the collection before export. External
source dependency completeness still needs a separate source audit.

| Arm | Four fixtures per parent | Declared answer effect |
| --- | --- | --- |
| M1 | Same semantic world in four prompt templates, including JSON-block and field-line renderings | Original certificate remains valid; semantic world unchanged |
| M2 R | Baseline, extended depth, changed endpoint, added request | Depth preserves the answer; endpoint/request interventions invalidate the old answer |
| M2 S | Baseline, changed target, changed cardinality/target, added item | Target/cardinality interventions invalidate the old answer; added item preserves the original certificate |

`preserve` means the original certificate remains valid. An added S item may
create additional valid subsets or change which subset is lexicographically
first. The audit does not mistake that for full solution-set invariance.
For each `change` intervention, the original certificate must pass the original
visible task and fail the changed visible task; both paths must recover valid
changed-task solutions. Surface variants must retain the entire parent spec.
Unsupported amplification (e.g. no spare R endpoint, alias/request capacity,
all-equal S values with no alternate target, or no extra item capacity) raises a
bounded error; no thresholds are relaxed and no rejected parent is silently
replaced. There are no stochastic teacher proposals in this milestone.

M1/M2 share source parents, split assignment, per-parent exposures, skill
allocation, answer format, rendering allocation, and acceptance policy.
They do **not** yet match measured training tokens, difficulty or answer
distributions. Equal row count is not evidence of an equal training budget.
Current fixtures bind each M2 intervention to one template, and the S
cardinality fixture constructs a solution at the first indices. Before native
corpus admission, cross conditions with renderings and balance answer positions
and difficulty strata under a frozen allocation; those cues are not a valid
transfer curriculum. The fixtures validate mechanics rather than approve data.
RS can only be recorded under `EVAL_RS` in a non-TRAIN component; both the
record validator and training export infer skills from visible semantics and
reject forged atomic labels. The default audit builds only atomic fixtures.

## Reproduce and preserve evidence

From the repository root, Python 3.10+ and standard library only:

```powershell
python experiments/rineya_synth_centralized_reasoning_v1/validate_manifest.py
python -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/tests -v
python experiments/rineya_synth_centralized_reasoning_v1/audit_cpu.py
```

To save another receipt, supply `--output` with a **new** JSON path inside this
package's `evidence` directory. Existing receipts cannot be overwritten. The
receipt records source HEAD plus current file hashes, deterministic catalog/row/
attempt hashes, explicit negative examples, transient export hashes, and test
output. Variable timestamps/timings are not part of the deterministic artifacts.
No training JSONL, model, network call, GPU operation, or teacher service is used.
All CPU `eval_fixture` labels are exposed development evidence and cannot be
reused as sealed confirmation or called a fresh model result.

## Remaining decision and launch gates

The study remains unchanged base M0 versus verified surface-varied M1 versus
verified condition-diverse M2, with RS held out for transfer. Before committing
to a run, freeze the **exact trainable parameter set and tokenizer-measured
non-padding student-training token budget**. Training cost is not meaningfully
estimable while those choices remain unset. Optimizer, schedule, masking,
answer format, source-parent allocation, replay, tokenizer/window, matching
tolerances, rejected-parent replacement, and checkpoint selection must also be
registered consistently. The fixture exporter is not a tokenized native exporter.

Fresh families, disjoint prior-correct guards, RS transfer cases, sample size,
seed-aware parent-cluster statistics, minimum useful gain, maximum retention
loss, actual host resource admission, checkpoint/tokenizer hashes, numerical
qualification, and explicit run approval remain open. M2 must beat M1 on fresh
families while preserving prior-correct behavior. Gains confined to familiar
templates do not establish transferable skill acquisition; RS transfer remains
a prespecified separate endpoint. No training or native comparison has run.

After the owner delegated parameter/budget selection, the
[first pilot decisions](PILOT_DECISIONS.md) froze unchanged dense Titan, rank-8 q/v
LoRA (1,146,880 trainables), and a 131,072 non-padding-token ceiling per adapted
arm. This supersedes the earlier unset choices above for the first development
screen only. Actual token scheduling, physical qualification and launch readiness
remain unestablished.

Preserve null findings, failed audits, and restart conditions. This milestone
does not retire or replace SYNTH-CRS-001, CRS-001, or COMP-RS, and does not open
CHECK/FINAL or alter the founding launch flags.
