# Binding signal: data readiness proposal

This prepares independently written synthetic TRAIN/DEV material for a possible
later experiment. It does not train a model, read a model endpoint, authorize a
worker, alter current LoRA predictions, or reopen a confirmation instrument.
The current high-dose LoRA experiment remains the active priority.

The intervention would teach a missing behavior: return a person's identity only
when the context determines an occupant of the relevant role. A known role does
not imply a known occupant. These are new supervised binding facts and explicit
UNKNOWN targets, not a replay of the historical preservation corpus.

## Original worlds and positive controls

Every base world has six roles, six listed people, four explicit appointments,
one explicitly vacant role and one role with no appointment fact. Two listed
people have no role assignment, preventing a set-complement solution from
identifying a unique missing occupant. A directory of names is explicitly not
an assignment. The duty is assigned to a role; each counterfactual designates
an occupied, missing or vacant role while retaining the same roster facts.

Every context contains both a vacancy statement and an unspecified role, even
when the question is answerable. Presence of a vacancy keyword alone therefore
cannot classify answerability. Roster lines, role lists, person lists and the
duty statement change order deterministically across views.

Each world contributes twelve fixed views:

| Category | Views | Target |
|---|---:|---|
| Occupied-role person questions | 3 paraphrases | Assigned full name |
| Missing-occupant person questions | 3 paraphrases | UNKNOWN |
| Vacant-role person questions | 3 paraphrases | UNKNOWN |
| Role questions | One per occupancy condition, three phrasings | Designated role |

The four categories are equally sized. Always UNKNOWN is correct on only half
the complete bank and fails all positive controls. Role questions remain
answerable in missing/vacant cases, which tests whether the model learned a
person-binding rule rather than blanket refusal. Roles with multiple conflicting
assignment facts are invalid builds; they are not silently relabeled as UNKNOWN.

The oracle operates on structured assignment facts and the designated role. It
does not inspect template text, stored reference strings or condition labels.
Rendering is a separate function. A second, independently specified parser reads
only the rendered user text. It reconstructs role/person domains, assignments,
vacancies, the duty-to-role designation and whether the question asks for a person
or a role. Its grammar is not derived from the formatter's template table. Every
fact and question must parse, and both the parsed facts and the answer computed
from them must match the structured-world oracle. Ordinary re-render equality is
an additional format check, not the proof that evidence reached the policy input.
Tests deliberately make generation and re-rendering share the same faulty
renderer: omitted assignments, swapped question kinds and swapped duty roles all
fail the independent semantic BUILD gate. Structured worlds/query metadata and assistant targets are
gold-side data; a future compiler must put only system/user messages in policy
inputs and supervise the assistant response plus EOS.

## Partition and integrity rules

Default scope is 64 TRAIN worlds and 32 DEV worlds: 768 TRAIN and 384 DEV views.
Root seed17 is fixed. Independent TRAIN/DEV salts determine world sampling;
render-order salts include split, parent world and query. All twelve views of a
world remain in one partition. The partitions have disjoint person names, role
names, action pools and template families. Template-content hashes are checked,
so changing only a family's name cannot disguise reused wording. The shared
system contract is the deliberate exception: it defines one unchanged task.

The builder rejects wrong answers, ambiguous/duplicate role bindings, duplicate
row IDs, missing views, inconsistent facts within one world, cross-partition
world derivatives, entity leaks, template-content overlap and identical rendered
prompts across partitions. Tests use tiny temporary worlds and no tokenizer,
model, torch, GPU or external dataset.

This generator does not open sealed prompts, calibration examples, RCI entity
lists or external evaluation rows as inputs. Accidental contact must still be
checked by the external exclusion gate below. The content is original
self-generated synthetic material with its provenance recorded. Its license field
is `UNSPECIFIED_INTERNAL_RESEARCH`: no licensing or release decision has been made,
and `release_permitted` is false.
`human_data_review` remains `NOT_PERFORMED`; automated checks are not human review.

Before any later training, add the actual experiment's complete exclusion
manifest: both acquisition banks, arithmetic panel, spent long-context/scout
material and any reserved instruments. Check parent IDs, entity pools, normalized
prompt hashes and meaningful text contact. Sealed instruments should contribute
only approved exclusion identifiers/hashes through their custodian; do not open
sealed prompts to invent or revise these examples. Today's build makes no claim
that this external exclusion step has already happened.

DEV is openly inspected development material. It is neither sealed nor a
confirmation instrument. Freeze its bytes and all analysis rules before its first
model read. Any later template edits create a new version with explicit lineage.

## Prospective later model bands — proposal, not permission

For the default384-row DEV bank, each category has96 rows. The following bands
are proposed now, before any model read, for a later candidate trained with this
signal. They must be copied unchanged into that experiment's registration or
explicitly amended before the first model endpoint exists:

| Endpoint | Proposed engineering band |
|---|---|
| Missing-occupant exact UNKNOWN | 72–96 of96 |
| Vacant-role exact UNKNOWN | 80–96 of96 |
| Occupied-person exact answer | 64–96 of96 |
| Role exact answer | 72–96 of96 |
| False UNKNOWN on all answerable controls | 0–10 of192 |
| Both acquisition banks, reported separately | 260–330 of384 each |
| Acquisition cost against frozen matched control | Net no worse than−20 on either bank |

The joint prediction requires both abstention bands, both positive-control bands,
the false-abstention limit and both acquisition requirements. A candidate does
not pass merely by emitting UNKNOWN more often. Report failed predictions as
failed. These are engineering thresholds, not equivalence or noninferiority
proofs. They are intentionally predictions with uncertainty, not prior results.

Every acquisition comparison reports net, discordant rows and exact paired
two-sided p-value. Cross-training-draw acquisition claims additionally need
absolute net at least20; a smaller observed loss cannot establish preservation.
Use paired correctness within each bank, never pair unrelated TRAIN/DEV worlds
or combine acquisition banks. The binding bank's paraphrases are correlated:
its 384 views are only **32 independent DEV worlds**. The row-count bands above
are engineering summaries. For binding inference, pre-register a Boolean
world-level endpoint per category (all three category views correct), and compare
the same 32 worlds by exact paired McNemar. For false abstention, use the
world-level event that none of its six positive-control views emits UNKNOWN.
Per-view paired discordances may be descriptive, but must not be presented as384
independent trials. Acquisition retains its existing distinct-parent instrument.
Report endpoint-specific detectable net at actual discordance; the best possible
two-sided significance threshold is six discordant wins, while actual discordance
can require more. Preserve answerable and UNKNOWN strata separately, including
false abstention. No claim of broad intelligence or confirmation follows from
this small developer-visible bank.

## Compilation and resource boundary

Candidate-v1 has completed tokenizer-only compilation: TRAIN has197064 full
teacher-forced tokens,2384 response-plus-EOS supervised tokens and103 native
update groups; DEV has107859 full tokens and1257 supervised tokens. Neither split
truncates. The compiled-v1/TOKEN_AUDIT.json receipt binds these counts and the
tokenizer assets. Mixture ordering, optimizer/runtime choice, qualification
charges, model-call limits and wall/disk budgets remain pending. The external
exclusion audit is incomplete and the draft remains barred from training. A later
registration must distinguish an engineering one-run signal test from a matched
control experiment that isolates added signal from extra data or compute.

The CLI is build-only and requires an explicit fresh output child beneath
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-signal-prep-20260920`:

```powershell
python research/binding_signal_v1/generate.py --out candidate-v1
```

It writes TRAIN.jsonl, DEV.jsonl and BUILD.json create-only. Candidate-v1 already
exists and must remain unchanged; rerunning that output name refuses. No
training/evaluation permission is carried by a build or tokenization manifest.
