# Every-row supervision contract

Rubric RIQF-RUBRIC-001. No row enters a research release with a pending or negative
review. Production training eligibility has additional gates.

## Roles and boundaries

The generator writes candidates. The independent verifier implements its own
public parser and algorithms. The semantic reviewer solves from public messages
without targets/world/derivations. The adversarial reviewer independently attacks
format, ambiguity, authority, missing evidence and plausible shortcuts. A retention
auditor tracks protected skill coverage and training-safe replay. The release
controller owns publication. Never assign the generator to review its own rows.

Managed agents currently share a model family. Their separation is procedural;
they can share systematic errors. Automated public parsing of every byte is not
human linguistic review of every sentence. Receipts state which review was
performed and its limits. No fabricated human review, calibrated grader, tool
execution, independent provider identity or cryptographic attestation is allowed.

## Row checks and defect severity

Critical defects: wrong/unsupported answer, missing evidence, leaked private gold,
forbidden source/split overlap, broken label/EOS boundary, or changed source bytes.
Major defects: ambiguity, wrong instruction scope, incomplete source rights,
misleading length claims, superficial shortcut dependence, unreasonable target,
or an untested verifier. Critical/major findings block their entire parent group.
Minor wording/style concerns may be repaired, but the changed row must be reviewed
again. Irrelevant distractor prose still needs bounded, typed QA; exclude malformed
records or accidental authority contradictions.

Every row needs: full public parse-back; independent solution; strict target
contract; at least malformed/extra-field/duplicate-key/type-confusion/wrong-answer
controls where applicable; a substantive evidence-changing control; complete
provenance; parent/split invariants; final-tokenizer measurement; and two assigned
row reviews. Absence of an applicable check is recorded explicitly, not counted
as a pass. Natural-language/code families need specialized verifiers and rubrics.

## Blind reviewer packet

Only row_id, row_sha256, family, profile and public system/user messages enter the
packet. No target, private world, generator reasoning, previous votes or model
results. Public text may contain deliberately hostile quoted instructions;
reviewers treat them as dataset content and assess the actual trusted contract.

Assignments are frozen before review and bind role, agent ID, packet SHA256,
rubric version and exact row IDs. Every receipt has:

```
row_id, row_sha256, role, reviewer_id, packet_sha256, rubric_version,
reviewer_model_identity, blind_answer, disposition, findings,
evidence, method, limitations
```

`blind_answer` is a strict answer JSON string, independently produced from the
public prompt. `findings` is a list; evidence identifies actual public record IDs,
conditions or spans and reasoning. An accept receipt requires findings=[] and an
independently correct answer. Repeating a generic accept phrase does not establish
semantic quality. Store reviewer helpers and output files in `reviews/`; keep
their byte hashes in the release. Tool-call transcript in this chat supports the
local execution history; file hashes alone cannot authenticate an actor.

## Repairs, controls and adjudication

Never overwrite candidate rows or old receipts. New prompt/target/specification
bytes create a new batch revision; source code changes also invalidate the batch
producer identity. A repaired row keeps its parent ancestry. Reassign both reviewers
and rerun public parsing, solution, mutations, tokenization and contamination.
If three repair attempts fail, quarantine the parent and investigate its family.

Before broader prose release, build a separate CAL reviewer-control bank containing
known correct examples and seeded flaws: missing support, subtle arithmetic error,
ignored exception, instruction violation, fabricated citation and leaked answer.
Blind graders to which are flawed. Estimate false acceptance and false rejection,
with uncertainty and defect-specific results. The current programmatic negative
controls exercise verifiers; they do not calibrate language reviewers.

## Continuous cycle and concurrency

Use `scripts/cycle_lock.py` to claim an exclusive dataset-cycle lock. It uses
exclusive file creation; another cycle exits without work. The operator may remove
a stale lock only after checking that its owner has ended. Release the lock at the
end of the actual cycle. No lock permits shared branch writes or model jobs.

Every scheduled wakeup reads the plan, quality policy, prior cycle status and
receipts. Select one quality improvement; use subagents with owned paths and no
shared edits. Work in immutable new candidate/audit/profile/review/release folders.
Do not import legacy whole-corpus validators that open reserved banks. All
new sources enter the registry before generation. A DEV failure can inform a task
class; never train on the failed prompt, answer or close paraphrase.

Notifications report meaningful new quality results, failures or required user
action. Quiet unchanged runs need no user-visible status. Every cycle records
what actually ran, attempted/rejected rows, changed files, review coverage and
unresolved gates. Continuous quality work is ongoing research, not automatic
training or a guarantee of eventual gain.
