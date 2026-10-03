# Proposed architecture: a data-production loop and a reasoning loop

This is a design, not an implemented runtime. "Centralized" means coordinated
shared interfaces, evidence, verification, and accounting around compact Rineya
models; it does not imply a new neural architecture or a deployed service.

**2026-10-03 addition:** the bounded offline CPU compiler/verifier subset is now
implemented; see [CPU_COMPILER.md](CPU_COMPILER.md). The architecture below remains
the founding proposal. The online runtime and native/training integration remain
unimplemented and unqualified.

## A. Offline learning-data production

1. **Source registry:** immutable, content-hashed source/world versions with
   provenance, license/attribution metadata, and parent-family ownership.
2. **Task compiler:** creates authorized atomic tasks and controlled variants.
   Split entire dependency-connected families before amplification. A harder
   variant must not smuggle a held-out skill combination into training.
3. **Candidate generator:** optional bounded teacher proposals or deterministic
   target construction. Record generator revision, prompt, seed, and failed
   attempts. No provider is selected or called by this registration.
4. **Independent verifier:** solve from student-visible information, not from the
   generator's answer. For initial R/S tasks, use independently checked reference
   resolution and finite subset enumeration. Check intermediate states before
   accepting process supervision; correct final output alone is insufficient.
5. **Accepted/rejected record store:** retain provenance and rejection reasons.
   A frozen acceptance policy is shared between experimental arms. Missing
   evidence, contradictions, malformed certificates, and budget limits get
   distinct labels, not an undifferentiated "correct" flag.
6. **Student exporter:** emits only allowed prompt fields and the selected target.
   Teacher-private hints, solver objects, test feedback, gold metadata, and source
   licensing sidecars never become accidental model inputs. Fit the actual
   tokenizer and qualified window; no silent truncation or guessed token counts.
7. **Training/evaluation boundary:** only approved TRAIN examples and disjoint
   replay enter adaptation. Hold selection data separate from fresh confirmation.

The teacher may know an answer while constructing a candidate explanation.
Every accepted intermediate step must nevertheless follow from what the student
can observe. Otherwise this is answer rationalization, not verified supervision.
A model grading its own free-form explanation is not independent verification.

## B. Online system-assisted reasoning (later, separate)

A frozen compact reader/controller retrieves permitted evidence, constructs
instance-local working state, calls typed primitive operations, verifies visible
claims, and returns an answer or a labeled abstention. A frozen procedure library
may help route operations. It is the CRS-001 system interface, not a shortcut
that gives an experimental arm access to gold answers.

**Two verifiers must not be conflated:** the deployable verifier checks permitted
source information; the isolated evaluation scorer can see hidden reference
answers. A deployable controller must never call the hidden scorer. Failure to
verify does not, on its own, prove an answer false; missing evidence is different
from demonstrated impossibility. Reject instructions embedded in retrieved data
as instructions to the controller. Reset all instance state between test parents.

For an online comparison, keep reader, retriever, primitive tools, visible
information, decoding, verifier, and resource ceilings matched. Compare frozen
validated procedure memory with no persisted procedure memory, as CRS-001
specifies. Record human-supplied versus model-discovered procedures separately.
Neither case is the first data-diversity experiment described in this package.

## Interfaces to existing research

| Track | Complementary interface | Claim that does NOT transfer |
| --- | --- | --- |
| CRS-001 external reasoning | Verified procedure records and future distillation targets | System gains do not establish student learning. |
| Centralized MiMo/Titan training | Qualified model binding, retention enrollment, native run receipts | A new corpus does not waive launch gates. |
| MAXRL composable skills | R/S task semantics, parent lineage, independent verification | Existing exposed banks do not become sealed; new demonstrations are not on-policy RL. |
| Skill deltas / independent coefficients | Later separately trained donors and retention/composition assays | Data gains do not establish lossless delta addition. |
| Retrieval / cache / long context | Exact evidence references, coverage and source-length accounting | Short-context supervision does not validate a larger native window. |
| Recurrence / TAPS / compute depth | Future tasks requiring observable iterative refinement | New data does not establish useful recurrence or a learned scheduler. |
| Abstention | Distinguish unsupported, contradicted, and impossible cases | Printed confidence is not calibrated confidence. |

## Cost and security boundaries

Account separately for data production (including rejected attempts and paid
teacher calls), online system execution, and student training/inference. Show
cold-start costs and any amortization denominator. Do not claim total compute
efficiency solely from fewer student-training tokens.

No automatic code execution from generated text. A future executor needs an
allowlist of typed operations, validation, resource limits, and no ambient
network/filesystem authority. Private source data must not leave the host through
a teacher API without separately reviewed consent. These are implementation
requirements, not controls implemented by the manifest validator.
