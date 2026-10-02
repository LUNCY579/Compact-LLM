# Centralized system design and method boundaries

**Status:** proposal only; all interfaces below are design contracts, not code.

## Paper anchor, not a Rineya result

Parallel Power Tempering couples replicas at different sequence-level powers
using corrected exchanges. The paper distinguishes tokenwise temperature
proposals from the whole-sequence target and uses a common fixed horizon with
EOS padding to permit termination decisions to be revised. Its local proposals
require forward/reverse probabilities; a finite run is not a guarantee of exact
equilibrium sampling. See [P1, sections 2-3 and appendices B-C](SOURCES.md).

For a prompt x, fixed horizon T, response record y, and frozen checkpoint theta:

    ell(y) = sum of base-model log probabilities through EOS or horizon T
    pi_alpha(y | x, T) is proportional to exp(alpha * ell(y))

Post-EOS deterministic padding contributes zero log probability. Do not replace
ell with average token log probability or silently omit EOS: those define a
different scoring convention. For replicas i and j the log swap acceptance is:

    min(0, (alpha_i - alpha_j) * (ell(y_j) - ell(y_i)))

The future local-move implementation must retain the full Metropolis-Hastings
proposal ratio. After exchanging records, any proposal score tied to a power
must be recomputed or retrieved under the receiving power, not reused blindly.
Restart positions must not silently depend on realized response length. These
are implementation review requirements, not evidence that this package already
implements the paper. The reference equations and conventions are from P1.

## Proposed centralized integration

Place a bounded search operation behind a controller interface, not above the
entire system. The proposed data flow is:

```text
immutable request + fixed evidence snapshot + explicit budget
                         |
                 centralized controller
                         |
              frozen-model search operation
                         |
      trajectory store <-> probability scorer <-> local refinements
                         |                         and replica exchanges
                 label-blind readout
                         |
              answer + complete cost receipt
                         |
              isolated evaluation / verifier
```

The controller may later choose direct answering, independent resampling,
retrieval, bounded interacting search, or abstention. First test PPT with routing
fixed; otherwise routing changes can confound a sampler comparison. Keep task
evidence fixed for the first experiment. A later tools-enabled protocol must
state exactly which tests or evidence are available at inference and give every
arm the same access. Hidden evaluation labels are never an inference tool.

## Proposed component contracts

| Component | Required contract |
|---|---|
| Request envelope | Stable item ID, prompt hash, evidence snapshot, checkpoint identity, budget and protocol hash; no gold answer fields |
| Frozen scorer | Base token log-probabilities, sequence score, tokenizer/EOS conventions and qualified precision/cache identity |
| Trajectory store | Full fixed-horizon record, realized length, proposal provenance, replica identity, state fingerprint and work ledger |
| Replica scheduler | Fixed declared power ladder, independent random streams, bounded local updates and declared exchange schedule |
| Readout | Predeclared highest-power terminal-state or other label-blind selection rule; never best-known-correct selection |
| Budget controller | Count initial generation, accepted and rejected proposals, rescoring and recomputation; stop before exceeding limits |
| Evaluation boundary | Labels available only after outputs are frozen; separate coverage, selection, retention and abstention metrics |
| Result receipt | Per-item output or failure, full costs, time/memory peaks, seeds, hashes and diagnostic transition statistics |

A direct-answer fallback must be budgeted, deterministic under the frozen
policy, and recorded. Do not hide a failed expensive search by reporting only a
successful fallback's costs. Gold-based stopping or picking the best seed is
not permitted. Never treat a likelihood score as proof of correctness.

## Complementary interfaces, not replacement claims

| Existing track | Relationship proposed for PPT-001 |
|---|---|
| CRS-001 | Optional search operation within shared evidence/procedure infrastructure; preserve CRS's original research question |
| Verified resampling | Reuse separately reviewed evaluation/qualification patterns; retain independent sampling as a strong baseline |
| Recurrence/TAPS | Different computation axis; isolate external trajectory search before any factorial combination |
| MiMo/post-training and retention | Keep weights frozen in the first pilot; test whether decoding changes behavior without claiming new weight-level skills |
| Skill composition and function-space work | Separate capability availability from elicitation; do not reinterpret composition results through sampler outcomes |
| SYNTH-CRS and harness distillation | Potential consumers of independently verified traces after a sampler win; compare against equally verified independent-sampling data |
| Retrieval and cache research | Fixed evidence and qualified cache initially; no native-context-extension or compression claim from this sampler |

These are planned relationships, not imported modules or demonstrated synergies.
The resampling and SYNTH-CRS snapshots are external branch references, not a
claim that their code has been merged into this base.

## Memory and practical limits

The proposed 0.6B pilot should share one weight instance and execute replicas
serially or in small batches on a qualified host. Multiple trajectories still
require storage and computation; serial scheduling saves no total work by
itself. Do not assume the owner's historical 8 GB GPU or available host memory
constitutes a current admission receipt. Measure peak GPU memory, host RAM,
wall-clock time, forward-token work and decoded proposal tokens independently.

Start with short reasoning contexts. Long-evidence integration, adaptive
routing, ladder tuning, speculative execution and distillation are later
questions. No launch budget or native adapter is supplied by this document.

## Interpretation boundaries

When candidate coverage is high but the returned answer is wrong, selection is
a plausible target for investigation. When no tested candidate is correct,
acquisition or evidence access may deserve attention, but finite failed search
cannot prove capability absence. More concentrated likelihood may favor a
wrong answer. A valid transition kernel does not ensure useful finite-budget
mixing, and better mixing does not ensure better task accuracy.
