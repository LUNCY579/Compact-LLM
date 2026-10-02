# PPT-001 controlled experiment proposal

**Design record, not a frozen executable protocol or authorization to launch.**
Every numerical setting below is a proposed Rineya pilot choice, not a paper
result or a demonstrated hardware limit. A later implementation must freeze a
separate dated protocol before access to held-out evaluation outcomes.

## Questions and preregisterable hypotheses

- **H1 / interacting-search value:** exchanges improve returned-answer accuracy
  or reduce work for a fixed accuracy relative to identical uncoupled replicas.
- **H2 / practical value:** the selected interacting-search arm improves the
  accuracy-cost tradeoff against the strongest simpler development-selected
  baseline, including independent sampling with a label-blind readout.
- **H3 / preservation:** decoding changes do not create an unacceptable loss of
  retained abilities, valid abstention, output validity or reliability.

A positive H1 does not imply H2 or H3. No hypothesis has been tested here.
All native stages below remain unimplemented and unlaunched.

## Stage 0: mechanics and exact-checkpoint qualification

First implement a tiny enumerable model with known fixed-horizon targets.
Calculate transition matrices and verify normalization, stationarity and local
and swap detailed balance where applicable. Do not require a composition of
invariant kernels to be reversible. Test full restart support, early EOS,
post-EOS self-transitions, lengthening after a restart before EOS, rejected
proposals, exchange orientation and power-specific proposal scores after swaps.
Demonstrate that a finite initialized chain need not already have its stationary
distribution. Include a deliberately broken truncation implementation as a
negative control, never as the native default.

Then qualify base log-probabilities on the exact frozen Rineya 0.6B checkpoint.
Compare cached incremental scoring against a reference full-sequence path,
including EOS and swapped states. Pin tokenizer, template, dtype, masking,
attention implementation and numerical tolerances. Matching the final argmax is
insufficient. Do not substitute an official upstream checkpoint for a Rineya
checkpoint without declaring a separate arm. Keep protected banks closed.

**Gate:** mathematical checks, native scoring qualification, host admission,
provenance and an explicit current launch grant must all exist before a native
pilot. Registration tests in this founding commit satisfy none of these gates.

## Stage 1: opportunity diagnosis on development data

Proposed screen: 32-64 public development problems, eight independent candidates
per problem, and short bounded completions. Use arithmetic/symbolic reasoning,
state tracking and selected acquired-skill probes, with separate answerable and
unanswerable strata. Select sources and check licensing/leakage before use.

Report candidate coverage, returned-answer accuracy under label-blind voting and
likelihood readouts, and likelihood-correctness alignment. Include valid and
unnecessary abstentions separately. Evaluator labels may measure diagnostics
only after candidates and selections are frozen. The sampler, controller and
selection rules may not inspect those labels.

The observed bank is a diagnostic, not a ceiling on every possible search.
Do not exclude problems from held-out evaluation because their gold outcome is
unfavorable. Freeze any development-based routing rule before held-out use.

## Stage 2: comparison arms

| ID | Intervention | Purpose |
|---|---|---|
| A0 | Ordinary checkpoint decoding | Operational single-generation baseline |
| A1 | Development-tuned low-temperature decoding | Strong inexpensive decoding baseline |
| A2 | Independent sampling with declared voting/likelihood readouts | Simpler competing use of compute |
| A3 | Fixed-horizon single-chain refinement | Isolate iterative revision |
| A4 | Multiple fixed-horizon chains, exchange disabled | Control replica count, power diversity and local proposals |
| A5 | Same as A4, exchange enabled | Isolate trajectory interaction |

For A4 versus A5, keep initialization, ladder, horizon, local-move schedule,
readout and budgets fixed; use separate random streams for local and exchange
decisions so toggling exchanges does not merely consume different local RNG
positions. Report remaining work differences caused by changed trajectories.
A4/A5 should read the terminal highest-power replica initially; changing this
to best-likelihood or voting is a separately declared readout ablation.
Independent-bank voting is not a definition of the sequence-level target.

Initial proposed settings: two or three replicas, a 512-token completion horizon
and very few refinement rounds. Power values, prompt limits, exact round counts,
seed count, runtime caps and resource thresholds are deliberately UNSET in the
founding registration. Select on development data, justify, and freeze before
held-out access. Never infer execution approval from these suggestions.

## Stage 3: held-out evidence and cost accounting

After a qualified development signal, propose 128-256 held-out items and multiple
seeds. Separate split membership by parent/template family where needed; record
exact content hashes and preprocessing. Keep development, held-out scoring and
future distillation training provenance distinct. Do not use sealed CHECK/FINAL
banks under the authority of this document.

Measure accuracy versus total work and versus elapsed time separately. Record
all initial generations, rejected proposals, rescoring, prefix recomputation,
cache transfers, synchronization overhead, padding conventions, timeouts and
OOM failures. Decoded-token counts alone are not a complete cross-schedule
compute comparison. Do not claim latency parity from matching tokens.

Maintain a per-item attempt ledger. Failed, truncated or timed-out items stay in
the denominator with explicit status; show complete-case diagnostics only as a
secondary view. Report accepted local moves, exchange rates and effective
trajectory diversity as diagnostics, not substitutes for accuracy.

Use paired item-level differences and uncertainty intervals with clustering by
problem across seeds. Report per-seed variation. Freeze the primary comparison,
selection rule, minimum worthwhile improvement, confidence procedure and
retention noninferiority margin before confirmatory use; these values remain
UNSET here. A 32-64-item screen is exploratory, not confirmation.

## Decisions without erasing the track

**Promote a configuration** only after H2's matched-budget comparison and H3's
preservation requirements have actual supporting evidence. A win only against
A0, when A2 is better at the same cost, is not practical promotion evidence.
An inconclusive confidence interval remains inconclusive.

**Reject a configuration** for numerical invalidity, budget violations,
label leakage, poor mixing, worsening accuracy or preservation failures.
Preserve the complete evidence and distinguish implementation failure from an
adverse result of a qualified method. Record a simpler baseline win honestly.

**Park the track** when qualified execution cannot be admitted or the current
budget regime is unproductive. Keep the open question, exact tip and restart
condition. Do not automatically obsolete or delete PPT-001 or any related track.

## Conditional extensions, outside the founding scope

Only after the standalone comparison, consider controller routing, long-context
evidence, a recurrence factorial, or verified-trace distillation. For distillation,
compare equal-sized, equally verified data from PPT and independent sampling,
record generation costs, freeze a disjoint held-out split, and report single-pass
student performance separately from teacher search. A future extension does not
retroactively turn the founding notes into an executed experiment.
