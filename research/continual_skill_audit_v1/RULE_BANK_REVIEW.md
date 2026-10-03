# Prospective rule worlds: review and intended use

No model has been run on either bank by this task. Both are visible development assets;
neither is a sealed confirmation set. The current training pilot remains unchanged.

## Preserve v1 as a failed construction

The first draft contains two perfect label shortcuts. Independent peer review found that
the register sufficient pair always has matching add/mix-value parity, while the insufficient
pair differs by one and has opposite parity. This classifies sufficiency without executing
the transition. Our subsequent review found that the prerequisite sufficient extra-add bit
is always the required pre_on bit, and the insufficient extra bit is always pre_off.

`rule_worlds_v1/SHORTCUTS.json` reproduces 100% accuracy for these features on both splits.
The original JSONL, build receipt and `builder_snapshot.py` are preserved. Do not use v1 to
claim compositional skill acquisition.

## V2 changes, before model use

- Register equivalent operations include shifts by multiples of 97. Inequivalent operations
  use symmetric, nonzero even offsets, matching the parity feature within each label pair.
- Prerequisite extra bits are neutral already-set/already-clear bits, distinct from both
  precondition masks. Their add masks have identical bit cardinality.
- Ledger equivalent and inequivalent multipliers come from the same support and have matched
  parity. The insufficient alternative is no longer always the next larger multiplier.

V2 still contains 696 rows across 72 worlds, with 16 train and 8 dev worlds per family.
Its 464 train rows and 232 dev rows contain direct transition, explicit-state comparison,
and joint two-event uncertainty tasks. All derivatives stay together in their world/split;
exact semantic duplicates across worlds are rejected. The build receipt binds the generator
and both JSONL files. Original data were not overwritten.

The detected register and prerequisite feature classifiers now score exactly 50% on both
splits, as does the ledger parity classifier. The strongest tested ledger magnitude cue is
62.5% on 16 dev joint rows; this small finite bank is not proven free of nuisance cues.
These are same-bank optimistic feature diagnostics, not generalization estimates or an
exhaustive shortcut search. Before execution, the remaining bounds must be recorded with
the intended endpoint rather than silently ignored.

Five generator tests pass, including an independent set-algebra oracle over all 256 bitmask
states, explicitly decisive precondition/order checks, register/ledger examples, grouping,
determinism, and regression checks for the identified shortcuts. Rendered examples from each
family have been read. There are still substantive limitations: negative register increments
and modulo-equivalent alternatives change task difficulty; templates differ only by a prefix;
the explicit-state comparison component is deliberately easy. These require model qualification.

## How to run a discriminating next experiment

1. Qualify source and specialist competence on **direct transition** and **boundary** components,
   separately by family. Count both exact state answers and discriminative UNKNOWN behavior.
2. Train component A on transition rows and component B on boundary rows, keeping all world
   derivations in the declared split. If the base already solves B, call it a preexisting skill.
3. Test joint rows without training on them for the compositional transfer arm. Separately
   train a joint-supervised positive control on **train** joint rows to assess reachability.
4. Condition joint analysis on whether each arm can solve that world's two direct event
   components and its boundary comparison. Include the unconditional joint score as well;
   conditioning must not hide failures or silently redefine the test population.
5. Compare single modules, naive merge, sequential orders and any proposed protection method
   under a fixed budget. Measure both specialist retention and held-out joint behavior.
6. A joint zero with failed components or a failed positive control is inconclusive about
   composition. It is not evidence that the model's capacity is exhausted.

Do not mix the generated train joint rows into component-only training and then call the
joint result zero-shot composition. Do not treat development counts as confirmation.
