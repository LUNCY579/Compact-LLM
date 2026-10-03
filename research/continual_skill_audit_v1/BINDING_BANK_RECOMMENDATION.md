# Richer binding coverage for the next retention experiment

The pilot's source scores 111/112 on simple role-roster development rows and solves all four
variants in 27/28 worlds. This is useful for detecting gross damage but weak coverage of the
historically fragile alias skill. A roster lookup is one relation; the older alias tasks require
following a role reference to an action and then recovering a name among salient distractors.

The closest existing mechanism generator is:

`C:/Users/clclc/BusinessProject/ChatBubble-dmrf/experiments/delta_multi_reasoning_v1/probe/build_alias_probe.py`

Read-only review confirms low/equal/high mention-count conditions, shuffled role and action
sentences, neutral distractor activities, and grouped base/paraphrase/reorder variants.
Its historical `ALIAS_BUILD.json` documents shortcut checks. Use its low/equal **design** as
a starting point for a fresh development generator with new world/entity partitions, not
the spent historical rows as a new selection or confirmation bank.

It is not ready to reuse unchanged. The binding sentence includes “who is not named again
in this account.” That is false in equal/high conditions, which deliberately repeat the
person. Remove that clause consistently in a new generator and qualify the changed
instrument. The same unique sentence structure may itself point to the intended answer;
make target and distractor bindings syntactically parallel and rotate which role is queried.

The historical high condition makes the answer the most frequent person by construction,
so its absolute accuracy does not demonstrate binding. A shared cue can also affect two
models differently if they exploit it differently; byte-identical paired prompts do not
automatically eliminate that construct-validity concern. The records themselves examine
this in `CUE_BOUND_TRANSFER_20260919.py` and `CUE_EXPLOITATION_CHECK_20260919.md`.

The related `build_pres4.py` broadens persons to objects and places and contains fixes for
doubled articles and capitalization. Its current proper-noun object design is not an
automatic improvement: the historical PRES-007 source reached 118/120 object rows, leaving
little dynamic range in that mechanism comparison. Replacing common nouns with distinctive
proper nouns changed difficulty. Neither an old confounded bank nor a ceilinged replacement
should be silently declared the validated general-purpose retention instrument.

A prospective development bank should cross two-hop versus direct binding, equal versus
distractor-heavy mention counts, persons/objects/places, early/late relevant statements,
and answerable/missing-evidence cases. Match phrasing, mention counts and ordering within
counterfactual pairs; move every variant of one world together. Predeclare minimum source
competence and report the observed ceiling. Keep genuine substantive paraphrase templates
separate across train and dev, not merely different template IDs. Inspect rendered cases
and simple extraction/position/frequency/structural predictors before model execution.

This is a reviewed recommendation, not a new model result or permission to alter the frozen pilot.
