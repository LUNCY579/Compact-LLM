# Frozen independent development banks

Each track gets its own generated bank, seed namespace, manifest, scorer hash,
generator hash, and immutable output paths. Nothing reads historical sealed
datasets. The generated bank is deliberately unsealed; unsealed does not license
candidate tuning on confirmatory answers or results.

| Role, per track | Examples | Independent scenarios | Purpose |
|---|---:|---:|---|
| Calibration | 128 | 128 | Engineering / Track B alignment only; 96 fit, 32 scenario holdout |
| Architecture development | 128 | 128 | Reject broken mechanisms and prospective gate grid |
| Confirmatory development | 1024 | 1024 | Frozen candidate efficacy |
| Confirmatory diagnostics | 128 | 0 additional | Linked robustness variants; separate analysis |

There are eight balanced family strata. Every confirmatory stratum has four
structurally distinct templates with 32 independently instantiated scenarios per
template: relational order/spatial/ancestry/containment; causal and arithmetic
counterfactuals; mapping/string/function/list composition; inventory/rate/modulo/
fraction arithmetic; Boolean/set/XOR/quantified logic; joined/dated/linked/noisy
retrieval; everyday grounded QA; and JSON/ordering/abstention/conflict instructions.
Template index 3 in every family appears only in confirmatory development (256
examples), so the unseen-template stratum is explicit. Calibration and architecture
development use templates 0-2 with disjoint scenario identities and contents.

The 128 diagnostics link to 128 distinct primary scenarios. Each track has 30
proposition paraphrases, 28 irrelevant-context insertions, 32 reordered-evidence
variants, 28 entity renamings, four reversed relation formulations (A taller than B
becomes B shorter than A), four longer ancestry chains (three to four generations),
and two counterfactual interventions with changed gold answers. Reordering selects
templates with independently interpretable records; sequential transformation
steps remain intact. Paraphrases rewrite propositions, with no preamble-only
substitutes. These diagnostics never inflate primary N. Insufficiency and conflict
also have dedicated primary templates. Direction, chain-length and intervention
strata are small exploratory probes and cannot establish broad robustness alone.

IDs encode track, role, family, a version/seed identity digest, index and variant;
another prospective seed/version never reuses IDs for changed examples. Randomized instance content
derives from SHA256 of the version, seed and identity. Manifests bind raw JSONL
hashes, the generator source, the strict scoring adapter and its reused legacy
scorer source. Reads reject sealed names before opening files and reject role,
track, counts, file hash, source hash, scorer hash and canonical manifest mismatch.
Writes use exclusive create. Reruns require a new output directory and prospective
version; no failed question may be silently edited after observing model output.

The bridge-facing `alignment_prompts` interface returns only Track B calibration
prompts and identities. Holdout assignment is deterministic at scenario level
(within-family index modulo four equals three). It exposes no answers or scorers.
Track A efficacy labels never enter bridge fitting.

Every original target is constructed programmatically before rendering the prompt.
A separate parser/solver recomputes every answer from the rendered facts and
question, without reading its gold answer, scorer, latent hash, or the builder.
Tests provide hand-authored cases and expected answers for all 32 templates and
verify that consistently corrupting both gold and scorer still fails validation.
The parser implements this controlled grammar; it is not an unrestricted natural
language verifier. Scoring
accepts one exact normalized answer or an exact JSON value. Prompts request only
the answer. Numeric labels are exact strings; extra explanation or a list ending
in the expected number fails. JSON rejects duplicate keys, nonfinite constants,
and Boolean/integer substitutions. Gold-answer consistency with the independent
solver, prompt-to-fact reconstruction, duplicate IDs, duplicate prompt hashes,
cross-role/track latent allocation collisions, calibration fit/holdout boundaries,
and engineering-template exclusion are checked. Each derived variant retains the
source scenario, split and latent allocation even when its intervention changes
the answer. Tests cover strict false positives and frozen manifest tampering.

These are synthetic, English, bounded-context development results. Everyday QA is
grounded preservation, not broad factual knowledge. No external factual corpus is
claimed. The bank has 32 structures rather than 1,024 unrelated task types.
Random instances share a template; small Boolean and ordinary-QA domains can repeat
logical assignments across independently drawn named scenarios. Thus they do not establish population
generalization to all reasoning tasks. Family and held-out-template effects must
accompany aggregate results. Retrieval is moderate length; no long-context claim
is licensed. Open-ended conversation quality is outside this bank's scope.

Primary accuracy analysis uses the 1,024 original independent scenarios. Report
paired wins/losses, exact McNemar, a scenario bootstrap interval, absolute and
relative error reduction, NLL differences and output identity. Linked diagnostics
use scenario means/bootstrap and scenario sign-flip tests; row McNemar there is
descriptive only. Family contrasts are exploratory unless the track protocol
registers them. Confirmatory candidate contrasts receive Holm annotations within
the independently specified track comparison family.

`power_planning.json` records sensitivity at N=256,512,1024,2048 for gains of 1,2,3
points and 5% relative error reduction (1.25 points under a hypothetical 75%
baseline). Discordance assumptions are 5%,10%,20%. This is an unadjusted two-sided
normal planning approximation, not observed power or a guarantee. At N=1024,
80%-power minimum detectable effects are approximately 1.96, 2.77 and 3.92 points
for those discordances. Thus a 1,024 bank can be informative without resolving all
plausible small effects. Multiplicity and correlated variants reduce sensitivity.
