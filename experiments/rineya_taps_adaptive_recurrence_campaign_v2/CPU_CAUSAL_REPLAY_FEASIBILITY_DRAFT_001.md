# DEVELOPMENT causal replay: metadata feasibility only

All 192 DEVELOPMENT cases/96 parents fit the proposed CPU shape and complete-count budget without narrowing. This is **FEASIBLE_METADATA_ONLY_PENDING_NEW_QUALIFICATION**, not a registered worker, native qualification, semantic efficacy result or dispatch permission. The author loaded only the exact offline fast-tokenizer JSON into a new campaign-local cache; no weights, Torch/model/GPU operation, network API, download or environment mutation was used. No validation payload was materialized: only the coordinator's source-bound DEVELOPMENT exports were read.

Inputs are `runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json` (`566e6d6eâ€¦`), `DEVELOPMENT_TOKEN_BINDING.json` (`6529b48bâ€¦`) and its manifest. They bind all rows with split DEVELOPMENT, from frozen bulk bank/binding 002, with no outcome selection. Frozen scorer `bulk_bank.py 32424392â€¦` and imported normalizer `open_bank.py 25db8ad7â€¦` remain unchanged. The metadata-only helper is `causal_replay_metadata.py`; its JSON packet records exact full hashes, source/cache and every proposed target/prefix.

## Exact target and prefix binding

There are 440 canonical answer-token positions. Answer lengths are one token for 80 cases, two for 2, three for 84 and four for 26. Category target counts are signed ledgers 108, deduction 32, program state 94, dialogue memory 78, document reconciliation 64 and answerability/source conflict 64. All 192 canonical strings are strict-correct/format-valid under their own frozen scorer and normalize to themselves. Each standalone answer token sequence decodes exactly to that string.

Every original rendered prompt reproduces its binding 002 IDs using the exact canonical fast-tokenizer JSON. Base input maximum is 232. For answer position i, input is exactly `original_prompt_ids + canonical_answer_ids[:i]`; the current and later target sequence positions are not appended. The largest answer-target prefix is 235, and the full-answer prefix for EOS is 236, bothâ‰¤256. Literal target token values can legitimately already occur in the prompt/earlier gold; the firewall concerns sequence-position construction, not banning equal token values. No truncation or selective exclusion is needed.

The packet also compares joint raw tokenization of `rendered_prompt + canonical_answer` with generation-style prompt/answer token concatenation: zero boundary differences across all192 cases. The specified replay convention nevertheless remains the fixed original prompt IDs plus separately bound continuation IDs; it must not be changed retrospectively. Hash every canonical string, original/extended prefix and input/target sequence before future model work.

EOS is separate: 192 post-answer positions. Canonical chat EOT is 151645, while actual generation accepts both 151645 and 151643. A single 151645 CE/derivative is a canonical EOT-string diagnostic, not probability of the complete accepted EOS set. An EOS-set objective would require an explicit prospective loss convention and its own numerical qualification. Neither EOS loss nor termination should silently replace answer loss.

## Complete counts and conservative forecast

For each target, the proposed mechanism measurement is three fully recomputed fixed-scale forwards at unit/1.01/0.99 and one full-suffix state adjoint at unit. No saved-state counterfactual or full future-gold input is used.

Proposed fresh qualification is 29 model-API attempts/three adjoints: the existing CPU H3 qualification structure 26 APIs/two adjoints plus one late-answer-prefix three-forward/one-adjoint FD check. This addition tests causal construction at the longest late prefix. Exact no-extra/unit/scaled values, max 256 tape/RMS/source/weights/gradient preservation, cache/normal/forced restoration and whole-suffix custom-head FD must qualify afresh. This count is a prospective proposal, not inherited native PASS.

| Plan | Target positions | Task forwards | Task adjoints | Total APIs with qualification | Total adjoints |
|---|---:|---:|---:|---:|---:|
| Answers only |440|1320|440|1349|443|
| Answers plus separately reported canonical EOT |632|1896|632|1925|635|

Observed CPU H3 max 256 costs are 0.9060053 seconds for a value forward and 1.3799383 seconds for a whole forward plus state adjoint. With factor 1.5, each target budgets `1.5*(1.3799383+2*.9060053)=4.78792335` seconds. Qualification pessimistically charges all 29 APIs at the slower whole-adjoint costÃ—1.5. Add 120 seconds setup/hash, 180 internal evidence reserve and 1800 campaign finalization reserve.

| Plan | Forecast task seconds | Qualification seconds | Total reserved seconds | Remaining campaign seconds |
|---|---:|---:|---:|---:|
| Answers only |2106.6863|60.0273|4266.7136|20225.563|
| Answers plus separate canonical EOT |3025.9676|60.0273|5185.9949|20225.563|

Both fit the complete remaining budget under these assumptions. This is a conservative forecast from observed CPU data, not a certified bound for new prompts/implementation or a timing claim. A new worker must measure fresh representative/late-prefix/max 256 CPU costs and admit the complete registered 192-case plan before any task results. If shape/resource/numerical/cost gates fail, retain an incomplete/refused run; do not drop difficult cases, targets, EOS or qualification to obtain PASS. CPU guards remain parent 8.4 GiB/preweights 7.4 GiB/RSS 6 GiB/running host reserve 1 GiB, with coordinator-owned containment and a preregistered deadline fitting the budget.

## First-prefix feature and interpretation firewall

All policy-predictive features must be computed from the original prompt-only first-position unit rollout and frozen before any gold continuation. That first target measurement can supply them without another model call. Later `fixed_rollout` moments, states and gradients are teacher-conditioned diagnostics; they must not enter an online scale policy, A5 pool, candidate fitting or claim of generation improvement. Every replay prefix uses only preceding gold, but teacher-forced histories differ from the model's own generated histories. The original v1 full-gold/NLL guards remain unchanged; this is a distinct source-bound causal diagnostic proposal.

Canonical strings do not exhaust scorer equivalence. Every case accepts additional surrounding whitespace, and YES/NO/UNKNOWN cases also accept case variants that tokenize differently. A loss change can reflect preference for the selected spelling/tokenization or length, not a change in accepted-answer accuracy or reasoning. Unknown's custom logit bias remains intact. EOS alternatives add another equivalence issue. Report canonical sequence sums/means and parent/category weights separately; the 440 target positions are not 440 independent cases. Future target-aggregation/statistical/feature conventions must freeze before native outcomes. No validation/TEST, fitting, new controller or broad IQ conclusion is included in this feasibility packet.
