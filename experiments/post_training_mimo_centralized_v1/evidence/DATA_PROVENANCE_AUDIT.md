# Data/provenance audit

Status: **NEWLY VERIFIED SOURCE INSPECTION; E1 DATA UNQUALIFIED.** Source inspected: `03db218ceebc61a2bc04f2482cf7e094045e01c1`. No native job, MiMo task execution, teacher collection, production dataset generation or FINAL-bank read occurred in this audit. Report reference: repository-root `Rineya_MiMo_PostTraining_Research_Report.pdf`, sections 2, 5, 8, 10, 11 and appendices A/C. The supplied Markdown counterpart has SHA-256 `851c9b0799067f13e942b73569b578cc516df8457ed5931e59f097d8fdd97168`.

## E1 does not already exist as an eligible bank

Report section 11.1 proposes **1,024 independent parents in each of four families**: bounded integer arithmetic, symbolic variable updates, finite-set queries, and instruction-constrained text transformations. It separately requires 1,024 training-safe retention rows, 512 acquisition CHECK parents and 512 retention CHECK parents. Existing row counts do not establish this population. `manifests/E1_DATA_REQUIREMENTS.json` records report constants and labels additional split choices as proposed amendments. It is intentionally not a runnable launch manifest.

The original report inspected separate branches; the current integration contains the assets below. This is source availability, not stronger evidence of efficacy or native qualification. Existing evaluation banks remain exposed. Re-splitting them cannot create fresh CHECK/FINAL evidence.

## Reusable source and recorded evidence

| Asset | Verified source location | Reuse and limit |
|---|---|---|
| Eight deterministic short-task families | `experiments/maxrl_verifiable_data_v1/generators.py:7`, `oracle.py:11`, `verifier.py:103` | Separate construction, public-spec solver and semantic reward. Arithmetic countdown and subset-sum may have many accepted certificates; these are not drop-in E1 unambiguous scalar/set-query targets. State tracking is a useful model for symbolic updates. |
| Canonical identity and split-before-render | `experiments/maxrl_verifiable_data_v1/specs.py:97`, `corpus.py:102` | Normalizes selected irrelevant ordering; does not prove general algebraic/semantic equivalence or historical exclusion. Roles are train/dev/test/challenge/warmstart, not the new five-role E1 protocol. |
| Full latent-world parent grouping | `experiments/maxrl_composable_skills_v2/corpus.py:143`, `contracts.py:104` | World digest is assigned before child views. All skill views, stage controls and twins inherit parent/role. Family-isomorphism and semantic contamination remain open. |
| Historical hash exclusions | `experiments/maxrl_composable_skills_v2/corpus.py:41` | Accepts a hash-only inventory as well as a ZIP. Prefer hash-only inventory for protected banks. Existing scope is only v1's 32,800 exact public instances. |
| Source-correct preservation | `experiments/maxrl_composable_skills_v2/behaviour.py:48`, `:67`, `:79` | Actual-response enrollment; separate repairs/damage and per-family/parent-cluster statistics. No baseline mastery has been established here. |
| Catalog boundary | `experiments/progressive_rl_curriculum_v1/data.py:18`, `schema.py:11` | Global parent and normalized-prompt cross-split checks. PRLC allows train/calibration/retention_train/retention_eval; do not relabel CHECK/FINAL to slip through its schema. |
| Trajectory eligibility | `experiments/rineya_harness_distillation_v1/datasets.py:11`, `:91`, `:131` | Replays actions/observations and recomputes success before SFT; excludes evaluation/private review; replacement pairs remain preference-ineligible until independently ranked. This is later harness work, not bridge supervision. |
| Adversarial verifier controls | `experiments/reasoning_posttraining_v1/verifier_audit.py:62`, `:157` | Reusable attack methodology, but its reward-module contract differs from E1. Do not call it on sealed banks. |
| Offline MiMo profiler | supplied report package `supporting_materials/profile_dataset.py:77` | Bounded schema/prompt statistics; no execution, semantic decontamination or native template qualification. Its recorded four passing tests are synthetic only. |

Prior evidence (recorded, not rerun in this audit): v1 `VALIDATION.json` reports 177 tests, 32,800 rows, 66,822 positive certificates and 360,800 negative controls; no native tokenizer/model/training. V2 `DATA_CARD.md:170` reports 462 combined tests and a 41,500-row audit across 10,450 parent clusters. V2 core TRAIN is 3,000 shared latent parents (seven skill views each), plus 500 answerability-twin parents; row count 22,000 must not become 22,000 independent parents. Optional v2 warmstart has only 150 parents/600 atomic rows. The new Q0 package records 46 additional synthetic qualification tests and deliberately exits native-blocked (`maxrl_comp002_pretraining_v1/QUALIFICATION.md:157`).

The local archive `C:/Users/clclc/Downloads/Rineya_MaxRL_Composable_Skills_v2.zip` was freshly hashed: **43,512,851 bytes**, SHA-256 `79b8171e83388ff0bf406328f90b5d74ab4cb6d94f29d0ba66af559cbd9b75da`, matching `maxrl_composable_skills_v2/evidence/download_artifact.json`. This verifies the archive file identity only; its contents were not re-solved. V1's archive receipt reports SHA-256 `7694ed0f654e8c23316ddc1546f845445eca1dedade2b79024ab64868c700129`; that archive was not located in the bounded Downloads filename search, so its local availability is unverified.

## Provenance and license limits

V1 README lines 110–112 and v2 DATA_CARD lines 192–195 identify original locally generated bounded tasks without copied external benchmark/teacher prose. They explicitly **do not assert a new blanket redistribution license** for repository code. Preserve source and notices; do not invent SPDX grants. The bridge will record user-owned project provenance with redistribution-license status unresolved, rather than treating prior data-mechanics checks as a license.

Report section 2.4 records the MiMo card's Apache-2.0 label. This audit has not refreshed that provider claim, downloaded the release or inspected every component license. The report's observed release revision is `639865fd3374018d6cb29b9fb82dd531406fcf5f`; it must be independently pinned before enrollment. Repository PRLC `data.py:78` accepts only a 40-hex revision and declared source parent field and quarantines every imported task behind an unavailable `mimo:<domain>` capability. Environment specs, empty answers and reward descriptors are not successful SFT traces, preference pairs or qualified RL episodes.

## Fresh bridge proposal and remaining gates

Build a separate sibling generator under this new experiment. Do not add helpers inside either MaxRL source tree: their manifests recursively hash Python files. Reuse design patterns and stable pure helpers only after explicitly recording dependencies. The proposed bridge uses original exact-answer arithmetic expressions, register programs, finite-set operations and text-token transformations with separate solver/verifier implementations. No stronger-model teacher is necessary. Short algorithmic/code tasks stay later factors; adding them to E1 would change its four-family design.

Canonical latent parents, not surface prompts, determine roles before rendering or solving. Arithmetic commutative operand order and finite-set irrelevant ordering are normalized; register names are index-based; text operation order is meaningful. All children share the same role. Reserved CHECK/FINAL operation compositions are explicitly distinct from training structures. Their difficulty is not assumed equal. This does not prove complete semantic equivalence exclusion.

The proposal reserves 256 acquisition TRAIN_SELECTION parents and 256 DEV parents, with 64 of each family; CHECK has 512 parents, half per family using familiar structures and half reserved compositions. These additions require protocol freeze before production generation. CHECK is reservation-only until the evaluation gate; FINAL remains closed and its confirmation count unresolved. No prompt, target or verifier payload for either bank is exported by preparation.

Every row needs source/parent lineage, specification and prompt hashes, target/verifier receipt, attempts/cost, tokenized prompt hash, and tokenizer/template identity. Actual 1,024/128-token fit is **unmeasured**. Character lengths cannot fill token fields. No silent truncation, held-out replacements, or teacher-success filtering. Historical hash inventories and an exact/normalized/semantic-overlap scope must be frozen before admission.

Retention is an independent blocker: v2 replay/guard only covers eight synthetic reasoning domains and does not certify retrieval, long-context, language quality or general instruction preservation. V3's 14,000-row preservation slice (`Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/README.md:19`) is historical training material, not fresh independent CHECK. E1 requires explicit source selection, ancestry, per-family roles and source-correct enrollment. No retention data have been silently manufactured or accepted.

## Source test commands and evidence interpretation

After root freezes the implementation policy, run the focused new bridge tests. Existing source-level tests can be run separately (they create only synthetic temporary fixtures):

```text
python -m pytest -q experiments/post_training_mimo_centralized_v1/tests/test_bridge_data.py
python -m pytest -q experiments/maxrl_verifiable_data_v1/tests experiments/maxrl_composable_skills_v2/tests
python -m pytest -q experiments/maxrl_comp002_pretraining_v1/tests
python -m pytest -q experiments/reasoning_posttraining_v1/tests/test_verifier_audit.py
```

The MaxRL tests cover public-only roundtrips, strict parsers, alternate valid certificates, deterministic regeneration, immutable export hashes, stage necessity, answerability twins, source-correct damage and guard disjointness. The Q0 suite covers padding/EOS/causal shifting, parameter inventory and write-once boundaries with synthetic tensors. Passing any of these is mechanics evidence, never pretrained efficacy. Do not run broad bank-audit commands over FINAL; use authorized fingerprint inventories only.
