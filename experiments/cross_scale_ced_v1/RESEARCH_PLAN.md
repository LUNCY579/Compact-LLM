# XSC-001 research plan and decision boundary

## Question

Can a native-width, intact Qwen3-4B decoder learn to use contextual representations from a native-width, full Supa E28 encoder while retaining its original behavior on the same decoder-visible input?

This is not yet a claim of better reasoning, long-context recovery, compression, efficiency, or cross-scale knowledge transfer. The pure shared-memory decoder is deferred: it would change the decoder's attention mechanism and input distribution simultaneously and needs a different registration.

## Frozen engineering comparison

B4 is the unmodified, ungated official decoder and teacher. H and P have identical unique/trainable tensor inventories and seeded initialization. H reads a full causal shared memory; P adds a contextual same-token sigmoid read. Each receives the same fixed paragraph order, token limit, 25 AdamW steps, learning rate, KL coefficient and clipping policy in `protocol.json`. Only interface parameters train. Both native sources, tokenizers and readout definitions are fixed by hashes.

Engineering PASS requires finite loss/gradients/updated parameters, nonzero gradient coverage for every interface tensor during the run, unchanged backbone digests, exact matched inventories, and an active checkpoint roundtrip. Loss must not explode beyond the declared engineering bound. A decreasing loss is not an efficacy result. An H-versus-P loss difference on these paragraphs is not a model-quality comparison.

## Gates

P0: inspect package/runtime/source requirements without allocating pretrained models. Local synthetic tests license nothing.

P1: require exact source assets, clean full checkout, pinned CPU runtime, native tokenizer identity, native encoder/decoder full and cached parity, zero-output and disabled-read identity, active-memory causal/padding/segment/physical-order checks, multiple chunk schedules, cache immutability, complete gradients, frozen donors and source-bound checkpoint roundtrips for H and P. A failed or absent check closes P1. A PASS permits only a **separately requested** fixed P2 smoke.

P2: the frozen paired engineering run above. Preserve every attempt and checkpoint. It opens no further permission.

## Registration required before efficacy

Do not opportunistically score existing Rineya sealed banks or tune on historical negative cases. Before a quality campaign, separately freeze the train/calibration/open-development/sealed banks, family/cluster identities, tokenization and context budgets, semantic versus formatting metrics, sample-size/power calculation, seed count, training budget, retention margins, multiplicity correction and promotion rules. The minimum meaningful effect and retention tolerances have **not** been registered here; inventing numerical thresholds after seeing scores is not allowed.

The primary quality comparisons should be H versus P and H versus B4, with clear accounting for extra parameters, compute and encoder-visible information. H beating a 0.6B baseline alone does not establish an architectural benefit. The same-scale E28/D28 lane can be compared only after its own source/runtime qualification and comparable data/budget registration.

An encoder-only-information diagnostic should use novel randomized facts and controlled answer changes. It requires a separately frozen input contract, not an undocumented change to the default same-token stream. Correct versus removed, within-segment shuffled, mismatched and counterfactual memory should establish content use. Read RMS diagnostics already exist; attention-entropy and task-specific information analyses are future additions and must not be reported as implemented measurements.

Possible outcomes: memory ignored; memory active but no useful quality gain; useful encoder evidence but retention loss; or a replicated gain over relevant controls with acceptable retention. None is decided by this implementation or its random fixtures. Do not merge into a Rineya release or open deployment based on P1/P2 alone.
