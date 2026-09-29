# RVS-001 implementation amendment 001

Date: 2026-09-29. Starting branch commit: `88efa852689c162c131d9c698923d4327cdec853`.

The user explicitly requested usable, runnable code to validate or falsify the repeated-sampling idea. This amendment supersedes planning-only restrictions on implementing RVS. Native execution remains opt-in through the documented commands, host/source checks, and inference qualification. No inherited training approval is used. Original planning files and receipt remain unchanged historical evidence.

## Implemented experiment

The CLI now prepares actual centralized bridge DEV/TRAIN_SELECTION tasks, freezes public-only inputs and policy, qualifies the exact frozen Titan checkpoint, samples independent complete answers, selects without labels, evaluates against a separately bound oracle, and produces replayable reports. A CPU fixture path runs without model dependencies. This is deliberately a bounded frozen-model inference experiment; it does not implement every future research idea.

Defaults are fixed prospectively in `configs/titan_pilot.json`: 128 parents at preparation, 16 samples plus one greedy per parent, temperature 0.7, no top-k/top-p truncation, max prompt/completion/context 896/128/1024, cache enabled only after parity qualification, 128-attempt segments, 1800 seconds per segment, 21600 seconds total supervised collection, 278528 reserved generation tokens, 256 MiB output budget, and 1000 parent-bootstrap replicates. These are engineering/exploratory choices, not measured fit or optimized hyperparameters. Smaller task/count/config variants are separate frozen studies. CHECK/FINAL and training remain inaccessible.

No calibration search is needed to run the fixed preset. A later temperature or selector search requires disjoint TRAIN_SELECTION parents, a new freeze, and all tuning cost reported. DEV outputs cannot be relabeled protected or used to choose a confirmation threshold. Reported confidence intervals are exploratory and unadjusted for multiple comparisons; no promotion thresholds or broad efficacy claim is introduced.

## Corrections to the earlier plan

Inspection of the actual inherited runtime shows that `worker_free_gib=6.0` means free system RAM. The external watchdog separately caps process-tree RSS at 4.0 GiB. These actual limits, plus GPU/disk guards, remain unchanged. The old planning label calling 6 GiB a worker-memory limit was imprecise.

The implementation uses the exact centralized stock-Qwen3 streamed BF16 loader design, not a generic HF loader that might reinterpret the custom checkpoint. The native path has no optimizer. A cached manual sampler is compared with an independent full-prefix reference, using the inherited likelihood tolerances. Qualification reports the actual short trace, its 32-token maximum, and a separate 1023-prefix/one-output-token maximum-envelope probe; it does not claim that every 128-token continuation was tested or that a cache of 1024 fed-back tokens was allocated.

Output limits are observed at segment, attempt-boundary, and native forward/resource boundaries. The sample journal has an exact pre-append cap. Native logs additionally cap at 16 MiB per segment and inherited stdout/stderr at 16 MiB. This is fail-closed bounded operation, not an OS disk quota; concurrent writes can cross the aggregate threshold between checks. Supervisor polling is 0.1 seconds and not an absolute instantaneous resource guarantee.

## Source/data interfaces

The existing bridge allocator selects roles before rendering. Only DEV or TRAIN_SELECTION receives a nonzero quota. Existing independent semantic verification checks targets; the evaluation process subsequently uses the bound strict canonical target. The runtime consumes only opaque IDs and public messages, not private targets or hidden family labels. The native worker does not import `evaluation.py`. This enforces ordinary pipeline separation, not protection from a malicious owner of the filesystem.

The source binding includes RVS runtime modules, original bridge, inherited watchdog, and its contract dependency. Qualification and resume reject active source changes. Documentation-only changes do not alter code digest. Only exact Git line-ending checkout normalization is tolerated for pinned upstream blobs; current bytes remain recorded.

Failed attempts are never discarded or resampled invisibly. Per-attempt seeds are independent of correctness and scheduling. A pending attempt needs explicit abandonment. Partial/corrupt records remain evidence. Complete-bank pass@k is not computed over an opportunistic subset of successful executions.

## Validation and remaining boundary

Local receipt: `receipts/CPU_VALIDATION.json`. Actual CPU tensor tests use a tiny causal fixture, not a pretrained model. Original-generator integration tests require the full repository; any local skips are disclosed. A GitHub CPU workflow is provided to execute those tests in a full checkout. Its actual execution status must be read separately rather than inferred from the workflow's existence.

No native Titan loads, inference tokens, training steps, paid model calls, or protected payload accesses occurred during local implementation validation. The actual host experiment remains scientifically unmeasured until its qualified collection/report executes. The measured CPU scenarios show that the instrument distinguishes a flat sampler, unrecoverable correct minority, usable voting gain, and ceiling behavior; they do not validate the paper on Titan.

The present report cannot establish reasoning rather than lucky guessing on small answer spaces, nor general language/long-context retention. Existing reference mutation checks validate input/scorer mechanics, not model causal evidence use. Follow-up chance, evidence-removal, counterfactual, preservation, and stronger-model comparisons need explicit additional instruments and cost allocations.
