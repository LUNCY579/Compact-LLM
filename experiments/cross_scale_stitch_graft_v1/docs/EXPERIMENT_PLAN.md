# Bounded experimental programme

This is a concrete proposal with working complementary mechanics, not a completed native registration. Numbers below are proposed caps to freeze after reviewing model assets and open data. No GPU, donor download or native training is authorized by this document.

## Stage R0 — exact subject, open data and runtime registration

Register the historical Supa-A1.1 subject and each donor separately. Preserve the custom host behavior rather than swapping in Supa Core 1.7 or stock Qwen3. Include weight-inventory hashes, all tokenizer/template assets, runtime source/version hashes, execution and storage precision, decoding settings, EOS behavior and logit policy. The current optional native helper cannot qualify the custom host.

Review fresh, open, task-level data. The proposed roles are FIT, MAP_CHECK, SELECT, CHECK and RETENTION. Split by independent scenario/document/problem-family group, not random token position. Canonical prompt hashes and example IDs may not cross roles. Capture both models on the identical physical token grid. Define the normalization used for the prompt digest before producing it.

Do not use the old Showcase tuning bank as an untouched reporting set. Do not read Invisible-Limit TEST2, sealed holdouts or their predictions. User authorization and provenance review cannot be replaced by editing `role` in a JSON file.

## Stage H0 — donor headroom, before any graft

Proposed first panel: 64 short, verifiable open SELECT cases plus 64 separate RETENTION cases. Fix a context cap (initially at most 512 tokens), answer cap, thinking setting, scorer and matched precision before collection. Distinguish task correctness, formatting and literal `UNKNOWN` behavior.

Start with 4B when a cheap engineering donor is useful; keep 8B as a separately registered donor. A larger model is not automatically a better donor for this specific target. Report candidate-only and receiver-only correct cases, rather than only total accuracy. Inspect complementary wins on the target skill, not merely the donor's global average.

A donor that has no useful measured advantage on the intended task does not justify a large transfer run. This is a local decision about that donor/panel, not a universal negative about cross-scale learning. Historical likelihood-only E7 capability claims were withdrawn; do not recreate that instrument as the principal success measure.

## Stage A0 — source parity and activation acquisition

First reproduce the exact unchanged host and donor forwards under the proposed native adapter/collector. Pass source reload checks, finite outputs, raw capture-site verification and causal-prefix invariance. Record missing dependencies, memory refusal and loader failure as blocked engineering, not negative model behavior.

Proposed capture budget: begin at 8,192 FIT token positions and 2,048 MAP_CHECK positions, spread across many independently grouped short examples. Escalate toward 16,384/4,096 only after the memory estimate and first capture succeed. Bound positions per example in advance. Store each source's captures separately and load donors sequentially. Weights need not remain resident for bridge fitting.

The bounded CLI refuses combined payloads above 256 MiB and ridge workspaces above 512 MiB. In particular, a full 4096→1024 8B map may exceed the default solver budget. Do not quietly lower rank or bypass the guard: either register a different resource envelope/solver implementation or keep that arm blocked.

## Stage G1 — one FFN graft, frozen inference first

Initial proposed pair: host block 13 and donor block 17, both raw post-attention sites. These are midpoint hypotheses, not layers proved optimal by the old eight-prompt ablation screen. Preserve the host's early sensitive layers and avoid an initial combinatorial depth sweep.

Fit both directional full affine maps on FIT; evaluate reconstruction on MAP_CHECK without selecting there. Retain only the linear part of the donor→host map for returning an FFN delta. Fit once before task selection.

Screen these clearly named arms:

| Arm | Role |
|---|---|
| Untouched host | Baseline |
| Explicit bypass and fixed-zero gain | Exact implementation controls, not efficacy evidence |
| Real frozen donor FFN, alpha 0.05 | Primary active candidate |
| Real frozen donor FFN, alpha 0.10 | One predeclared strength sensitivity arm |
| Same-shaped random donor with declared seed and norm convention | Pretrained-content negative control |
| Bridge fit on deliberately shuffled pairings | Alignment negative control |

Controls must receive unique identities and must not be mislabeled as real paired source captures. A random or shuffled arm has its own explicit transformation manifest. Control-generation orchestration for native runs is a separate collector prerequisite; this package does not fabricate such measurements.

Use SELECT for exploratory comparisons, then freeze at most one active candidate before fresh CHECK. Zero/bypass parity alone cannot open a quality claim. A candidate must be finite, visibly nonzero and functionally compared against the relevant controls.

## Stage S1 — bidirectional stitch as a separate study

Initial arms: 4B `[0:18]` → bridge → host `[14:28]`, and host `[0:14]` → bridge → 4B `[18:36]`. Use separate raw block-output maps rather than graft-site maps. Add same-source split/rejoin identity controls at the exact cut and a shuffled-pair connection control.

Measure teacher-forced next-token loss as a compatibility diagnostic, then task-level generated answers with a qualified bounded collector. Do not equate teacher-forced likelihood of one spelling with semantic task success. The delivered runners are uncached; no latency claim can be made by comparing them against a cached baseline.

Quarter-depth and three-quarter-depth cuts are queued sensitivity studies only. They do not run automatically after a midpoint result. Headroom, graft and stitch conclusions remain separate; failure of one does not close the others.

## Stage T1 — optional interface-only training

Proceed only after a new explicit training authorization, reviewed train data and frozen budget. Proposed initial cap: 50 optimizer updates for one selected architecture and its ordinary-adapter control; cap both number of updates and actual supervised tokens. Keep original backbone weights frozen. Record optimizer, learning rate, data order, loss mask, teacher targets, total tokens, initialization and early stopping policy before execution.

The supplied `masked_next_token_loss` supervises target-token positions, including EOS. Position zero cannot be a target. The explicit optimizer allowlist rejects missing/duplicate interfaces or accidentally trainable backbone parameters. Tests demonstrate that frozen downstream layers still pass gradients to a bridge; frozen parameters do not make the backward pass free.

Use the no-donor `OrdinaryAdapter` with the same teacher supervision and trainable parameter count. Report calibration/capture compute separately. Do not claim FLOP matching when only trainable parameters and optimizer updates are matched. Add a genuinely compute-matched comparison before making efficiency claims.

There is no native training command in this package. Its four synthetic updates verify gradient plumbing, not the value of 50-step native training. WSD, LoRA, backbone recovery, new architectural composition and new data recipes are not bundled into this contrast.

## Stage C1 — reporting and retention

Proposed CHECK size: 128 independent open cases, disjoint from SELECT and fitting data, plus the frozen RETENTION panel. Freeze the chosen operator, gains, boundary, maps and checkpoint before opening CHECK. Repeated selection on a reporting set consumes its holdout status.

Report paired wins/losses, per-family changes, format regressions, literal-UNKNOWN regressions, output length, memory, prefill/decode costs, data/capture cost and total retained parameters. The offline scorer provides paired counts, an explicitly descriptive row-wise exact McNemar value and a group-resampled bootstrap interval. With one independent group it reports the interval as undefined. Selection, clustering and multiple arms must be accounted for in any later confirmatory statistics.

No automated GO is implemented. The smallest useful result is an honest decision about one nonzero connection on one registered subject and task—not a declaration that the whole architecture family is good or bad.
