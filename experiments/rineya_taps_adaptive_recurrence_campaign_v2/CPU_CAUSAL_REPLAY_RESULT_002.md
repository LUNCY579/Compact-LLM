# CPU causal replay attempt 002: qualification passed, task matrix not run

The directly authorized corrected attempt passed native CPU FP32/eager scalar
and accepted-EOS-mass gradient qualification, then refused the full runtime
forecast before any of the 192 DEVELOPMENT cases. There is no task mechanism
result, EOS task result, generated-performance result or controller-fit evidence.

The supervisor charged 66.109 native seconds. Qualification made 29 model API
attempts, 27 full-forward attempts, 26 completed forwards, three state adjoints
and 4,764 completed prefix positions; one expected partial exception and two
precompute cache rejections are retained. Eight exact value-parity pairs and
three persisted G/raw-update vector pairs support independent review. Numerical
checks passed, including accepted-EOS-mass absolute FD error 0.000065889675.

Fresh maximum-shape forward plus adjoint took 1.7388156999986677 seconds; a value
forward took 0.9941931000030308. The registered factor 1.5 and actual setup elapsed
64.91039009999804 imply 64.9103901 + 632*(1.7388157 + 2*0.9941931)*1.5 + 180 =
3,778.2977913044815 seconds. This exceeded the approved 3,600-second cap by
178.2977913044815 seconds. The run stopped without extending the deadline,
reducing the bank, changing the forecast or retrying.

At qualification, loaded state before/after was b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551;
all 96 bound sources and ten canonical assets matched. Worker peak RSS was
4,456,894,464 bytes and minimum recorded host availability 6,196,453,376 bytes.
No resource violation was recorded. Qualification verification is the latest
resident-state proof: final post-forecast resident-state/parameter verification
was not written and remains unavailable. Canonical assets stayed unchanged.

The approved 5397 allowance was used once and is consumed. Cumulative native
use is 1,448.515 of 21,600 seconds, leaving 20,151.485. No worker or lock remains.
Independent audits: audits/cpu_causal_replay_native_execution_002.json,
audits/cpu_causal_replay_execution_math_002.json and
reviews/cpu_causal_replay_execution_review_002.json.

A request-only successor preserves all 192 cases/96 parents, 440 answer targets,
192 separate EOS targets, 1,925 API attempts, 635 adjoints, max256 shape, native
operator, endpoints, predictor, numerical thresholds, factor and resource floors.
It proposes one 4,500-second attempt (4,320 body + 180 internal reserve), no further
retry. That cap gives 721.7022086955185 seconds over the measured forecast; with
the 1,800-second global reserve its 6,300-second allocation fits remaining time.
Fresh qualification and complete-cost admission remain required. The template
and source reviews grant no execution authority; another direct human approval
is mandatory. This is the same study's retry/resource expansion, not new science.

The larger GPU validation remains incomplete and primary comparisons null.
Provisional complete DEV gives best fixed/unit 69/192, TAPS/random 67/192 and
constant-grid oracle 70/192, with preservation failures. The goal remains unmet.
