# Halo evidence-retention endpoint calibration

Status: **exploratory metric calibration complete; definition frozen; one
narrow local prospective validation subsequently completed**.

The original exact diagnostic remains unchanged: every required token must be available
in at least one KV head in every evaluated layer. It was zero for all 33 condition rows
in the Halo retention pilot, so it cannot rank policies in this pilot. It remains useful
as the strongest all-or-nothing diagnostic.

## Frozen primary endpoint

For required fact `f` with exact annotated token set `T_f`, let `U_l` be the union of
resident absolute positions over KV heads in layer `l`. The v1 primary endpoint is:

```text
layer_fact_exact_completion_rate
  = mean over (layer l, required fact f) of 1[T_f is a subset of U_l]
```

This changes only the aggregation, not the evidence annotation. Each fact must still be
complete in a layer to earn that layer–fact cell. Macro-averaging makes facts and layers
equal units, prevents longer facts from dominating, and avoids a fitted threshold. It
also avoids rewarding retention of only the repeated template language in a fact.

The following are secondary sensitivity diagnostics:

- `layer_fact_macro_token_recall`: macro-mean partial token recall per layer and fact.
- `layer_balanced_minimum_fact_token_recall`: mean over layers of the worst required-fact
  token recall in each layer.
- `minimum_layer_fact_token_recall`: the worst individual layer–fact token recall.

Partial-token diagnostics are not semantic-survival claims. The strict global Boolean
continues to be reported separately.

## Outcome-blind calibration audit

The calibration script read only each row's `evidence` mapping. It pooled all 33 records
without condition labels and did not access correctness, predictions, accepted answers,
failure labels, or task family. The source file is frozen by SHA-256
`5c37102aa839053026d63b4e2c29f063d340cc97a737ec6ca9ad517863bd3dcf`.

The primary endpoint had 14 distinct values, ranged from 0 to 0.482143, and had a 0.4242
floor fraction with no ceiling cases. It passed the predeclared non-saturation screen.
The strict diagnostic had one distinct value and a 1.0 floor fraction, confirming its
saturation in this pilot.

Authoritative files:

- `configs/halo_evidence_endpoint_v1.yaml`
- `results/statistics/halo_evidence_endpoint_calibration_v1.json`
- `src/rineya_significance_lab/evaluation/evidence_endpoints.py`

## Existing-pilot application

Applying the already-frozen definition back to the same 11 B0-eligible clusters is
necessarily post-hoc and exploratory:

| condition | mean exact layer–fact completion |
|---|---:|
| B2 raw accumulation | 0.1477 |
| B3 observation normalization | 0.2094 |
| B4 positional control | 0.0000 |

The paired B3-minus-B2 difference was 0.06169 (cluster bootstrap 95% interval
0.03084–0.09416; exact cluster sign-flip `p=0.0078125`). B3-minus-B4 was 0.20942
(0.13636–0.28571; `p=0.001953125`). These values show that the endpoint is capable of
resolving retained-state differences in this calibration sample. They do not establish
a confirmatory quality gain or causal mediation of answers.

The application artifact is
`results/statistics/halo_retention_endpoint_analysis_v1.json`.

## Limits and freeze rule

- Endpoint selection reused final resident-state traces from the existing pilot.
- The pilot clusters were selected conditional on B0 capability and cover R1/R3/R4 only.
- Existing annotations cover exact rendered fact statements; they do not identify a
  smaller semantic payload token subset.
- Current artifacts describe final resident state across layers, not survival at every
  compression boundary or time-to-eviction.
- No result here connects retention to answer correctness.

Version `layer-fact-retention-v1` remains frozen. Prospective v1 later stopped
at B0 `9/24`. Difficulty-calibrated v2 passed B0 `18/24` and locally registered
12 clusters before compressed execution; B3-B2 was `+0.0401786`, interval
`[0.0133929, 0.0758929]`, `p=0.01953125`, while answers were 5/12 versus 6/12
(`p=1.0`). See [ENDPOINT_VALIDATION_V2_RESULT.md](ENDPOINT_VALIDATION_V2_RESULT.md).

That result does not retroactively make this calibration confirmatory, nor does
it register the broad protocol. Any endpoint-definition change still requires
a new version and calibration artifact; any broad replication requires its own
pre-outcome registration.
