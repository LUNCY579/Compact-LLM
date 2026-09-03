# Canonical TGSA + MTP Programme Path — v5.1

## Status

`research/tgsa-v5-1-unified-programme` is the coordination branch for the next
TGSA programme. It starts from `research/tgsa-v5-competence-and-mechanism` at
`b09149002a7892a87a461af0b3c852ed14842905` and leaves every source branch
unchanged.

The branch is not a monolithic merge of Phase 2A and Phase 2B. Those stages use
different MTP causal graphs and remain separately identified implementations.
V5.1 unifies their prerequisites, evidence identity, promotion rules, and claim
ceilings.

## Frozen lineage

| Role | Branch | Commit |
|---|---|---|
| Rejected v4 regime boundary | historical branch `research/tgsa-phase1-v4-competence-first`; live tag `archive/tgsa-phase1-v4-regime-rejected` | `8b5737e260b5478427b8bb033cffa394a5f3fc5d` |
| v5 design/execution source | `research/tgsa-v5-competence-execution` | `de3bd632b491df294f9b8736079bdcbbdd64e4d3` |
| v5 executable/mechanism source | `research/tgsa-v5-competence-and-mechanism` | `b09149002a7892a87a461af0b3c852ed14842905` |
| Phase 2A source | `research/tgsa-phase2a-ci-and-competence` | `de3bd632b491df294f9b8736079bdcbbdd64e4d3` |
| Phase 2B source | `research/tgsa-mtp-rineya-complementarity-v1` | `49aa961265c7b248be5c81244cc5d80d49db13c7` |
| Pretrained retrofit source | `research/tgsa-rineya-pretrained-retrofit` | `ef5194b36583f31bd59508cfe04aec1f5dcfdaec` |

The v5 executable/mechanism source contains the v5 design/execution source.
There is therefore no missing merge between the two v5 branches.

## Prospective registration boundary

The inherited v5 plan remains byte-identical at Git blob
`0538faf4d2ad142dbce9c408910f9dcbcf1fd064`. Its internal status remains
`implementation_complete_not_registered`; v5.1 does not rewrite history.
Instead, `configs/v5_1_programme_contract.json` freezes the reviewed bytes,
lineage, stage order, and promotion boundaries before nondevelopment execution.

A nondevelopment v5 run must use `scripts/phase1_v5_1_qualify.py` and include the
operator attestation that no nondevelopment outcome was inspected before the
v5.1 lock. If that statement is not true, do not use the flag: create and review
a new prospective amendment instead.

## Stage identities

### V5 apparatus

- `study_kind`: `apparatus_qualification`
- MTP: absent
- arms: `cascade`, `history_control`, `learned_delta`
- purpose: find a competent, non-saturated, stable scratch-model regime
- forbidden interpretation: TGSA mechanism, Rineya quality, serial-round, or
  wall-clock evidence

### MTP-free mechanism study

- `study_kind`: `mechanism_identification`
- MTP: absent
- controls: cascade, history, aligned trajectory, replay, deranged alignment
- purpose: isolate trajectory alignment from serial depth, history, capacity,
  and pathway availability

### Phase 2A

- `study_kind`: `mechanism_bridge`
- `mtp_semantics`: `sequential_teacher_forced_recursive_v1`
- modes: `dormant`, `backbone`, `trajectory`
- gradient order: `frozen_probe`, `stop_gradient`, `joint`
- purpose: test incremental future-token information in the registered
  trajectory coordinate
- not a speculative-decoding speed experiment

### Phase 2B

- `study_kind`: `system_factorial`
- `mtp_semantics`: `parallel_independent_offsets_v1`
- cells: `parallel_ar`, `parallel_mtp`, `tgsa_ar`, `tgsa_mtp`
- purpose: test exact-decoding quality/serial-round complementarity
- no mechanism interpretation; no wall-clock inference

### Real Rineya checkpoint qualification

- retains the complete Phase 2B four-cell block;
- adds source-checkpoint hashing, real-dtype initialization drift, exact decode,
  within-scope competence, registered quality, serial-round, and closed-evidence
  gates;
- may support only checkpoint-scoped conclusions.

### Wall-clock benchmark

A separate post-qualification preregistration. Serial target-round reduction is
never accepted as a substitute.

## Evidence separation

The following are evidence-integrity requirements, not documentation advice:

1. Phase 1, Phase 2A, Phase 2B, real-checkpoint, and wall-clock artifacts use
   distinct evidence roots and schema identities.
2. Checkpoints are not promoted across stages without a manifest recording the
   originating stage and scope.
3. Task, model, objective, competence, and environment scope hashes must match.
4. Development evidence cannot certify confirmatory execution.
5. A sealed bank is single-use for the preregistration that first opens it.
6. Phase 2A and Phase 2B evidence may not be pooled or described as
   replications.
7. A system-level `GO` after a mechanism `NO_GO` is permitted, but mechanism
   interpretation remains forbidden.

## Promotion graph

```text
v5 apparatus GO
       |
       +--> MTP-free mechanism GO --> Phase 2A GO --> mechanism-interpretable Rineya lane
       |
       +--> within-scope pretrained competence --> Phase 2B GO --> system-utility Rineya lane
                                                                    |
                                                                    +--> wall-clock benchmark
```

The two downstream lanes answer different questions. A system-utility result
cannot be used to rescue or reinterpret a negative mechanism result.

## Canonical entry points

```text
scripts/verify_v5_1_programme_contract.py
scripts/phase1_v5_1_qualify.py
scripts/select_phase1_v5_1.py
```

The lower-level v5 scripts remain available for development smoke and historical
reproduction. Nondevelopment apparatus execution and selection use the v5.1
wrappers so the prospective lock is attached before any evidence root is
created.

## Present claim ceiling

The current branch is implementation and protocol evidence only. Real v5
apparatus evidence, TGSA mechanism evidence, Phase 2A/2B real evidence, Rineya
quality evidence, and wall-clock evidence remain unrun unless a later immutable
result bundle says otherwise.
