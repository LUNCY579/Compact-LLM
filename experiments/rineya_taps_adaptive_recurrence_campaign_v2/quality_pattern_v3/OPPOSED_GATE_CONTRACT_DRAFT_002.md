# Exact proposed contracts before exposed DEV scoring

Status: DRAFT, independent review pending. This resolves definitions in draft001; it does not freeze source, increment candidates or permit dispatch. No candidate outcomes have been scored. The choice of confidence and cutoff is a post-evidence DEV hypothesis, even though no numeric threshold search is allowed.

## Online API and three specifications

The pure policy API takes only a policy name, raw prompt string and, for confidence policies, the actual same-run original-prefix intact probability. It accepts no case/parent/split/category/gold/scorer/status/teacher/loss/outcome metadata. Identifiers remain journal join keys outside the policy. K is fixed before token1 for the entire answer. All K1 generation uses the inherited exact unit whole L12-L13 operator on its own trajectory; each full-prefix forward resets state. There are no online lookup tables or token rewriting.

CONFIDENT_RECURRENCE selects K1 iff p >= 0.5. UNCERTAIN_RECURRENCE selects K1 iff p < 0.5. Equality belongs to the confident arm. Each physical policy makes its own actual original-prefix intact pilot. Exact feature: z = actual full-head last-token logits.detach().float(); logp = z.log_softmax(0); p_vector = logp.exp(); p = float(p_vector[z.argmax()]). Require finite z and p in [0,1]; violations fail the run. No LSE reconstruction, smoothing, head-row filtering or alternate approximation. K0 reuses the live pilot's first logits; K1 makes a fresh unit forward on unchanged original IDs. Reuse is restricted to a bound same-run model/source/state/prefix handle. It is not KV cache or serialized evidence reuse. Charge pilot, probability computation and all later calls. For T generated tokens and L input tokens, confidence full calls are T+K; positions are sum_{j=0}^{T-1}(L+j)+K*L; extra-layer calls are 2*K*T. The two confidence arms have combined call cap17N at cap8 only if physical pilots agree exactly and decisions are complementary; otherwise stop the registered study.

EVIDENCE_CONSISTENCY_GUARD uses only this strict existing rendered contract:

    DOCUMENTS: use only A-approved records[; latest approved date per entity]. Equal approved values give that value; absent/conflicting values give UNKNOWN.
    ENTITY A VALUE;ENTITY D VALUE[...].
    Question: ENTITY? Reply one integer or UNKNOWN.

The alternate final line is exactly `Query: ENTITY. Reply one integer or UNKNOWN.` ENTITY is three ASCII uppercase letters. VALUE is a canonical integer matching `0|-?[1-9][0-9]*`. Undated headers require undated records. The optional latest header requires every record to have `dDATE` between A/D and VALUE, with DATE a positive canonical integer. Semicolons separate one or more records; no extra whitespace, trailing material, mixed dates, unknown flags or ambiguous query is accepted. Preserve exact header punctuation and newline boundaries. Repeated records are allowed. Parser rejection returns UNRECOGNIZED and K1; no partial parsing or generic natural-language inference.

For recognized text, retain only queried entity's A records. No relevant A records yields ABSENT. For latest mode retain records at the largest relevant A date. More than one distinct retained value yields CONFLICT; one yields CONSISTENT. K0 iff ABSENT or CONFLICT; otherwise K1. The online return contains K, recognized flag and certificate only, with no solved value. Parse the complete prompt before any model call and charge parsing cost. New lexical contracts would be a new registered candidate, not a repair after outcomes.

## Fixed controls and label firewall

Replay K0 and exact unit K1 using complete recorded answers. The prior strongest fixed-scale comparator is unit1. Header cue selects K0 whenever the first line starts `DOCUMENTS:` and otherwise K1; it never reads records or correctness. This checks whether template detection alone explains the guard.

Report the standalone same-grammar certificate classifier with coverage and false certificate counts. Independently implemented prompt-only full solver is the positive control, with its own bank's constant UNKNOWN/YES/NO, first/last integer and query-blind floors. Compare every model result with its own bank floors; solver success cannot be called neural reasoning or recurrence value. If an ordinary solver dominates, preserve that limitation.

For each candidate, freeze its equal-item DEV K1 rate n1/192 after decisions are made without outcomes. Matched random controls use two streams, seeds17 and29, independent per item and per seed. Input order is the frozen 192-row prompt export order. The exact RNG is Python stdlib random.Random(seed); draw random() once per item and use K1 iff draw < frozen rate. No correctness, category or parent conditioning. Average seed outcomes within item before parent/family aggregation; repeats are not independent cases. Record actual marginal and seed coverage, without adjusting draws. This is case-level random K, not the old token-level random-alpha control.

## One exposed replay and mechanical nomination

Join only the exact192 old DEVELOPMENT cases, actual A0 pilot statistics and complete A0/unit response records. The policy decisions are made first with prompt/probability-only objects; grading joins labels in a separate phase. Require exact input/source/hash/case completeness and unit fixed1 token-response identity. No old VAL outcomes enter selection. Label results EXPOSED_DEV_POLICY_REPLAY. No generation, latency, validation or causal execution claim follows from selection of stored answers.

Equal six family weights; average the two variants within each of96 parents before family means. Primary endpoint is strict full-response correctness. Report all case gains/losses versus both K0 and unit, correct-case retention, answerability, false UNKNOWN, unsupported answers, format, EOS and cap stops separately. Keep literal uppercase UNKNOWN distinct from normalized UNKNOWN. No interval or p-value will turn this exposed selection exercise into confirmation.

Useful DEV gain threshold remains >=0.05 with no observed unit correct-case losses, negative family correctness, format decrease, unsupported-answer increase or false-abstention increase. Binary replay oracle75/192 versus unit69/192 caps improvement at0.03125, so useful DEV superiority is impossible in this replay. No margin reduction is permitted.

A separate mechanical nomination may choose at most one frozen policy for prospective fresh research if it has strictly more correct answers than unit, zero unit correct-case losses, no negative family correctness, no format decrease, no unsupported-answer increase, no false UNKNOWN increase, and strictly beats its matched-random mean. The semantic guard must also strictly beat the header cue. Choose highest strict correctness among eligible policies; ties use fixed priority CONFIDENT, UNCERTAIN, GUARD. If none qualifies, nominate none, preserve rejection and design a different distinct hypothesis prospectively. This screen is a resource allocation decision, not evidence of useful efficacy. Even a nominee does not satisfy the gain goal or promise five-point improvement.

Before fresh measurement, freeze policy source, data construction, comparators, scorer, parent/family units, multiplicity, preservation and feasibility. Distinct new worlds or agent roles do not themselves establish independent source/human authorship. Fresh corpus and provenance must be separately reviewed; old DEV/VAL stay exposed. New native worker source, maximum shape, full matrix forecast, dispatch review and resources must be qualified. No model runs are authorized by this draft alone.
