# Explicit post-screen scope decision

Made after the L27 failure diagnosis and before fixed-arm development outputs.
This record clarifies the continuation decision; it is not a retroactive claim
that the full campaign passed or that this explanation was preregistered.

The original campaign at `01_middle_reapplication_screen/20260905T040946_172756Z`
is FAIL and remains unchanged. It completed all ten prespecified middle layers
at both depths, eight cases each, plus the L25 late control. The first L27
repeat-2 prefill crossed the emergency RMS guard, aborting the stage. L27's
remaining prompts and repeat-3 were not run. No threshold was relaxed and no
generation continued through its norm failure.

L27 was already excluded from fixed middle selection by the frozen 3–24
eligibility range. A single isolated diagnostic retained the original bounds,
reproduced RMS growth 22.7502 -> 277.0247 (12.1768x), and then demonstrated
bit-identical baseline restoration and unchanged state/checkpoint hashes.

The original README gate is per candidate: an arm qualifies when all eight
cases pass its stated norms/logits/validity/score criteria. All middle candidates
are complete; there is no missing-middle selection bias introduced by stopping
the final late control. The completed-candidate audit recomputes those exact
gates and requires equality with every archived summary. Nine middle repeat-2
arms pass; only L16 passes repeat-3. L12 is selected by the frozen cosine rank,
and L12–L13 by the frozen adjacent-pair rank. L12 repeat-3 remains blocked.

On that basis, a fresh fixed-development run is licensed for those independently
qualified operators. This does not turn the original campaign into PASS, make
L27 safe, authorize a screen retry, or license learned routing, PEFT or sealed
evaluation. The derived atlas labels both the original campaign failure and
the completed middle-candidate gate explicitly.
