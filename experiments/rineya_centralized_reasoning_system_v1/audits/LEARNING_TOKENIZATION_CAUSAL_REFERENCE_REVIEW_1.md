# Learning tokenization positional reference review

2026-10-02. Reviewer `/root/scientific_audit`; reference author `/root`.
Disposition: **REFERENCE_POSITIONAL_CONSISTENCY_ONLY; NOT INSTRUMENT ACCEPTANCE.**
No consequential arithmetic error found. I did not inspect or approve the mutable
instrument. Shared model/workspace/design context is not independent ground truth.

Read the frozen scientific and operations tokenization scope reviews and
`LEARNING_TOKENIZATION_CAUSAL_REFERENCE_001.json`, SHA256
`11a1afdae2f5b431352fbe1926ba07efdd7742eb479f7b71561a6831e4485970`.
Only JSON/stdlib arithmetic was used; no tokenizer/framework/model, host database,
holdout, token measurement, tensor, loss or training execution occurred.

The declared prompt has three tokens, including its user EOS at position2. All
three prompt labels are ignored. Response content occupies positions3/4, followed
by one supervised assistant EOS at position5. Thus labels are
`[-100,-100,-100,123,456,151645]`, unshifted ownership is `[0,0,0,1,1,1]`, and
the causal shifted prediction mask is `[0,0,1,1,1]`. Tokens at3/4/5 are predicted
by logits at2/3/4, respectively. There are three supervised tokens, two EOS IDs in
the complete sequence, and exactly one EOS in supervised response ownership.
All supplied counts, positions and vectors were independently recomputed and agree.

The full teacher input has N=6 positions. The served full prefix needed to predict
the last EOS has N-1=5 positions. This arithmetic does not establish numerical
equality between these different forward shapes; the historical bulk/prefix gate
and native/training qualification remain separate and unchanged.

The four rejected references are correct under the frozen scope: supervising the
prompt EOS leaks prompt loss; omitting assistant EOS loses a required response
target; using unshifted ownership for logit rows omits the first response prediction;
and retaining/supervising position6's post-EOS newline violates the predeclared
training-sequence transformation. The final template newline must remain recorded
in full-render bytes while being explicitly excluded from training IDs/labels.
These are negative positional references, not tested instrument rejection outcomes.

A later fresh frozen-instrument review must replay the accepted and negative
vectors against actual production validation and verify whole-sequence/prefix
token association. This reference supplies no encoding-of-text evidence, token
fit, mask implementation acceptance, numerical/gradient correctness, loss value,
model eligibility, acquisition or retention result.
