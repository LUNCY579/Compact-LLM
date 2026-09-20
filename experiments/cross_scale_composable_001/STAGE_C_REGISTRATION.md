# CSC-001 Stage C: bounded native capture

This prospective amendment follows the user's instruction to proceed toward testing
model improvement and to commit/push progress. That instruction authorizes the next
registered native stage. The previous first-pass zero-call records remain historical
and immutable. This amendment does not authorize DOSELORA or change its resource gates.

The existing calibration and scientific hypotheses are unchanged: 32 exact open parents,
24 FIT/eight MAP_CHECK, student layers0/13/27 and donor layers0/17/35; post-block residuals
at the last nonpadding prompt token. The cap remains256 tokens. Tokenizer-only preflight
must reproduce byte-identical IDs in the qualified student and donor tokenizer runtimes.
Those calls are accounted separately from model forwards. No evaluation bank is opened.
Each worker also re-encodes its32 frozen rendered prompts before loading a model to
verify token identity; those32 tokenizer encodings per launched role are reported.
They are additional tokenizer checks, not extra model forwards.

The execution envelope permits **96 forwards total**, at most32 in each of student_base,
donor_base and donor_skilled. There are **zero optimizer steps and zero generated tokens**.
Each role has one launch reservation, each forward is durably reserved before execution,
and failures consume their reservation. No retry, resume, warm-up or uncounted smoke call
is permitted. A fresh create-only session binds a clean source commit, exact source and
asset hashes, runtime pins, row/token hashes, output roots, and a deadline at most60 minutes
after registration. Each supervised child additionally has a maximum600-second timeout.

Student capture uses the source-qualified FP32/eager Supa loader in `.venv-mq001`.
Donor capture uses the existing donor environment, NF4 double quantization and BF16
compute. Both donor base and donor-skilled apply the same k-bit preparation, including
its FP32 treatment of nonquantized parameters, before eval/no-grad capture. The skilled
role adds only the bound existing adapter. The donor attention backend is explicitly
SDPA in this new qualification. The historical donor training omitted an explicit
attention backend; no historical arithmetic-equivalence claim is made.

Check the current GPU lease and resource snapshot before each launch. The existing
cooperative lease and foreground watchdog are reused; a suspected stale foreign lease
is never deleted. Source changes, expired deadlines, occupied lease, or insufficient
memory stop before dispatch. Required free host RAM stays7 GiB before student imports,
6 GiB after them; donor requires12 GiB before and after imports. Preload free GPU minima
are3 GiB student/6 GiB donor. Per-row floors are1.5 GiB host/.5 GiB GPU. These are
prerequisites, not measured peak-memory certificates. No precision change or package
installation is used to get around a guard.

Roles run in separate processes and models are released between captures. Independent
role outputs may be retained when a later role cannot launch. A partial capture session
is not a transport result: all three role identities and prompt positions must pass
comparability before fitting. Failures retain their cause, durable reservations and
observed terminal state. The controller cannot report COMPLETE from a failed child.

The intended native decision remains the final layer's heldout representation MSE
relative to its frozen FIT-mean baseline, plus nonzero donor skill residual and finite
transport. Record all three layer pairs and fixed shuffle/random residual controls.
Passing this stage establishes only representation mechanics. Testing actual student
acquisition requires Stage D's ordinary-versus-donor adaptation and magnitude controls;
it is never inferred from a successful base capture or a low representation error.

On completion or a terminal refusal, append receipts and a source-bound status update,
commit it to the same research branch, and push. Never rewrite the initial bundle.
