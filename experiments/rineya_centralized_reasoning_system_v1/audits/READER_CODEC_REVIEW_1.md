# Reader codec review, round 1

Reviewer `/root`; builder `/root/data_audit`, 2026-10-02. Disposition:
**REPAIR_REQUIRED for the CPU preservation bound; native execution remains closed.**
The reviewer inspected source and the draft before accepting any interpretation.
Reviewers share workspace privileges, model lineage and exposed DEV evidence;
this is code review, not independent neural ground truth.

Frozen code SHA256 `79ca7018ca801321d822410135e51edde039a78ba968b90bc1f14ed0cc5e9e6f`,
tests `80975714eab30ec5a6f261191df7d8f7ce87a954e0f029075e57e05c65c0f1de`,
draft `1443d5e3f958ce7fd8a894c4d08d808719dd4cc6fa2cf527aaad65855a17cf80`.
All16 supplied CPU tests passed in0.007s. No framework, tokenizer or model loaded.

The output fields originate in the supplied response; the grammar verifier only
rejects unsupported fields and maps the scheduled ordinal to original source
bytes. It never fills missing model output. Prompts contain the supplied schema
and one quoted source line; case/parent/family/query/gold labels stay outside.
The existing16 DEV variants represent8 parents. Their90 planned calls and
512+64 bounds are prospective dose ceilings, not measured tokenizer fit or
native serving/capacity evidence. This controlled-grammar task does not measure
broad language understanding or autonomous discovery.

## RC-01: preservation bypasses the declared output bound

`decode_case` calls `_preserved(response['output'])` before the bounded decoder.
`_preserved` encodes, decodes, hashes and/or hex-expands the entire value, despite
the4096-byte decoding ceiling. A malformed oversized payload can therefore
allocate and scan large buffers before its eventual rejection. This affects the
bounded CPU interface even though qualified64-token generation is still closed.

Bounded independent reproducer: take `data.reader_cases()['cases'][0]` and one
response for span0 with output `b'\xff' * 1048576`; other cost fields are null and
`operational_error` null. Spy on `_sha` input sizes without changing results.
The final case is FAILED, but its preserved HEX string contains2097152 characters
and `_sha` reads1048576 bytes, both before output-ceiling rejection. A preliminary
reproducer used the wrong bundle key `public` and failed before calling the codec;
the corrected `cases` version reproduced the finding. No original dataset,
protected bank or result was changed.

Verification criterion: reject oversized values before unbounded conversion,
hex expansion, hashing or parsing. Retain bounded diagnostic prefix and known
length, explicitly labeling incomplete raw retention and unavailable full hash;
the future runner owns a separately bounded raw artifact if needed. Accepted
outputs within4096 bytes must remain lossless. Cover large bytes, Unicode and
invalid UTF8/surrogates, wrong outputs, missing spans and no input-derived
fallback. Keep source/contract versions and this failure available. Independent
review of the repair is required; no source inspection may grant native eligibility.
