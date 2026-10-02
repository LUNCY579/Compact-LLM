# Reader codec repair review, round 2

Reviewer `/root`; builder `/root/data_audit`, 2026-10-02. Disposition:
**CPU_PREPARATION_AND_DECODING_HELPER_ACCEPTED; RC-01 CLOSED.** This grants no
tokenizer fit, native collection, reader qualification or efficacy claim.

Frozen code SHA256 `4cf31a69aac9a3ed78a06db91d16288ebf8f7e4fcffdc8385a3f6034aff24e14`,
tests `c5bfd6c0d971ecf1e12c3539d543be943080bccba8b3cdc6af6367d12b2944da`,
draft `ec3aed2fa9b211d8121eef76f115c792575cad0291ace3c0dde77284a5ea2e2a`.
The withdrawn first freeze is byte-preserved in `history/reader_codec_round1/`.
Shared workspace/model/evidence limitations from round1 still apply.

The reviewer reran all21 lightweight tests (0.009s) and the independently authored
1MiB invalid-UTF8 response counterexample. The case remained FAILED; retention
now contains256 bytes of prefix, actual byte length1048576, `sha256:null`,
`full_raw_preserved:false` and an explicit external raw-artifact requirement.
The largest `_sha` input was635 bytes from the ordinary prompt preparation;
the oversized response was never hashed or expanded in full.

Source character admission now precedes Evidence encoding, source byte admission
precedes hashing/span copies, oversized response character/byte admission
precedes preservation/parsing, and operational error text is bounded before
stripping or concatenation. Supported outputs remain lossless; rejected oversized
outputs explicitly retain incomplete diagnostics. Wrong fields, incomplete
coverage and unsupported fit never become successful events or abstention.

The helper is supplied controlled-grammar machinery. Model-derived fields are
checked against public source and mapped to original UTF8 references; no field
is filled by a source-derived fallback. This does not measure broad language
understanding, Rineya procedure discovery, native truth or learning. No framework,
tokenizer, model, GPU call or protected bank was accessed by this review.

Future collection must bind the exact qualified serving bundle, measured complete
chat-template/tokenizer input fit, raw generated IDs/EOS/output, actual calls,
worker time and immutable case/parent bank/scorer. The draft's16 variants/8 parents,
90 calls,5760 possible forwards and3130560 input-position ceiling remain planned
counts; aggregate sustained execution and runner admission are separate gates.
The inherited runtime quarantine remains closed.
