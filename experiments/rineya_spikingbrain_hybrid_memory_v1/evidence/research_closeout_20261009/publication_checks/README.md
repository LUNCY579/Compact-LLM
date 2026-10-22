# Publication verification records

The final independent reviews are:

- [Native code review](native_publication_code_review.json): no material code findings;
  frozen code/proposal/anchor bindings and preserved guards verified.
- [Archive documentation review](final_archive_docs_review.json): 48 passing checks,
  resolved links, correct scientific limitations and accurate fixture history.
- [Final provenance/scope review](provenance_archive_scope_review_final.json): 26
  passing checks, read-only verifier and matching byte copies, zero issues.
- [Fixture receipt](publication-tests-receipt.json): final 222 passed, zero skips,
  with both earlier operator setup failures preserved in their JUnit files.

Reviews ran before these review files were added to the publication. Their
initial archive-manifest fingerprint is preserved by
[reviewed-initial-MANIFEST.json](reviewed-initial-MANIFEST.json). The enclosing
[final manifest](../MANIFEST.json) seals the extended archive, including these
unchanged review records. Counts differ only because publication added reviews
and this index; the frozen study code and original receipts did not change.

[provenance_archive_scope_review.json](provenance_archive_scope_review.json) is
a superseded draft. Its two flags were auditor predicates: a literal prose
match and treating immutable string `replace` as a filesystem mutation. The
final review corrected those assumptions without changing the document or
verifier. [archive_copy_regex_adjudication.json](archive_copy_regex_adjudication.json)
likewise resolves two loose credential-pattern matches inside ordinary words;
no token-bounded credential matched. Draft flags are not native run failures.

Every record keeps its original bytes, status and scope. These reviews concern
publication and engineering integrity, with no model run, protected-bank read,
new native permission, capability confirmation or promotion.
