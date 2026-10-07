# Full owned external record joins at terminal audit

Gate002 pipeline writes a five-field schema2 external record containing config,
nine caller anchors, grant reference and owner-receipt reference. Its auditor
passed only the historical two-field projection to the preservation consumer,
which compared that projection against the full recorded file and would refuse
every V2 result. The cd09b0d9 source is retained in pre_V2terminal_join002.

The successor passes the actual original/preserved full V2 bytes. It reads them
through the externally anchored preservation descriptor, compares independent
config/nine-anchor selections, and binds both document references to the actual
already-enrolled source gate references. It reads both documents by their exact
file/body anchors. No caller boolean or raw record can replace that selection.
The original source/runtime/Git and owner grant validation remains unchanged.

Four CPU controls run the actual native preservation source body with invented,
explicitly labeled SUP/death/gate documents. Full V2 evidence passes; missing
schema, rebound reference, ninth-anchor change and changed document bytes refuse.
The fixture only sets package location for the real input path and uses original
writer/core bytes; it is not a native event or human approval witness. Controls001
contains two fixture-location errors. Controls004 contains a test-pattern failure
because the added separate-document check rejected earlier; that refusal is
retained. Final controls005 has four passes, zero failures/errors.

Only prefix_input_owned_auditor.py and new test_V2terminal_join.py were edited.
C/P/W and runtime/gate permissions were not changed. Existing native resource,
lease, death, model prohibition and purpose checks remain. The source fix makes
the recorded V2 shape consumable; it supplies no new run authorization.
