# Actual registered source loading exercised with inert files

The [private source-loader route](evidence/inert_registered_loader_reviewed_cpu_20261005/prefix_registered_loader.py)
now binds the context's selected temporary layout to a capability-checked
resolver used consistently by finder and installation. The public native
resolver still requires the exact separately reviewed host layout.

Five new controls install the actual finder into an isolated importer/module
universe, import the selected inert `psutil/__init__.py` through the actual
registered source loader, observe source execution, and remove the finder.
Source mutation, unregistered files, foreign capabilities and layout changes
fail; the original meta-path state is restored. The externally assumed
bootstrap module is explicitly not marked as observed execution.

Root passed29 controls including the existing24 regressions. The
[independent29-control review](evidence/inert_registered_loader_reviewed_cpu_20261005/cpu_loader_integration_review_1791239056162974100.json)
also passes on unchanged selected bytes with no actual host runtime/model/cache
reads or native/framework imports. Four workflow sources are auxiliary metadata
fixture inputs, with no new execution acceptance.
Root reproduced all29 controls from the
[preserved copy](evidence/inert_registered_loader_root_replay_20261005.json)
in2.617 seconds, with unchanged source hashes and no failures/errors/skips.

The first test incorrectly expected `ModuleSpec.cached` to remainNone. Python
may derive a documentary cache filename; that does not show cache execution.
The corrected test verifies no cache file is created and actual source-loader
execution. Its previous source and a tool-chunk observation are preserved;
the initial console was not saved as a file and is not reconstructed.

This closes private inert loader mechanics only. Public worker-main/registered
context consumption, actual current runtime842, bootstrap/DLL/environment and
owned resource/GPU qualification remain open. No native gate, model, M0 row,
training, grading or transfer result is added.
