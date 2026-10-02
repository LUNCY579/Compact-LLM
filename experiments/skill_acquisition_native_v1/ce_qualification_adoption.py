"""Reviewed fixed canonical qualification metadata paths; no IO or model grant.

Only executable source entry points call adopt_source_paths(). Imports and
ordinary test/API use remain closed. The declaration has no request/source
hashes; those are generated after this source commit and admitted independently.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from . import ce_native_authority as authority

OUTPUT = r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\r-oracle-sft-qualification-20261002\canonical'
CHECKOUT = r'C:\Users\clclc\BusinessProject\ChatBubble-comp-rs-s1-dev-20260930'
CHOICE = '9e39a9d443f4f4343bb6ad6c21192b97e1a7964d6195dd360ac0a3b5e5377fe5'
RECIPE = '32c9a1e326a6ac58d3ac320fd85a215ffc4b94476b3f23bcc8288f0c32940f40'


def declarations():
    root = Path(OUTPUT)
    bounds = {'read_bytes': 64 * 1024 ** 2, 'write_bytes': 1024 ** 2, 'hash_attempts': 10000,
        'hash_bytes': 256 * 1024 ** 2, 'metadata_bytes': 1024 ** 2, 'operation_attempts': 10000, 'wall_seconds': 120}
    common = {'source_checkout': CHECKOUT, 'output_root': OUTPUT, 'choice_sha256': CHOICE,
              'static_recipe_sha256': RECIPE, 'bounds': bounds}
    parent = dict(common, schema='CE8-QUALIFICATION-METADATA-ADOPTION-1',
        configuration_path=str(root / 'registration' / 'PARENT_CONFIGURATION.json'),
        owned_directory=str(root / 'metadata-intake' / 'parent'))
    child = dict(common, schema='CE8-CHILD-CONFIGURATION-ADOPTION-1',
        configuration_root=str(root / 'checkpoints'), owned_directory=str(root / 'metadata-intake' / 'children'))
    return parent, child


def adopt_source_paths():
    """Fixed compiled declaration only, no arguments, files, argv or env."""
    from . import ce_qualification_configuration as parent, ce_checkpoint_configuration as child
    first, second = declarations()
    parent.validate_declaration(first); child.validate_declaration(second)
    for module, value in ((parent, first), (child, second)):
        authority.require(module._SOURCE_ADOPTION is None or dict(module._SOURCE_ADOPTION) == value,
                          'source metadata adoption cannot switch experiment')
    parent._SOURCE_ADOPTION, child._SOURCE_ADOPTION = MappingProxyType(first), MappingProxyType(second)
