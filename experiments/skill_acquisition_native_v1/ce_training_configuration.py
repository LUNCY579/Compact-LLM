"""Paid fixed-source production configuration intake; closed by default."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
from types import MappingProxyType

from . import ce_qualification_configuration as intake, ce_qualification_artifacts as raw

DECLARATION='CE8-TRAINING-METADATA-ADOPTION-1'
_SOURCE_DECLARATION=_SOURCE_DECLARATION_SHA256=None
_LOADED=_EVIDENCE=None
require,canonical,digest=raw.require,raw.canonical,raw.digest


def validate_declaration(value):
    require(type(value) is dict and value.get('schema')==DECLARATION and raw._hex(value.get('qualification_selection_sha256')),
            'fixed completed-qualification source selection pin required')
    parent=copy.deepcopy(value);parent.pop('qualification_selection_sha256');parent['schema']=intake.DECLARATION
    intake.validate_declaration(parent)
    return copy.deepcopy(value)


def load_source_configuration():
    global _LOADED,_EVIDENCE
    require(type(_SOURCE_DECLARATION) is MappingProxyType and raw._hex(_SOURCE_DECLARATION_SHA256),
            'fixed production source declaration remains closed')
    declaration=validate_declaration(copy.deepcopy(dict(_SOURCE_DECLARATION)))
    require(digest(declaration)==_SOURCE_DECLARATION_SHA256,'literal source declaration pin drift')
    if _LOADED is not None:
        require(_EVIDENCE['declaration_sha256']==_SOURCE_DECLARATION_SHA256,'production source intake cannot switch')
        return copy.deepcopy(_LOADED),copy.deepcopy(_EVIDENCE)
    payment=intake.reserve_metadata_intake(declaration,'parent')
    counts,bounds=payment['counts'],payment['bounds'];head=payment['head']
    try:
        path=Path(declaration['configuration_path'])
        require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=intake.MAX_CONFIGURATION,
                'regular bounded fixed production configuration')
        with path.open('rb') as stream:blob=stream.read(intake.MAX_CONFIGURATION+1)
        counts['read_bytes']+=len(blob);counts['hash_attempts']+=1;counts['hash_bytes']+=len(blob)
        require(0<len(blob)<=intake.MAX_CONFIGURATION and counts['read_bytes']<=bounds['read_bytes'],'production intake read bound')
        value=raw._json(blob)
        from . import ce_training_entry as entry
        counts['hash_attempts']+=2;counts['hash_bytes']+=2*intake.MAX_CONFIGURATION
        require(counts['hash_attempts']<=bounds['hash_attempts'] and counts['hash_bytes']<=bounds['hash_bytes'],'production intake digest envelope')
        entry.validate_configuration(value)
        policy=value['service_spec']['scope_adapters']['worker']
        require(value['request']['selectedoperatorchoice_ref']['sha256']==declaration['choice_sha256']
            and value['service_spec']['scope_adapters']['static_recipe_sha256']==declaration['static_recipe_sha256']
            and policy['source_checkout']==declaration['source_checkout']
            and value['service_spec']['resource']['resource_probe']['output_root']==declaration['output_root']
            and value['qualification_selection_sha256']==declaration['qualification_selection_sha256'],
            'production source choice/recipe/checkout/output/qualification pin mismatch')
        for owned in (value['service_spec']['bootstrap']['owned_directory'],value['service_spec']['constructor_prepayment']['owned_directory']):
            require(Path(owned).is_relative_to(Path(declaration['output_root'])),'production metadata owner escaped fixed output')
        reference={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
        head=intake._append(payment['journal'],payment['declaration_sha256'],head,'RETURNED',
            {'configuration_ref':reference,'configuration_sha256':value['configuration_sha256'],
             'qualification_selection_sha256':value['qualification_selection_sha256'],'metadata_only':True,
             'native_qualified':False},bounds,counts,payment['origin'])
        evidence={'schema':intake.RECORD,'declaration_sha256':payment['declaration_sha256'],
            'journal_path':str(payment['journal']),'ledger_head':head,'configuration_ref':reference,
            'permanent_bounds':copy.deepcopy(bounds),'actual_tracked_counts':counts,'metadata_only':True,
            'native_qualified':False,'execution_verified':False}
        _LOADED,_EVIDENCE=copy.deepcopy(value),copy.deepcopy(evidence)
        return value,evidence
    except BaseException as primary:
        try:intake._append(payment['journal'],payment['declaration_sha256'],head,'FAILED',
            {'type':type(primary).__name__,'message':str(primary)[:1000],'charges_refunded':False},bounds,counts,payment['origin'])
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise
