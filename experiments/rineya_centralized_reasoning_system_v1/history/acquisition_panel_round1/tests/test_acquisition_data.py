"""One fixed panel CPU receipt; graph/lineage negatives, no model/token work."""
import ast
import copy
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import acquisition_data as panel


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = panel.build_panel()
        cls.receipt = panel.cpu_receipt(cls.bundle)

    def reseal(self, bundle):
        manifest = bundle['manifest']
        for key, value in (('public',bundle['cases']),('gold',bundle['private']['gold']),('latent',bundle['private']['latent']),
                           ('lineage',bundle['private']['lineage']),('assessments',bundle['private']['assessments'])):
            manifest[key+'_sha256'] = panel.digest(value)
        return bundle

    def test_planned32_counterbalanced_one_variant_root_identity_before_render(self):
        bundle = self.bundle
        self.assertEqual(len(bundle['cases']),32)
        self.assertEqual(len({case['parent_id'] for case in bundle['cases']}),32)
        self.assertEqual(bundle['manifest']['planned_roots'],32)
        self.assertEqual(bundle['manifest']['source_candidate_roots']+bundle['manifest']['unqualified_roots'],32)
        self.assertEqual(bundle['manifest']['status_counts_all_planned'],{state:8 for state in panel.STATUSES})
        for family, cells in bundle['manifest']['family_status_counts_all_planned'].items():
            self.assertEqual(cells,{state:2 for state in panel.STATUSES})
        for slot in range(8):
            self.assertEqual(Counter(panel.assigned_evidence(family,slot) for family in panel.FAMILIES),Counter({state:1 for state in panel.STATUSES}))
        for row in bundle['private']['lineage']:
            ancestry=row['ancestry'];self.assertEqual(ancestry['root_id'],panel.root_identity(ancestry['parameters']))
            self.assertEqual(ancestry['ancestor_roots'],[ancestry['root_id']]);self.assertEqual(ancestry['variant'],0)
        self.assertFalse(bundle['manifest']['LA_I1_002_closed']);self.assertFalse(bundle['manifest']['H01_tested_or_amended'])

    def test_direct_constructor_cannot_use_optimizer_world(self):
        params=panel.parameters(panel.FAMILIES[0],0)
        with patch.object(panel.optimizer,'_world',side_effect=AssertionError('old world must not be copied')):
            world=panel.construct(params,panel.assigned_evidence(params['family'],params['slot']))
        self.assertEqual(sum(row['mode']=='CONFIRMED' for row in world['bindings']),2)
        self.assertEqual(len(world['conditions']),4)
        self.assertEqual(len(world['facts']),6)

    def test_exact32_core_scorer_comparisons_and32_rename_controls(self):
        self.assertEqual(self.receipt['main_comparisons'],32)
        self.assertEqual(self.receipt['rename_checks'],32)
        self.assertEqual(self.receipt['main_passed'],32)
        self.assertEqual(self.receipt['rename_passed'],32)
        self.assertTrue(self.receipt['population_validity_review_open'])
        self.assertFalse(self.receipt['native_or_training_eligible'])

    def test_all_actual_caps_no_padded_or_pruned_rows(self):
        for case in self.bundle['cases']:
            cid=case['id'];world=self.bundle['private']['latent'][cid]
            self.assertLessEqual(len(case['source'].encode()),8192)
            self.assertLessEqual(sum(len(world[group]) for group in panel.GROUPS),256)
            signature=self.bundle['private']['assessments'][cid]['signature']
            self.assertLessEqual(signature['states'],4096)
        self.assertLessEqual(len(panel.canonical(self.bundle).encode()),8*1024*1024)
        self.assertEqual(len(self.receipt['rows']),32)

    def test_public_input_no_gold_root_family_or_provenance_optimizer_api_rejects(self):
        for case in self.bundle['cases']:
            value=panel.model_input(case)
            self.assertEqual(set(value),{'source','query','rules'})
            text=panel.canonical(value)
            self.assertNotIn(case['id'],text);self.assertNotIn(case['parent_id'],text);self.assertNotIn(case['family'],text)
        self.assertFalse(hasattr(panel,'training_targets'))
        with self.assertRaises(ValueError):panel.optimizer.training_targets(self.bundle)
        with self.assertRaises(ValueError):panel.scoring_records(self.bundle,intent='TRAIN_OPTIMIZATION')
        record=panel.scoring_records(self.bundle,intent='ACQUISITION_PROBE_SCORING')
        self.assertEqual(len(record['membership']),32);self.assertFalse(record['optimizer_exposure'])

    def test_optimizer_census32_and_cosmetic_rename_retime_order_duplicate(self):
        roots,deny,unavailable=panel.optimizer_deny()
        self.assertEqual(len(roots),32);self.assertEqual(len(deny),32);self.assertEqual(unavailable,[])
        old=panel.optimizer.build_train32();case=old['cases'][0];world=old['private']['latent'][case['id']]
        changed,_=panel.rename(world)
        self.assertEqual(panel.fingerprints(world)['codes'],panel.fingerprints(changed)['codes'])
        ancestry=copy.deepcopy(self.bundle['private']['lineage'][0]['ancestry'])
        self.assertEqual(panel.assess(changed,ancestry)['state'],'UNQUALIFIED')

    def test_query_cone_and_kinship_ignore_unreferenced_filler(self):
        case=self.bundle['cases'][0];world=self.bundle['private']['latent'][case['id']]
        changed=copy.deepcopy(world)
        changed['facts'].append({'id':'FillerNew','time':3,'predicate':'UnusedNew','status':'SUPPORTED'})
        before,after=panel.fingerprints(world),panel.fingerprints(changed)
        self.assertNotEqual(before['codes']['full'],after['codes']['full'])
        self.assertEqual(before['codes']['cone'],after['codes']['cone'])
        self.assertEqual(before['codes']['kinship'],after['codes']['kinship'])
        ancestry=self.bundle['private']['lineage'][0]['ancestry']
        self.assertEqual(panel.assess(changed,ancestry)['reason'],'NOT_REGISTERED_DIRECT_LATENT_ROOT')

    def test_declared_fact_state_boundary_and_modality_counterfactual_kinship(self):
        old=panel.optimizer.build_train32();case=next(case for case in old['cases'] if case['family']==panel.FAMILIES[0] and old['private']['gold'][case['id']]['status']=='SUPPORTED')
        world=old['private']['latent'][case['id']];before=panel.fingerprints(world)
        predicate=next(row['predicate'] for row in world['conditions'] if row['relation']=='EXCEPT')
        modified=[]
        state=copy.deepcopy(world);next(row for row in state['facts'] if row['predicate']==predicate)['status']='SUPPORTED';modified.append(state)
        boundary=copy.deepcopy(world);fact=next(row for row in boundary['facts'] if row['predicate']==predicate)
        fact['time']=max(row['time'] for row in world['bindings'])+1;modified.append(boundary)
        modality=copy.deepcopy(world);next(row for row in modality['bindings'] if row['mode']=='CONFIRMED')['mode']='PROPOSED';modified.append(modality)
        for changed in modified:
            after=panel.fingerprints(changed)
            self.assertNotEqual(before['codes']['full'],after['codes']['full'])
            self.assertEqual(before['codes']['kinship'],after['codes']['kinship'])
            ancestry=copy.deepcopy(self.bundle['private']['lineage'][0]['ancestry'])
            self.assertEqual(panel.assess(changed,ancestry)['state'],'UNQUALIFIED')

    def test_canonical_ties_and_binding_cutoffs_not_erased(self):
        params=panel.parameters(panel.FAMILIES[0],0)
        supported=panel.construct(params,'SUPPORTED');unknown=panel.construct(params,'UNKNOWN');contrad=panel.construct(params,'CONTRADICTORY')
        codes=[panel.fingerprints(world)['codes']['cone'] for world in (supported,unknown,contrad)]
        self.assertEqual(len(set(codes)),3)
        self.assertEqual(len({panel.fingerprints(world)['codes']['kinship'] for world in (supported,unknown,contrad)}),1)

    def test_duplicate_registered_sibling_cannot_be_new_parent(self):
        case=self.bundle['cases'][0];world=self.bundle['private']['latent'][case['id']]
        ancestry=self.bundle['private']['lineage'][0]['ancestry']
        signature=self.bundle['private']['assessments'][case['id']]['signature']
        self.assertEqual(panel.assess(world,ancestry,[(case['parent_id'],signature)])['state'],'UNQUALIFIED')
        bad=copy.deepcopy(ancestry);bad['operation']='COUNTERFACTUAL_DESCENDANT'
        with self.assertRaises(ValueError):panel.assess(world,bad)

    def test_resealed_family_root_case_lineage_seed_and_world_mutations_reject(self):
        mutations=(lambda b:b['cases'][0].update(family=panel.FAMILIES[1]),
                   lambda b:b['cases'][0].update(parent_id='new-root-id'),
                   lambda b:b['private']['lineage'][0]['ancestry'].update(ancestor_roots=['forged']),
                   lambda b:b['manifest']['registration'].update(draw_seed=float(panel.DRAW)),
                   lambda b:b['manifest']['ordered_case_ids'].reverse(),
                   lambda b:b['private']['latent'][b['cases'][0]['id']]['facts'][0].update(status='SUPPORTED'),
                   lambda b:b['cases'].__setitem__(0,b['cases'][1]))
        for mutate in mutations:
            bad=copy.deepcopy(self.bundle);mutate(bad);self.reseal(bad)
            with self.assertRaises(ValueError):panel.validate_bundle(bad)

    def test_unknown_alias_provenance_and_exhausted_canonicalization_never_distinct(self):
        case=self.bundle['cases'][0];world=self.bundle['private']['latent'][case['id']]
        with self.assertRaises(panel.Unavailable):panel.fingerprints(world,limit=1)
        with self.assertRaises(ValueError):panel.fingerprints(world,limit=True)
        cycle=copy.deepcopy(world);name=cycle['query']['entity']
        cycle['aliases']=[{'id':'Cycle1','time':0,'alias':name,'canonical':'Second'}, {'id':'Cycle2','time':0,'alias':'Second','canonical':name}]
        with self.assertRaises(ValueError):panel.fingerprints(cycle)
        ancestry=copy.deepcopy(self.bundle['private']['lineage'][0]['ancestry']);ancestry['parameters']['draw_seed']=float(panel.DRAW)
        with self.assertRaises(ValueError):panel.assess(world,ancestry)
        good=self.bundle['private']['lineage'][0]['ancestry']
        with patch.object(panel,'fingerprints',side_effect=panel.Unavailable('bounded exhausted fixture')):
            self.assertEqual(panel.assess(world,good)['state'],'UNQUALIFIED')

    def test_source_size_latent_count_oversized_private_and_finite_types_reject(self):
        bad=copy.deepcopy(self.bundle);bad['cases'][0]['source']='雪'*8192
        with self.assertRaises(ValueError):panel.validate_bundle(bad)
        bad=copy.deepcopy(self.bundle);bad['private']['assessments'][bad['cases'][0]['id']]['error']='x'*131073
        with self.assertRaises(ValueError):panel.validate_bundle(bad)
        case=self.bundle['cases'][0];world=copy.deepcopy(self.bundle['private']['latent'][case['id']])
        world['facts']=[{'id':'F'+str(i),'time':0,'predicate':'P','status':'SUPPORTED'} for i in range(257)]
        with self.assertRaises(ValueError):panel.fingerprints(world)
        for seed in (307.0,True,308):
            with self.assertRaises(ValueError):panel.build_panel(seed,panel.ORDER)
        bad=copy.deepcopy(self.bundle);bad['cases'][0]['query']['time']=True
        with self.assertRaises(ValueError):panel.validate_bundle(bad)

    def test_observed_shortcuts_retained_not_absence_claim(self):
        scores=self.receipt['shortcut_controls'];self.assertEqual(scores['denominator_all_planned'],32)
        self.assertEqual(scores['status_correct']['family_label_fitted_majority'],8)
        self.assertEqual(scores['status_correct']['FACT_count_label_fitted_majority'],8)
        self.assertFalse(scores['shortcut_absence_claim'])
        self.assertTrue(all(value<32 for value in scores['status_correct'].values()))

    def test_source_imports_and_optimizer_bytes_preserved(self):
        tree=ast.parse(Path(panel.__file__).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[row.name for row in node.names]
            elif isinstance(node,ast.ImportFrom):names=[node.module or '']
            else:continue
            self.assertFalse(any(name.split('.')[0] in {'torch','transformers','tokenizers','native','operations','reader','subprocess'} for name in names))
        self.assertEqual(panel.file_sha(panel.PACKAGE/'learning_data.py'),panel.OPTIMIZER_SOURCE_SHA)
        self.assertEqual(panel.file_sha(panel.PACKAGE/'contracts/LEARNING_DATA_V1.json'),panel.OPTIMIZER_CONTRACT_SHA)


if __name__=='__main__':unittest.main()
