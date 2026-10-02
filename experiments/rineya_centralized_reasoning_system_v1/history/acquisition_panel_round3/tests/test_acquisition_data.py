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
            self.assertEqual(ancestry,panel.ancestry_record(ancestry['parameters']));self.assertEqual(ancestry['variant'],0)
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


class AcquisitionRepairTargetedTests(unittest.TestCase):
    """Never builds a panel or calls oracle/core/scorer/full receipt."""
    def setUp(self):
        for owner,name in ((panel,'cpu_receipt'),(panel,'build_panel'),(panel,'_registered'),
                           (panel,'optimizer_deny'),(panel.optimizer,'oracle'),(panel.optimizer,'_world')):
            target=patch.object(owner,name,side_effect=AssertionError('full-bank/oracle/core work forbidden in targeted checks'))
            target.start();self.addCleanup(target.stop)

    def deep(self):
        root={};cursor=root
        for _ in range(1200):cursor['x']={};cursor=cursor['x']
        return root

    def case(self,world,params):
        return {'id':'targeted-case','parent_id':panel.root_identity(params),'family':params['family'],'split':'TRAIN',
                'source':panel.render(world,panel.root_identity(params)),'query':copy.deepcopy(world['query']),'rules':copy.deepcopy(panel.RULES)}

    def test_rules_nested_cyclic_huge_and_non_scalar_reject_before_JSON(self):
        params=panel.parameters(panel.FAMILIES[0],0);world=panel.construct(params,'REFUTED');base=self.case(world,params)
        cycle={};cycle['cycle']=cycle;loop=[];loop.append(loop)
        bad_values=(self.deep(),cycle,loop,{'cancel_mode':self.deep(),'equal_time':'contradictory','complete':True},
                    dict(panel.RULES,complete=1),dict(panel.RULES,cancel_mode='x'*131073),
                    {'k'+str(index):True for index in range(4097)},dict(panel.RULES,**{'x'*131073:True}))
        with patch.object(panel.json,'dumps',side_effect=AssertionError('invalid rules entered JSON serializer')):
            for rules in bad_values:
                value=dict(base,rules=rules)
                with self.assertRaises(ValueError):panel.validate_public(value)
                with self.assertRaises(ValueError):panel.model_input(value)

    def test_parameters_all_entrypaths_nested_cyclic_extra_and_bool_reject(self):
        good=panel.parameters(panel.FAMILIES[0],0);world=panel.construct(good,'REFUTED');ancestry=panel.ancestry_record(good)
        cycle={};cycle['cycle']=cycle
        bad=(dict(good,extra=self.deep()),dict(good,draw_seed=True),dict(good,slot=0.0),
             dict(good,guarded_candidates=self.deep()),dict(good,recipe_version=cycle),
             {'x'+str(index):1 for index in range(4097)},dict(good,**{'x'*131073:cycle}))
        with patch.object(panel.json,'dumps',side_effect=AssertionError('invalid parameters entered JSON serializer')):
            for params in bad:
                with self.assertRaises(ValueError):panel.root_identity(params)
                with self.assertRaises(ValueError):panel.construct(params,'REFUTED')
                with self.assertRaises(ValueError):panel.assess(world,dict(ancestry,parameters=params))

    def test_generic_JSON_and_bundle_bound_traversal_precede_consumers(self):
        params=panel.parameters(panel.FAMILIES[0],0);world=panel.construct(params,'REFUTED');case=self.case(world,params)
        cycle=[];cycle.append(cycle)
        with patch.object(panel.json,'dumps',side_effect=AssertionError('bad tree entered serializer')):
            for value in (self.deep(),cycle,'x'*131073,{'x'*129:1},[None]*4097,float('nan')):
                with self.assertRaises(ValueError):panel.canonical(value)
            bundle={'cases':[case]*32,'private':{'gold':self.deep(),'latent':{},'lineage':[],'assessments':{}},'manifest':{}}
            with self.assertRaises(ValueError):panel.validate_bundle(bundle)

    def test_seen_signature_and_ancestry_shapes_reject_before_census_or_JSON(self):
        params=panel.parameters(panel.FAMILIES[0],0);world=panel.construct(params,'REFUTED');ancestry=panel.ancestry_record(params)
        nested=self.deep()
        cases=([('root',nested)], [('root',{'codes':{'full':nested,'cone':'x','kinship':'x'},'sha256':{},'states':1})],
               [(nested,{})], [('root',{'codes':{},'sha256':{},'states':True})], [None]*33)
        with patch.object(panel.json,'dumps',side_effect=AssertionError('invalid seen/ancestry entered JSON')):
            for seen in cases:
                with self.assertRaises(ValueError):panel.assess(world,ancestry,seen)
            for value in (dict(ancestry,ancestor_roots=[nested,ancestry['root_id']]),
                          dict(ancestry,prior_panel_root=nested),dict(ancestry,variant=True),dict(ancestry,extra=nested)):
                with self.assertRaises(ValueError):panel.assess(world,value)

    def test_prospective_cells_balanced_and_positions_predeclared_not_depth(self):
        cells=Counter();positions=set()
        for family in panel.FAMILIES:
            recipes=[];interiors=[]
            for slot in range(8):
                params=panel.parameters(family,slot);evidence=panel.assigned_evidence(family,slot);recipe=panel.selection_recipe(params,evidence)
                cells[family,evidence]+=1;recipes.append(recipe)
                if evidence=='SUPPORTED':
                    position=recipe['earlier_supported_position'];positions.add(position)
                    self.assertLess(position,params['guarded_candidates']-1)
                    self.assertEqual(recipe['terminal_disposition'],'EXCLUDED')
                    if position not in {0,params['guarded_candidates']-2,params['guarded_candidates']-1}:interiors.append(position)
            self.assertEqual(sum(row['earlier_supported_position'] is not None for row in recipes),2)
            self.assertEqual(sum(row['terminal_disposition']=='EXCLUDED' for row in recipes),4)
            self.assertGreaterEqual(len(interiors),1)
        self.assertTrue(all(cells[family,status]==2 for family in panel.FAMILIES for status in panel.STATUSES))
        self.assertGreater(len(positions),1)

    def test_constructor_realizes_prior_gate_applicability_opposed_to_terminal(self):
        for family in panel.FAMILIES:
            slot=next(slot for slot in range(8) if panel.assigned_evidence(family,slot)=='SUPPORTED')
            params=panel.parameters(family,slot);recipe=panel.selection_recipe(params,'SUPPORTED')
            positive=panel.construct(params,'SUPPORTED');negative=panel.construct(params,'REFUTED')
            candidates=[row for row in positive['bindings'] if row['mode']=='CONFIRMED' and row['id'] not in {row['target'] for row in positive['cancellations']}]
            for world,expected in ((positive,'SUPPORTED'),(negative,'REFUTED')):
                selected=candidates[recipe['earlier_supported_position']]
                guard=next(row for row in world['conditions'] if row['target']==selected['id'] and row['relation']=='REQUIRE')
                evidence=[row for row in world['facts'] if row['predicate']==guard['predicate'] and row['time']<=selected['time']]
                latest=max(row['time'] for row in evidence)
                self.assertEqual({row['status'] for row in evidence if row['time']==latest},{expected})
                public=panel.public_position_controls(self.case(world,params))
                self.assertEqual(public['controls']['terminal_only']['state'],'EXCLUDED')
            self.assertEqual(len(positive['facts']),len(negative['facts']))

    def test_canonical_relatedness_and_versioned_v1_v2_v3_ancestry_retained(self):
        params=panel.parameters(panel.FAMILIES[0],1)
        positive=panel.construct(params,'SUPPORTED');negative=panel.construct(params,'REFUTED')
        first,second=panel.fingerprints(positive),panel.fingerprints(negative)
        self.assertNotEqual(first['codes']['cone'],second['codes']['cone'])
        self.assertEqual(first['codes']['kinship'],second['codes']['kinship'])
        ancestry=panel.ancestry_record(params)
        self.assertNotEqual(ancestry['root_id'],ancestry['prior_panel_root'])
        self.assertEqual(ancestry['operation'],'RECIPE_REVISION_OF_UNQUALIFIED_PANEL')
        previous=copy.deepcopy(params);previous['recipe_version']=panel.PREVIOUS_VERSION
        self.assertEqual(ancestry['prior_panel_root'],'ap-root-'+panel.digest(previous))
        original=copy.deepcopy(params);original['recipe_version']=panel.ORIGINAL_VERSION
        self.assertEqual(ancestry['original_panel_root'],'ap-root-'+panel.digest(original))
        self.assertEqual(ancestry['ancestor_roots'],[ancestry['original_panel_root'],ancestry['prior_panel_root'],ancestry['root_id']])


class AcquisitionFinalRepairTargetedTests(AcquisitionRepairTargetedTests):
    """The sole candidate3 class; all whole-bank/oracle/core/scorer work fatal."""
    def setUp(self):
        super().setUp()
        import builtins
        original_import=builtins.__import__
        def bounded_import(name,*args,**kwargs):
            if name.rsplit('.',1)[-1] in {'core','scoring','scoring_v2'}:
                raise AssertionError('core/scorer imports forbidden in final targeted checks')
            return original_import(name,*args,**kwargs)
        target=patch.object(builtins,'__import__',side_effect=bounded_import)
        target.start();self.addCleanup(target.stop)

    def test_exact_v2_assignments_independent_of_future_recipe_version(self):
        expected={
            panel.FAMILIES[0]:('SUPPORTED','UNKNOWN','UNKNOWN','SUPPORTED','CONTRADICTORY','REFUTED','CONTRADICTORY','REFUTED'),
            panel.FAMILIES[1]:('REFUTED','CONTRADICTORY','CONTRADICTORY','REFUTED','SUPPORTED','UNKNOWN','SUPPORTED','UNKNOWN'),
            panel.FAMILIES[2]:('UNKNOWN','SUPPORTED','SUPPORTED','UNKNOWN','REFUTED','CONTRADICTORY','REFUTED','CONTRADICTORY'),
            panel.FAMILIES[3]:('CONTRADICTORY','REFUTED','REFUTED','CONTRADICTORY','UNKNOWN','SUPPORTED','UNKNOWN','SUPPORTED'),
        }
        self.assertEqual(panel.FROZEN_V2_EVIDENCE,expected)
        with patch.object(panel,'VERSION','UNOBSERVED_FUTURE_RECIPE_NAME'):
            self.assertEqual({family:tuple(panel.assigned_evidence(family,slot) for slot in range(8)) for family in panel.FAMILIES},expected)
        self.assertEqual(panel.ASSIGNMENT_VERSION,panel.PREVIOUS_VERSION)

    def test_four_predeclared_interior_worlds_defeat_complete_three_position_rule(self):
        planned=((panel.FAMILIES[0],3,1),(panel.FAMILIES[1],4,2),(panel.FAMILIES[2],2,1),(panel.FAMILIES[3],5,3))
        for family,slot,position in planned:
            params=panel.parameters(family,slot);self.assertEqual(panel.assigned_evidence(family,slot),'SUPPORTED')
            self.assertEqual(panel.selection_recipe(params,'SUPPORTED')['earlier_supported_position'],position)
            world=panel.construct(params,'SUPPORTED');controls=panel.public_position_controls(self.case(world,params))
            count=controls['candidate_count'];self.assertTrue(1<=position<=count-3)
            combined=controls['controls']['combined_three_position']
            self.assertEqual(combined['inspected_positions'],sorted({0,count-2,count-1},reverse=True))
            self.assertNotIn(position,combined['inspected_positions'])
            self.assertEqual(combined['prediction'],{'status':'REFUTED','value':None})
            self.assertEqual(combined['candidate_guard_evaluations']+combined['skipped_candidate_guard_evaluations'],count)
            self.assertEqual(controls['controls']['terminal_only']['state'],'EXCLUDED')
        self.assertEqual(panel.registration()['combined_three_position_max_exact'],28)


if __name__=='__main__':unittest.main()
