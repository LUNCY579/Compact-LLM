"""Hand latent counterexamples and CPU runtime cross-check; no native work."""
import ast
import copy
import json
import random
from pathlib import Path
import unittest

from experiments.rineya_centralized_reasoning_system_v1 import learning_data as data
from experiments.rineya_centralized_reasoning_system_v1.core import run_case
from experiments.rineya_centralized_reasoning_system_v1.data import DEV_FAMILIES


def world(bindings=(), facts=(), conditions=(), aliases=(), cancellations=(), query="parcel", time=9):
    return {"bindings": list(bindings), "facts": list(facts), "conditions": list(conditions),
            "aliases": list(aliases), "cancellations": list(cancellations),
            "query": {"entity": query, "time": time}, "complete": True}


def binding(eid, time, owner=None, mode=None, entity="parcel"):
    return {"id": eid, "time": time, "owner": owner, "mode": mode, "entity": entity}


def fact(eid, time, predicate, status):
    return {"id": eid, "time": time, "predicate": predicate, "status": status}


def guard(eid, relation, target, predicate):
    return {"id": eid, "time": 0, "relation": relation, "target": target, "predicate": predicate}


class LearningDataTests(unittest.TestCase):
    def reseal(self, bundle):
        for key, value in (('public_sha256',bundle['cases']), ('gold_sha256',bundle['private']['gold']),
                           ('latent_sha256',bundle['private']['latent']), ('lineage_sha256',bundle['private']['lineage'])):
            bundle['manifest'][key] = data.digest(value)
        ordered = [c['id'] for c in bundle['cases']];random.Random(211).shuffle(ordered)
        bundle['manifest']['ordered_case_ids'] = ordered
        return bundle

    def test_draw32_distinct_parents_balanced_status_within_every_family(self):
        bundle = data.build_train32()
        self.assertTrue(data.validate_bundle(bundle))
        self.assertEqual(len({c["parent_id"] for c in bundle["cases"]}), 32)
        self.assertEqual(bundle["manifest"]["status_counts"], {s: 8 for s in data.STATUSES})
        for family in data.FAMILIES:
            cases = [c for c in bundle["cases"] if c["family"] == family]
            self.assertEqual(len(cases), 8)
            self.assertEqual({s: sum(bundle["private"]["gold"][c["id"]]["status"] == s for c in cases) for s in data.STATUSES}, {s: 2 for s in data.STATUSES})
        self.assertEqual(set(data.FAMILIES) & set(DEV_FAMILIES), set())
        self.assertEqual(tuple(DEV_FAMILIES), data.V1_DEV_FAMILIES)

    def test_fixed_seed_and_order_reproduce_no_extra_draw_or_roles(self):
        first, second = data.build_train32(), data.build_train32()
        self.assertEqual(first, second)
        self.assertEqual(len(first["manifest"]["ordered_case_ids"]), 32)
        self.assertNotEqual(first["manifest"]["ordered_case_ids"], [c["id"] for c in first["cases"]])
        for args in ({"seed": 102}, {"seed": True}, {"order_seed": 212}, {"order_seed": 211.0}):
            with self.assertRaises(ValueError): data.build_train32(**args)

    def test_generated_sources_match_independent_latent_oracle_in_CPU_core(self):
        bundle = data.build_train32()
        for c in bundle["cases"]:
            with self.subTest(case=c["id"]):
                result = run_case(c)
                self.assertIsNone(result["error"])
                answer = {k: result["answer"][k] for k in ("status", "value")}
                self.assertEqual(answer, bundle["private"]["gold"][c["id"]])
        # Runtime common grammar/rules remain a declared limitation; this does
        # not make either supplied solver a native or independent world truth.

    def test_absence_empty_and_known_owner_are_distinct(self):
        self.assertEqual(data.oracle(world()), {"status": "UNKNOWN", "value": None})
        self.assertEqual(data.oracle(world([binding("e0", 0)])), {"status": "REFUTED", "value": None})
        self.assertEqual(data.oracle(world([binding("e0", 0, "Alice")])), {"status": "SUPPORTED", "value": "Alice"})

    def test_definite_exclusion_overrides_missing_or_contradictory_other_guard(self):
        base = [binding("e0", 0), binding("e1", 2, "Bob", "CONFIRMED")]
        gates = [guard("g0", "REQUIRE", "e1", "permit"), guard("g1", "EXCEPT", "e1", "ban")]
        self.assertEqual(data.oracle(world(base, [fact("f0", 1, "permit", "REFUTED")], gates))["status"], "REFUTED")
        self.assertEqual(data.oracle(world(base, [fact("f0", 1, "ban", "SUPPORTED"), fact("f1", 1, "permit", "CONTRADICTORY")], gates))["status"], "REFUTED")

    def test_future_fact_is_not_evidence_at_transfer_time(self):
        w = world([binding("e0", 0), binding("e1", 2, "Bob", "CONFIRMED")],
                  [fact("f0", 3, "permit", "SUPPORTED")], [guard("g0", "REQUIRE", "e1", "permit")])
        self.assertEqual(data.oracle(w)["status"], "UNKNOWN")
        w["facts"][0]["time"] = 1
        self.assertEqual(data.oracle(w), {"status": "SUPPORTED", "value": "Bob"})

    def test_latest_distinct_facts_create_contradiction_without_echoing_label(self):
        w = world([binding("e0", 0), binding("e1", 2, "Bob", "CONFIRMED")],
                  [fact("f0", 1, "permit", "SUPPORTED"), fact("f1", 1, "permit", "REFUTED")],
                  [guard("g0", "REQUIRE", "e1", "permit")])
        self.assertEqual(data.oracle(w)["status"], "CONTRADICTORY")
        self.assertNotIn('CONTRADICTORY', '\n'.join(data.event_lines(w)))
        w["facts"].append(fact("f2", 2, "permit", "SUPPORTED"))
        self.assertEqual(data.oracle(w)["status"], "SUPPORTED")

    def test_cancellation_replays_empty_and_aliases_do_not_guess(self):
        w = world([binding("e0", 0), binding("e1", 2, "Bob", "CONFIRMED")],
                  cancellations=[{"id": "c0", "time": 3, "target": "e1"}])
        self.assertEqual(data.oracle(w)["status"], "REFUTED")
        w["cancellations"][0]["time"] = 1
        with self.assertRaises(ValueError): data.oracle(w)
        aliases = [{"id": "a0", "time": 0, "alias": "Item", "canonical": "parcel"},
                   {"id": "a1", "time": 0, "alias": "Item", "canonical": "other"}]
        self.assertEqual(data.oracle(world([binding("e0", 0, "Alice")], aliases=aliases, query="Item"))["status"], "UNKNOWN")

    def test_oracle_changes_with_premises_not_generator_desired_status(self):
        w = data._world(data.FAMILIES[0], 0, "SUPPORTED")
        self.assertEqual(data.oracle(w)["status"], "SUPPORTED")
        permit = w["conditions"][0]["predicate"]
        next(f for f in w["facts"] if f["predicate"] == permit)["status"] = "REFUTED"
        self.assertEqual(data.oracle(w)["status"], "REFUTED")
        w["wanted_status"] = "SUPPORTED"
        with self.assertRaises(ValueError): data.oracle(w)

    def test_private_targets_are_explicit_TRAIN_only_and_no_explanations_or_masks(self):
        b = data.build_train32(); targets = data.training_targets(b)
        self.assertEqual([t["case_id"] for t in targets], b["manifest"]["ordered_case_ids"])
        for target in targets:
            value = json.loads(target["target"])
            self.assertEqual(set(value), {"status", "value"})
            self.assertEqual(target["target"], data.canonical(value))
            self.assertIsNone(target["response_tokens"]); self.assertIsNone(target["loss_mask"])
            self.assertFalse(target["native_eligible"])
        for role in ("DEV", "CONFIRM", "CHECK", "FINAL"):
            with self.assertRaises(ValueError): data.training_targets(b, role=role)
        bad = copy.deepcopy(b); bad["cases"][0]["split"] = "DEV"
        with self.assertRaises(ValueError): data.training_targets(bad)
        self.assertTrue(all(set(c) == data.PUBLIC_FIELDS for c in b["cases"]))

    def test_modified_source_gold_counts_order_and_parent_leakage_reject(self):
        base = data.build_train32()
        changes = [lambda b:b["cases"][0].update(source="e0 @0 INIT ldParcel0 Mallory"),
                   lambda b:b["private"]["gold"][b["cases"][0]["id"]].update(value="Mallory"),
                   lambda b:b["manifest"].update(unique_parents=16),
                   lambda b:b["manifest"]["ordered_case_ids"].reverse(),
                   lambda b:b["cases"][1].update(parent_id=b["cases"][0]["parent_id"]),
                   lambda b:b["manifest"].update(native_eligible=True)]
        for change in changes:
            bad = copy.deepcopy(base); change(bad)
            with self.assertRaises(ValueError): data.validate_bundle(bad)

    def test_shortcuts_are_measured_with_all32_denominator(self):
        probes = data.shortcut_probes(data.build_train32())
        self.assertEqual(probes["always_UNKNOWN"]["semantic_exact"], 8)
        self.assertEqual(probes["always_REFUTED"]["semantic_exact"], 8)
        self.assertEqual(probes["last_source_owner"]["semantic_exact"], 0)
        self.assertEqual(probes["last_literal_fact_status"]["semantic_exact"], 4)
        self.assertTrue(all(p["scheduled"] == 32 and len(p["predictions"]) == 32 for p in probes.values()))

    def test_identifier_position_family_and_fact_count_do_not_perfectly_encode_status(self):
        bundle = data.build_train32(); probes = data.shortcut_probes(bundle)
        self.assertTrue(all(not any(ch.isdigit() for ch in c["query"]["entity"]) for c in bundle["cases"]))
        self.assertEqual({sum(' FACT ' in line for line in c["source"].splitlines()) for c in bundle["cases"]}, {4})
        for name in ('query_identifier_mod4', 'query_identifier_hash_mod4', 'draw_position_mod4', 'order_position_mod4', 'family_majority_status', 'FACT_count_majority_status'):
            self.assertLess(probes[name]["status_exact"], 32, name)
        self.assertEqual(probes['query_identifier_mod4']['status_exact'], 8)
        self.assertEqual(probes['family_majority_status']['status_exact'], 8)
        self.assertEqual(probes['FACT_count_majority_status']['status_exact'], 8)

    def test_public_names_times_and_owners_do_not_depend_on_requested_label(self):
        for family in data.FAMILIES:
            worlds = [data._world(family, 7, status) for status in data.STATUSES]
            self.assertTrue(all(w['query'] == worlds[0]['query'] for w in worlds))
            self.assertTrue(all(w['aliases'] == worlds[0]['aliases'] for w in worlds))
            self.assertTrue(all(w['bindings'] == worlds[0]['bindings'] for w in worlds))
            self.assertTrue(all(w['conditions'] == worlds[0]['conditions'] for w in worlds))
            self.assertTrue(all(len(w['facts']) == 4 for w in worlds))

    def test_consistent_opaque_rename_preserves_semantics_and_grounded_source(self):
        bundle = data.build_train32()
        for c in bundle['cases']:
            original = bundle['private']['latent'][c['id']]; renamed = copy.deepcopy(original)
            names = {original['query']['entity']}
            groups = {'bindings':('entity','owner'), 'conditions':('predicate',), 'facts':('predicate',), 'aliases':('alias','canonical')}
            for group, fields in groups.items():
                names.update(row[field] for row in original[group] for field in fields if row[field] is not None)
            mapping = {name:'Renamed'+chr(65+i) for i,name in enumerate(sorted(names))}
            for group, fields in groups.items():
                for row in renamed[group]:
                    for field in fields:
                        if row[field] is not None: row[field] = mapping[row[field]]
            renamed['query']['entity'] = mapping[renamed['query']['entity']]
            before, after = data.oracle(original), data.oracle(renamed)
            self.assertEqual(after, {'status':before['status'], 'value':mapping[before['value']] if before['value'] is not None else None})
            public = copy.deepcopy(c);public['query'] = renamed['query'];public['source'] = '\n'.join(data.event_lines(renamed)+[data.DISTRACTOR])
            result = run_case(public)
            self.assertIsNone(result['error'])
            self.assertEqual({k:result['answer'][k] for k in ('status','value')}, after)

    def test_model_input_omits_case_parent_family_split_and_private_labels(self):
        bundle = data.build_train32()
        for case in bundle['cases']:
            value = data.model_input(case)
            self.assertEqual(set(value), {'source','query','rules'})
            for key in ('id','parent_id','family','split'):
                self.assertNotIn(case[key], data.canonical(value))
            value['query']['entity'] = 'mutated_copy'
            self.assertNotEqual(value['query'], case['query'])

    def test_coherently_resealed_false_family_counts_and_cells_reject_target_exposure(self):
        for mode in ('collapsed', 'status_family'):
            bundle = data.build_train32()
            for case in bundle['cases']:
                case['family'] = data.FAMILIES[0] if mode == 'collapsed' else data.FAMILIES[data.STATUSES.index(bundle['private']['gold'][case['id']]['status'])]
            by_id = {c['id']:c for c in bundle['cases']}
            for row in bundle['private']['lineage']:row['family'] = by_id[row['case_id']]['family']
            self.reseal(bundle)
            with self.assertRaises(ValueError):data.validate_bundle(bundle)
            with self.assertRaises(ValueError):data.training_targets(bundle)

    def test_coherently_resealed_forged_unique_parents_and_cases_reject(self):
        for mode in ('parent','case'):
            bundle = data.build_train32()
            for i, case in enumerate(bundle['cases']):
                if mode == 'parent':case['parent_id'] = 'forged-parent-'+str(i)
                else:
                    previous = case['id'];case['id'] = 'forged-case-'+str(i)
                    for key in ('gold','latent'):bundle['private'][key][case['id']] = bundle['private'][key].pop(previous)
                    bundle['private']['lineage'][i]['case_id'] = case['id']
                bundle['private']['lineage'][i]['parent_id'] = case['parent_id']
            self.reseal(bundle)
            with self.assertRaises(ValueError):data.training_targets(bundle)

    def test_loaded_seed_provenance_requires_exact_int_types(self):
        for key in ('draw_seed','order_seed'):
            for value in (float(data.SEED if key=='draw_seed' else data.ORDER_SEED), True, '101', None):
                bundle = data.build_train32();bundle['manifest'][key] = value
                with self.assertRaises(ValueError):data.training_targets(bundle)

    def test_valid_resealed_alternate_world_is_not_the_registered_bank(self):
        bundle = data.build_train32();case = bundle['cases'][0];cid = case['id'];w = bundle['private']['latent'][cid]
        old = w['bindings'][0]['entity'];new = 'OpaqueAlternateParcel'
        for row in w['bindings']:
            if row['entity'] == old:row['entity'] = new
        for row in w['aliases']:
            if row['alias'] == old:row['alias'] = new
            if row['canonical'] == old:row['canonical'] = new
        if w['query']['entity'] == old:w['query']['entity'] = new
        case['query'] = copy.deepcopy(w['query']);case['source'] = '\n'.join(data.event_lines(w)+[data.DISTRACTOR])
        bundle['private']['gold'][cid] = data.oracle(w)
        row = bundle['private']['lineage'][0];row['world_sha256'] = data.digest(w)
        import hashlib
        row['source_sha256'] = hashlib.sha256(case['source'].encode()).hexdigest()
        self.reseal(bundle)
        # Its controlled semantics are valid, but renaming is an independent
        # invariance control, not permission to replace registered supervision.
        self.assertIsNone(run_case(case)['error'])
        with self.assertRaisesRegex(ValueError,'immutable registered101/211'):data.training_targets(bundle)

    def test_latent_composition_not_just_family_label_must_match(self):
        bundle = data.build_train32();case = bundle['cases'][0];cid = case['id'];w = bundle['private']['latent'][cid]
        w['conditions'] = [g for g in w['conditions'] if g['relation'] != 'EXCEPT']
        case['source'] = '\n'.join(data.event_lines(w)+[data.DISTRACTOR]);bundle['private']['gold'][cid] = data.oracle(w)
        self.reseal(bundle)
        with self.assertRaisesRegex(ValueError,'composition|status cells'):data.validate_bundle(bundle)

    def test_exact_fixed_bank_anchor_rejects_resealed_presentation_and_extra_metadata(self):
        bundle = data.build_train32()
        # Valid JSON/model-input association is not authority to invent another
        # renderer order or an unregistered manifest under the same draw label.
        case = bundle['cases'][0];case['source'] = '\n'.join(reversed(case['source'].splitlines()))
        import hashlib
        bundle['private']['lineage'][0]['source_sha256'] = hashlib.sha256(case['source'].encode()).hexdigest()
        self.reseal(bundle)
        with self.assertRaisesRegex(ValueError,'immutable registered101/211'):data.training_targets(bundle)
        bundle = data.build_train32();bundle['manifest']['unregistered_role'] = 'newbank'
        with self.assertRaisesRegex(ValueError,'immutable registered101/211'):data.training_targets(bundle)

    def test_oracle_module_imports_no_runtime_parser_scorer_or_model(self):
        tree = ast.parse(Path(data.__file__).read_text())
        imports = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
        imports |= {alias.name for n in tree.body if isinstance(n, ast.Import) for alias in n.names}
        self.assertFalse(any(i and any(x in i for x in ('core', 'scoring', 'native', 'torch', 'transformers', 'tokenizers')) for i in imports))


if __name__ == '__main__':
    unittest.main()
