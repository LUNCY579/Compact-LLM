"""Synthetic evaluator controls only; no student responses or native measurements."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import dev_scorer as s


def encoded(value):
    return json.dumps({"answer": value}, separators=(",", ":"), ensure_ascii=False)


def parent(module, family, role="DEV"):
    for index in range(10000):
        candidate = module.allocate_parent(module.generate_spec(family, 17, index), 17, index)
        if candidate.role == role:
            return candidate
    raise AssertionError("no mechanical fixture")


def wrap(source, panel):
    language = panel == "language"
    return {
        "row_id": source["parent_id"] + "-dev" if language else source["row_id"],
        "parent_id": source["parent_id"], "family": "general_language" if language else source["family"],
        "stratum": source["structure_id"] if panel == "acquisition" else source["evidence_position"] if panel == "long_context" else source["stratum"],
        "variant": "primary" if language else f"{source['prompt_tokens']}-{source['variant_id']}" if panel == "long_context" else str(source["variant_id"]),
        "role": "DEV", "panel": panel,
        "prompt_token_hash": source["input_token_ids_sha256"] if panel == "long_context" else "a" * 64,
        "scoring_payload": {"schema_version": 1, "panel": panel, "source_row_sha256": s.digest(source), "source_row": deepcopy(source)},
    }


def short(family, variant=None):
    module = s.bridge if family in s.bridge.FAMILIES else s.retention
    p = parent(module, family)
    if variant is None:
        variant = 0 if module is s.bridge else "answerable" if family == "answerability_abstention" else "standard"
    row = module.make_row(p, "a" * 40, variant)
    return wrap(row, "acquisition" if module is s.bridge else "deterministic_retention")


def rehash(row):
    source = row["scoring_payload"]["source_row"]
    if "row_sha256" in source:
        source["row_sha256"] = s.digest({k:v for k,v in source.items() if k != "row_sha256"})
    row["scoring_payload"]["source_row_sha256"] = s.digest(source)


def lc(family=None, answerable=True, adversarial=False, length=2024):
    """Synthetic token metadata; never claimed to be native tokenization or fit."""
    t = s.long_context
    family = family or t.FAMILIES[0]
    for index in range(10000):
        p = t.allocate_parent(t.generate_spec(family, 17, index))
        if p.role == "DEV":
            break
    else:
        raise AssertionError("no DEV fixture")
    variant = t.render_variant(p, answerable, adversarial)
    body = "\n".join(variant["support_slots"] + variant["decoys"]) + "\n"
    prompt = t.SYSTEM_PROMPT + "\nCONTEXT\n" + body + "QUERY\n" + variant["query"]
    rendered = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    variant_id = ("answerable" if answerable else "unanswerable") + ("_adversarial" if adversarial else "_neutral")
    char_start = rendered.index(body)
    source = {
        "schema_version":1, "row_id":f"{p.parent_id}-{length}-{variant_id}", "parent_id":p.parent_id, "role":"DEV",
        "generator_version":"long-context-native-additive-note-dp-v1", "semantic_generator_version":t.VERSION,
        "verifier_version":t.VERIFIER_VERSION, "support_unit":"record_block", "split_sha256":p.split_sha256,
        "family":family, "evidence_position":p.placement, "variant_id":variant_id,
        "distractor_modes":variant["distractor_modes"], "answerable":answerable, "adversarial":adversarial,
        "missing_index":p.missing_index, "canonical_parent_sha256":s._text_hash(p.canonical_json),
        "structural_cluster_id":p.structural_cluster_id, "query":variant["query"], "context_body":body,
        "prompt":prompt, "rendered_prompt":rendered, "input_ids":[17]*length, "prompt_tokens":length,
        "body_tokens":100, "total_sequence_envelope":length+24, "generation_reserve_tokens":24,
        "target":variant["target"], "target_token_ids":[42], "target_tokens":1,
        "prompt_sha256":s._text_hash(prompt), "rendered_prompt_sha256":s._text_hash(rendered),
        "context_body_sha256":s._text_hash(body), "target_sha256":s._text_hash(variant["target"]),
        "input_token_ids_sha256":s.digest([17]*length), "target_token_ids_sha256":s.digest([42]),
        "template_kwargs":{"add_generation_prompt":True,"enable_thinking":False},
        "support_spans":[{"dependency_index":i,"region":p.placement,"required_fact_present":answerable or i!=p.missing_index} for i in range(3)],
        "controlled_slots":[{"dependency_index":i,"present":answerable or i!=p.missing_index} for i in range(3)],
        "body_whole_prompt_token_start":5, "body_whole_prompt_token_end":105,
        "body_whole_prompt_char_start":char_start, "body_whole_prompt_char_end":char_start+len(body),
        "native_offset_mapping":[[0,0] for _ in range(length)], "visible_text_oracle":variant["target"],
        "semantic_verification":t.verify_variant(p,variant), "individual_ablations_unknown":True,
        "policy_input_only":["prompt","input_ids"], "training_eligibility":False,
        "source_correct_membership":None, "native_qualification_inferred":False, "e1_launch_ready":False,
    }
    source["row_sha256"] = s.digest(source)
    return wrap(source,"long_context")


def language():
    world = {"parent_id":"LDEV-explanation-01", "role":"DEV", "stratum":"explanation", "conversation_form":"single",
        "author_id":"/root/baseline_enrollment", "semantic_world":{
            **{field:["Synthetic fixture fact"] for field in ("actors","facts","timeline","constraints","potential_confusions","exclusion_tags")},
            "task_intent":"Mechanical fixture only", "decisive_structure":"No native scoring claim"}}
    payload = {"parent_id":world["parent_id"], "world_spec_sha256":s.digest(world),
        "messages":[{"role":"user","content":"Synthetic fixture prompt."}],
        "semantic_contract":{
            **{field:["Fixture guidance"] for field in ("must_include","must_avoid","acceptable_variation")},
            "dimensions":dict.fromkeys(("adequacy","relevance","fluency","role_consistency"),"Unqualified fixture guidance")},
        "dev_exemplar":"A synthetic illustrative reply, not a model output.", "cal_controls":[], "authorship":"assistant_source_candidate"}
    source = {**world, **payload, "spec_file_sha256":"a"*64,
        "spec_source_path":"sources/language_dev_cal_v1/SPECS_A.json", "payload_file_sha256":"b"*64,
        "payload_source_path":"sources/language_dev_cal_v1/PAYLOADS_A.json", "payload_world_sha256":s.digest(payload)}
    source["row_sha256"] = s.digest(source)
    return wrap(source,"language")


class DevScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shorts = {family:short(family) for family in (*s.bridge.FAMILIES,*s.retention.DOMAINS)}

    def test_correct_targets_for_every_short_family(self):
        for family,row in self.shorts.items():
            with self.subTest(family=family):
                target = row["scoring_payload"]["source_row"]["target"]
                self.assertIsNone(s.validate_evaluator_row(row))
                result = s.score_response(row," \n"+target+"\t")
                self.assertEqual(result,{"correct":True,"malformed":False})
                self.assertTrue(all(type(v) is bool for v in result.values()))

    def test_wrong_well_shaped_values_are_not_malformed(self):
        wrongs = {"bounded_integer_arithmetic":10**100,"symbolic_variable_updates":[10**80,0,0],
            "finite_set_queries":[10**60],"constrained_text_transformations":"SNOWMAN-☃",
            "retrieval":["IMPOSSIBLE","IMPOSSIBLE"],"instruction_following":"WRONG", "answerability_abstention":None}
        for family,value in wrongs.items():
            with self.subTest(family=family):
                self.assertEqual(s.score_response(self.shorts[family],encoded(value)),{"correct":False,"malformed":False})

    def test_family_types_are_malformed_including_boolean_integers(self):
        wrongs = {"bounded_integer_arithmetic":[True,1.0,None,"1",[1]],
            "symbolic_variable_updates":[[1,2],[1,2,3,4],[1,True,3],None],
            "finite_set_queries":[[2,1],[1,1],[False],{},"[]"],
            "constrained_text_transformations":[None,1,[]], "retrieval":[None,["one"],["one",2]],
            "instruction_following":[None,["a","b"]], "answerability_abstention":[False,{},["one"],["one",None]]}
        for family,values in wrongs.items():
            for value in values:
                with self.subTest(family=family,value=value):
                    self.assertEqual(s.score_response(self.shorts[family],encoded(value)),{"correct":False,"malformed":True})

    def test_empty_strings_and_empty_set_remain_valid_shapes(self):
        for family,value in (("constrained_text_transformations",""),("instruction_following",""),("finite_set_queries",[]),("retrieval",["",""])):
            with self.subTest(family=family):
                self.assertFalse(s.score_response(self.shorts[family],encoded(value))["malformed"])

    def test_answerability_twins_share_nullable_shape(self):
        answerable = self.shorts["answerability_abstention"]
        missing = short("answerability_abstention","missing_evidence")
        attempted = answerable["scoring_payload"]["source_row"]["target"]
        self.assertEqual(s.score_response(answerable,'{"answer":null}'),{"correct":False,"malformed":False})
        self.assertEqual(s.score_response(missing,'{"answer":null}'),{"correct":True,"malformed":False})
        self.assertEqual(s.score_response(missing,attempted),{"correct":False,"malformed":False})

    def test_json_envelope_duplicates_constants_extra_prose(self):
        row = self.shorts["bounded_integer_arithmetic"]
        invalid = ['{"answer":1,"answer":1}','{"answer":1,"reason":"x"}', '[1]', '{}',
            '{"answer":NaN}','{"answer":Infinity}','{"answer":-Infinity}','{"answer":1e999}',
            'Answer: {"answer":1}','{"answer":1} trailing','{"answer":1}{"answer":1}',None,True]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(s.score_response(row,raw),{"correct":False,"malformed":True})

    def test_short_character_limit_is_exact(self):
        row = self.shorts["bounded_integer_arithmetic"]
        target = row["scoring_payload"]["source_row"]["target"]
        raw = " "*(4096-len(target))+target
        self.assertEqual(s.score_response(row,raw),{"correct":True,"malformed":False})
        self.assertEqual(s.score_response(row," "+raw),{"correct":False,"malformed":True})

    def test_scoring_never_calls_reference_generators(self):
        row = self.shorts["bounded_integer_arithmetic"]
        target = row["scoring_payload"]["source_row"]["target"]
        with patch.object(s.bridge,"reference_answer",side_effect=AssertionError("reference used")), \
             patch.object(s.bridge,"make_row",side_effect=AssertionError("generator used")):
            self.assertTrue(s.score_response(row,target)["correct"])
        row = self.shorts["retrieval"]
        with patch.object(s.retention,"reference_answer",side_effect=AssertionError("reference used")), \
             patch.object(s.retention,"make_row",side_effect=AssertionError("generator used")):
            self.assertTrue(s.score_response(row,row["scoring_payload"]["source_row"]["target"])["correct"])

    def test_source_target_corruption_raises_even_when_rehashed(self):
        for family in ("bounded_integer_arithmetic","retrieval"):
            row = deepcopy(self.shorts[family]);source=row["scoring_payload"]["source_row"]
            source["target"] = encoded(999999 if family=="bounded_integer_arithmetic" else ["IMPOSSIBLE","IMPOSSIBLE"])
            source["target_sha256"] = s._text_hash(source["target"]);rehash(row)
            with self.subTest(family=family),self.assertRaisesRegex(ValueError,"independent"):
                s.score_response(row,"malformed model text")

    def test_source_role_parent_spec_prompt_and_envelope_drift_raise(self):
        for field,value in (("role","CHECK"),("parent_id","bridge-"+"0"*64),("split_sha256","0"*64),
                            ("prompt","altered"),("verifier_version","wrong"),("source_revision","short")):
            row=deepcopy(self.shorts["bounded_integer_arithmetic"]);row["scoring_payload"]["source_row"][field]=value;rehash(row)
            with self.subTest(field=field),self.assertRaises(ValueError):s.score_response(row,"bad")
        row=deepcopy(self.shorts["bounded_integer_arithmetic"]);row["scoring_payload"]["source_row"]["extra"]=True;rehash(row)
        with self.assertRaisesRegex(ValueError,"schema"):s.validate_evaluator_row(row)
        row=deepcopy(self.shorts["bounded_integer_arithmetic"]);row["scoring_payload"]["source_row"]["reward"]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):s.score_response(row,"bad")

    def test_each_outer_metadata_mapping_is_checked(self):
        for field in ("row_id","parent_id","family","stratum","variant"):
            row=deepcopy(self.shorts["retrieval"]);row[field]="wrong"
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"mapping"):s.validate_evaluator_row(row)
        for field,value in (("role","FINAL"),("prompt_token_hash","bad")):
            row=deepcopy(self.shorts["retrieval"]);row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):s.validate_evaluator_row(row)
        row=deepcopy(self.shorts["retrieval"]);row["scoring_payload"]["source_row_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"hash drift"):s.score_response(row,"bad")

    def test_long_context_all_families_lengths_and_eight_variants(self):
        for family in s.long_context.FAMILIES:
            for length in (2024,4072):
                for answerable in (True,False):
                    for adversarial in (True,False):
                        row=lc(family,answerable,adversarial,length);target=row["scoring_payload"]["source_row"]["target"]
                        with self.subTest(family=family,length=length,answerable=answerable,adversarial=adversarial):
                            with patch.object(s.long_context,"reference_answer",side_effect=AssertionError("reference used")):
                                self.assertEqual(s.score_response(row," \t"+target+"\n"),{"correct":True,"malformed":False})

    def test_long_context_parser_is_strict_not_diagnostic(self):
        row=lc();target=row["scoring_payload"]["source_row"]["target"]
        for raw in ("Answer: "+target,encoded(target),target.lower(),target+".",target+" extra",None):
            with self.subTest(raw=raw):self.assertEqual(s.score_response(row,raw),{"correct":False,"malformed":True})
        self.assertEqual(s.score_response(row,"UNKNOWN"),{"correct":False,"malformed":False})
        missing=lc(answerable=False)
        self.assertEqual(s.score_response(missing,target),{"correct":False,"malformed":False})

    def test_long_context_visible_oracle_and_token_binding_corruption_raise(self):
        row=lc();source=row["scoring_payload"]["source_row"]
        source["target"]="UNKNOWN";source["target_sha256"]=s._text_hash("UNKNOWN");rehash(row)
        with self.assertRaisesRegex(ValueError,"oracle"):s.score_response(row,"UNKNOWN")
        row=lc();row["prompt_token_hash"]="0"*64
        with self.assertRaisesRegex(ValueError,"mapping"):s.validate_evaluator_row(row)
        row=lc();source=row["scoring_payload"]["source_row"];source["input_ids"][0]=True;rehash(row)
        with self.assertRaisesRegex(ValueError,"token schema"):s.score_response(row,"bad")
        row=lc();row["scoring_payload"]["source_row"]["context_body"]+="unregistered evidence\n";rehash(row)
        with self.assertRaises(ValueError):s.validate_evaluator_row(row)

    def test_language_identity_only_never_matches_exemplar(self):
        row=language();self.assertIsNone(s.validate_evaluator_row(row))
        with self.assertRaises(s.UnqualifiedLanguageGraderError):
            s.score_response(row,row["scoring_payload"]["source_row"]["dev_exemplar"])
        row["scoring_payload"]["source_row"]["messages"][0]["content"]="Changed without payload hash";rehash(row)
        with self.assertRaisesRegex(ValueError,"hash chain"):s.validate_evaluator_row(row)

    def test_language_metadata_and_closed_role_rejected(self):
        row=language();row["family"]="arithmetic"
        with self.assertRaisesRegex(ValueError,"mapping"):s.validate_evaluator_row(row)
        row=language();row["scoring_payload"]["source_row"]["role"]="CAL";rehash(row)
        with self.assertRaisesRegex(ValueError,"DEV"):s.validate_evaluator_row(row)

    def test_dependencies_are_fixed_bound_and_guard_every_entry(self):
        binding=s.dependency_binding();self.assertIsNone(s.validate_dependency_binding(binding))
        self.assertEqual(set(binding["files"]),{"dev_scorer.py",*s.HELPER_SHA256})
        self.assertEqual(binding["dependency_sha256"],s.digest(binding["files"]))
        for name,expected in s.HELPER_SHA256.items():
            self.assertEqual(binding["files"][name]["sha256"],expected)
        bad=deepcopy(binding);bad["files"]["dev_scorer.py"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"binding drift"):s.validate_dependency_binding(bad)
        identity=s._identity
        def drift(path):
            return {"sha256":"0"*64,"bytes":1} if path.name=="retention_data.py" else identity(path)
        with patch.object(s,"_identity",side_effect=drift):
            for call in (lambda:s.validate_evaluator_row(self.shorts["retrieval"]),
                         lambda:s.score_response(self.shorts["retrieval"],"bad"), s.dependency_binding):
                with self.assertRaises(s.DependencyDriftError):call()

    def test_source_and_caller_inputs_not_mutated(self):
        row=self.shorts["retrieval"];before=deepcopy(row)
        s.score_response(row,row["scoring_payload"]["source_row"]["target"])
        self.assertEqual(row,before)

    def test_import_has_no_native_or_preparation_side_effects(self):
        root=Path(__file__).resolve().parents[3]
        code=("import os,sys;before=dict(os.environ);"
              "from experiments.post_training_mimo_centralized_v1.scripts import dev_scorer;"
              "assert dict(os.environ)==before;"
              "assert not any(n.startswith(('torch','transformers','tokenizers')) or 'tokenize_' in n or 'build_long_context' in n for n in sys.modules)")
        done=subprocess.run([sys.executable,"-B","-c",code],cwd=root,env=dict(os.environ),capture_output=True,text=True,timeout=20)
        self.assertEqual(done.returncode,0,done.stderr)

    def test_dependency_bytes_checked_before_local_import_execution(self):
        # An isolated copy exercises startup failure without editing a real helper.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            copied=root/"dev_scorer.py"
            copied.write_bytes(Path(s.__file__).read_bytes())
            for name in s.HELPER_SHA256:
                (root/name).write_text("raise AssertionError('unregistered helper executed')\n",encoding="utf-8")
            code=("import runpy,sys;\ntry: runpy.run_path(sys.argv[1])\n"
                  "except ValueError as e: assert type(e).__name__=='DependencyDriftError' and 'unregistered verifier' in str(e)\n"
                  "else: raise AssertionError('changed helper was accepted')\n")
            done=subprocess.run([sys.executable,"-B","-c",code,str(copied)],capture_output=True,text=True,timeout=20)
            self.assertEqual(done.returncode,0,done.stderr)


if __name__ == "__main__":
    unittest.main()
