"""Complete synthetic DEV artifact schemas; no tokenizer/model or production writes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from experiments.post_training_mimo_centralized_v1.scripts import dev_inputs as d


def jsonl(rows):
    return b"".join(d.canonical(row) + b"\n" for row in rows)


def text_hash(text):
    return d.identity(text.encode())["sha256"]


def wrapper(prompt):
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def short_row(panel, family, ordinal, stratum, variant):
    parent = f"SYNTHETIC-{panel}-{family}-{ordinal}"
    row_id = parent + ("-v0" if panel == "acquisition" else "-" + variant)
    prompt = "Synthetic contract prompt " + row_id
    target = '{"answer":"REFERENCE_ONLY_FIXTURE"}'
    seed = 20260927 if panel == "acquisition" else 20260928
    spec = {"family" if panel == "acquisition" else "domain": family, "fixture": ordinal}
    source = {"schema_version": 1, "row_id": row_id, "parent_id": parent, "role": "DEV",
              "family": family, "variant_id": variant, "source_revision": "a" * 40,
              "environment_revision": "a" * 40, "generator_seed": seed, "candidate_index": ordinal,
              "spec": spec, "prompt": prompt, "prompt_sha256": text_hash(prompt),
              "normalized_prompt_sha256": text_hash(" ".join(prompt.split())),
              "target": target, "target_sha256": text_hash(target),
              "canonical_parent_sha256": d.digest(spec), "split_sha256": "b" * 64,
              "training_role_eligible": False, "training_eligibility": False,
              "source_id": "original-rineya-e1-bridge" if panel == "acquisition" else "original-rineya-retention",
              "policy_or_teacher_identity": "DETERMINISTIC_REFERENCE_NO_MODEL", "tokenized_prompt_sha256": None,
              "tokenization_status": "UNMEASURED", "verifier_result": True, "mutation_control_rejected": True,
              "source_correct_membership": None, "cost": {"model_calls": 0, "teacher_calls": 0}}
    if panel == "acquisition":
        source["structure_id"] = stratum
    else:
        source.update(domain=family, stratum=stratum)
    prompt_ids, completion = [151644, 10 + ordinal], [4242, d.EOS]
    ids = prompt_ids + completion
    token = {"schema_version": 1, **{k: source[k] for k in ("row_id", "parent_id", "role", "family", "variant_id")},
             "training_role_eligible": False, "training_eligibility": False,
             "source_row_sha256": d.digest(source), "source_prompt_sha256": source["prompt_sha256"],
             "source_target_sha256": source["target_sha256"], "prompt_tokens": 2, "completion_tokens": 2,
             "nonpadding_tokens": 4, "target_json_tokens": 1, "padding_tokens": 0, "completion_eos_id": d.EOS,
             "input_ids": ids, "attention_mask": [1] * 4, "completion_mask": [0, 0, 1, 1],
             "labels": [-100, -100, 4242, d.EOS], "input_token_ids_sha256": d.digest(ids),
             "tokenized_prompt_sha256": d.digest(prompt_ids), "completion_token_ids_sha256": d.digest(completion),
             "template_kwargs": deepcopy(d.TEMPLATE_KWARGS), "rendered_prompt_sha256": text_hash(wrapper(prompt)),
             "token_hash_encoding": "UTF8 canonical compact JSON integer list; SHA256",
             "mask_contract": "prompt=-100; completion including EOS=token ID; right padding=-100"}
    if panel == "deterministic_retention":
        token.update({k: source[k] for k in ("domain", "stratum", "canonical_parent_sha256", "split_sha256", "source_revision")})
        token.update(native_qualification_inferred=False, source_correct_membership=None)
    return source, token


def long_row(family, position, ordinal, length, answerable, adversarial):
    parent = f"SYNTHETIC-LC-{family}-{position}-{ordinal}"
    variant = ("answerable" if answerable else "unanswerable") + ("_adversarial" if adversarial else "_neutral")
    prompt = f"Synthetic long-context input {parent} {length} {variant}"
    ids = [8] * (length - 1) + [9]
    target = "SABCDE" if answerable else "UNKNOWN"
    row = {"schema_version": 1, "row_id": f"{parent}-{length}-{variant}", "parent_id": parent, "role": "DEV",
           "family": family, "evidence_position": position, "variant_id": variant, "answerable": answerable,
           "adversarial": adversarial, "missing_index": ordinal % 3, "split_sha256": d.digest(parent),
           "canonical_parent_sha256": d.digest([parent]), "structural_cluster_id": family,
           "query": "Synthetic query", "context_body": "Synthetic supplied evidence", "prompt": prompt,
           "rendered_prompt": wrapper(prompt), "input_ids": ids, "prompt_tokens": length,
           "input_token_ids_sha256": d.digest(ids), "generation_reserve_tokens": 24,
           "total_sequence_envelope": length + 24, "target": target, "target_token_ids": [4242],
           "target_tokens": 1, "target_token_ids_sha256": d.digest([4242]),
           "template_kwargs": deepcopy(d.TEMPLATE_KWARGS), "policy_input_only": ["prompt", "input_ids"],
           "training_eligibility": False, "native_qualification_inferred": False, "source_correct_membership": None,
           "support_spans": [{"fixture_only": True}], "visible_text_oracle": target}
    for field in ("prompt", "rendered_prompt", "context_body", "target"):
        row[field + "_sha256"] = text_hash(row[field])
    row["row_sha256"] = d.digest(row)
    return row


def language_row(stratum, ordinal, author):
    parent = f"LDEV-{stratum}-{ordinal:02d}"
    single = ordinal <= 4
    semantic = {"actors": ["synthetic actor"], "facts": ["synthetic fact"], "timeline": ["synthetic change"],
                "constraints": ["synthetic limit"], "task_intent": parent, "decisive_structure": "fixture only",
                "potential_confusions": ["fixture confusion"], "exclusion_tags": ["synthetic test"]}
    world = {"parent_id": parent, "role": "DEV", "stratum": stratum,
             "conversation_form": "single" if single else "multi", "author_id": author, "semantic_world": semantic}
    part = "A" if author == "/root/baseline_enrollment" else "B"
    world_hash = d.digest(world)
    messages = [{"role": "user", "content": "Synthetic initial request"}]
    if not single:
        messages += [{"role": "assistant", "content": "Authored prior context, never a model result"},
                     {"role": "user", "content": "Synthetic corrected final request"}]
    payload = {"parent_id": parent, "world_spec_sha256": world_hash, "messages": messages,
               "semantic_contract": {"must_include": ["fixture"], "must_avoid": ["fixture invention"],
                                     "acceptable_variation": ["fixture alternatives"], "dimensions": dict.fromkeys(
                                         ("adequacy", "relevance", "fluency", "role_consistency"), "fixture guidance")},
               "dev_exemplar": "REFERENCE_ONLY_LANGUAGE_FIXTURE", "cal_controls": [], "authorship": "assistant_source_candidate"}
    source = {**world, **payload, "spec_file_sha256": ("a" if part == "A" else "b") * 64,
              "spec_source_path": f"sources/language_dev_cal_v1/SPECS_{part}.json",
              "payload_source_path": f"sources/language_dev_cal_v1/PAYLOADS_{part}.json",
              "payload_file_sha256": ("c" if part == "A" else "d") * 64, "payload_world_sha256": d.digest(payload)}
    source["row_sha256"] = d.digest(source)
    ids = [151644, 40 + ordinal]
    completion = [4243, d.EOS]
    token = {"parent_id": parent, "role": "DEV", "stratum": stratum,
             "source_row_sha256": d.digest(source), "messages_sha256": d.digest(messages),
             "prompt_token_ids": ids, "prompt_tokens": len(ids), "prompt_token_ids_sha256": d.digest(ids),
             "prior_authored_assistant_turns": len(messages) // 2,
             "references": [{"reference_id": "dev_exemplar", "response_sha256": text_hash(source["dev_exemplar"]),
                             "completion_token_ids": completion, "completion_tokens_including_EOS": 2,
                             "completion_token_ids_sha256": d.digest(completion), "qualified_reference_labels": False}],
             "truncation": False, "native_qualified": False, "training_eligibility": False, "e1_launch_ready": False}
    return source, token


def fixture_bytes():
    raw = {}
    def put(name, value):
        raw[name] = d.canonical(value) + b"\n"
    def ident(name):
        return d.identity(raw[name])
    template = "SYNTHETIC_REGISTERED_TEMPLATE_NOT_NATIVE"
    assets = {name: {"sha256": str(i) * 64, "bytes": i + 10} for i, name in enumerate(
        ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"), 1)}
    pin = {"weight_sha256": d.WEIGHT_SHA256, "tokenizer_sha256": assets["tokenizer.json"]["sha256"],
           "chat_template_sha256": text_hash(template), "chat_template": template,
           "tokenizer_files": assets, "eos_token_id": d.EOS}
    put("tokenizer", pin)
    put("primary", {"weight_sha256": d.WEIGHT_SHA256, "parameter_count": 596049920,
                    "tokenizer_sha256": pin["tokenizer_sha256"], "chat_template_sha256": pin["chat_template_sha256"]})
    for panel in ("acquisition", "deterministic_retention"):
        sources, tokens = [], []
        if panel == "acquisition":
            for family in d.ACQUISITION_FAMILIES:
                for i in range(64):
                    s, t = short_row(panel, family, i, "fixture-structure", 0)
                    sources.append(s); tokens.append(t)
            registration = {"status": "FROZEN_FAST_LANE_DATA_ONLY", "selection_seed": 20260927,
                            "counts": {"DEV": {"acquisition_parents": 256, "per_family": 64}}}
            statuses = ("MEASURED_DETERMINISTIC_DATA_MECHANICS_ONLY", "MEASURED_OPEN_ROLE_TOKENIZATION_ONLY")
        else:
            for family in d.RETENTION_FAMILIES:
                for i in range(64):
                    strata = ["format_" + str(j) for j in range(4)] if family == "instruction_following" else d.POSITIONS
                    variants = ("answerable", "missing_evidence") if family == "answerability_abstention" else ("standard",)
                    for variant in variants:
                        s, t = short_row(panel, family, i, strata[i % 4], variant)
                        sources.append(s); tokens.append(t)
            registration = {"status": "FROZEN_FAST_LANE_RETENTION_ONLY", "selection_seed": 20260928,
                            "counts": {"DEV": dict.fromkeys(d.RETENTION_FAMILIES, 64)}, "primary_student_weight_sha256": d.WEIGHT_SHA256}
            statuses = ("MEASURED_DETERMINISTIC_RETENTION_MECHANICS_ONLY", "MEASURED_OPEN_RETENTION_ROLE_TOKENIZATION_ONLY")
        raw[panel + "_source"], raw[panel + "_tokens"] = jsonl(sources), jsonl(tokens)
        put(panel + "_registration", registration)
        receipt = {"status": statuses[0], "source_revision": "a" * 40, "open_row_counts": {"DEV": 256},
                   "registration_sha256": ident(panel + "_registration")["sha256"],
                   "files": {"DEV.jsonl": ident(panel + "_source"), "REGISTRATION.json": ident(panel + "_registration")}}
        put(panel + "_receipt", receipt)
        profile = {"status": statuses[1], "source_data_receipt_sha256": ident(panel + "_receipt")["sha256"],
                   "source_registration_sha256": ident(panel + "_registration")["sha256"],
                   "source_open_files_verified": {"DEV.jsonl": ident(panel + "_source")},
                   "files": {"DEV.tokenized.jsonl": ident(panel + "_tokens")}, "source_row_counts": {"DEV": 256},
                   "source_role_parent_counts": {"DEV": d.COUNTS[panel][1]}, "tokenizer_receipt_sha256": ident("tokenizer")["sha256"],
                   "tokenizer_files": assets, "chat_template_sha256": pin["chat_template_sha256"],
                   "template_kwargs": d.TEMPLATE_KWARGS, "truncation": False, "native_qualification_inferred": False}
        put(panel + "_profile", profile)
    lc = [long_row(f, s, i, p, a, adv) for f in d.LC_FAMILIES for s in d.POSITIONS for i in range(4)
          for p in (2024, 4072) for a in (True, False) for adv in (False, True)]
    raw["long_context_source"] = jsonl(lc)
    reg = {"status": "FROZEN_FAST_LANE_LONG_CONTEXT_ONLY", "counts": {"DEV": 32}, "prompt_lengths": [2024, 4072],
           "primary_student_weight_sha256": d.WEIGHT_SHA256, "tokenizer_sha256": pin["tokenizer_sha256"],
           "chat_template_sha256": pin["chat_template_sha256"]}
    put("long_context_registration", reg)
    raw["long_context_export_registration"] = json.dumps(reg, indent=2).encode()
    put("long_context_receipt", {"status": "MEASURED_LONG_CONTEXT_DEV_TOKENIZER_MECHANICS_ONLY",
                                 "files": {"DEV.jsonl": ident("long_context_source"), "REGISTRATION.json": ident("long_context_export_registration")},
                                 "registration_identity": ident("long_context_registration"), "open_row_counts": {"DEV": 256},
                                 "tokenizer_receipt_identity": ident("tokenizer"), "tokenizer_files": assets,
                                 "chat_template_sha256": pin["chat_template_sha256"], "template_kwargs": d.TEMPLATE_KWARGS})
    language_sources, language_tokens = [], []
    for index, stratum in enumerate(d.LANGUAGE_STRATA):
        author = "/root/baseline_enrollment" if index < 4 else "/root/e1_review"
        for i in range(1, 9):
            s, t = language_row(stratum, i, author)
            language_sources.append(s); language_tokens.append(t)
    raw["language_source"], raw["language_tokens"] = jsonl(language_sources), jsonl(language_tokens)
    put("language_registration", {"status": "ACCEPTED_DATA_ONLY_AUTHORING_NOT_GRADING_OR_NATIVE_EXECUTION", "DEV_parent_count": 64})
    frozen_fields = ("parent_id", "role", "stratum", "conversation_form", "author_id", "world_spec_sha256", "spec_file_sha256")
    put("language_freeze", {"status": "FROZEN_LANGUAGE_WORLD_SPECS_BEFORE_RENDERING", "registration": ident("language_registration"),
                            "worlds": [{k: s[k] for k in frozen_fields} for s in language_sources],
                            "source_files": {f"sources/language_dev_cal_v1/SPECS_{part}.json": {"sha256": letter * 64, "bytes": 100}
                                             for part, letter in (("A", "a"), ("B", "b"))}})
    put("language_receipt", {"status": "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED", "files": {"DEV.jsonl": ident("language_source")},
                             "counts": {"DEV_parents": 64}, "grading_qualified": False, "reference_labels_qualified": False,
                             "input_files": {d.PATHS[name][len(d.PACKAGE):]: ident(name)
                                             for name in ("language_registration", "language_freeze")}})
    put("language_profile", {"status": "MEASURED_LANGUAGE_DEV_CAL_TOKENIZER_ONLY_UNQUALIFIED_GRADING",
                             "files": {"DEV.prompts.jsonl": ident("language_tokens")}, "source_files": {"DEV.jsonl": ident("language_source")},
                             "bank_receipt": ident("language_receipt"), "tokenizer_receipt": ident("tokenizer"), "tokenizer_assets": assets,
                             "chat_template_sha256": pin["chat_template_sha256"], "template_kwargs": d.TEMPLATE_KWARGS,
                             "parents": {"DEV": 64}, "qualified_reference_labels": False, "native_qualified": False, "truncation": False})
    assert set(raw) == set(d.PATHS)
    return raw


class DevInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_bytes()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.package = self.root / d.PACKAGE
        for name, raw in self.fixture.items():
            path = self.root / d.PATHS[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.config = {"schema_version": 1, "status": "FROZEN_DEV_INPUT_BINDING_ONLY", "primary_weight_sha256": d.WEIGHT_SHA256,
                       "file_identities": {d.PATHS[name]: d.identity(raw) for name, raw in self.fixture.items()},
                       "native_execution": False, "language_scoring": False}

    def first(self, name):
        return d._parse(self.fixture[name].splitlines()[0])

    def test_full_population_exact_policy_whitelist_and_target_exclusion(self):
        result = d.bind(self.package, self.config)
        self.assertEqual(result["status"], "BOUND_DEV_INPUTS_NOT_NATIVE_ADMISSION")
        self.assertEqual(len(result["policy_rows"]), 832)
        self.assertEqual(sum(c["parents"] for c in result["counts"].values()), 544)
        evaluators = {r["row_id"]: r for r in result["evaluator_rows"]}
        for policy in result["policy_rows"]:
            self.assertEqual(set(policy), d.POLICY_FIELDS)
            self.assertEqual(policy["prompt_tokens"], len(policy["input_ids"]))
            self.assertEqual(policy["attention_mask"], [1] * policy["prompt_tokens"])
            self.assertEqual(policy["prompt_token_hash"], d.digest(policy["input_ids"]))
            self.assertEqual(set(evaluators[policy["row_id"]]), d.EVALUATOR_FIELDS)
            self.assertTrue(set(policy).isdisjoint({"target", "labels", "completion_mask", "references", "spec",
                                                  "answerable", "variant", "family", "stratum", "parent_id", "panel", "role"}))
            self.assertNotIn("REFERENCE_ONLY", json.dumps(policy))
        short = result["policy_rows"][0]
        original = self.first("acquisition_tokens")
        self.assertEqual(short["input_ids"], original["input_ids"][:original["prompt_tokens"]])
        self.assertNotIn(4242, short["input_ids"])
        self.assertEqual(evaluators[short["row_id"]]["scoring_payload"]["source_row"], self.first("acquisition_source"))
        self.assertFalse(result["native_qualified"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["grading_qualified"])
        self.assertFalse(result["language_scoring"])
        self.assertEqual(result["data_sha256"], d.digest({k: v for k, v in result.items() if k != "data_sha256"}))

    def test_language_keeps_authored_conversation_without_exemplar_tokens(self):
        result = d.bind(self.package, self.config)
        policy = next(r for r in result["policy_rows"] if r["row_id"] == "LDEV-explanation-05-dev")
        self.assertEqual([m["role"] for m in policy["messages"]], ["user", "assistant", "user"])
        self.assertNotIn(4243, policy["input_ids"])
        evaluator = next(r for r in result["evaluator_rows"] if r["row_id"] == policy["row_id"])
        self.assertEqual((evaluator["family"], evaluator["variant"]), ("general_language", "primary"))
        self.assertEqual(evaluator["scoring_payload"]["source_row"]["dev_exemplar"], "REFERENCE_ONLY_LANGUAGE_FIXTURE")

    def test_fixed_inventory_and_flags_rejected_before_any_read(self):
        for mutation in ("extra", "missing", "native", "language", "bool_version"):
            cfg = deepcopy(self.config)
            if mutation == "extra": cfg["file_identities"]["artifacts/CHECK.jsonl"] = {"sha256": "a" * 64, "bytes": 1}
            elif mutation == "missing": cfg["file_identities"].pop(d.FILES[0])
            elif mutation == "native": cfg["native_execution"] = True
            elif mutation == "language": cfg["language_scoring"] = True
            else: cfg["schema_version"] = True
            with self.subTest(mutation=mutation), mock.patch.object(d, "_read") as read, self.assertRaises(ValueError):
                d.bind(self.package, cfg)
            read.assert_not_called()

    def test_file_drift_and_repin_without_receipt_join_are_rejected(self):
        path = self.root / d.PATHS["acquisition_tokens"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "frozen file"):
            d.bind(self.package, self.config)
        self.config["file_identities"][d.PATHS["acquisition_tokens"]] = d.identity(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "profile token DEV identity"):
            d.bind(self.package, self.config)

    def test_resolved_redirect_and_unlisted_closed_path_block_before_read(self):
        with mock.patch.object(Path, "read_bytes") as read, self.assertRaisesRegex(ValueError, "fixed DEV inventory"):
            d._read(self.root, "artifacts/mimo_central/CAL.jsonl")
        read.assert_not_called()
        redirected = self.root / "CHECK.jsonl"
        with mock.patch.object(Path, "resolve", return_value=redirected), mock.patch.object(Path, "read_bytes") as read:
            with self.assertRaisesRegex(ValueError, "redirected DEV path"):
                d._read(self.root, d.PATHS["acquisition_source"])
        read.assert_not_called()

    def test_only_allowlisted_dev_files_are_read_and_snapshot_rechecked(self):
        original = d._read
        calls = []
        def recording(root, label):
            calls.append(label)
            return original(root, label)
        with mock.patch.object(d, "_read", side_effect=recording):
            d.bind(self.package, self.config)
        self.assertEqual(calls, list(d.FILES) * 2)
        self.assertFalse(any(Path(p).name in ("CAL.jsonl", "CAL.prompts.jsonl", "CHECK.jsonl", "FINAL.jsonl", "SEALED_RESERVATIONS.jsonl") for p in calls))

    def test_snapshot_change_during_validation_fails_closed(self):
        original = d._population
        def mutate(panel, rows):
            result = original(panel, rows)
            if panel == "language":
                path = self.root / d.PATHS["language_tokens"]
                path.write_bytes(path.read_bytes() + b" ")
            return result
        with mock.patch.object(d, "_population", side_effect=mutate), self.assertRaisesRegex(ValueError, "snapshot changed"):
            d.bind(self.package, self.config)

    def test_short_array_masks_hashes_types_and_source_join(self):
        source, token = self.first("acquisition_source"), self.first("acquisition_tokens")
        receipt = d._parse(self.fixture["acquisition_receipt"])
        reg = d._parse(self.fixture["acquisition_registration"])
        mutations = [{"input_token_ids_sha256": "0" * 64}, {"tokenized_prompt_sha256": "0" * 64},
                     {"completion_token_ids_sha256": "0" * 64}, {"source_row_sha256": "0" * 64},
                     {"prompt_tokens": 3}, {"completion_tokens": True}, {"input_ids": [True, 10, 4242, d.EOS]},
                     {"attention_mask": [True] * 4}, {"completion_mask": [0, 1, 1, 1]},
                     {"labels": [-100, 10, 4242, d.EOS]}, {"completion_eos_id": 151643},
                     {"rendered_prompt_sha256": "0" * 64}]
        for mutation in mutations:
            changed = deepcopy(token); changed.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d._short(source, changed, "acquisition", receipt, reg)

    def test_changed_source_semantics_invalidates_canonical_token_join(self):
        source = self.first("acquisition_source")
        source["spec"]["fixture"] = "changed"
        with self.assertRaisesRegex(ValueError, "canonical row join"):
            d._short(source, self.first("acquisition_tokens"), "acquisition",
                     d._parse(self.fixture["acquisition_receipt"]), d._parse(self.fixture["acquisition_registration"]))

    def test_json_duplicate_nonfinite_role_and_duplicate_rows_rejected(self):
        for raw in (b'{"row_id":"a","row_id":"b","role":"DEV"}\n',
                    b'{"row_id":"a","role":"DEV","value":NaN}\n',
                    b'{"row_id":"a","role":"CAL"}\n', b'\n',
                    b'{"row_id":"a","role":"DEV"}\n{"row_id":"a","role":"DEV"}\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                d._rows(raw, "row_id")

    def test_population_missing_twins_and_long_variants_do_not_become_independent(self):
        result = d.bind(self.package, self.config)
        for panel in ("deterministic_retention", "long_context"):
            rows = [deepcopy(r) for r in result["evaluator_rows"] if r["panel"] == panel]
            target = next(r for r in rows if r["variant"] in {"answerable", "2024-answerable_neutral"})
            target["variant"] = "missing_evidence" if panel == "deterministic_retention" else "4072-answerable_neutral"
            with self.subTest(panel=panel), self.assertRaisesRegex(ValueError, "coverage"):
                d._population(panel, rows)

    def test_long_context_self_hash_prompt_and_factors_are_verified(self):
        original = self.first("long_context_source")
        for field, value in (("input_ids", [5]), ("prompt_tokens", 1024), ("answerable", 1),
                             ("variant_id", "unanswerable_neutral"), ("rendered_prompt", "wrong"),
                             ("target_token_ids_sha256", "0" * 64)):
            row = deepcopy(original); row[field] = value
            row["row_sha256"] = d.digest({k: v for k, v in row.items() if k != "row_sha256"})
            with self.subTest(field=field), self.assertRaises(ValueError):
                d._long(row)

    def test_language_join_conversation_and_reference_exclusion_contract(self):
        source = self.first("language_source")
        token = self.first("language_tokens")
        freeze = d._parse(self.fixture["language_freeze"])
        for mutation in ({"source_row_sha256": "0" * 64}, {"messages_sha256": "0" * 64},
                         {"prompt_tokens": 897}, {"native_qualified": True}, {"references": []}):
            changed = deepcopy(token); changed.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d._language(source, changed, freeze)
        changed = deepcopy(source)
        changed["messages"][0]["role"] = "system"
        changed["row_sha256"] = d.digest({k: v for k, v in changed.items() if k != "row_sha256"})
        with self.assertRaises(ValueError):
            d._language(changed, token, freeze)

    def test_import_does_not_load_framework_preparation_or_change_environment(self):
        root = Path(__file__).resolve().parents[3]
        code = ("import os,sys; before=dict(os.environ); "
                "from experiments.post_training_mimo_centralized_v1.scripts import dev_inputs; "
                "assert dict(os.environ)==before; "
                "assert not any(n.startswith(('torch','transformers','tokenizers')) or "
                "n.endswith(('tokenize_bridge','tokenize_retention','tokenize_language_eval','build_long_context','dev_scorer')) for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                env=dict(os.environ, USE_TORCH="FIXTURE_UNCHANGED"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
