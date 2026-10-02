"""CPU-only checks of the founding design record; NOT sampler qualification."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
CHARTER = ROOT / 'RINEYA_PPT_CENTRALIZED_REASONING.md'


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = json.loads((PACKAGE / 'registration.json').read_text(encoding='utf-8'))

    def test_identity_and_design_status(self) -> None:
        r = self.record
        self.assertEqual(r['schema_version'], 1)
        self.assertEqual(r['experiment_id'], 'PPT-001')
        self.assertEqual(r['repository'], 'LUNCY579/ChatBubble')
        self.assertEqual(r['branch'], 'research/rineya-ppt-centralized-reasoning-v1')
        self.assertEqual(r['package_path'], 'experiments/rineya_ppt_centralized_reasoning_v1')
        self.assertEqual(r['record_kind'], 'historical_design_not_launch_config')
        self.assertEqual(r['status'], 'DESIGN_RECORDED_NATIVE_NOT_IMPLEMENTED')

    def test_parent_pin(self) -> None:
        self.assertEqual(self.record['base'], {
            'branch': 'research/rineya-centralized-reasoning-system-v1',
            'commit': '7459b614bcb017ae4f0160b15e0ccd53473bb990',
            'tree': '9223d7b22e4bbbab867107a241c67ea7410a0ede',
        })
        self.assertIn(self.record['base']['commit'], CHARTER.read_text(encoding='utf-8'))

    def test_versioned_paper(self) -> None:
        p = self.record['paper']
        self.assertEqual(p['arxiv_id'], '2609.38104')
        self.assertEqual(p['version'], 'v1')
        for key in ('abstract_url', 'html_url'):
            self.assertTrue(p[key].startswith('https://arxiv.org/'))
            self.assertTrue(p[key].endswith('2609.38104v1'))

    def test_complementary_non_expiring_scope(self) -> None:
        self.assertEqual(self.record['scope']['target_parameter_scale'], '0.6B')
        self.assertEqual(self.record['scope']['research_track'], 'centralized_system_design_and_reasoning')
        p = self.record['preservation']
        self.assertIs(p['complementary'], True)
        self.assertIs(p['automatic_sunset'], False)
        self.assertIsNone(p['expiry_date'])
        self.assertIs(p['preserve_negative_results'], True)
        self.assertIs(p['github_branch_protection_configured'], False)

    def test_owner_controlled_disposition(self) -> None:
        self.assertEqual(set(self.record['preservation']['owner_approval_required_for']), {
            'retirement', 'archival', 'branch_deletion', 'replacement', 'obsolescence_label',
        })
        charter = CHARTER.read_text(encoding='utf-8')
        self.assertIn('explicit owner approval', charter)
        self.assertIn('near future', charter)
        self.assertIn('Scientific hypotheses remain falsifiable', charter)

    def test_reversible_parking(self) -> None:
        self.assertEqual(set(self.record['preservation']['parking_requires']), {
            'reason', 'exact_source_tip', 'open_questions', 'dependencies', 'restart_condition',
        })
        self.assertIn('restart condition', (PACKAGE / 'AGENTS.md').read_text(encoding='utf-8'))

    def test_no_execution_permissions(self) -> None:
        permissions = self.record['permissions_granted_by_this_record']
        self.assertEqual(set(permissions), {
            'native_inference', 'training', 'dataset_generation', 'protected_bank_access',
            'paid_compute', 'model_upload', 'merge', 'branch_deletion', 'force_push',
        })
        for key, value in permissions.items():
            with self.subTest(permission=key):
                self.assertIs(value, False)

    def test_no_native_or_sampler_result_claims(self) -> None:
        for key in ('native_sampler_implemented', 'native_runtime_implemented', 'new_model_results'):
            self.assertIs(self.record['scope'][key], False)
        e = self.record['evidence']
        self.assertIs(e['cpu_registration_checks_only'], True)
        for key in ('cpu_sampler_qualified', 'native_scoring_qualified',
                    'protected_banks_opened', 'external_results_are_rineya_results'):
            self.assertIs(e[key], False)
        for key in ('model_runs', 'training_runs'):
            self.assertIs(type(e[key]), int)
            self.assertEqual(e[key], 0)

    def test_future_native_values_remain_unset(self) -> None:
        unresolved = self.record['unresolved_before_native']
        required = {
            'checkpoint_sha256', 'tokenizer_sha256', 'prompt_template_sha256',
            'public_split_sha256', 'native_adapter_revision', 'dtype',
            'numerical_tolerances', 'power_ladder', 'refinement_rounds', 'seed_count',
            'resource_limits', 'minimum_worthwhile_improvement',
            'retention_noninferiority_margin', 'statistical_procedure', 'launch_authorization',
        }
        self.assertEqual(set(unresolved), required)
        self.assertTrue(all(value is None for value in unresolved.values()))

    def test_related_sources_are_pinned_not_replaced(self) -> None:
        refs = self.record['related_snapshots']
        self.assertEqual(len(refs), 3)
        self.assertEqual(len({ref['branch'] for ref in refs}), len(refs))
        sources = (PACKAGE / 'SOURCES.md').read_text(encoding='utf-8')
        for ref in refs:
            with self.subTest(source=ref['id']):
                self.assertRegex(ref['commit'], r'^[0-9a-f]{40}$')
                self.assertIn(ref['commit'], sources)
                self.assertNotEqual(ref['branch'], self.record['branch'])
        self.assertEqual(refs[0]['relationship'], 'baseline_reference_not_merged')
        self.assertEqual(refs[1]['relationship'], 'sibling_reference_not_merged')
        self.assertEqual(refs[2]['relationship'], 'ancestor_no_permission_transfer')

    def test_interaction_control_and_baseline(self) -> None:
        arms = self.record['arms']
        self.assertEqual([arm['id'] for arm in arms], [f'A{i}' for i in range(6)])
        by_id = {arm['id']: arm for arm in arms}
        self.assertEqual(self.record['primary_control'], ['A4', 'A5'])
        self.assertEqual(self.record['practical_baseline'], 'A2')
        self.assertIs(by_id['A4']['exchange_enabled'], False)
        self.assertIs(by_id['A5']['exchange_enabled'], True)
        self.assertEqual(by_id['A4']['readout'], by_id['A5']['readout'])
        plan = (PACKAGE / 'EXPERIMENT_PLAN.md').read_text(encoding='utf-8')
        for arm in arms:
            self.assertIn(arm['id'], plan)
        for hypothesis in ('H1', 'H2', 'H3'):
            self.assertIn(hypothesis, plan)

    def test_documents_are_present_and_local_links_resolve(self) -> None:
        expected = {'README.md', 'AGENTS.md', 'DESIGN.md', 'EXPERIMENT_PLAN.md',
                    'SOURCES.md', 'STATUS.md', 'VALIDATION.md'}
        self.assertEqual({path.name for path in PACKAGE.glob('*.md')}, expected)
        docs = sorted(PACKAGE.glob('*.md')) + [CHARTER]
        for doc in docs:
            text = doc.read_text(encoding='utf-8')
            self.assertTrue(text.endswith('\n'), doc.name)
            self.assertNotIn('\r', text, doc.name)
            self.assertNotIn('\ufffd', text, doc.name)
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                resolved = (doc.parent / target.split('#', 1)[0]).resolve()
                with self.subTest(document=doc.name, target=target):
                    self.assertTrue(resolved.is_relative_to(ROOT))
                    self.assertTrue(resolved.is_file())


if __name__ == '__main__':
    unittest.main()
