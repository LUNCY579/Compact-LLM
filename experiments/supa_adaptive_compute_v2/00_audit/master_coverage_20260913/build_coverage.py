"""Read-only, standard-library master requirement coverage snapshot.

No models, tensors, evaluation rows, test execution or Git mutations. Existing
audits are cited within their recorded scope, not promoted into model replays.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
ATTACHMENT = Path('C:/Users/clclc/.codex/attachments/5a82b879-f744-47eb-950f-693096a4d8ba/pasted-text-1.txt')

PROOF = {
    'master': '00_audit/master_goal.md',
    'inventory': '00_audit/existing_work_inventory.md',
    'history': '00_audit/integrity_20260913T0323/receipt.json',
    'plan': 'README.md',
    'boundary': 'AGENTS.md',
    'roadmap': 'ROADMAP.md',
    'data': 'data/DESIGN.md',
    'A_bank': 'data/generated/v2/native_armr/manifest.json',
    'B_bank': 'data/generated/v2/cross_scale/manifest.json',
    'A_validation': 'native_armr/evidence/01_dataset_validation/20260905T171537_568883Z_e2e20e72/validation.json',
    'B_validation': 'cross_scale/evidence/01_dataset_validation/20260905T171720_931623Z_4247ee5b/validation.json',
    'datasets': 'shared/datasets.py',
    'scoring': 'shared/scoring.py',
    'statistics': 'shared/statistics.py',
    'power': 'data/generated/v2/power_planning.json',
    'frontier': 'shared/reporting.py',
    'guards': 'shared/guards.py',
    'integrity': 'shared/integrity.py',
    'evaluation': 'shared/evaluation.py',
    'manifests': 'shared/manifests.py',
    'runtime': 'shared/runtime.py',
    'A_protocol': 'native_armr/protocol/preregistration.md',
    'A_amendment': 'native_armr/protocol/gpu_fp32_v2_4_amendment.md',
    'A_config': 'native_armr/configs/protocol_gpu_v2_4.json',
    'A_registration': 'native_armr/evidence/00_preregistration/081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e/registration.json',
    'A_operator': 'native_armr/src/operators.py',
    'A_control': 'native_armr/src/learned_control.py',
    'A_control_audit': '00_audit/native_v2_4_complete_control_review_20260913/receipt.json',
    'A_baseline_audit': '00_audit/native_v2_4_baseline_review_20260913/receipt.json',
    'A_analysis': 'native_armr/src/analysis.py',
    'A_audit': 'native_armr/src/audit.py',
    'A_caller': 'native_armr/scripts/run_large_eval.py',
    'A_campaign': 'scripts/run_native_campaign.py',
    'A_qualification': 'native_armr/evidence/00_runtime_qualification/20260906T203929_861509Z_7b0b0fc8/report.json',
    'A_tests': '00_audit/native_v2_4_validation_20260906/receipt.json',
    'B_protocol': 'cross_scale/protocol/transformation_alignment_v2.md',
    'B_delta': 'cross_scale/src/delta_alignment.py',
    'B_fit': 'cross_scale/src/bridge_fit.py',
    'B_matrix': 'cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/aggregate/attempt_000/matrix.json',
    'B_selection': 'cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/aggregate/attempt_000/selection.json',
    'B_matrix_audit': 'cross_scale/evidence/07_independent_audit/matrix_audit_20260907T004813_5299633Z/audit_receipt.json',
    'B_matrix_manifest': 'cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/aggregate/attempt_000/run_manifest.json',
    'B_screen_protocol': 'cross_scale/generation_stage/v1/protocol/generation_screen_v1.md',
    'B_screen_config': 'cross_scale/generation_stage/v1/configs/generation_screen_gpu_v1.json',
    'B_screen_registration': 'cross_scale/generation_stage/v1/evidence/00_preregistration/dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3/registration.json',
    'B_screen_execution': 'cross_scale/generation_stage/v1/src/execution.py',
    'B_screen_analysis': 'cross_scale/generation_stage/v1/src/analysis.py',
    'B_qualification': 'cross_scale/generation_stage/v1/src/qualification.py',
    'B_control': 'cross_scale/generation_stage/v1/src/compute_control.py',
    'B_control_audit': '00_audit/cross_control_calibration_review_20260913/attempt_001/receipt.json',
    'B_tests': 'cross_scale/generation_stage/v1/evidence/00_preparation_validation/20260913_screen_retry1/receipt.json',
    'status': 'STATUS.md',
    'report': 'CURRENT_REPORT.md',
}

# Status concerns the complete requirement, not just whether a filename exists.
ROWS = [
 (1, 'Preserve historical Native and donor evidence', 'proved', 'inventory history', '198 historical files and canonical checkpoint rehashed in the cited integrity audit; preserve this boundary during remaining runs.'),
 (2, 'Audit existing work before implementation', 'proved', 'inventory', 'Exact component classifications, branches, SHAs and untracked graft provenance recorded; no new repository-wide rediscovery needed.'),
 (3, 'Single codebase with two independent pipelines', 'implemented', 'plan boundary A_caller B_screen_execution', 'Shared utilities and independent track/stage lifecycles exist; continue through their separate scientific registrations.'),
 (4, 'Share engineering; separate science and data', 'implemented', 'boundary datasets evaluation B_protocol B_screen_protocol', 'Track/role guards and separate registrations are implemented; every future run must retain these boundaries.'),
 (5, 'Fresh stratified approximately 1,024-example bank', 'proved', 'data A_bank B_bank A_validation B_validation', 'Each bank has 1,024 primary originals in eight families/32 templates. Synthetic bounded English scope must accompany results.'),
 (6, 'Immutable scenarios, multiple templates, linked variants', 'proved', 'data datasets A_validation B_validation', '128 linked diagnostics add zero independent primary N; relation reversal/longer-chain/intervention subsets are only 4/4/2 cases.'),
 (7, 'Calibration / architecture / confirmation separation', 'implemented', 'data datasets A_registration B_protocol B_screen_protocol', '128 calibration (96 fit/32 holdout), 128 architecture originals, 1,024 confirmation originals; confirmation outputs remain unobserved.'),
 (8, 'Answer Native recurrence generalization question', 'pending', 'A_protocol A_control_audit', 'Finish the five-arm mechanical screen, then registered confirmation and independent analysis; mechanics alone cannot answer the question.'),
 (9, 'Evaluate A0/A1/A3/A16 prospectively', 'pending', 'A_config A_operator A_control_audit A_baseline_audit', 'A0 is complete; A1/A3/A16 remaining campaign evidence is root-owned. All four fixed arms must reach required complete roles.'),
 (10, 'Native approximate matched-compute control and measured cost', 'pending', 'A_amendment A_control A_control_audit A_audit', 'AM architecture 128/128 is audited. Complete paired confirmation and disclose 90% of A3 added matrix FLOPs, copied/active parameters, latency and RAM/VRAM.'),
 (11, 'Native paired primary statistics', 'implemented', 'statistics A_analysis A_protocol', 'Real complete confirmation accuracy, wins/losses, McNemar, scenario CI and absolute/relative error reduction remain pending.'),
 (12, 'Native family accuracy, paired effects, NLL, identity, CI', 'implemented', 'statistics A_analysis', 'paired_analysis returns all required family metrics. No completed confirmatory family effects exist yet.'),
 (13, 'Native unseen-template and linked generalization', 'implemented', 'data statistics A_analysis A_audit', 'Report pooled seen/unseen strata (256 unseen originals) and original-versus-linked-variant advantage changes. No broad robustness claim from tiny variant strata.'),
 (14, 'Prospective meaningful Native GO conjunction', 'implemented', 'A_protocol A_config A_analysis', 'Thresholds and hierarchical A3>A0 then A3>AM tests, preservation, validity and cost disclosure are frozen; actual GO/NO-GO unresolved.'),
 (15, 'Retain Native positive, generic-compute, specialized or negative outcomes', 'pending', 'A_analysis A_protocol', 'Use actual complete outcomes; secondary/family effects cannot replace failed A3 primary or justify an untested router.'),
 (16, 'Address donor fidelity before expanding inert historical graft', 'proved', 'B_protocol B_matrix B_matrix_audit', 'The fresh alignment matrix completed; historical inert G was not promoted to a large efficacy run.'),
 (17, 'Prioritize transformation over absolute correspondence', 'proved', 'B_delta B_fit B_matrix B_selection', 'Actual delta losses and separate state/delta measurements exist and were executed; donor usefulness remains separate.'),
 (18, 'Use native and projected donor transformation deltas', 'proved', 'B_delta B_fit B_matrix_audit', 'Full-prompt actual decoder L(z)-z versus L(y)-y, original positions and replay provenance are covered by implementation/recorded audit; this audit does not replay tensors.'),
 (19, 'Compare B0/B1/B2/B3/BS/BR, retain controls', 'proved', 'B_protocol B_matrix B_matrix_audit', 'All 96 cells completed; keep all negative controls, including any surprising favorable future efficacy.'),
 (20, 'Token balancing and all/ordinary/special/final reporting', 'proved', 'B_protocol B_delta B_matrix', 'All four strata measured. Selected B3 special-token fidelity is worse than same-pair B0 and must remain explicit.'),
 (21, 'Cheap held-out depth-pair matrix and stability metrics', 'proved', 'B_protocol B_matrix B_matrix_audit', 'All 16 depth pairs x six objectives completed without per-pair efficacy generation.'),
 (22, 'Select/freeze pair using alignment only', 'proved', 'B_selection B_matrix_audit B_screen_registration', 'B3 Supa L19 / donor L24 selected under frozen complete-matrix ranking; exported and bound before generation.'),
 (23, 'Preregister fidelity thresholds; relate fidelity to usefulness', 'pending', 'B_protocol B_selection B_screen_analysis', 'Mechanism thresholds passed. No explicit efficacy-to-fidelity joined analysis is implemented or executed; add a separate descriptive evidence report after screen, without changing selection.'),
 (24, 'Prospective gate-dose study with behavior/NLL/cost diagnostics', 'pending', 'B_screen_protocol B_screen_config B_screen_analysis', '17-cell screen is frozen (.03/.10/.20/.40 plus X0). Runtime qualification and actual fixed-cell generation remain pending.'),
 (25, 'Mechanism -> small screen -> fresh confirmation funnel', 'pending', 'B_matrix_audit B_screen_execution B_screen_protocol', 'Stage 1 complete; Stage 2 implemented. Stage 3 confirmation lifecycle/protocol/config/audit caller is not implemented or frozen and is gated by SCREEN_GO.'),
 (26, 'B X0/XG/XS/XR/XM and frozen Native XN when available', 'pending', 'B_screen_protocol B_control B_control_audit', '17-cell screen includes required non-Native controls; XN unavailable at freeze. Reassess XN solely from independently frozen Native results at B confirmation freeze; no Native efficacy rows imported.'),
 (27, 'B paired/family/preservation/cost and efficacy-alignment analysis', 'pending', 'statistics B_screen_analysis frontier', 'Screen paired statistics are implemented; confirmatory hierarchy/effect threshold/multiplicity/preservation and fidelity linkage are not yet frozen or measured.'),
 (28, 'Freeze backbones; train only registered bridges; verify integrity', 'proved', 'integrity B_matrix_audit A_control_audit B_control_audit', 'Completed audits support frozen-state/gradient/checkpoint boundaries within their scope. Every remaining real run needs its own before/after evidence.'),
 (29, 'Keep 8B gated', 'implemented', 'boundary B_protocol B_screen_protocol', '8B remains closed; mechanism GO alone is not donor efficacy or license for escalation.'),
 (30, 'Shared paired statistics with roles and multiplicity', 'implemented', 'statistics A_tests B_tests', 'Coverage, exact paired tests, scenario bootstrap, NLL/family summaries and Holm annotations are implemented/tested. Real confirmation reports remain pending.'),
 (31, 'Power at N256/512/1024/2048 for plausible effects', 'proved', 'power data statistics', 'All requested N/effect/discordance scenarios recorded prospectively; observed discordance is not yet available and assumed power is not a guarantee.'),
 (32, 'Freeze bank/scorer/source/preregistration before execution', 'implemented', 'A_registration B_screen_registration datasets evaluation', 'Current stages are frozen. B confirmation must receive its own registration; no post-output item edits or retuning on confirmation.'),
 (33, 'Deterministic high-quality scoring and ambiguity handling', 'implemented', 'data scoring datasets A_validation B_validation A_tests', 'Independent controlled-grammar solver and strict exact/JSON scoring validated; no opaque LLM judge or open-ended quality claim.'),
 (34, 'Immutable resumable scheduling; qualify GPU against reference', 'pending', 'evaluation runtime A_campaign A_qualification B_qualification', 'Native runtime qualification passed. B stage-owned CPU/GPU per-cell qualification remains necessary; queues/process liveness is operational root evidence, not inferred here.'),
 (35, 'Parallel research, resource-safe sequential model execution', 'implemented', 'boundary runtime B_screen_execution', 'Shared lease and available-resource guards exist; continue serial heavy execution, preserving admission failures.'),
 (36, 'Separate immutable evidence roots', 'implemented', 'manifests evaluation B_screen_execution', 'Independent Native/B/stage output directories implemented; all failed and interrupted attempts remain preserved.'),
 (37, 'Complete run manifests and failure receipts', 'implemented', 'manifests evaluation B_screen_execution B_matrix_manifest A_control_audit', 'Actual completed stages have audited provenance; future run and abandoned-worker receipts still required. This audit does not synthesize post-run integrity.'),
 (38, 'Hard failure on instability, mutation, leakage and incomplete controls', 'implemented', 'guards integrity evaluation A_audit B_screen_execution A_tests B_tests', 'Guards/tests and preserved failures support fail-closed behavior; failed cells must not become partial denominators or disappear from reports.'),
 (39, 'Explicit Native versus donor capability/specialization/cost frontier', 'pending', 'frontier A_audit B_screen_analysis data', 'Per-track frontier code exists. Joint audited comparison is absent; different track banks cannot support a same-example paired superiority claim.'),
 (40, 'Choose next mechanism from both tracks actual outcomes', 'pending', 'roadmap A_protocol B_screen_protocol', 'Neither track has a confirmatory efficacy outcome. Preserve unresolved/negative results; do not force a winner.'),
 (41, 'Routing only after positive/complementary operator effects', 'implemented', 'boundary roadmap', 'Routing remains closed and unimplemented; fixed/heterogeneous choice waits for actual admissible evidence.'),
 (42, 'No MLA/MHA/MTP mixing during study', 'implemented', 'boundary roadmap', 'All remain closed; no tests or benefits claimed for these future mechanisms.'),
 (43, 'Distillation only after demonstrated donor efficacy', 'implemented', 'boundary B_protocol B_screen_protocol', 'Distillation remains intentionally unimplemented; fidelity GO and energy calibration do not satisfy its efficacy gate.'),
 (44, 'Meaningful shared/Native/B tests', 'proved', 'A_tests B_tests', 'Recorded Native/shared 331 tests +148 subtests and B Stage 2 243 tests +6 subtests passed. They are overlapping tiny-fixture/mechanics suites, not capability evidence; no repeats run here.'),
 (45, 'Maintain clear stage/gate status', 'implemented', 'status report', 'Current docs distinguish mechanics/alignment from efficacy. Root should refresh moving execution counts/HEAD after each meaningful stage.'),
 (46, 'Minimum audit/framework/data/protocol/caller/test deliverables', 'proved', 'inventory plan data A_caller B_protocol A_tests B_tests report', 'Minimum requested implementation deliverables exist. This does not complete the active full empirical research goal.'),
 (47, 'Final full report with exact provenance, evidence and one next experiment per track', 'pending', 'report status', 'Current report has required structure as a snapshot. Final closure needs fresh repository state, actual results/gates, limitations and exactly one causal next experiment per track.'),
 (48, 'Resolve both questions under fresh data, controls, power and audit', 'pending', 'master A_protocol B_screen_protocol report', 'Full objective remains incomplete. Await legitimate gated empirical outcomes; do not substitute implementation/test completion for scientific resolution.'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    missing = [f'{key}: {path}' for key, path in PROOF.items() if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError('Missing exact proof files: ' + '; '.join(missing))
    if ATTACHMENT.read_text(encoding='utf-8-sig') != (ROOT / PROOF['master']).read_text(encoding='utf-8-sig'):
        raise RuntimeError('Preserved master differs from user attachment')
    assert [row[0] for row in ROWS] == list(range(1, 49))
    rows = [{'section': n, 'requirement': title, 'status': status,
             'proof_paths': [PROOF[key] for key in keys.split()], 'remaining_work_or_scope': notes}
            for n, title, status, keys, notes in ROWS]
    hashes = {path: digest(ROOT / path) for path in sorted(set(PROOF.values()))}
    receipts = {}
    for key in ('history', 'A_tests', 'B_tests', 'A_control_audit', 'B_matrix_audit', 'B_control_audit'):
        source = json.loads((ROOT / PROOF[key]).read_text(encoding='utf-8-sig'))
        receipts[key] = {k: source[k] for k in ('status', 'tests_passed', 'subtests_passed', 'checks_passed', 'check_count', 'audit_checks', 'historical_files_verified', 'matrix_pairs', 'objective_cells') if k in source}
    payload = {'schema': 'master_completion_coverage_v1', 'utc': stamp,
        'repository_head_at_snapshot': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'scope': 'Source, protocol, manifest and existing audit inspection; no model/tensor loads, scorer execution, tests, efficacy rows or Git mutations.',
        'master_attachment_path': ATTACHMENT.as_posix(), 'master_attachment_sha256': digest(ATTACHMENT),
        'master_text_matches_preserved_copy': True, 'full_objective_complete': False,
        'live_process_status_checked': False, 'input_sha256': hashes, 'cited_receipt_summaries': receipts,
        'requirements': rows, 'auditor_sha256': digest(Path(__file__))}
    with (HERE / 'coverage.json').open('x', encoding='utf-8', newline='\n') as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write('\n')
    text = ['# Master goal completion coverage', '', f'Snapshot: {stamp}. Full objective **incomplete**. All 48 numbered sections are mapped below.', '',
        'This independent read-only audit inspected current source, protocols, frozen manifest metadata and existing receipts. It loaded no models or tensors, opened no efficacy rows or sealed data, ran no tests and made no Git changes. It does not replace the underlying model-integrity audits or certify a queue as live. Current execution counts belong to the root campaign snapshot.', '',
        'Statuses: **proved** = relevant deliverable/measurement has authoritative evidence within the stated scope; **implemented** = code/protocol exists but future execution must still honor it; **pending** = required empirical work or gated implementation remains; **unverified** = evidence is insufficient. A proved foundation item is not proof of programme completion.', '',
        'Every proof path is relative to `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/`; each exact file was checked for existence and byte-hashed in `coverage.json`. Historical `00_audit/requirements_audit.md` is a preserved earlier snapshot, not current completion evidence.', '',
        '| Section | Requirement | Status | Existing proof | Remaining work / scope |',
        '|---:|---|---|---|---|']
    for row in rows:
        links = '<br>'.join(f'[{path}]({(ROOT / path).as_posix()})' for path in row['proof_paths'])
        text.append(f"| {row['section']} | {row['requirement']} | {row['status']} | {links} | {row['remaining_work_or_scope']} |")
    text += ['', '## Requirements most likely to be omitted', '',
        '1. **B confirmation is not an implemented caller.** The current stage explicitly accepts only 128 architecture originals and denies confirmation. After SCREEN_GO, create a separate confirmation protocol/caller/audit with the selected frozen gate, unchanged mandatory controls, prospective primary hierarchy/multiplicity/effect threshold, preservation, and exact 1,024-case coverage. Reuse low-level mechanics without editing the frozen Stage 2 source.',
        '2. **Fidelity and efficacy still need an explicit joined report.** Stage 1 supplies held-out fidelity; Stage 2 supplies NLL/identity/accuracy by map and dose. Current analysis does not join them. A descriptive comparison of the three same-pair map families can answer whether the high-fidelity selected map is more useful than BS/BR. Four doses reuse the same alignment estimate and are not four independent fidelity observations. With one selected depth pair, a general across-depth predictive correlation is not identifiable; state that limitation instead of adding unregistered mechanism generation.',
        '3. **Native generalization code exists; measurements do not.** Shared paired analysis reports family effects and pooled seen/unseen-template results; Native linked_robustness reports changes in paired advantage for each linked intervention. Execute and audit the registered diagnostics only after complete original evidence; do not present 128 variants as extra independent N. Per-template individual estimates are not currently returned by shared paired_analysis, but the registered pooled unseen-template test is implemented.',
        '4. **Cross-track comparison requires comparable estimands.** A and B have independently generated banks. Their per-track deltas and costs can be described side by side; they cannot establish same-example wins or complementary-case overlap. An independently frozen XN on B confirmation, if available then, permits paired Native-versus-donor comparisons on B examples. Never pool A and B result rows as paired samples or let B outputs select XN.',
        '5. **Carry measured accelerator costs into the final frontier.** Native independent audit adds run-level RAM plus allocated/reserved CUDA peaks; B screen analysis reports RAM/allocated CUDA. Include active/resident parameters and dominant matrix work separately from attention/instrumentation and variable generated-token cost. Contended wall times are not a controlled speedup estimate. AM uses 90% of A3 added matrix work.',
        '6. **Keep inference labels and gates honest.** Architecture P values/intervals are descriptive after selection. N=1,024 is only approximately 82/53/30% powered for +2 points under 5/10/20% discordance; no rejection is not equivalence. Strong ordinary-token B fidelity coexists with worse special-token fidelity. Routing, 8B, reread mechanisms and distillation remain closed without their stated empirical conditions.', '',
        'These are completion requirements and reporting limitations, not authorization to retune any frozen experiment. No new scientific threshold, candidate, dataset edit or model run is proposed by this audit.', '']
    with (HERE / 'coverage.md').open('x', encoding='utf-8', newline='\n') as output:
        output.write('\n'.join(text))
    print(json.dumps({'status': 'COMPLETE_COVERAGE_AUDIT', 'full_objective_complete': False,
        'sections': len(rows), 'input_files_hashed': len(hashes),
        'status_counts': {state: sum(row['status'] == state for row in rows) for state in ('proved', 'implemented', 'pending', 'unverified')},
        'outputs': [str(HERE / 'coverage.md'), str(HERE / 'coverage.json')]}))


if __name__ == '__main__':
    main()
