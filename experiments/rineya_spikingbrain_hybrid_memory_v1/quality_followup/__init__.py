"""Isolated, prospective update-order quality experiment; no native work on import."""

SCHEMA = 'RINEYA-UPDATE-ORDER-QUALITY/v1'
CONDITIONS = ('anchor', 'control', 'candidate')
DATA_SEED = 170105
MODEL_SEED = 17
RECIPE = {
    'device': 'cuda:0', 'dtype': 'float32', 'layer': 14, 'rank': 8,
    'window': 128, 'seed': MODEL_SEED, 'learning_rate': 0.0001,
    'steps': 512, 'max_train_tokens': 259, 'max_prompt_tokens': 1158,
    'qualification_tokens': 2048, 'max_new_tokens': 32, 'prefill_chunk': 64,
    'optimizer': 'AdamW(.9,.95), wd=0, eps=1e-8, clip=1',
}
REGISTRATION = {
    'primary': 'typed_update_greatest_not_first', 'primary_rows': 256,
    'minimum_net_correct': 26, 'minimum_gain_percentage_points': 10.0,
    'middle_and_last_each_improve': True, 'all_six_correct_groups_increase': True,
    'greatest_first_no_decline': True, 'json_contract_no_decline': True,
    'retention_metrics': ['strict_exact_match', 'typed_value_correct'],
    'baseline_minimum_correct_per_metric': 16, 'baseline_minimum_skills_per_metric': 4,
    'maximum_source_correct_damage_per_metric': 0,
    'answerable_and_absent_each_no_decline': True,
    'update_worlds_dev': 16, 'training_steps_per_condition': 512,
    'dev_rows_per_condition': 672, 'seed_count': 1, 'model_promotion': False,
    'automatic_followup': False,
}
