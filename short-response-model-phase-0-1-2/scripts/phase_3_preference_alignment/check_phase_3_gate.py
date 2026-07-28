from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json, write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--sft-summary',type=Path,required=True); p.add_argument('--dpo-summary',type=Path,required=True); p.add_argument('--out',type=Path,default=ROOT/'reports/PHASE_3_STOP_GATE.md'); a=p.parse_args()
 s,d=read_json(a.sft_summary),read_json(a.dpo_summary)
 checks={'score_not_regressed':d['mean_heuristic_score']>=s['mean_heuristic_score']-0.02,'no_critical_failures':d.get('critical_failure_count',0)==0,'completion':d['complete_utterance_rate']>=.99,'p90_length':d['p90_reply_tokens']<=24}
 status='PASS_PENDING_HUMAN_REVIEW' if all(checks.values()) else 'FAIL'
 lines=['# PHASE 3 STOP GATE','','## Automated checks','',*['- '+k+': `'+('PASS' if v else 'FAIL')+'`' for k,v in checks.items()],'',f'## Gate status\n\n`{status}`','','- Human decision: `PENDING`','- Reviewer:','- Date:','- Notes:']
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text('\n'.join(lines),encoding='utf-8'); print(a.out)
if __name__=='__main__': main()
