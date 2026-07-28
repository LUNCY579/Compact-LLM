from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import write_json
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--dense-times',type=Path,required=True);p.add_argument('--sparse-times',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'artifacts/phase_7_router/runtime_comparison.json');a=p.parse_args();d=json.loads(a.dense_times.read_text());s=json.loads(a.sparse_times.read_text());dm=statistics.median(d['decode_ms']);sm=statistics.median(s['decode_ms']);write_json(a.out,{'dense_median_decode_ms':dm,'sparse_median_decode_ms':sm,'speedup':dm/sm if sm else None,'dense_peak_vram_gb':d.get('peak_vram_gb'),'sparse_peak_vram_gb':s.get('peak_vram_gb'),'passes_runtime_target':(dm/sm>=1.2) if sm else False});print(a.out)
if __name__=='__main__':main()
