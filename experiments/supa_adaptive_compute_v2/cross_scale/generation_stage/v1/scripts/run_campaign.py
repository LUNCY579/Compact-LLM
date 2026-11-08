"""Run/resume the fixed Stage 2 screen in serial fresh subprocesses."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main():
    from cross_scale.generation_stage.v1.src.execution import run_campaign
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(run_campaign(args.config), flush=True)


if __name__ == '__main__':
    main()
