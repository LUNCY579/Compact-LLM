"""Audit only completed or explicitly rejected Stage 2 architecture cells."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main():
    from cross_scale.generation_stage.v1.src.execution import audit_screen
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(audit_screen(args.config), flush=True)


if __name__ == '__main__':
    main()
