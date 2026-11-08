"""Export the audited Stage 1 selected bridge and exact same-pair controls."""
import argparse
from pathlib import Path
import sys

PROGRAMME = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROGRAMME))


def main():
    from cross_scale.generation_stage.v1.src.upstream import export_upstream
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--aggregate-attempt', type=int, default=0)
    parser.add_argument('--audit-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    print(export_upstream(args.run_id, args.aggregate_attempt,
                          audit_directory=args.audit_directory, output_directory=args.output_directory))


if __name__ == '__main__':
    main()
