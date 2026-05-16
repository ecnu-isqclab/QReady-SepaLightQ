from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qnas.common.io import load_circuit_dir
from qnas.sample_nas.deduplicate import deduplicate_circuits
from qnas.sample_nas.rank import rank_circuits
from qnas.sample_nas.select import save_rank_results


def parse_args():
    parser = argparse.ArgumentParser(description="Rank sampled QNAS circuits by basic metrics.")
    parser.add_argument("--circuits", required=True, help="Directory containing candidate json circuits.")
    parser.add_argument("--out", required=True, help="Output experiment directory.")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--deduplicate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = load_circuit_dir(args.circuits)
    if args.deduplicate:
        specs = deduplicate_circuits(specs)
    ranked = rank_circuits(specs)
    save_rank_results(ranked, out_dir=args.out, top_k=args.top_k)


if __name__ == "__main__":
    main()
