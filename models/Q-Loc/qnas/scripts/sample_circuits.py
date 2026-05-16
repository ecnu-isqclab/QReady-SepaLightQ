from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qnas.sample_nas.sample_pool import sample_circuit_pool


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a pool of QNAS circuits.")
    parser.add_argument("--out", required=True, help="Output candidates directory.")
    parser.add_argument("--num-circuits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--entangle-pattern", default="linear", choices=["linear", "ring"])
    return parser.parse_args()


def main():
    args = parse_args()
    sample_circuit_pool(
        out_dir=args.out,
        num_circuits=args.num_circuits,
        seed=args.seed,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        entangle_pattern=args.entangle_pattern,
    )


if __name__ == "__main__":
    main()
