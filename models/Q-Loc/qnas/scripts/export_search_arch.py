from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qnas.search_nas.export import export_search_arch


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained search layer to CircuitSpec json.")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    raise NotImplementedError(
        "This script needs a project checkpoint loader. Use qnas.search_nas.export.export_search_arch(layer, path) "
        "inside the training script for now."
    )


if __name__ == "__main__":
    main()
