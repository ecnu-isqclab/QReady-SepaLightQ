from __future__ import annotations

import csv
from pathlib import Path

from qnas.common.io import save_circuit
from qnas.common.schema import CircuitSpec


def save_rank_results(
    ranked: list[tuple[CircuitSpec, dict]],
    *,
    out_dir: str | Path,
    top_k: int,
) -> None:
    out_dir = Path(out_dir)
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    if ranked:
        fieldnames = list(ranked[0][1].keys())
        with (out_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for _, metrics in ranked:
                writer.writerow(metrics)

    for idx, (spec, _) in enumerate(ranked[:top_k]):
        top_spec = CircuitSpec(
            name=f"top_{idx:03d}",
            n_qubits=spec.n_qubits,
            n_inputs=spec.n_inputs,
            n_weights=spec.n_weights,
            measured_qubits=spec.measured_qubits,
            gates=spec.gates,
            meta={**spec.meta, "source_name": spec.name, "rank": idx},
        )
        save_circuit(top_spec, best_dir / f"top_{idx:03d}.json")
