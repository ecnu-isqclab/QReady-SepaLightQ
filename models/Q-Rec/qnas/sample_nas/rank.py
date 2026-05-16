from __future__ import annotations

from qnas.common.schema import CircuitSpec
from qnas.metrics.hardware import hardware_cost
from qnas.metrics.score import score_metrics
from qnas.metrics.structural import structural_metrics


def evaluate_circuit(spec: CircuitSpec, weights: dict[str, float] | None = None) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {"name": spec.name}
    metrics.update(structural_metrics(spec))
    metrics["hardware_cost"] = hardware_cost(spec)
    metrics["score"] = score_metrics(metrics, weights=weights)
    return metrics


def rank_circuits(
    specs: list[CircuitSpec],
    *,
    weights: dict[str, float] | None = None,
    reverse: bool = True,
) -> list[tuple[CircuitSpec, dict[str, float | int | str]]]:
    rows = [(spec, evaluate_circuit(spec, weights=weights)) for spec in specs]
    return sorted(rows, key=lambda item: float(item[1]["score"]), reverse=reverse)

