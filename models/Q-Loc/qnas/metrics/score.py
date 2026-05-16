from __future__ import annotations


DEFAULT_WEIGHTS = {
    "n_2q_gates": -0.25,
    "depth_proxy": -0.15,
    "hardware_cost": -0.60,
}


def score_metrics(metrics: dict, weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * float(metrics.get(key, 0.0))
    return score

