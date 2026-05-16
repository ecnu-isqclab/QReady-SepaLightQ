from __future__ import annotations

from qnas.common.schema import CircuitSpec


def expressibility_proxy(spec: CircuitSpec, *, num_samples: int = 64) -> float:
    """Placeholder for TensorCircuit-based expressibility.

    The first implementation should sample random inputs/weights, run
    qnas.backends.tensorcircuit_backend.expectation_fn, and measure output
    coverage or entropy. Kept explicit so structural ranking works without
    TensorCircuit installed.
    """
    raise NotImplementedError("expressibility_proxy requires the TensorCircuit metric implementation.")

