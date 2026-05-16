from __future__ import annotations

from qnas.common.schema import CircuitSpec


def entanglement_proxy(spec: CircuitSpec, *, num_samples: int = 32) -> float:
    """Placeholder for TensorCircuit-based entanglement metrics."""
    raise NotImplementedError("entanglement_proxy requires the TensorCircuit metric implementation.")
