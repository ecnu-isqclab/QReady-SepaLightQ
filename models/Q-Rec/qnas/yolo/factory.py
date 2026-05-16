from __future__ import annotations

import torch.nn as nn

from qnas.common.io import load_circuit
from qnas.search_nas.search_space import SearchSpaceSpec
from qnas.search_nas.searchable_layer import SearchableQuantumLayer
from qnas.search_nas.tensorcircuit_search_layer import TensorCircuitSearchableQuantumLayer
from qnas.yolo.blocks import QuantumFeatureBlock
from qnas.yolo.layers import SampledQuantumLayer


def build_quantum_block(
    *,
    mode: str,
    channels: int,
    circuit_path: str | None = None,
    search_space: SearchSpaceSpec | None = None,
    search_backend: str = "tensorcircuit",
    tc_backend: str = "pytorch",
    residual_scale: float = 1.0,
) -> nn.Module:
    mode = mode.lower()
    if mode == "none":
        return nn.Identity()

    if mode == "sampled":
        if not circuit_path:
            raise ValueError("mode='sampled' requires circuit_path.")
        spec = load_circuit(circuit_path)
        q_layer = SampledQuantumLayer(spec)
        return QuantumFeatureBlock(channels, q_layer, spec.n_inputs, len(spec.measured_qubits), residual_scale)

    if mode == "search":
        if search_space is None:
            search_space = SearchSpaceSpec()
        if search_backend == "tensorcircuit":
            q_layer = TensorCircuitSearchableQuantumLayer(search_space, tc_backend=tc_backend)
        elif search_backend == "torch":
            q_layer = SearchableQuantumLayer(search_space)
        else:
            raise ValueError(f"Unknown search_backend {search_backend!r}. Use 'tensorcircuit' or 'torch'.")
        return QuantumFeatureBlock(
            channels,
            q_layer,
            search_space.n_qubits,
            search_space.n_qubits,
            residual_scale,
        )

    raise ValueError(f"Unknown quantum block mode {mode!r}.")
