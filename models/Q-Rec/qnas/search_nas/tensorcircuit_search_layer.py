from __future__ import annotations

from contextlib import nullcontext

import torch

from qnas.search_nas.search_space import SearchSpaceSpec
from qnas.search_nas.searchable_layer import SearchableQuantumLayer


def _load_tensorcircuit(tc_backend: str):
    try:
        import numpy as np

        if not hasattr(np, "ComplexWarning"):
            np.ComplexWarning = RuntimeWarning
        import tensorcircuit as tc
    except ImportError as exc:
        raise ImportError("TensorCircuit search layer requires tensorcircuit.") from exc

    if tc_backend == "tensorflow":
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:
            raise ImportError("TensorCircuit tensorflow backend requires tensorflow.") from exc

    backend = tc.set_backend(tc_backend)
    return tc, backend


class TensorCircuitSearchableQuantumLayer(SearchableQuantumLayer):
    """Differentiable search layer executed by TensorCircuit.

    This is the YOLO-facing search layer. It keeps alpha/theta as torch
    parameters, but the quantum state evolution and Z expectations are computed
    by TensorCircuit through tc.interfaces.torch_interface.
    """

    def __init__(self, search_space: SearchSpaceSpec, tc_backend: str = "pytorch"):
        super().__init__(search_space)
        self.tc_backend = tc_backend
        self.tc_layer = self._build_tensorcircuit_layer()

    def _build_tensorcircuit_layer(self):
        tc, backend = _load_tensorcircuit(self.tc_backend)
        space = self.search_space
        n_qubits = space.n_qubits
        n_layers = space.n_layers
        oneq_ops = tuple(space.oneq_ops)
        twoq_ops = tuple(space.twoq_ops)
        edges = tuple(space.twoq_edges)
        input_encoding = space.input_encoding

        unsupported_twoq = set(twoq_ops) - {"crx", "cry", "crz", "skip", "identity"}
        if unsupported_twoq:
            raise NotImplementedError(
                "TensorCircuit search currently supports parameterized two-qubit candidates "
                f"crx/cry/crz only; got {sorted(unsupported_twoq)}."
            )

        def qml(sample, theta_1q, theta_2q, prob_1q, prob_2q):
            circuit = tc.Circuit(n_qubits)

            for qubit_idx in range(n_qubits):
                getattr(circuit, input_encoding)(qubit_idx, theta=sample[qubit_idx])

            for layer_idx in range(n_layers):
                for qubit_idx in range(n_qubits):
                    for op_idx, op in enumerate(oneq_ops):
                        if op in {"skip", "identity"}:
                            continue
                        theta = prob_1q[layer_idx, qubit_idx, op_idx] * theta_1q[layer_idx, qubit_idx]
                        getattr(circuit, op)(qubit_idx, theta=theta)

                for edge_idx, (control, target) in enumerate(edges):
                    for op_idx, op in enumerate(twoq_ops):
                        if op in {"skip", "identity"}:
                            continue
                        theta = prob_2q[layer_idx, edge_idx, op_idx] * theta_2q[layer_idx, edge_idx]
                        getattr(circuit, op)(control, target, theta=theta)

            outputs = backend.stack(
                [
                    backend.real(circuit.expectation([tc.gates.z(), [qubit_idx]]))
                    for qubit_idx in range(n_qubits)
                ]
            )
            return backend.reshape(outputs, [-1])

        qml_vmap = backend.vmap(qml, vectorized_argnums=0)
        if self.tc_backend == "pytorch":
            return qml_vmap
        return tc.interfaces.torch_interface(qml_vmap, jit=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        prob_1q = torch.softmax(self.alpha_1q, dim=-1)
        prob_2q = torch.softmax(self.alpha_2q, dim=-1)

        theta_1q = self.theta_1q
        theta_2q = self.theta_2q

        # TensorCircuit's PyTorch backend creates gate constants on the current
        # default torch device. Match it to the feature tensor so QNN search can
        # run on CUDA with the rest of the NAS model.
        context = torch.device(x.device) if self.tc_backend == "pytorch" and x.device.type == "cuda" else nullcontext()
        with context:
            output = self.tc_layer(x, theta_1q, theta_2q, prob_1q, prob_2q)

        return output.to(device=x.device, dtype=output_dtype)
