from __future__ import annotations

import torch
import torch.nn as nn

from qnas.backends.tensorcircuit_backend import expectation_fn
from qnas.common.io import load_circuit
from qnas.common.schema import CircuitSpec


class SampledQuantumLayer(nn.Module):
    """PyTorch module for a fixed CircuitSpec selected by sample NAS."""

    def __init__(self, spec: CircuitSpec):
        super().__init__()
        self.spec = spec
        self.num_inputs = spec.n_inputs
        self.num_outputs = len(spec.measured_qubits)
        self.weights = nn.Parameter(torch.randn(spec.n_weights) * 0.02)
        self._runner = None

    @classmethod
    def from_path(cls, path: str) -> "SampledQuantumLayer":
        return cls(load_circuit(path))

    def _get_runner(self):
        if self._runner is None:
            self._runner = expectation_fn(self.spec)
        return self._runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        runner = self._get_runner()
        outputs = [runner(sample, self.weights) for sample in x]
        return torch.stack(outputs, dim=0).to(device=x.device, dtype=x.dtype)

