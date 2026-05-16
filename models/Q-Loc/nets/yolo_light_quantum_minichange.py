from __future__ import annotations

import torch
import torch.nn as nn

from nets.yolo_light_quantumchanel import (
    LightQuantumChannelSPPCSPC,
    YoloLightQuantumChanelBody,
)
from nets.yolov7quantumnewchanel6qubitsanglecoding64output import (
    QuantumNewChannel6QubitsRepConv,
    TrueQNN6QubitsAngleCoding64Output,
)


class TrueQNN6QubitsAngleCoding6ZOutput(TrueQNN6QubitsAngleCoding64Output):
    """Keep the old 6-qubit circuit, but read six Pauli-Z expectations."""

    def __init__(self, channels: int, n_qubits: int = 6, upload_rounds: int = 5):
        super().__init__(channels, n_qubits=n_qubits, upload_rounds=upload_rounds)
        self.readout = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, channels),
        )

    def _build_qml_layer(self):
        try:
            import numpy as np

            if not hasattr(np, "ComplexWarning"):
                np.ComplexWarning = RuntimeWarning
            import tensorcircuit as tc
            import tensorflow as tf  # noqa: F401
        except Exception as exc:
            raise RuntimeError("TrueQNN6QubitsAngleCoding6ZOutput requires tensorcircuit and tensorflow.") from exc

        backend = tc.set_backend("tensorflow")
        n_qubits = self.n_qubits
        upload_rounds = self.upload_rounds

        def qml(encoded, angles):
            circuit = tc.Circuit(n_qubits)
            for round_idx in range(upload_rounds):
                round_values = encoded[round_idx * 12 : (round_idx + 1) * 12]
                for qubit in range(n_qubits):
                    circuit.ry(qubit, theta=round_values[qubit])
                    circuit.rz(qubit, theta=round_values[qubit + n_qubits])
                for qubit in range(n_qubits):
                    circuit.rx(qubit, theta=angles[round_idx, qubit, 0])
                    circuit.ry(qubit, theta=angles[round_idx, qubit, 1])
                    circuit.rz(qubit, theta=angles[round_idx, qubit, 2])
                for qubit in range(n_qubits):
                    circuit.cnot(qubit, (qubit + 1) % n_qubits)
            return backend.stack([backend.real(circuit.expectation_ps(z=[qubit])) for qubit in range(n_qubits)])

        qml_vmap = backend.vmap(qml, vectorized_argnums=0)
        return tc.interfaces.torch_interface(qml_vmap, jit=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, got {channels}.")
        encoded = self.encoder(x).reshape(batch, 60)
        encoded = torch.tanh(encoded).float()
        if self._qml_layer is None:
            self._qml_layer = self._build_qml_layer()
        z_expectations = self._qml_layer(encoded, self.q_angles.float())
        z_expectations = z_expectations.to(dtype=x.dtype, device=x.device)
        gate = torch.sigmoid(self.readout(z_expectations)).reshape(batch, channels, 1, 1)
        return x * gate


class QuantumMiniChange6QubitsRepConv(QuantumNewChannel6QubitsRepConv):
    """Existing 1x1 + 2x2 structure with the mini-change QNN readout."""

    def __init__(self, c1: int, c2: int):
        super().__init__(c1, c2)
        self.qnn_branch = TrueQNN6QubitsAngleCoding6ZOutput(c1)


class LightQuantumMiniChangeSPPCSPC(LightQuantumChannelSPPCSPC):
    """Existing light SPPCSPC with the mini-change QNN readout."""

    def __init__(self, c1: int, c2: int, e: float = 0.5, k: tuple[int, int, int] = (5, 9, 13)):
        super().__init__(c1, c2, e=e, k=k)
        self.qnn_branch = TrueQNN6QubitsAngleCoding6ZOutput(c2)


class YoloLightQuantumMiniChangeBody(YoloLightQuantumChanelBody):
    """yolo_light_quantumchanel with 64-state output replaced by six Z measurements."""

    def __init__(self, anchors_mask, num_classes: int = 1, phi: str = "light", pretrained: bool = False, **kwargs):
        super().__init__(anchors_mask, num_classes=num_classes, phi=phi, pretrained=pretrained, **kwargs)
        transition_channels = 16
        self.sppcspc = LightQuantumMiniChangeSPPCSPC(transition_channels * 32, transition_channels * 16)
        self.rep_conv_1 = QuantumMiniChange6QubitsRepConv(transition_channels * 4, transition_channels * 8)
        self.rep_conv_2 = QuantumMiniChange6QubitsRepConv(transition_channels * 8, transition_channels * 16)
        self.rep_conv_3 = QuantumMiniChange6QubitsRepConv(transition_channels * 16, transition_channels * 32)


MODEL_CLASS = YoloLightQuantumMiniChangeBody
