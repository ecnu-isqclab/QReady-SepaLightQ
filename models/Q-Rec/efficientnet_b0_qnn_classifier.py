"""EfficientNet-B0 classifier with a TensorCircuit quantum classification head.

The EfficientNet feature extractor is kept classical.  Only the final linear
classifier is replaced by:

    Linear(1280, n_qubits) -> angle-encoded VQC -> Linear(n_qubits, classes)

This keeps the quantum simulation at a small, high-level feature dimension.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def _load_tensorcircuit():
    """Load TensorCircuit lazily so importing the package fails with context."""
    try:
        import tensorcircuit as tc
    except ImportError as exc:
        raise ImportError(
            "EfficientNetB0QNNClassifier requires TensorCircuit. "
            "Install it with `pip install tensorcircuit` before using this model."
        ) from exc

    backend = tc.set_backend("pytorch")
    return tc, backend


def _ensure_qnas_on_path(qnas_root: str | None = None) -> None:
    """Make the YOLO-side qnas package importable from this classifier package."""
    candidates = []
    if qnas_root:
        root = Path(qnas_root)
        candidates.append(root.parent if root.name == "qnas" else root)

    workspace_root = Path(__file__).resolve().parents[2]
    candidates.append(workspace_root / "yolov7-pytorch-master")

    for candidate in candidates:
        if (candidate / "qnas").is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return

    raise ImportError(
        "Could not find the qnas package. Set model.qnas_root to the directory "
        "that contains qnas/, for example /srv/share/hackathon/yolov7-pytorch-master."
    )


def _build_qnas_search_layer(
    *,
    n_qubits: int,
    quantum_depth: int,
    qnas_backend: str = "torch",
    qnas_tc_backend: str = "pytorch",
    qnas_entangle_pattern: str = "ring",
    qnas_oneq_ops: Iterable[str] = ("skip", "rx", "ry", "rz"),
    qnas_twoq_ops: Iterable[str] = ("skip", "crx", "cry", "crz"),
    qnas_input_encoding: str = "ry",
    qnas_root: str | None = None,
) -> nn.Module:
    _ensure_qnas_on_path(qnas_root)
    from qnas.search_nas.search_space import SearchSpaceSpec
    from qnas.search_nas.searchable_layer import SearchableQuantumLayer
    from qnas.search_nas.tensorcircuit_search_layer import TensorCircuitSearchableQuantumLayer

    search_space = SearchSpaceSpec(
        n_qubits=n_qubits,
        n_layers=quantum_depth,
        oneq_ops=tuple(qnas_oneq_ops),
        twoq_ops=tuple(qnas_twoq_ops),
        entangle_pattern=qnas_entangle_pattern,
        input_encoding=qnas_input_encoding,
    )
    if qnas_backend == "torch":
        return SearchableQuantumLayer(search_space)
    if qnas_backend == "tensorcircuit":
        return TensorCircuitSearchableQuantumLayer(search_space, tc_backend=qnas_tc_backend)
    raise ValueError("qnas_backend must be either 'torch' or 'tensorcircuit'.")


def _build_qnas_sampled_layer(circuit_path: str, qnas_root: str | None = None) -> nn.Module:
    _ensure_qnas_on_path(qnas_root)
    from qnas.yolo.layers import SampledQuantumLayer

    return SampledQuantumLayer.from_path(circuit_path)


def _build_angle_qnn_layer(n_qubits: int, quantum_depth: int):
    """Create a TensorCircuit TorchLayer for angle encoding."""
    tc, backend = _load_tensorcircuit()

    def qnn_circuit(inputs, weights):
        circuit = tc.Circuit(n_qubits)

        for qubit_idx in range(n_qubits):
            circuit.ry(qubit_idx, theta=inputs[qubit_idx])

        for layer_idx in range(quantum_depth):
            for qubit_idx in range(n_qubits - 1):
                circuit.cnot(qubit_idx, qubit_idx + 1)
            if n_qubits > 2:
                circuit.cnot(n_qubits - 1, 0)

            for qubit_idx in range(n_qubits):
                circuit.rx(qubit_idx, theta=weights[layer_idx, qubit_idx, 0])
                circuit.ry(qubit_idx, theta=weights[layer_idx, qubit_idx, 1])
                circuit.rz(qubit_idx, theta=weights[layer_idx, qubit_idx, 2])

        measurements = backend.stack(
            [
                circuit.expectation_ps(z=[qubit_idx])
                for qubit_idx in range(n_qubits)
            ]
        )
        return backend.real(measurements)

    return tc.TorchLayer(
        qnn_circuit,
        weights_shape=[quantum_depth, n_qubits, 3],
        use_interface=False,
    )


class TensorCircuitAngleQNN(nn.Module):
    """Angle-encoded variational quantum circuit.

    Input shape: ``[batch_size, n_qubits]``.
    Output shape: ``[batch_size, n_qubits]`` with Z-basis expectation values.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        quantum_depth: int = 3,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2.")
        if quantum_depth <= 0:
            raise ValueError("quantum_depth must be positive.")

        self.n_qubits = int(n_qubits)
        self.quantum_depth = int(quantum_depth)
        self.num_inputs = self.n_qubits
        self.num_outputs = self.n_qubits
        self.q_layer = _build_angle_qnn_layer(self.n_qubits, self.quantum_depth)

        for parameter in self.q_layer.parameters():
            nn.init.uniform_(parameter, -init_scale, init_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        if inputs.device.type != "cuda" or not hasattr(torch, "set_default_device"):
            return self.q_layer(inputs)

        previous_device = torch.get_default_device()
        try:
            torch.set_default_device(inputs.device)
            return self.q_layer(inputs)
        finally:
            torch.set_default_device(previous_device)


class HybridQuantumClassifierHead(nn.Module):
    """Low-dimensional quantum head for EfficientNet pooled features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits: int = 8,
        quantum_depth: int = 3,
        quantum_init_scale: float = 0.02,
        use_measurement_norm: bool = True,
        quantum_mode: str = "fixed",
        qnas_backend: str = "torch",
        qnas_tc_backend: str = "pytorch",
        qnas_entangle_pattern: str = "ring",
        qnas_oneq_ops: Iterable[str] = ("skip", "rx", "ry", "rz"),
        qnas_twoq_ops: Iterable[str] = ("skip", "crx", "cry", "crz"),
        qnas_input_encoding: str = "ry",
        circuit_path: str | None = None,
        qnas_root: str | None = None,
    ) -> None:
        super().__init__()

        self.quantum_mode = quantum_mode.lower()
        if self.quantum_mode == "fixed":
            self.quantum = TensorCircuitAngleQNN(
                n_qubits=n_qubits,
                quantum_depth=quantum_depth,
                init_scale=quantum_init_scale,
            )
            quantum_inputs = n_qubits
            quantum_outputs = n_qubits
        elif self.quantum_mode == "search":
            self.quantum = _build_qnas_search_layer(
                n_qubits=n_qubits,
                quantum_depth=quantum_depth,
                qnas_backend=qnas_backend,
                qnas_tc_backend=qnas_tc_backend,
                qnas_entangle_pattern=qnas_entangle_pattern,
                qnas_oneq_ops=qnas_oneq_ops,
                qnas_twoq_ops=qnas_twoq_ops,
                qnas_input_encoding=qnas_input_encoding,
                qnas_root=qnas_root,
            )
            quantum_inputs = n_qubits
            quantum_outputs = n_qubits
        elif self.quantum_mode == "sampled":
            if not circuit_path:
                raise ValueError("circuit_path is required when quantum_mode='sampled'.")
            self.quantum = _build_qnas_sampled_layer(circuit_path, qnas_root=qnas_root)
            quantum_inputs = int(self.quantum.num_inputs)
            quantum_outputs = int(self.quantum.num_outputs)
        else:
            raise ValueError("quantum_mode must be one of 'fixed', 'search', or 'sampled'.")

        self.input_projection = nn.Linear(in_features, quantum_inputs)
        self.measurement_norm = (
            nn.LayerNorm(quantum_outputs) if use_measurement_norm else nn.Identity()
        )
        self.classifier = nn.Linear(quantum_outputs, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.input_projection(features)) * math.pi
        measurements = self.quantum(angles)
        measurements = self.measurement_norm(measurements)
        return self.classifier(measurements)


class EfficientNetB0QNNClassifier(nn.Module):
    """EfficientNet-B0 image classifier with a TensorCircuit QNN head.

    Args:
        num_classes: Number of aircraft classes to predict.
        pretrained: Whether to load ImageNet-1K pretrained weights.
        dropout: Dropout probability before the hybrid quantum head.
        freeze_features: If True, freezes the EfficientNet feature extractor.
        input_shape: Kept for config compatibility.
        n_qubits: Number of qubits in the variational circuit.
        quantum_depth: Number of repeated entangling/rotation blocks.
        quantum_init_scale: Initial range for trainable quantum parameters.
        use_measurement_norm: Whether to apply LayerNorm to QNN measurements.
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_features: bool = False,
        input_shape: Iterable[int] = (128, 128),
        n_qubits: int = 8,
        quantum_depth: int = 3,
        quantum_init_scale: float = 0.02,
        use_measurement_norm: bool = True,
        quantum_mode: str = "fixed",
        qnas_backend: str = "torch",
        qnas_tc_backend: str = "pytorch",
        qnas_entangle_pattern: str = "ring",
        qnas_oneq_ops: Iterable[str] = ("skip", "rx", "ry", "rz"),
        qnas_twoq_ops: Iterable[str] = ("skip", "crx", "cry", "crz"),
        qnas_input_encoding: str = "ry",
        circuit_path: str | None = None,
        qnas_root: str | None = None,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.n_qubits = int(n_qubits)
        self.quantum_depth = int(quantum_depth)

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights, dropout=dropout)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = HybridQuantumClassifierHead(
            in_features=in_features,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            quantum_depth=self.quantum_depth,
            quantum_init_scale=quantum_init_scale,
            use_measurement_norm=use_measurement_norm,
            quantum_mode=quantum_mode,
            qnas_backend=qnas_backend,
            qnas_tc_backend=qnas_tc_backend,
            qnas_entangle_pattern=qnas_entangle_pattern,
            qnas_oneq_ops=qnas_oneq_ops,
            qnas_twoq_ops=qnas_twoq_ops,
            qnas_input_encoding=qnas_input_encoding,
            circuit_path=circuit_path,
            qnas_root=qnas_root,
        )

        if freeze_features:
            self.freeze_features()

    def freeze_features(self) -> None:
        """Freeze all feature-extraction layers and keep the QNN head trainable."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

    def unfreeze_features(self) -> None:
        """Unfreeze the EfficientNet feature extractor."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape ``[batch_size, num_classes]``."""
        return self.backbone(x)

    def export_qnas_circuit_spec(self, name: str = "efficientnet_b0_qnn_search"):
        """Export the searched QNAS architecture when quantum_mode='search'."""
        quantum = self.backbone.classifier[1].quantum
        if not hasattr(quantum, "export_circuit_spec"):
            raise RuntimeError("QNAS export is only available when quantum_mode='search'.")
        return quantum.export_circuit_spec(name=name)


MODEL_CLASS = EfficientNetB0QNNClassifier
