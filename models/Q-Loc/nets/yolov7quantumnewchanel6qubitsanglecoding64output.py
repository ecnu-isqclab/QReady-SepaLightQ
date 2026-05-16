from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbone import Backbone, Conv, Multi_Concat_Block, SiLU, Transition_Block
from nets.yolo import SPPCSPC, RepConv, fuse_conv_and_bn


class TrueQNN6QubitsAngleCoding64Output(nn.Module):
    """6-qubit branch: 60 angle-coded inputs, 64 basis-state probabilities out."""

    def __init__(self, channels: int, n_qubits: int = 6, upload_rounds: int = 5):
        super().__init__()
        if n_qubits != 6:
            raise ValueError("This branch is fixed to 6 qubits.")
        if upload_rounds != 5:
            raise ValueError("60 input features require exactly 5 upload rounds of 12 angles.")
        self.channels = channels
        self.n_qubits = n_qubits
        self.upload_rounds = upload_rounds
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 15, 1, 1, 0, bias=False),
            nn.BatchNorm2d(15, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.q_angles = nn.Parameter(torch.randn(upload_rounds, n_qubits, 3) * 0.02)
        self.readout = nn.Linear(64, channels)
        self._qml_layer = None

    def _build_qml_layer(self):
        try:
            import numpy as np

            if not hasattr(np, "ComplexWarning"):
                np.ComplexWarning = RuntimeWarning
            import tensorcircuit as tc
            import tensorflow as tf  # noqa: F401
        except Exception as exc:
            raise RuntimeError("TrueQNN6QubitsAngleCoding64Output requires tensorcircuit and tensorflow.") from exc

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
            state = circuit.state()
            return backend.real(state * backend.conj(state))

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
        probabilities = self._qml_layer(encoded, self.q_angles.float())
        probabilities = probabilities.to(dtype=x.dtype, device=x.device)
        gate = torch.sigmoid(self.readout(probabilities)).reshape(batch, channels, 1, 1)
        return x * gate


class QuantumNewChannel6QubitsRepConv(nn.Module):
    """Frozen 1x1 branch + trainable 2x2 branch + trainable 6-qubit auxiliary branch."""

    def __init__(self, c1: int, c2: int, act=SiLU()):
        super().__init__()
        self.rbr_2x2 = nn.Sequential(
            nn.Conv2d(c1, c2, 2, 1, 0, bias=False),
            nn.BatchNorm2d(c2, eps=0.001, momentum=0.03),
        )
        self.qnn_branch = TrueQNN6QubitsAngleCoding64Output(c1)
        self.quantum_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2, eps=0.001, momentum=0.03),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2, eps=0.001, momentum=0.03),
        )
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_2x2 = self.rbr_2x2(F.pad(x, (0, 1, 0, 1)))
        branch_qnn = self.quantum_1x1(self.qnn_branch(x))
        branch_1x1 = self.rbr_1x1(x)
        return self.act(branch_2x2 + branch_qnn + branch_1x1)


class YoloBodyQuantumNewChanel6QubitsAngleCoding64Output(nn.Module):
    """YOLOv7 with P3/P4/P5 1x1 + 2x2 + 6-qubit auxiliary branches."""

    def __init__(self, anchors_mask, num_classes, phi, pretrained: bool = False, **_unused_kwargs):
        super().__init__()
        transition_channels = {"l": 32, "x": 40}[phi]
        block_channels = 32
        panet_channels = {"l": 32, "x": 64}[phi]
        e = {"l": 2, "x": 1}[phi]
        n = {"l": 4, "x": 6}[phi]
        ids = {"l": [-1, -2, -3, -4, -5, -6], "x": [-1, -3, -5, -7, -8]}[phi]

        self.backbone = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(
            transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids
        )
        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(
            transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids
        )
        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Multi_Concat_Block(
            transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids
        )
        self.down_sample2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Multi_Concat_Block(
            transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids
        )

        self.rep_conv_1 = QuantumNewChannel6QubitsRepConv(transition_channels * 4, transition_channels * 8)
        self.rep_conv_2 = QuantumNewChannel6QubitsRepConv(transition_channels * 8, transition_channels * 16)
        self.rep_conv_3 = QuantumNewChannel6QubitsRepConv(transition_channels * 16, transition_channels * 32)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        for module in self.modules():
            if isinstance(module, RepConv):
                module.fuse_repvgg_block()
            elif type(module) is Conv and hasattr(module, "bn"):
                module.conv = fuse_conv_and_bn(module.conv, module.bn)
                delattr(module, "bn")
                module.forward = module.fuseforward
        return self

    def forward(self, x: torch.Tensor):
        feat1, feat2, feat3 = self.backbone.forward(x)
        p5 = self.sppcspc(feat3)
        p5_conv = self.conv_for_P5(p5)
        p4 = torch.cat([self.conv_for_feat2(feat2), self.upsample(p5_conv)], 1)
        p4 = self.conv3_for_upsample1(p4)
        p4_conv = self.conv_for_P4(p4)
        p3 = torch.cat([self.conv_for_feat1(feat1), self.upsample(p4_conv)], 1)
        p3 = self.conv3_for_upsample2(p3)
        p4 = torch.cat([self.down_sample1(p3), p4], 1)
        p4 = self.conv3_for_downsample1(p4)
        p5 = torch.cat([self.down_sample2(p4), p5], 1)
        p5 = self.conv3_for_downsample2(p5)
        p3 = self.rep_conv_1(p3)
        p4 = self.rep_conv_2(p4)
        p5 = self.rep_conv_3(p5)
        out2 = self.yolo_head_P3(p3)
        out1 = self.yolo_head_P4(p4)
        out0 = self.yolo_head_P5(p5)
        return [out0, out1, out2]


MODEL_CLASS = YoloBodyQuantumNewChanel6QubitsAngleCoding64Output
