from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbone import Conv, Multi_Concat_Block, SiLU, Transition_Block
from nets.yolo import RepConv, fuse_conv_and_bn
from nets.yolo_light import LightBackbone
from nets.yolov7quantumnewchanel6qubitsanglecoding64output import (
    QuantumNewChannel6QubitsRepConv,
    TrueQNN6QubitsAngleCoding64Output,
)


class Conv2x2Same(nn.Module):
    """2x2 Conv + BN + activation with output spatial size kept unchanged."""

    def __init__(self, c1: int, c2: int, act=SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 2, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(F.pad(x, (0, 1, 0, 1)))))


class LightQuantumChannelSPPCSPC(nn.Module):
    """Light SPPCSPC with two 2x2 spatial branches and a 6-qubit auxiliary channel."""

    def __init__(self, c1: int, c2: int, e: float = 0.5, k: tuple[int, int, int] = (5, 9, 13)):
        super().__init__()
        hidden_channels = int(2 * c2 * e)
        self.cv1 = Conv(c1, hidden_channels, 1, 1)
        self.cv2 = Conv(c1, hidden_channels, 1, 1)
        self.cv3 = Conv2x2Same(hidden_channels, hidden_channels)
        self.cv4 = Conv(hidden_channels, hidden_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in k])
        self.cv5 = Conv(4 * hidden_channels, hidden_channels, 1, 1)
        self.cv6 = Conv2x2Same(hidden_channels, hidden_channels)
        self.cv7 = Conv(2 * hidden_channels, c2, 1, 1)
        self.qnn_branch = TrueQNN6QubitsAngleCoding64Output(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [pool(x1) for pool in self.m], 1)))
        y2 = self.cv2(x)
        base = self.cv7(torch.cat((y1, y2), dim=1))
        return base + self.qnn_branch(base)


class YoloLightQuantumChanelBody(nn.Module):
    """YOLO light with q6 SPPCSPC channel and q6 P3/P4/P5 pre-head blocks."""

    def __init__(
        self,
        anchors_mask,
        num_classes: int = 1,
        phi: str = "light",
        pretrained: bool = False,
        **_unused_kwargs,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("YoloLightQuantumChanelBody is aircraft/object-only and requires num_classes=1.")

        transition_channels = 16
        block_channels = 16
        panet_channels = 16
        e = 1
        n = 2
        ids = [-1, -2, -3, -4]

        self.num_classes = 1
        self.phi = phi
        self.pretrained = pretrained
        self.backbone = LightBackbone(transition_channels, block_channels, n)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc = LightQuantumChannelSPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(
            transition_channels * 16,
            panet_channels * 4,
            transition_channels * 8,
            e=e,
            n=n,
            ids=ids,
        )

        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(
            transition_channels * 8,
            panet_channels * 2,
            transition_channels * 4,
            e=e,
            n=n,
            ids=ids,
        )

        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Multi_Concat_Block(
            transition_channels * 16,
            panet_channels * 4,
            transition_channels * 8,
            e=e,
            n=n,
            ids=ids,
        )

        self.down_sample2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Multi_Concat_Block(
            transition_channels * 32,
            panet_channels * 8,
            transition_channels * 16,
            e=e,
            n=n,
            ids=ids,
        )

        self.rep_conv_1 = QuantumNewChannel6QubitsRepConv(transition_channels * 4, transition_channels * 8)
        self.rep_conv_2 = QuantumNewChannel6QubitsRepConv(transition_channels * 8, transition_channels * 16)
        self.rep_conv_3 = QuantumNewChannel6QubitsRepConv(transition_channels * 16, transition_channels * 32)

        out_channels = 5 + self.num_classes
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * out_channels, 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * out_channels, 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * out_channels, 1)

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


MODEL_CLASS = YoloLightQuantumChanelBody
