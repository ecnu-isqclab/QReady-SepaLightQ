import torch
import torch.nn as nn

from nets.backbone import Conv, Multi_Concat_Block, Transition_Block
from nets.yolo import RepConv, SPPCSPC, fuse_conv_and_bn


class LightBackbone(nn.Module):
    def __init__(self, transition_channels=16, block_channels=16, n=2):
        super().__init__()
        ids = [-1, -2, -3, -4]

        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(
                transition_channels * 4,
                block_channels * 2,
                transition_channels * 8,
                n=n,
                ids=ids,
            ),
        )
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(
                transition_channels * 8,
                block_channels * 4,
                transition_channels * 16,
                n=n,
                ids=ids,
            ),
        )
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            Multi_Concat_Block(
                transition_channels * 16,
                block_channels * 8,
                transition_channels * 32,
                n=n,
                ids=ids,
            ),
        )
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 32, transition_channels * 16),
            Multi_Concat_Block(
                transition_channels * 32,
                block_channels * 8,
                transition_channels * 32,
                n=n,
                ids=ids,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


class YoloLightBody(nn.Module):
    def __init__(
        self,
        anchors_mask,
        num_classes=1,
        phi="light",
        pretrained=False,
        **_unused_kwargs,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("YoloLightBody is aircraft/object-only and requires num_classes=1.")

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

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
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

        self.rep_conv_1 = RepConv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = RepConv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = RepConv(transition_channels * 16, transition_channels * 32, 3, 1)

        out_channels = 5 + self.num_classes
        self.yolo_head_P3 = nn.Conv2d(
            transition_channels * 8,
            len(anchors_mask[2]) * out_channels,
            1,
        )
        self.yolo_head_P4 = nn.Conv2d(
            transition_channels * 16,
            len(anchors_mask[1]) * out_channels,
            1,
        )
        self.yolo_head_P5 = nn.Conv2d(
            transition_channels * 32,
            len(anchors_mask[0]) * out_channels,
            1,
        )

    def fuse(self):
        print("Fusing layers... ")
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone.forward(x)

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)
        return [out0, out1, out2]


MODEL_CLASS = YoloLightBody
