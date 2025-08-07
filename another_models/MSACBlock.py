
import torch
import torch.nn as nn
import torch.nn.functional as F
from another_models.MSAA import CombinedAttentionModule

class MSACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors=[1, 2, 4], reduction=4, kernel_size=7):
        super(CSBlock, self).__init__()

        # 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                CombinedAttentionModule(out_channels, reduction, kernel_size)  # 将注意力模块集成到每个卷积层中
            ) for _ in scale_factors
        ])

        # 注意力模块应用后，再融合不同尺度特征
        self.scale_factors = scale_factors
        self.fusion_conv = nn.Conv2d(out_channels * len(scale_factors), out_channels, kernel_size=1)

    def forward(self, x):
        features = []
        for i, conv in enumerate(self.convs):
            scale_factor = self.scale_factors[i]

            # 缩放至目标尺度
            scaled_x = F.interpolate(x, scale_factor=1.0 / scale_factor, mode='bilinear', align_corners=False)

            # 经过卷积和注意力处理
            scaled_feat = conv(scaled_x)

            # 恢复为原始尺度
            features.append(F.interpolate(scaled_feat, size=x.size()[2:], mode='bilinear', align_corners=False))

        # 融合多尺度特征
        fusion_features = torch.cat(features, dim=1)
        out = self.fusion_conv(fusion_features)

        return out

