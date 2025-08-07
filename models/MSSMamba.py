import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead
from another_models.backup_models.HLAEM import HLAEM

class MSSMamba(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]),
                              w=(self.img_size // self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)
        # 特征融合：将两个处理后的特征图相加得到C
        # C = reshaped_embed[0] + reshaped_embed[1]

        HLAEM1 = HLAEM(dim=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        HLAEM1.to(device)
        C1 = HLAEM1(reshaped_embed[0])
        C2 = HLAEM1(reshaped_embed[1])
        C = C1 + C2
        # 预测卷积：使用conv_pred模块对C进行卷积和上采样，以匹配SegmentationHead的输入尺寸
        C = self.conv_pred(C)

        # 分割头：将处理后的特征图传递给segmentation_head，生成最终的分割图
        out = self.segmentation_head(C)

        return out  