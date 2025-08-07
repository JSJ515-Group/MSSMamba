

import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CombinedAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super(CombinedAttentionModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        # 先应用通道注意力

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_attention.to(device)  # 将模块移到相同设备上
        self.spatial_attention.to(device)  # 同样处理

        channel_weights = self.channel_attention(x)  # 输出形状: (10, 96, 1, 1) 或其他
        x = x * channel_weights  # 形状保持 (10, C, H, W)

        # 再应用空间注意力
        spatial_weights = self.spatial_attention(x)  # 输出形状: (10, 1, H, W)
        x = x * spatial_weights  # 形状保持 (10, C, H, W)

        return x