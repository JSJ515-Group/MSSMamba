import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_




class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))

        return out
class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)

        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.enhance = FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        ctx = self.norm3(self.act(self.ctx_conv(x)))  #SCM模块

        enh_x = self.enhance(x)                       #FRM模块
        x = self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_dcn=False):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x
class AtrousMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos1 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=1, groups=dim * 2)
        self.pos2 = nn.Conv2d(dim * mlp_ratio, dim * 2, 3, padding=2, dilation=2, groups=dim * 2)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.act(self.fc1(x))
        x1 = self.act(self.pos1(x))
        x2 = self.act(self.pos2(x))
        x_a = torch.cat([x1, x2], dim=1)
        x = self.fc2(x_a)

        return x

# 二次创新
class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out,kernel_size,stride)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class HLAEM(nn.Module): #高低频特征自适应模块
    def __init__(self, dim):
        super(HLAEM, self).__init__()
        n_feats = dim
        self.down = nn.AvgPool2d(kernel_size=2)
        self.cmunextblock = CMUNeXtBlock(n_feats,n_feats)
        self.meem = MEEM(dim,dim)
        self.alise1 = nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)
        self.att = CALayer(n_feats)
    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        lowf = self.meem(low)
        highfeat =  self.cmunextblock(high)
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out = self.alise2(self.att(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))) + x
        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #随机生成一张输入图片张量
    HLAEM  = HLAEM(dim=32)
    output = HLAEM(input)  # 进行前向传播
    # 输出结果的形状
    print("HLAEM_输入张量的形状：", input.shape)
    print("HLAEM_输出张量的形状：", output.shape)
