from timm.models.layers import trunc_normal_
from utils import *
from einops import rearrange
from models.vmamba import VSSM
from another_models.MSACBlock import MSACBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3):
        super().__init__()

        model_path = './weights/vmamba_small_e238_ema.pth'
        self.vmunet = VSSM(in_chans=in_channels,
                           num_classes=9,
                           depths=[2, 2, 9, 2],
                           depths_decoder=[2, 9, 2, 2],
                           drop_path_rate=0.2)

        local_weights_path = "./weights/resnet50-0676ba61.pth"
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained=False)")
        resnet.load_state_dict(torch.load(local_weights_path))
        resnet.eval()

        self.resnet_layers = nn.ModuleList(resnet.children())[:7]

        # Conv layers for channel alignment
        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)

        # CS Blocks for each pyramid level
        self.cs_block1 = MSACBlock(in_channels=256, out_channels=config.swin_pyramid_fm[0])
        self.cs_block2 = MSACBlock(in_channels=config.swin_pyramid_fm[1], out_channels=config.swin_pyramid_fm[1])
        self.cs_block3 = MSACBlock(in_channels=config.swin_pyramid_fm[2], out_channels=config.swin_pyramid_fm[2])

        self.p2 = self.resnet_layers[5]
        self.p3 = self.resnet_layers[6]

        # Normalization and pooling layers
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv192_96 = nn.Conv2d(192, 96, kernel_size=1)
        self.conv96_192 = nn.Conv2d(96, 192, kernel_size=1)
        self.conv384_192 = nn.Conv2d(384, 192, kernel_size=1)
        self.conv192_384 = nn.Conv2d(192, 384, kernel_size=1)
        self.conv768_384 = nn.Conv2d(768, 384, kernel_size=1)

    def forward(self, x):
        # ResNet feature extraction
        for i in range(5):
            x = self.resnet_layers[i](x)

        # Level 1
        fm1 = x  # Shape: (10, 256, 56, 56)
        fm1_ch = self.p1_ch(self.upsample(fm1))  # Align channels, Shape: (10, 96, 112, 112)
        fm1_processed = self.cs_block1(fm1)  # CS Block, Shape: (10, 96, 56, 56)

        m1 = self.vmunet.layers[0](fm1_ch.permute(0, 2, 3, 1))  # VSSM Layer 0, Shape: (10, 56, 56, 192)
        m1 = self.conv192_96(m1.permute(0, 3, 1, 2)) + fm1_processed  # Align and add, Shape: (10, 96, 56, 56)

        m1_CLS = self.avgpool_1(self.norm_1(m1.flatten(2).transpose(1, 2)).transpose(1, 2)).squeeze(-1)  # CLS token
        m1_skipped = self.conv96_192(m1).permute(0, 2, 3, 1)  # Prepare for next level, Shape: (10, 56, 56, 192)

        # Level 2
        fm2 = self.p2(fm1)  # Shape: (10, 512, 28, 28)
        fm2_ch = self.cs_block2(self.p2_ch(fm2))  # Align channels and process, Shape: (10, 192, 28, 28)

        m2 = self.vmunet.layers[1](m1_skipped)  # VSSM Layer 1, Shape: (10, 28, 28, 384)
        m2 = self.conv384_192(m2.permute(0, 3, 1, 2)) + fm2_ch  # Align and add, Shape: (10, 192, 28, 28)

        m2_skipped = self.conv192_384(m2).permute(0, 2, 3, 1)  # Prepare for next level, Shape: (10, 28, 28, 384)

        # Level 3
        fm3 = self.p3(fm2)  # Shape: (10, 1024, 14, 14)
        fm3_ch = self.cs_block3(self.p3_ch(fm3))  # Align channels and process, Shape: (10, 384, 14, 14)

        m3 = self.vmunet.layers[2](m2_skipped)  # VSSM Layer 2, Shape: (10, 14, 14, 768)
        m3 = self.conv768_384(m3.permute(0, 3, 1, 2)) + fm3_ch  # Align and add, Shape: (10, 384, 14, 14)

        m3_CLS = self.avgpool_2(self.norm_2(m3.flatten(2).transpose(1, 2)).transpose(1, 2)).squeeze(-1)  # CLS token

        # Concatenate CLS tokens and features for level 1 and level 3
        m1_output = torch.cat((m1_CLS.unsqueeze(1), m1.flatten(2).transpose(1, 2)), dim=1)  # Shape: (10, 3136, 96)
        m3_output = torch.cat((m3_CLS.unsqueeze(1), m3.flatten(2).transpose(1, 2)), dim=1)  # Shape: (10, 196, 384)

        return [m1_output, m3_output]

class All2Cross(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        # 从 config 中提取的一个参数，控制是否在跨分支时使用位置嵌入
        self.cross_pos_embed = config.cross_pos_embed
        # 从PyramidFeatures中提取不同尺度的特征
        self.pyramid = PyramidFeatures(config=config, img_size=img_size, in_channels=in_chans)

        n_p1 = (config.image_size // config.patch_size) ** 2  # default: 3136
        n_p2 = (config.image_size // config.patch_size // 4) ** 2  # default: 196
        num_patches = (n_p1, n_p2)
        self.num_branches = 2

        self.pos_embed = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0

        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads,
                                  mlp_ratio=config.mlp_ratio,
                                  qkv_bias=config.qkv_bias,  drop=config.drop_rate,
                                  attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        xs = self.pyramid(x)

        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs
