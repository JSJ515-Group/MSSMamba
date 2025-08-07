
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Block
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


############   SparseAttention（稀疏注意力） ############
# 定义稀疏注意力机制
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, block_size=16, attention_dropout=0.1):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.block_size = block_size
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        block_size = self.block_size
        num_blocks = N // block_size
        attn_output = torch.zeros_like(q)

        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            q_block = q[:, :, start:end, :]
            k_block = k[:, :, start:end, :]
            v_block = v[:, :, start:end, :]
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_block, k_block) / (C // self.num_heads) ** 0.5
            attn_probs = self.attention_dropout(torch.softmax(attn_scores, dim=-1))
            attn_output[:, :, start:end, :] = torch.einsum('bhqk,bhvd->bhqd', attn_probs, v_block)

        attn_output = attn_output.reshape(B, N, C)
        out = self.proj(attn_output)
        return out


# 修改 CrossAttentionBlock 以使用 SparseAttention
class SparseAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SparseAttention(dim=dim, num_heads=num_heads, block_size=16, attention_dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 修改 MultiScaleBlock 以使用 CrossAttentionBlockWithSparse
class MultiScaleBlock(nn.Module):
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = [Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                         attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer) for i in range(depth[d])]
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            self.projs.append(
                nn.Sequential(norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d + 1) % num_branches])))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            nh = num_heads[(d + 1) % num_branches]
            tmp = [SparseAttentionBlock(dim=dim[d], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                 norm_layer=norm_layer, has_mlp=False) for _ in range(depth[-1])]
            self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            self.revert_projs.append(nn.Sequential(norm_layer(dim[(d + 1) % num_branches]), act_layer(),
                                                   nn.Linear(dim[(d + 1) % num_branches], dim[d])))

    def forward(self, x):
        inp = x
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(inp, self.projs)]
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], inp[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, inp[i][:, 1:, ...]), dim=1)
            outs.append(tmp)

        outs = [block(x_) for x_, block in zip(outs, self.blocks)]
        return outs

############ Test ############
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            B, C, H, W = input.shape
            input = input.expand(B, 3, H, W)

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
