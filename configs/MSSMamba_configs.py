import ml_collections
import os

os.makedirs('./weights', exist_ok=True)


def get_mssmamba_configs():

    cfg = ml_collections.ConfigDict()
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

