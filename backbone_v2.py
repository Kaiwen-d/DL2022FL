from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
# from detectron2.modeling import ViT, SimpleFeaturePyramid
from vitdet.vit import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

# Creates Simple Feature Pyramid from ViT backbone
model = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=512,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=512,
)
