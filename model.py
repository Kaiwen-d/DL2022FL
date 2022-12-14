import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import vit


MODEL_PATH = "./fasterrcnn-11-13.pth"

def get_model():

    # init backbone
    backbone = vit.__dict__['vit_base_patch16'](
        img_size=512,
        num_classes=100,
        drop_path_rate=0.1
    )

    # build fpn network
    backbone_with_fpn = vit.FPN(
        backbone=backbone,
        embed_dim=768,
        out_dim=256,
        extra_pool=1
    )

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=100,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((32,),(64,),(128,),(256,),(512,)),
            aspect_ratios=((0.5, 1.0, 2.0),)*5
        ),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=8,
            sampling_ratio=2
        )
    )
    model.transform = GeneralizedRCNNTransform(
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        min_size=512,
        max_size=512,
        fixed_size=(512, 512),
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"])
    print(msg)

    return model
