import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate


# PRE_TRAINED_PATH = "./checkpoint-50.pth"
# PRE_TRAINED_PATH = "./checkpoints/fasterrcnn-7-11.pth"
# PRE_TRAINED_PATH = "./fasterrcnn-2-2.pth"

def get_model():

    cfg = LazyConfig.load("./backbone_v2.py")
    backbone = instantiate(cfg.model)
    DetectionCheckpointer(backbone).load("./checkpoint-50-net.pth")
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(sizes=((16,),(32,),(64,),(128,),(256,)),
                                aspect_ratios=((0.5, 1.0, 2.0),)*5)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["p2", "p3", "p4", "p5"],
                                                    output_size=7,
                                                    sampling_ratio=0)
    roi_pooler.scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                num_classes=100,
                min_size=512,
                max_size=512,
                fixed_size=(512, 512),
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler)

    # checkpoint = torch.load(PRE_TRAINED_PATH, map_location='cpu')
    # print("Load pre-trained checkpoint from: %s" % PRE_TRAINED_PATH)
    # checkpoint_model = checkpoint['model']
    # msg = model.load_state_dict(checkpoint_model)
    # print(msg)

    return model

