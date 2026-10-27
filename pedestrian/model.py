"""Construct the pretrained reference model; project fine-tuning comes later."""

from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

WEIGHTS = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"
PERSON_LABEL = WEIGHTS.meta["categories"].index("person")


def load_pretrained_detector():
    """Download/cache the official COCO weights and return a CPU eval model."""
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=WEIGHTS)
    return model.to("cpu").eval()
