"""The inference path shared by the CLI and the future application."""

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from pedestrian.model import PERSON_LABEL


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Orient an image and convert it to RGB float [C, H, W] in [0, 1]."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return pil_to_tensor(rgb).to(dtype=torch.float32) / 255.0


@torch.inference_mode()
def predict(image: Image.Image, model, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    """Return person boxes, labels, and scores in oriented-image pixel coordinates."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    model.eval()
    output = model([prepare_image(image)])[0]
    keep = (output["labels"] == PERSON_LABEL) & (output["scores"] >= threshold)
    return {key: output[key][keep].cpu() for key in ("boxes", "labels", "scores")}
