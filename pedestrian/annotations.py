"""Turn instance-mask pixel IDs into person detection targets."""

import numpy as np
import torch
from PIL import Image
from torchvision.ops import masks_to_boxes


def extract_instances(mask: Image.Image) -> dict[str, torch.Tensor]:
    """Convert a single-channel instance mask into half-open xyxy person boxes.

    Mask value 0 is background; every other value identifies one person.
    Right and bottom box coordinates are exclusive, as with Python slices.
    """
    values = np.array(mask)
    if values.ndim != 2:
        raise ValueError("Expected a single-channel instance-ID mask")

    pixels = torch.from_numpy(values.astype(np.int64))
    instance_ids = torch.unique(pixels)
    instance_ids = instance_ids[instance_ids != 0]
    if len(instance_ids) == 0:
        boxes = torch.empty((0, 4), dtype=torch.float32)
    else:
        binary_masks = pixels == instance_ids[:, None, None]
        boxes = masks_to_boxes(binary_masks)
        # masks_to_boxes returns the greatest occupied pixel index. Add 1 so
        # right/bottom are exclusive and a one-pixel instance has nonzero area.
        boxes[:, 2:] += 1

    return {
        "instance_ids": instance_ids,
        "boxes": boxes,
        "labels": torch.ones(len(instance_ids), dtype=torch.int64),
    }
