"""Load image/annotation pairs listed in the fixed split manifest."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from pedestrian.annotations import extract_instances
from pedestrian.inference import prepare_image


def horizontal_flip(image, target):
    """Flip an image and its half-open xyxy boxes without changing the input."""
    width = image.shape[-1]
    boxes = target["boxes"].clone()
    left = boxes[:, 0].clone()
    boxes[:, 0] = width - boxes[:, 2]
    boxes[:, 2] = width - left
    return image.flip(-1), {**target, "boxes": boxes}


class PennFudanDataset(Dataset):
    def __init__(
        self,
        root: Path,
        manifest: Path,
        split: str,
        image_ids=None,
        flip_probability: float = 0.0,
        augmentation_seed: int = 43,
    ):
        self.root = Path(root)
        if not 0 <= flip_probability <= 1:
            raise ValueError("Flip probability must be between 0 and 1")
        if split != "train" and flip_probability:
            raise ValueError("Augmentation is allowed on training data only")
        self.flip_probability = flip_probability
        self.augmentation_generator = torch.Generator().manual_seed(augmentation_seed)
        splits = json.loads(Path(manifest).read_text(encoding="utf-8"))["splits"]
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        allowed = splits[split]
        self.image_ids = list(allowed if image_ids is None else image_ids)
        if len(set(self.image_ids)) != len(self.image_ids) or not set(self.image_ids) <= set(
            allowed
        ):
            raise ValueError("Image IDs must be unique members of the selected split")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(self.root / "PNGImages" / f"{image_id}.png") as image:
            tensor = prepare_image(image)
        with Image.open(self.root / "PedMasks" / f"{image_id}_mask.png") as mask:
            if tuple(tensor.shape[-2:]) != (mask.height, mask.width):
                raise ValueError(f"Image and mask dimensions differ for {image_id}")
            annotation = extract_instances(mask)
        target = {"boxes": annotation["boxes"], "labels": annotation["labels"]}
        if (
            self.flip_probability
            and torch.rand((), generator=self.augmentation_generator).item() < self.flip_probability
        ):
            tensor, target = horizontal_flip(tensor, target)
        return tensor, target, image_id


def collate_detection(batch):
    """Keep differently sized images and variable numbers of boxes in lists."""
    images, targets, image_ids = zip(*batch, strict=True)
    return list(images), list(targets), list(image_ids)
