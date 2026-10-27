"""Inspect one Penn-Fudan image, its instance mask, and derived ground-truth boxes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pedestrian.annotations import extract_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--image-id", required=True, help="Image name without .png")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path(args.image_id).name != args.image_id or args.image_id in {"", ".", ".."}:
        parser.error("--image-id must be a filename without a path")

    image_path = args.dataset_root / "PNGImages" / f"{args.image_id}.png"
    mask_path = args.dataset_root / "PedMasks" / f"{args.image_id}_mask.png"
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.copy()
    except OSError as error:
        parser.error(f"Could not open the matching image and mask: {error}")
    if image.size != mask.size:
        parser.error(f"Image and mask sizes differ: {image.size} vs {mask.size}")
    try:
        target = extract_instances(mask)
    except ValueError as error:
        parser.error(str(error))

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for instance_id, box in zip(
        target["instance_ids"].tolist(), target["boxes"].tolist(), strict=True
    ):
        left, top, right, bottom = box
        # Pillow draws inclusive endpoints; our box's right/bottom are exclusive.
        draw.rectangle((left, top, right - 1, bottom - 1), outline="lime", width=2)
        draw.text((left, max(0, top - 12)), f"person {instance_id}", fill="lime")

    report = {
        "image_id": args.image_id,
        "image_size_wh": list(image.size),
        "mask_mode": mask.mode,
        "background_id": 0,
        "instance_ids": target["instance_ids"].tolist(),
        "boxes_xyxy_exclusive": target["boxes"].tolist(),
        "class_labels": target["labels"].tolist(),
        "note": "Ground-truth boxes derived from instance masks; not model predictions.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output / "ground-truth-boxes.png")
    # IDs in the original grayscale mask are small integers, so displaying them
    # directly would look nearly black. Colorize a copy without changing the IDs.
    colors = [(255, 90, 90), (80, 190, 255), (255, 220, 70), (170, 110, 255)]
    mask_values = np.array(mask)
    colored = np.zeros((*mask_values.shape, 3), dtype=np.uint8)
    for index, instance_id in enumerate(target["instance_ids"].tolist()):
        colored[mask_values == instance_id] = colors[index % len(colors)]
    Image.fromarray(colored).save(args.output / "instance-mask.png")
    (args.output / "annotation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved annotation images and JSON to {args.output}")


if __name__ == "__main__":
    main()
