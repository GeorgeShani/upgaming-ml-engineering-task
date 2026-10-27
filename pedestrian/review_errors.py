"""Render validation examples with annotation and prediction boxes side by side."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    result = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if result.get("split") != "validation":
        parser.error("This review currently accepts validation results only")
    ranked = sorted(
        result["images"], key=lambda item: (-(item["fp"] + item["fn"]), item["image_id"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for item in ranked[: args.top]:
        image_id = item["image_id"]
        with Image.open(args.dataset_root / "PNGImages" / f"{image_id}.png") as image:
            image = image.convert("RGB")
        ground_truth, predicted = image.copy(), image.copy()
        truth_draw, prediction_draw = ImageDraw.Draw(ground_truth), ImageDraw.Draw(predicted)
        for box in item["ground_truth_boxes"]:
            truth_draw.rectangle(box, outline="lime", width=3)
        for box, score in zip(item["predicted_boxes"], item["predicted_scores"], strict=True):
            prediction_draw.rectangle(box, outline="red", width=3)
            prediction_draw.text((box[0], max(0, box[1] - 12)), f"{score:.2f}", fill="red")
        for panel in (ground_truth, predicted):
            panel.thumbnail((600, 500))
        width, height = (
            max(ground_truth.width, predicted.width),
            max(ground_truth.height, predicted.height),
        )
        canvas = Image.new("RGB", (width * 2, height + 30), "white")
        canvas.paste(ground_truth, (0, 30))
        canvas.paste(predicted, (width, 30))
        label = ImageDraw.Draw(canvas)
        label.text((5, 8), f"{image_id}: annotations", fill="black")
        label.text(
            (width + 5, 8),
            f"predictions: TP {item['tp']}, FP {item['fp']}, FN {item['fn']}",
            fill="black",
        )
        path = args.output / f"{image_id}.png"
        canvas.save(path)
        print(path)


if __name__ == "__main__":
    main()
