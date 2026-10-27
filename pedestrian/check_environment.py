"""Check native detection and metric dependencies without downloading a model."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the JSON report")
    args = parser.parse_args()

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30.0]])
    kept = nms(boxes, torch.tensor([0.9, 0.8, 0.7]), iou_threshold=0.5)
    if kept.tolist() != [0, 2]:
        raise RuntimeError("TorchVision's compiled NMS operator returned an unexpected result")

    # An exact, single-box detection should have perfect AP. This checks the
    # metric backend installation, not the quality of our pretrained detector.
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    target = {"boxes": boxes[:1], "labels": torch.tensor([1])}
    metric.update([{**target, "scores": torch.tensor([0.9])}], [target])
    scores = metric.compute()
    if abs(float(scores["map"]) - 1.0) > 1e-6:
        raise RuntimeError("The evaluation dependency failed its known-answer check")

    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "device": "cpu",
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "torchmetrics", "pycocotools", "pillow", "numpy")
        },
        "nms_kept_indices": kept.tolist(),
        "known_answer_map": float(scores["map"]),
        "known_answer_ap50": float(scores["map_50"]),
        "note": "Synthetic dependency checks; these are not dataset evaluation results.",
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
