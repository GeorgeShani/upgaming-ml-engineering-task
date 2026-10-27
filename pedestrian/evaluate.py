"""Measure a detector on the validation split without tuning on the test split."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from pedestrian.data import PennFudanDataset
from pedestrian.model import PERSON_LABEL, load_pretrained_detector


def match_counts(prediction, target, threshold=0.5, iou_threshold=0.5):
    """Greedily match score-ordered detections to distinct ground-truth boxes."""
    indices = torch.where(prediction["scores"] >= threshold)[0]
    indices = indices[torch.argsort(prediction["scores"][indices], descending=True)]
    overlaps = box_iou(prediction["boxes"][indices], target["boxes"])
    matched = set()
    true_positives = 0
    for row in overlaps:
        candidates = sorted(range(len(row)), key=lambda i: float(row[i]), reverse=True)
        for index in candidates:
            if index not in matched and float(row[index]) >= iou_threshold:
                matched.add(index)
                true_positives += 1
                break
    return {
        "tp": true_positives,
        "fp": len(indices) - true_positives,
        "fn": len(target["boxes"]) - true_positives,
    }


@torch.inference_mode()
def evaluate_model(model, dataset, threshold=0.5, progress=True):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    totals = {"tp": 0, "fp": 0, "fn": 0}
    images = []
    started = time.perf_counter()
    for index, (image, target, image_id) in enumerate(dataset):
        output = model([image])[0]
        person = output["labels"] == PERSON_LABEL
        pred = {key: output[key][person].cpu() for key in ("boxes", "scores", "labels")}
        # TorchMetrics uses zero-indexed classes; the model's person ID is 1.
        metric_pred = {**pred, "labels": torch.zeros_like(pred["labels"])}
        metric_target = {**target, "labels": torch.zeros_like(target["labels"])}
        metric.update([metric_pred], [metric_target])
        counts = match_counts(pred, target, threshold=threshold)
        for key, value in counts.items():
            totals[key] += value
        chosen = pred["scores"] >= threshold
        images.append(
            {
                "image_id": image_id,
                **counts,
                "ground_truth_boxes": target["boxes"].tolist(),
                "predicted_boxes": pred["boxes"][chosen].tolist(),
                "predicted_scores": pred["scores"][chosen].tolist(),
            }
        )
        if progress:
            print(f"Evaluated {index + 1}/{len(dataset)}: {image_id}", flush=True)
    if not images:
        raise ValueError("Cannot evaluate an empty dataset")
    values = metric.compute()
    precision_denominator = totals["tp"] + totals["fp"]
    recall_denominator = totals["tp"] + totals["fn"]
    return {
        "metrics": {key: float(values[key]) for key in ("map", "map_50", "map_75", "mar_100")},
        "counts": totals,
        "precision_at_threshold": totals["tp"] / precision_denominator
        if precision_denominator
        else None,
        "recall_at_threshold": totals["tp"] / recall_denominator if recall_denominator else None,
        "confidence_threshold_for_counts": threshold,
        "matching_iou_for_counts": 0.5,
        "elapsed_seconds": time.perf_counter() - started,
        "images": images,
    }


def configure_evaluation(model):
    # Keep low-scoring predictions for AP; the reporting threshold is separate.
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 300
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/splits.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Trained checkpoint; omit for COCO baseline"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    dataset = PennFudanDataset(args.dataset_root, args.manifest, "validation")
    if args.checkpoint:
        # Import here to avoid a module cycle: training also calls evaluate_model.
        from pedestrian.train import load_checkpoint

        model, checkpoint_metadata = load_checkpoint(args.checkpoint)
        experiment = checkpoint_metadata.get("config", {}).get("experiment", "trained")
    else:
        model = load_pretrained_detector()
        experiment = "E0"
    model = configure_evaluation(model)
    result = evaluate_model(model, dataset)
    result.update(
        {
            "experiment": experiment,
            "split": "validation",
            "score_floor": 0.001,
            "threads": args.threads,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "checkpoint_sha256": (
                hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                if args.checkpoint
                else None
            ),
            "checkpoint_epoch": checkpoint_metadata["epoch"] if args.checkpoint else None,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "metrics",
                    "counts",
                    "precision_at_threshold",
                    "recall_at_threshold",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
