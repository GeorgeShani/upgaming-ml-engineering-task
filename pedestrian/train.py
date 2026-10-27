"""Fine-tune the detector on training images and select by validation mAP."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from pedestrian.data import PennFudanDataset, collate_detection
from pedestrian.evaluate import configure_evaluation, evaluate_model
from pedestrian.model import MODEL_NAME, load_pretrained_detector


def _replace_batchnorm_with_frozen(module):
    # An uninitialized TorchVision model uses BatchNorm2d, but the pretrained
    # detector uses FrozenBatchNorm2d. Match its architecture without downloading.
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
        else:
            _replace_batchnorm_with_frozen(child)


def build_empty_model():
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=2
    )
    _replace_batchnorm_with_frozen(model)
    return model


def build_training_model():
    model = load_pretrained_detector()
    inputs = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inputs, 2)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    return model


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_state": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = build_empty_model()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint["metadata"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, learning_rate=0.005):
    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )


def tiny_learning_check(dataset, seed):
    """Check finite losses, actual updates, frozen backbone, and small-set learning."""
    set_seed(seed)
    model = build_training_model()
    model.train()
    optimizer = make_optimizer(model)
    images, targets, _ = collate_detection([dataset[0], dataset[1]])
    backbone_parameter = next(model.backbone.parameters())
    frozen_before = backbone_parameter.detach().clone()
    head_parameter = model.roi_heads.box_predictor.cls_score.weight
    head_before = head_parameter.detach().clone()
    losses = []
    started = time.perf_counter()
    for step in range(5):
        # RPN samples proposals randomly; repeating its seed makes this small
        # diagnostic easier to interpret, but does not make training deterministic.
        torch.manual_seed(seed + 100)
        components = model(images, targets)
        loss = sum(components.values())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Tiny-set check encountered a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise RuntimeError("Frozen backbone unexpectedly received gradients")
        if head_parameter.grad is None or not bool(torch.isfinite(head_parameter.grad).all()):
            raise RuntimeError("Prediction head did not receive finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"Tiny-set step {step + 1}: loss {losses[-1]:.4f}", flush=True)
    if torch.equal(head_before, head_parameter.detach()):
        raise RuntimeError("Prediction head did not update")
    if not torch.equal(frozen_before, backbone_parameter.detach()):
        raise RuntimeError("Frozen backbone parameter unexpectedly changed")
    if min(losses[1:]) >= losses[0]:
        raise RuntimeError("Tiny-set loss did not decrease; investigate before the full run")
    return {
        "image_ids": dataset.image_ids[:2],
        "losses": losses,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_backbone_unchanged": True,
        "head_updated": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/splits.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size, and threads must be positive")
    torch.set_num_threads(args.threads)
    flip_probability = 0.5 if args.experiment == "B" else 0.0
    train = PennFudanDataset(
        args.dataset_root,
        args.manifest,
        "train",
        flip_probability=flip_probability,
        augmentation_seed=args.seed + 1,
    )
    diagnostic_train = PennFudanDataset(args.dataset_root, args.manifest, "train")
    validation = PennFudanDataset(args.dataset_root, args.manifest, "validation")
    if len(train) < 2:
        parser.error("Training split needs at least two images for the diagnostic")
    if any(
        (args.output / name).exists()
        for name in ("tiny-check.json", "history.json", "best.pt", "last.pt")
    ):
        parser.error("Output already contains an experiment; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    tiny = tiny_learning_check(diagnostic_train, args.seed)
    (args.output / "tiny-check.json").write_text(
        json.dumps(tiny, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Tiny check completed in {tiny['elapsed_seconds']:.1f}s", flush=True)

    # Start the actual experiment from the original pretrained weights, not the
    # weights modified by the diagnostic above.
    set_seed(args.seed)
    model = build_training_model()
    optimizer = make_optimizer(model)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_detection,
    )
    config = {
        "experiment": args.experiment,
        "architecture": MODEL_NAME,
        "initial_weights": "COCO_V1",
        "classes": {"background": 0, "person": 1},
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "threads": args.threads,
        "augmentation": "horizontal_flip(p=0.5)" if flip_probability else "none",
        "augmentation_seed": args.seed + 1 if flip_probability else None,
        "frozen": "entire backbone including FPN",
        "train_image_ids": train.image_ids,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "preprocessing": {
            "image_mode": "RGB",
            "input_dtype": "float32",
            "input_range": [0, 1],
            "min_size": list(model.transform.min_size),
            "max_size": model.transform.max_size,
            "image_mean": model.transform.image_mean,
            "image_std": model.transform.image_std,
        },
        "validation_score_floor": 0.001,
    }
    history = []
    best = float("-inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        sums = {}
        for batch_index, (images, targets, _) in enumerate(loader, 1):
            components = model(images, targets)
            loss = sum(components.values())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}, batch {batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in components.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"Epoch {epoch}/{args.epochs}, batch {batch_index}/{len(loader)}: "
                    f"loss {float(loss.detach()):.4f}",
                    flush=True,
                )
        mean_losses = {key: value / len(loader) for key, value in sums.items()}
        result = evaluate_model(configure_evaluation(model), validation, progress=False)
        entry = {
            "epoch": epoch,
            "mean_losses": mean_losses,
            "validation": result,
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        history.append(entry)
        metadata = {"config": config, "epoch": epoch, "validation_metrics": result["metrics"]}
        save_checkpoint(args.output / "last.pt", model, metadata)
        if result["metrics"]["map"] > best:
            best = result["metrics"]["map"]
            save_checkpoint(args.output / "best.pt", model, metadata)
        print(
            f"Epoch {epoch}: validation mAP {result['metrics']['map']:.4f}, "
            f"AP50 {result['metrics']['map_50']:.4f}",
            flush=True,
        )
        (args.output / "history.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "tiny_check": tiny,
                    "epochs": history,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Saved best checkpoint and history to {args.output}")


if __name__ == "__main__":
    main()
