"""Freeze the selected configuration, then evaluate it on the held-out split."""

import argparse
import hashlib
import json
from pathlib import Path

from pedestrian.app_support import (
    PROJECT_ROOT,
    load_selected_model,
    matches_manifest_hash,
    read_selection,
    sha256_file,
    sha256_text_file,
)
from pedestrian.data import PennFudanDataset
from pedestrian.evaluate import evaluate_model

FROZEN_SOURCES = (
    "app.py",
    "pedestrian/annotations.py",
    "pedestrian/app_support.py",
    "pedestrian/data.py",
    "pedestrian/evaluate.py",
    "pedestrian/final_evaluate.py",
    "pedestrian/inference.py",
    "pedestrian/model.py",
    "pedestrian/train.py",
)


def dataset_fingerprint(dataset_root, image_ids):
    """Hash held-out file bytes without running a model or inspecting results."""
    digest = hashlib.sha256()
    for image_id in sorted(image_ids):
        for relative in (f"PNGImages/{image_id}.png", f"PedMasks/{image_id}_mask.png"):
            path = Path(dataset_root) / relative
            digest.update(relative.encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_contract(root, dataset_root):
    root = Path(root)
    selection_path = root / "manifests" / "selected-model.json"
    splits_path = root / "manifests" / "splits.json"
    selection, checkpoint = read_selection(selection_path, root)
    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum does not match selection")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    sets = [set(splits[name]) for name in ("train", "validation", "test")]
    if any(not group for group in sets) or any(
        sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise ValueError("Train, validation, and test splits must be non-empty and disjoint")
    if any(
        len(sets[i]) != len(splits[name]) for i, name in enumerate(("train", "validation", "test"))
    ):
        raise ValueError("Split manifest contains duplicate image IDs")
    return {
        "version": 1,
        "selection_sha256": sha256_text_file(selection_path),
        "split_manifest_sha256": sha256_text_file(splits_path),
        "dependency_lock_sha256": sha256_text_file(root / "uv.lock"),
        "source_sha256": {path: sha256_text_file(root / path) for path in FROZEN_SOURCES},
        "test_dataset_sha256": dataset_fingerprint(dataset_root, splits["test"]),
        "test_image_count": len(splits["test"]),
        "model": {
            key: selection[key]
            for key in ("experiment", "checkpoint_sha256", "selected_epoch", "min_size", "max_size")
        },
        "evaluation": {
            "split": "test",
            "score_floor": 0.001,
            "detections_per_image": 300,
            "confidence_threshold_for_counts": 0.5,
            "iou_threshold_for_counts": 0.5,
            "map_iou_range": "0.50:0.95",
            "threads": 4,
        },
    }


def verify_contract(contract, root, dataset_root):
    root = Path(root)
    files = {
        "selection": (root / "manifests" / "selected-model.json", "selection_sha256"),
        "split manifest": (root / "manifests" / "splits.json", "split_manifest_sha256"),
        "dependency lock": (root / "uv.lock", "dependency_lock_sha256"),
    }
    for name, (path, key) in files.items():
        if sha256_text_file(path) != contract[key]:
            raise ValueError(f"Frozen {name} changed")
    for path, expected in contract["source_sha256"].items():
        if sha256_text_file(root / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    actual = build_contract(root, dataset_root)
    if actual != contract:
        raise ValueError("Frozen configuration or held-out dataset changed")
    return actual


def write_new_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Record choices without evaluating the test set")
    freeze.add_argument("--dataset-root", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="Evaluate the frozen model on the held-out split")
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--freeze", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.command == "freeze":
        write_new_json(args.output, build_contract(PROJECT_ROOT, args.dataset_root))
        print(f"Frozen configuration saved to {args.output}; no test evaluation was performed.")
        return
    contract = json.loads(args.freeze.read_text(encoding="utf-8"))
    verify_contract(contract, PROJECT_ROOT, args.dataset_root)
    torch.set_num_threads(contract["evaluation"]["threads"])
    model, metadata = load_selected_model(
        PROJECT_ROOT / "manifests" / "selected-model.json", PROJECT_ROOT
    )
    if not matches_manifest_hash(
        PROJECT_ROOT / "manifests" / "splits.json", metadata["config"]["manifest_sha256"]
    ):
        raise ValueError("Checkpoint was trained with a different split manifest")
    if (
        model.roi_heads.score_thresh != contract["evaluation"]["score_floor"]
        or model.roi_heads.detections_per_img != contract["evaluation"]["detections_per_image"]
    ):
        raise ValueError("Model postprocessing differs from the frozen evaluation settings")
    dataset = PennFudanDataset(
        args.dataset_root, PROJECT_ROOT / "manifests" / "splits.json", "test"
    )
    result = evaluate_model(
        model, dataset, threshold=contract["evaluation"]["confidence_threshold_for_counts"]
    )
    result.update(
        {
            "split": "test",
            "freeze_sha256": sha256_text_file(args.freeze),
            "checkpoint_sha256": contract["model"]["checkpoint_sha256"],
            "evaluation_settings": contract["evaluation"],
        }
    )
    write_new_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("metrics", "counts", "precision_at_threshold", "recall_at_threshold")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
