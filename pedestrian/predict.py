"""Run a pretrained pedestrian prediction and save an image plus JSON."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps

from pedestrian.inference import predict, prepare_image
from pedestrian.model import MODEL_NAME, WEIGHTS, load_pretrained_detector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/prediction"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    try:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except OSError as error:
        parser.error(f"Cannot read image {args.image}: {error}")

    torch.set_num_threads(args.threads)
    print("Loading COCO weights (downloaded once, then cached)...", flush=True)
    model = load_pretrained_detector()
    result = predict(image, model, threshold=args.threshold)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, score in zip(result["boxes"].tolist(), result["scores"].tolist(), strict=True):
        draw.rectangle(box, outline="lime", width=3)
        draw.text((box[0], max(0, box[1] - 12)), f"person {score:.3f}", fill="lime")

    tensor = prepare_image(image)
    report = {
        "image": args.image.as_posix(),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_size_wh": list(image.size),
        "tensor_shape_chw": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_range": [float(tensor.min()), float(tensor.max())],
        "model": MODEL_NAME,
        "weights": WEIGHTS.name,
        "device": "cpu",
        "cpu_threads": args.threads,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "threshold": args.threshold,
        "count": len(result["scores"]),
        "predictions": {key: value.tolist() for key, value in result.items()},
        "note": "Pretrained smoke prediction; no fine-tuning or accuracy measurement.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output / "annotated.png")
    (args.output / "predictions.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"People detected: {report['count']} at threshold {args.threshold}")
    print(f"Tensor: {report['tensor_shape_chw']}, {report['tensor_dtype']}, [0, 1]")
    print(f"Saved {args.output / 'annotated.png'} and {args.output / 'predictions.json'}")


if __name__ == "__main__":
    main()
