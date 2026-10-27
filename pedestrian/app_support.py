"""Input and selected-model helpers shared by the local demonstration."""

import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from pedestrian.evaluate import configure_evaluation
from pedestrian.train import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTION = Path(
    os.environ.get("PEDESTRIAN_SELECTION_FILE", PROJECT_ROOT / "manifests" / "selected-model.json")
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path):
    """Hash equivalent LF/CRLF text consistently across Git checkouts."""
    normalized = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def matches_manifest_hash(path, expected):
    """Accept historical raw manifest hashes from either common line ending."""
    normalized = Path(path).read_bytes().replace(b"\r\n", b"\n")
    variants = (normalized, normalized.replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(value).hexdigest() == expected for value in variants)


def read_selection(config_path=DEFAULT_SELECTION, project_root=PROJECT_ROOT):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    checkpoint = Path(project_root) / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    minimum, maximum = config["min_size"], config["max_size"]
    if not isinstance(minimum, int) or not isinstance(maximum, int) or not 0 < minimum <= maximum:
        raise ValueError("Invalid selected resize dimensions")
    return config, checkpoint


def load_selected_model(config_path=DEFAULT_SELECTION, project_root=PROJECT_ROOT):
    config, checkpoint = read_selection(config_path, project_root)
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum does not match the manifest")
    model, metadata = load_checkpoint(checkpoint)
    if (
        metadata["epoch"] != config["selected_epoch"]
        or metadata["config"]["experiment"] != config["experiment"]
    ):
        raise ValueError("Selected checkpoint metadata does not match the manifest")
    model.transform.min_size = (config["min_size"],)
    model.transform.max_size = config["max_size"]
    return configure_evaluation(model), config


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Please upload a JPEG or PNG image")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not read that image. Please choose a valid JPEG or PNG.") from error


def draw_predictions(image: Image.Image, prediction) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width = max(2, min(image.size) // 200)
    for box, score in zip(prediction["boxes"].tolist(), prediction["scores"].tolist(), strict=True):
        draw.rectangle(box, outline="#4eff90", width=width)
        text = f"person {score:.2f}"
        xy = (box[0], max(0, box[1] - 15))
        bounds = draw.textbbox(xy, text)
        draw.rectangle(bounds, fill="#10291a")
        draw.text(xy, text, fill="#aaffc3")
    return result
