import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from pedestrian.app_support import decode_image, draw_predictions, load_selected_model


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_decode_image_converts_supported_modes_to_rgb(mode):
    image = Image.new(mode, (6, 4), color=128)
    data = io.BytesIO()
    image.save(data, format="PNG")
    result = decode_image(data.getvalue())
    assert result.mode == "RGB"
    assert result.size == (6, 4)


def test_corrupt_image_returns_helpful_error():
    with pytest.raises(ValueError, match="image"):
        decode_image(b"not an image")


def test_draw_predictions_keeps_source_image_unchanged():
    source = Image.new("RGB", (20, 20), "black")
    result = {"boxes": torch.tensor([[2, 2, 15, 15.0]]), "scores": torch.tensor([0.9])}
    rendered = draw_predictions(source, result)
    assert rendered.getpixel((2, 2)) != source.getpixel((2, 2))
    assert source.getpixel((2, 2)) == (0, 0, 0)


def test_selected_model_rejects_wrong_checksum_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"different weights")
    selection = {
        "checkpoint": "model.pt",
        "checkpoint_sha256": "0" * 64,
        "experiment": "B",
        "selected_epoch": 3,
        "min_size": 320,
        "max_size": 640,
    }
    config = tmp_path / "selected.json"
    config.write_text(json.dumps(selection))
    monkeypatch.setattr(
        "pedestrian.app_support.load_checkpoint", lambda _: pytest.fail("must not load")
    )
    with pytest.raises(ValueError, match="checksum"):
        load_selected_model(config, tmp_path)


def test_selected_model_checks_metadata_and_applies_frozen_resolution(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"known weights")
    selection = {
        "checkpoint": "model.pt",
        "checkpoint_sha256": hashlib.sha256(b"known weights").hexdigest(),
        "experiment": "B",
        "selected_epoch": 3,
        "min_size": 320,
        "max_size": 640,
    }
    config = tmp_path / "selected.json"
    config.write_text(json.dumps(selection))
    model = SimpleNamespace(
        transform=SimpleNamespace(min_size=(256,), max_size=512),
        roi_heads=SimpleNamespace(score_thresh=0.05, detections_per_img=100),
    )
    meta = {"epoch": 3, "config": {"experiment": "B"}}
    monkeypatch.setattr("pedestrian.app_support.load_checkpoint", lambda _: (model, meta))
    loaded, actual = load_selected_model(config, tmp_path)
    assert actual == selection
    assert loaded.transform.min_size == (320,)
    assert loaded.transform.max_size == 640
    assert loaded.roi_heads.score_thresh == 0.001
