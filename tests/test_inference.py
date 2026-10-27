import pytest
import torch
from PIL import Image

from pedestrian.inference import predict, prepare_image


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_image_is_rgb_chw_float_in_zero_one_range(mode):
    image = Image.new(mode, (12, 8), color=255)
    tensor = prepare_image(image)
    assert tensor.shape == (3, 8, 12)
    assert tensor.dtype == torch.float32
    assert 0 <= tensor.min() <= tensor.max() <= 1


def test_pixel_values_preserve_rgb_channel_order_and_scaling():
    tensor = prepare_image(Image.new("RGB", (1, 1), color=(255, 128, 0)))
    assert tensor[:, 0, 0].tolist() == pytest.approx([1, 128 / 255, 0])


def test_camera_orientation_is_applied_before_tensor_conversion():
    image = Image.new("RGB", (12, 8))
    image.getexif()[274] = 6  # EXIF: rotate 90 degrees clockwise.
    assert prepare_image(image).shape == (3, 12, 8)


class FixedDetector(torch.nn.Module):
    """Avoid weight downloads in tests while exercising the inference boundary."""

    def forward(self, images):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert images[0].shape == (3, 8, 12)
        return [
            {
                "boxes": torch.tensor([[1, 1, 4, 7], [2, 1, 8, 7], [3, 1, 9, 7.0]]),
                "labels": torch.tensor([1, 18, 1]),
                "scores": torch.tensor([0.8, 0.99, 0.3]),
            }
        ]


def test_prediction_keeps_only_people_at_or_above_threshold():
    result = predict(Image.new("RGB", (12, 8)), FixedDetector(), threshold=0.8)
    assert result["labels"].tolist() == [1]
    assert result["boxes"].tolist() == [[1, 1, 4, 7]]
    assert result["scores"].tolist() == pytest.approx([0.8])


def test_no_qualifying_people_returns_correctly_shaped_empty_tensors():
    result = predict(Image.new("RGB", (12, 8)), FixedDetector(), threshold=1.0)
    assert result["boxes"].shape == (0, 4)
    assert result["labels"].shape == (0,)
    assert result["scores"].shape == (0,)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        predict(Image.new("RGB", (12, 8)), FixedDetector(), threshold=threshold)


def test_cli_saves_an_annotated_image_and_consistent_prediction_json(tmp_path, monkeypatch):
    import json

    from pedestrian import predict as cli

    image_path = tmp_path / "person.png"
    output = tmp_path / "results"
    Image.new("RGB", (12, 8)).save(image_path)
    monkeypatch.setattr(cli, "load_pretrained_detector", lambda: FixedDetector())
    cli.main(["--image", str(image_path), "--output", str(output)])
    report = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert report["count"] == len(report["predictions"]["boxes"]) == 1
    assert report["threshold"] == 0.5
    assert report["tensor_shape_chw"] == [3, 8, 12]
    with Image.open(output / "annotated.png") as annotated:
        assert annotated.size == (12, 8)


def test_unreadable_cli_input_is_rejected_before_model_download(tmp_path, capsys):
    from pedestrian.predict import main

    image = tmp_path / "corrupt.png"
    image.write_text("not an image", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--image", str(image)])
    assert error.value.code == 2
    assert "Cannot read image" in capsys.readouterr().err
