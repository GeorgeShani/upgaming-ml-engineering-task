import json

import numpy as np
import pytest
import torch
from PIL import Image

from pedestrian.data import PennFudanDataset
from pedestrian.evaluate import evaluate_model, match_counts


def test_dataset_reads_manifest_image_and_mask(tmp_path):
    (tmp_path / "PNGImages").mkdir()
    (tmp_path / "PedMasks").mkdir()
    Image.new("RGB", (5, 4), (255, 128, 0)).save(tmp_path / "PNGImages" / "PennPed00001.png")
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:5] = 4
    Image.fromarray(mask).save(tmp_path / "PedMasks" / "PennPed00001_mask.png")
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"splits": {"train": ["PennPed00001"]}}))
    dataset = PennFudanDataset(tmp_path, manifest, "train")
    image, target, image_id = dataset[0]
    assert image.shape == (3, 4, 5)
    assert image.dtype == torch.float32
    assert image[:, 0, 0].tolist() == pytest.approx([1, 128 / 255, 0])
    assert target["boxes"].tolist() == [[2, 1, 5, 3]]
    assert target["labels"].tolist() == [1]
    assert image_id == "PennPed00001"


def test_match_counts_does_not_count_duplicate_boxes_twice():
    target = {"boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30.0]])}
    predicted = {
        "boxes": torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [40, 40, 50, 50.0]]),
        "scores": torch.tensor([0.9, 0.8, 0.2]),
    }
    assert match_counts(predicted, target, threshold=0.5) == {"tp": 1, "fp": 1, "fn": 1}


class KnownDetector(torch.nn.Module):
    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[0, 0, 5, 5], [5, 5, 9, 9.0]]),
                "labels": torch.tensor([1, 18]),
                "scores": torch.tensor([0.9, 0.99]),
            }
            for _ in images
        ]


def test_evaluation_filters_non_person_and_measures_perfect_match():
    sample = (
        torch.zeros((3, 10, 10)),
        {"boxes": torch.tensor([[0, 0, 5, 5.0]]), "labels": torch.tensor([1])},
        "example",
    )
    result = evaluate_model(KnownDetector(), [sample], progress=False)
    assert result["metrics"]["map"] == pytest.approx(1.0)
    assert result["metrics"]["map_50"] == pytest.approx(1.0)
    assert result["counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert result["images"][0]["image_id"] == "example"
