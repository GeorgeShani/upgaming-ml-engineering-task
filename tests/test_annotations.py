import numpy as np
import torch
from PIL import Image

from pedestrian.annotations import extract_instances


def test_mask_values_become_separate_boxes_with_background_excluded():
    # Instance IDs need not be consecutive, and their shapes need not be rectangles.
    mask = Image.fromarray(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 0, 7, 0],
                [0, 2, 0, 0, 7, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    )

    result = extract_instances(mask)

    assert result["instance_ids"].tolist() == [2, 7]
    assert result["boxes"].tolist() == [[1, 1, 3, 3], [4, 1, 5, 3]]
    assert result["boxes"].dtype == torch.float32
    assert result["labels"].tolist() == [1, 1]
    assert result["labels"].dtype == torch.int64


def test_background_only_mask_returns_empty_boxes_and_labels():
    result = extract_instances(Image.new("L", (4, 3)))
    assert result["instance_ids"].shape == (0,)
    assert result["boxes"].shape == (0, 4)
    assert result["labels"].shape == (0,)


def test_single_pixel_instance_still_has_positive_area():
    values = np.zeros((3, 4), dtype=np.uint8)
    values[1, 2] = 5
    result = extract_instances(Image.fromarray(values))
    assert result["boxes"].tolist() == [[2, 1, 3, 2]]
