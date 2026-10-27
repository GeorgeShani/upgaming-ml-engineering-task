from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pedestrian.dataset_audit import audit_dataset, group_images, split_groups


def write_pair(root: Path, image_id: str, value: int = 80):
    (root / "PNGImages").mkdir(exist_ok=True)
    (root / "PedMasks").mkdir(exist_ok=True)
    Image.new("RGB", (4, 3), (value, 50, 20)).save(root / "PNGImages" / f"{image_id}.png")
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 1:3] = 4
    Image.fromarray(mask).save(root / "PedMasks" / f"{image_id}_mask.png")


def test_audit_matches_images_and_masks_and_counts_instances(tmp_path):
    write_pair(tmp_path, "FudanPed00001")
    rows = audit_dataset(tmp_path)
    assert len(rows) == 1
    assert rows[0]["image_id"] == "FudanPed00001"
    assert rows[0]["source"] == "Fudan"
    assert rows[0]["size_wh"] == [4, 3]
    assert rows[0]["instances"] == 1


def test_audit_rejects_missing_mask_and_mismatched_dimensions(tmp_path):
    write_pair(tmp_path, "FudanPed00001")
    mask = tmp_path / "PedMasks" / "FudanPed00001_mask.png"
    mask.unlink()
    with pytest.raises(ValueError, match="missing mask"):
        audit_dataset(tmp_path)
    Image.new("L", (2, 2)).save(mask)
    with pytest.raises(ValueError, match="dimensions"):
        audit_dataset(tmp_path)


def test_grouped_split_is_seeded_complete_and_keeps_each_group_together():
    rows = [
        {"image_id": f"{source}Ped{i:05d}", "source": source}
        for source in ("Fudan", "Penn")
        for i in range(10)
    ]
    groups = [[row["image_id"]] for row in rows]
    groups[0].append(groups.pop(1)[0])
    split = split_groups(rows, groups, seed=42)
    assert split == split_groups(rows, groups, seed=42)
    members = [image_id for ids in split.values() for image_id in ids]
    assert len(members) == len(set(members)) == len(rows)
    assert set(members) == {row["image_id"] for row in rows}
    assert any(set(groups[0]).issubset(ids) for ids in split.values())
    for source in ("Fudan", "Penn"):
        assert [sum(image_id.startswith(source) for image_id in split[name]) for name in split] == [
            6,
            2,
            2,
        ]


def test_exact_duplicates_and_reviewed_groups_are_kept_together():
    rows = [
        {"image_id": "FudanPed00001", "pixel_sha256": "same", "dhash": "0000000000000000"},
        {"image_id": "FudanPed00002", "pixel_sha256": "same", "dhash": "0000000000000000"},
        {"image_id": "FudanPed00003", "pixel_sha256": "other", "dhash": "ffffffffffffffff"},
    ]
    groups, candidates = group_images(rows, [["FudanPed00002", "FudanPed00003"]])
    assert groups == [["FudanPed00001", "FudanPed00002", "FudanPed00003"]]
    assert candidates == [{"images": ["FudanPed00001", "FudanPed00002"], "distance": 0}]
