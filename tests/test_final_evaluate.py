import hashlib
import json

import pytest
import torch

from pedestrian.final_evaluate import build_contract, verify_checkpoint_manifest, verify_contract


def test_frozen_contract_detects_changed_split_or_checkpoint(tmp_path, monkeypatch):
    import pedestrian.final_evaluate as final

    (tmp_path / "manifests").mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    selection = {
        "experiment": "B",
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_epoch": 3,
        "min_size": 320,
        "max_size": 640,
    }
    (tmp_path / "manifests" / "selected-model.json").write_text(json.dumps(selection))
    splits = tmp_path / "manifests" / "splits.json"
    split_text = json.dumps(
        {"splits": {"train": ["a"], "validation": ["b"], "test": ["c"]}}, indent=2
    )
    splits.write_bytes(split_text.replace("\n", "\r\n").encode())
    dataset = tmp_path / "data"
    (dataset / "PNGImages").mkdir(parents=True)
    (dataset / "PedMasks").mkdir()
    (dataset / "PNGImages" / "c.png").write_bytes(b"image")
    (dataset / "PedMasks" / "c_mask.png").write_bytes(b"mask")
    (tmp_path / "uv.lock").write_text("dependencies")
    (tmp_path / "source.py").write_text("source")
    monkeypatch.setattr(final, "FROZEN_SOURCES", ("source.py",))
    contract = build_contract(tmp_path, dataset)
    verify_contract(contract, tmp_path, dataset)
    splits.write_bytes(split_text.encode())
    verify_contract(contract, tmp_path, dataset)
    from pedestrian.app_support import matches_manifest_hash

    old_hash = hashlib.sha256(split_text.replace("\n", "\r\n").encode()).hexdigest()
    assert matches_manifest_hash(splits, old_hash)
    splits.write_text("changed")
    with pytest.raises(ValueError, match="split"):
        verify_contract(contract, tmp_path, dataset)
    splits.write_bytes(split_text.encode())
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        verify_contract(contract, tmp_path, dataset)
    checkpoint.write_bytes(b"selected checkpoint")
    (dataset / "PedMasks" / "c_mask.png").write_bytes(b"different mask")
    with pytest.raises(ValueError, match="held-out"):
        verify_contract(contract, tmp_path, dataset)


def test_checkpoint_manifest_preflight_reads_checkpoint_metadata(tmp_path):
    split = tmp_path / "splits.json"
    split.write_bytes(b'{"splits": {}}\n')
    recorded_hash = hashlib.sha256(split.read_bytes().replace(b"\n", b"\r\n")).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"metadata": {"config": {"manifest_sha256": recorded_hash}}}, checkpoint)
    verify_checkpoint_manifest(checkpoint, split)
    split.write_bytes(b'{"changed": true}\n')
    with pytest.raises(ValueError, match="different split"):
        verify_checkpoint_manifest(checkpoint, split)
