import pytest
import torch

from pedestrian.optimize import choose_resolution, select_checkpoint, summarize_latencies


def test_choose_reduced_only_if_faster_and_within_accuracy_limit():
    assert choose_resolution(100, 80, 0.70, 0.68)["selected"] == "reduced"
    assert choose_resolution(100, 80, 0.70, 0.679)["selected"] == "standard"
    assert choose_resolution(100, 101, 0.70, 0.70)["selected"] == "standard"


def test_latency_summary_uses_median_and_nearest_rank_p95():
    result = summarize_latencies(list(range(1, 31)))
    assert result["median_ms"] == 15.5
    assert result["p95_ms"] == 29


def test_invalid_metrics_cannot_select_an_optimization():
    with pytest.raises(ValueError):
        choose_resolution(100, float("nan"), 0.7, 0.69)


def test_select_checkpoint_uses_validation_map_and_rejects_changed_training_settings(tmp_path):
    import json

    config = {
        "architecture": "model",
        "initial_weights": "weights",
        "classes": {"person": 1},
        "seed": 42,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "threads": 4,
        "frozen": "backbone",
        "train_image_ids": ["one"],
        "manifest_sha256": "hash",
        "torch": "v",
        "torchvision": "v",
        "preprocessing": {"min_size": [320]},
        "validation_score_floor": 0.001,
    }
    for name, score in (("A", 0.6), ("B", 0.7)):
        root = tmp_path / name
        root.mkdir()
        run_config = {
            **config,
            "experiment": name,
            "augmentation": "none" if name == "A" else "horizontal_flip(p=0.5)",
        }
        metrics = {"map": score, "map_50": score}
        history = {
            "config": run_config,
            "epochs": [{"epoch": 1, "validation": {"metrics": metrics}}],
        }
        (root / "history.json").write_text(json.dumps(history))
        torch.save(
            {"metadata": {"epoch": 1, "config": run_config, "validation_metrics": metrics}},
            root / "best.pt",
        )
    winner, _, _ = select_checkpoint(tmp_path / "A", tmp_path / "B")
    assert winner == "B"
    path = tmp_path / "B" / "history.json"
    changed = json.loads(path.read_text())
    changed["config"]["batch_size"] = 4
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="batch_size"):
        select_checkpoint(tmp_path / "A", tmp_path / "B")
