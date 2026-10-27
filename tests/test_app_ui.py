import json
from pathlib import Path

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_example_flow_and_confidence_slider(tmp_path, monkeypatch):
    import pedestrian.app_support as support
    import pedestrian.inference as inference

    (tmp_path / "data").mkdir()
    Image.new("RGB", (20, 20), "white").save(tmp_path / "data" / "person1.jpg")
    config = tmp_path / "selected.json"
    config.write_text(json.dumps({"checkpoint": "model.pt"}))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(support, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(support, "DEFAULT_SELECTION", config)
    monkeypatch.setattr(support, "read_selection", lambda: ({}, checkpoint))
    monkeypatch.setattr(support, "load_selected_model", lambda: (object(), {}))

    def fixed_prediction(image, model, threshold):
        keep = 0.8 >= threshold
        return {
            "boxes": torch.tensor([[1, 1, 15, 18.0]]) if keep else torch.empty((0, 4)),
            "scores": torch.tensor([0.8]) if keep else torch.empty((0,)),
        }

    monkeypatch.setattr(inference, "predict", fixed_prediction)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30
    )
    app.run()
    assert not app.exception
    assert "Choose a JPEG or PNG image" in app.info[0].value
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.metric[0].value == "1"
    app.slider[0].set_value(1.0).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert "No people were detected" in app.info[0].value


def test_missing_checkpoint_shows_recovery_message(tmp_path, monkeypatch):
    import pedestrian.app_support as support

    config = tmp_path / "selected.json"
    config.write_text(json.dumps({"checkpoint": "missing.pt"}))
    monkeypatch.setattr(support, "DEFAULT_SELECTION", config)

    def missing():
        raise FileNotFoundError("checkpoint missing")

    monkeypatch.setattr(support, "read_selection", missing)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30
    )
    app.run()
    assert not app.exception
    assert "checkpoint missing" in app.error[0].value
    assert "README" in app.info[0].value
