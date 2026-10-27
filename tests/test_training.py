import torch
import torchvision.models._api as torchvision_api
from torchvision.ops.misc import FrozenBatchNorm2d

from pedestrian.train import build_empty_model, load_checkpoint, save_checkpoint


def test_checkpoint_reloads_same_predictions_without_downloading(tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("Checkpoint reload must not download pretrained weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_download)
    monkeypatch.setattr(torchvision_api, "load_state_dict_from_url", no_download)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_empty_model().eval()
    image = torch.rand((3, 80, 96))
    with torch.inference_mode():
        before = model([image])[0]
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"epoch": 1, "seed": 42})
    loaded, metadata = load_checkpoint(path)
    with torch.inference_mode():
        after = loaded([image])[0]
    assert metadata == {"epoch": 1, "seed": 42}
    assert any(isinstance(layer, FrozenBatchNorm2d) for layer in loaded.modules())
    for key in ("boxes", "scores", "labels"):
        assert torch.equal(before[key], after[key])
