# Pedestrian detection: build and learn

A CPU-first computer vision project implemented one milestone at a time.
The full assignment includes data preparation, fine-tuning, evaluation, an
optimization attempt, and a local Streamlit demonstration.

**Current scope: milestone 1, pretrained inference.** No project-specific
training or dataset evaluation has been performed. Your independent review
comes before milestone 2. See [the schedule](docs/implementation-plan.md).

## Set up on Windows

Run commands from this repository in PowerShell. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.

```powershell
uv sync --locked
uv run python -m pedestrian.check_environment
uv run pytest
uv run ruff check .
```

`uv` selects Python 3.12 and creates `.venv`; shell activation is unnecessary.
The committed lockfile pins transitive dependencies. PyTorch and TorchVision
come from the official CPU wheel index; other packages come from PyPI.
This environment is verified on Windows x86-64. The first setup needs internet.

## Run the first prediction

Download one sample from the pinned TorchVision release, or use your own image:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/pytorch/vision/v0.24.1/gallery/assets/person1.jpg' -OutFile 'data/person1.jpg'
uv run python -m pedestrian.predict --image data/person1.jpg --output runs/milestone1
```

The first prediction downloads the pretrained COCO weights to PyTorch's cache.
Later predictions reuse that cache. You get `annotated.png` and
`predictions.json` under the requested output directory. The JSON records image
identity, tensor details, model identity, dependency versions, and predictions.
Boxes are `[left, top, right, bottom]` in pixels on the EXIF-oriented image.

```powershell
uv run python -m pedestrian.predict --image data/person1.jpg --output runs/threshold-090 --threshold 0.90
```

`--threshold` defaults to `0.5`; `--threads` defaults to `4` CPU threads.
Threshold changes only filter detections. They do not train the network.
Outputs are overwritten if you reuse an output directory; use a new directory
for each comparison.

## What to read

- [Learning journal and your first exercise](docs/learning-journal.md)
- [Two-day implementation plan](docs/implementation-plan.md)
- [Technical report, currently in progress](docs/technical-report.md)
- [AI assistance and verification record](docs/ai-usage.md)
- [Actual milestone 1 output and verification evidence](docs/evidence/milestone-01/README.md)

`pedestrian/model.py` loads the detector; `inference.py` prepares images and
filters predictions; `predict.py` handles CLI input and saved outputs.
`check_environment.py` checks native detection and evaluation dependencies.
`tests/` protects image conversion and filtering without downloading weights.

## Limits and troubleshooting

This is an existing COCO-trained detector, not yet our trained solution.
The confidence score is not a calibrated probability. One sample prediction
does not establish accuracy, generalization, or production readiness.

- If setup fails, check that the lockfile is present and run `uv sync --locked`.
  Do not independently upgrade torch or torchvision: their compiled operators
  must be compatible.
- If Windows reports a missing DLL when importing torch, check the official
  [PyTorch Windows FAQ](https://docs.pytorch.org/docs/stable/notes/windows.html).
- If weights cannot download, restore network access and rerun the prediction.
- Dataset downloads, model weights, caches, and generated runs are gitignored.
  Learning notes and selected evidence under `docs/` are committed.
