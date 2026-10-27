# Milestone 1 evidence

Recorded by AI-run verification on 2026-10-27. George's own checks remain pending
in the learning journal. These files are selected outputs copied from the
gitignored run directory; they are not manually invented results.

- `environment.json`: Python/package versions and native dependency checks.
  Its perfect mAP is for a synthetic exact-box dependency check only.
- `predictions.json`: actual pretrained output, image hash, and tensor details.
- `annotated.png`: visually inspected output with one person detection.

Reproduction:

```powershell
uv sync --locked
uv run --locked python -m pedestrian.check_environment --output runs/milestone1/environment.json
uv run --locked python -m pedestrian.predict --image data/person1.jpg --output runs/milestone1
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Download instructions for the input image are in the root README. Source:
[TorchVision v0.24.1 sample](https://raw.githubusercontent.com/pytorch/vision/v0.24.1/gallery/assets/person1.jpg).
This annotated output derives from that upstream sample; it is not an image
collected by George. Input SHA256:
`a5fff88c6b4a378657cd763c788053ca084f093d9a67d05c93172536881d4de5`.

The output demonstrates person detection on an external smoke input. It does
not establish dataset accuracy, generalization, training quality, or performance.
