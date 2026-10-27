# Pedestrian detection: build and learn

A CPU-first computer vision project implemented one milestone at a time.
The full assignment includes data preparation, fine-tuning, evaluation, an
optimization attempt, and a local Streamlit demonstration.

**Current scope: validation experiments A/B and resolution optimization.** The
final held-out test and Streamlit remain upcoming. See
[the schedule](docs/implementation-plan.md) and the learning journal.

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

## Milestone 2: inspect one image and its annotation

Download the [Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
and extract it into `data/`, so that `data/PennFudanPed/PNGImages` and
`data/PennFudanPed/PedMasks` exist. The downloaded archive and extracted dataset
are ignored by Git; the dataset's readme restricts reposting its materials.

```powershell
uv run --locked python -m pedestrian.inspect_annotation --dataset-root data/PennFudanPed --image-id FudanPed00001 --output runs/milestone2-example
Invoke-Item .\runs\milestone2-example\ground-truth-boxes.png
Invoke-Item .\runs\milestone2-example\instance-mask.png
Get-Content .\runs\milestone2-example\annotation.json
```

The first image shows boxes derived from the annotations; the second uses
artificial colors to make the original mask's instance IDs visible. Generated
images stay local in `runs/`. No model is run by this command. See the learning
journal for a worked example and the next verification exercise.

To audit every image/mask pair and reproduce the grouped splits:

```powershell
uv run --locked python -m pedestrian.dataset_audit --dataset-root data/PennFudanPed --reviewed-groups manifests/reviewed_groups.json --output runs/audit-check
Get-Content .\runs\audit-check\splits.json
```

The committed `manifests/audit.json` records pairing and mask metadata; the
committed `manifests/splits.json` records the seed and image IDs. Related-image
groups come from hash screening and a limited visual review. These checks do
not prove all repeated people or scenes were found.

## Validation baseline and experiment A

The evaluator uses only the validation IDs. It computes COCO-style mAP across
scores and separately reports detection counts at confidence 0.5 and IoU 0.5.
The held-out test split is not used for model selection.

```powershell
uv run --locked python -m pedestrian.evaluate --dataset-root data/PennFudanPed --output runs/e0-recheck/validation.json
uv run --locked python -m pedestrian.train --dataset-root data/PennFudanPed --output runs/experiment-a-repro
uv run --locked python -m pedestrian.evaluate --dataset-root data/PennFudanPed --checkpoint runs/experiment-a-repro/best.pt --output runs/experiment-a-repro/reloaded-validation.json
uv run --locked python -m pedestrian.review_errors --dataset-root data/PennFudanPed --evaluation runs/experiment-a-repro/reloaded-validation.json --output runs/experiment-a-repro/review
uv run --locked python -m pedestrian.predict --image data/person1.jpg --checkpoint runs/experiment-a-repro/best.pt --output runs/experiment-a-repro/sample
```

Training starts with a five-step learning check, then restarts from pretrained
weights for three epochs. Choose a **new** output directory for each training
run; existing experiment output is protected against accidental overwrite.
The best checkpoint is selected by validation mAP and saved locally with its
configuration. Loading it does not download pretrained weights. Checkpoints,
dataset imagery, and side-by-side reviews are gitignored; run instructions
recreate them. Numerical summaries are committed under `docs/evidence/experiments/`.

## Experiment B and resolution comparison

Experiment B uses the same settings as A, plus horizontal flips on training
images. After reproducing A, run B and the comparison into new output paths:

```powershell
uv run --locked python -m pedestrian.train --dataset-root data/PennFudanPed --experiment B --output runs/experiment-b-repro
uv run --locked python -m pedestrian.optimize --dataset-root data/PennFudanPed --run-a runs/experiment-a-repro --run-b runs/experiment-b-repro --output runs/optimization-repro/validation.json
```

The recorded run selected experiment B at 320/640; see
`manifests/selected-model.json`. To use the current local checkpoint:

```powershell
uv run --locked python -m pedestrian.predict --image data/person1.jpg --checkpoint runs/experiment-b/best.pt --min-size 320 --max-size 640 --output runs/selected-example
```

The original checkpoint is local and ignored by Git. A retrained checkpoint may
have a different checksum; re-run model selection after retraining rather than
assuming the recorded checkpoint identity or benchmark will be identical.

`pedestrian/model.py` loads the detector; `inference.py` prepares images and
filters predictions; `predict.py` handles CLI input and saved outputs.
`check_environment.py` checks native detection and evaluation dependencies.
`tests/` protects image conversion and filtering without downloading weights.

## Limits and troubleshooting

The default prediction command uses the original COCO-trained detector. Pass
`--checkpoint` to use a locally trained model.
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
