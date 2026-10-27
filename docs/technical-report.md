# Technical report (in progress)

## Current status

Milestone 1: environment and pretrained inference. No fine-tuning, Penn-Fudan
validation/test evaluation, or optimization measurements have been performed.
Single-image predictions are smoke checks, not evidence of model accuracy.

## Problem and intended application

Identify pedestrians in a photograph and show boxes, scores, and a count.
The final interface will be a local Streamlit application. Video tracking and
unique-person counting across images are outside the scope.

## Decisions and alternatives

- Penn-Fudan is planned because it is small enough for inspection and CPU-first
  experiments. Its small size and absence of person-free images limit evaluation.
- A pretrained MobileNet-backed Faster R-CNN avoids learning visual features
  from a tiny dataset. A heavier ResNet model costs more computation; a network
  trained from scratch needs more data and a larger experimental budget.
- Plain Python modules keep training and inference visible. A large training
  framework would hide more mechanisms from a beginner.
- `uv` and a committed lockfile reproduce the Python environment. An unpinned
  pip install is shorter initially but can resolve incompatible compiled builds.
- CPU wheels fit the available Intel integrated-graphics laptop. GPU-specific
  packages would add download/setup cost without usable CUDA hardware here.

## Results

Milestone 1 ran successfully on Windows AMD64, Python 3.12.13, CPU, using torch
2.9.1+cpu and torchvision 0.24.1+cpu. The compiled NMS operator and pycocotools
evaluation backend passed synthetic known-answer checks. Twelve focused tests
passed for preprocessing, filtering, and the CLI boundary.

The external TorchVision sample produced one person detection with score
0.999205 at threshold 0.5. The input tensor was `[3, 640, 480]`, float32, [0, 1].
See [saved evidence](evidence/milestone-01/README.md). This is a smoke prediction;
it is neither the E0 validation experiment nor an accuracy measurement.

E0, A, B, optimization, and held-out test results remain pending.

## Known limitations and next improvements

There is no project-trained model yet. No accuracy claim is supported.
Next: inspect masks and boxes, audit duplicates, save split manifests, and run
the validation baseline before training. See the journal for actual evidence
and the implementation plan for the acceptance criteria.

## Primary references

- [Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
- [Versioned TorchVision model](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)
- [Fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Official compatible CPU builds](https://pytorch.org/get-started/previous-versions/#v2-9-1)
- [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Detection metric](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)
