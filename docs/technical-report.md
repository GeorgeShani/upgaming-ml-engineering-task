# Technical report (in progress)

## Current status

The dataset audit and split, pretrained E0 baseline, fine-tuning experiments A/B,
and a resolution optimization attempt are complete. The final application and
held-out test remain pending.
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

The data audit found 170 paired images and masks with matching dimensions, and
423 nonzero mask instance IDs. The original dataset webpage's 345-pedestrian
headline differs from this mask count; the release readme mentions additional
annotations, but the precise cause has not been independently established.

Using seed 42 and grouping 13 visually related image sets yielded 102 training,
34 validation, and 34 test images. No decoded-pixel exact duplicates or pairs
within difference-hash distance 3 were found. This similarity screen and visual
review are not exhaustive, so related scenes or people may still span splits.

The external TorchVision sample produced one person detection with score
0.999205 at threshold 0.5. The input tensor was `[3, 640, 480]`, float32, [0, 1].
See [saved evidence](evidence/milestone-01/README.md). This is a smoke prediction;
it is neither the E0 validation experiment nor an accuracy measurement.

### Validation comparison

Both runs evaluated the same 34 validation images with TorchMetrics/pycocotools,
person-only predictions, model score floor 0.001, and up to 300 detections per
image before COCO's metric limits. Counts use confidence 0.5 and IoU 0.5.

| Run | mAP 0.50:0.95 | AP50 | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0: pretrained | 0.6912 | 0.9338 | 73 | 31 | 4 | 0.7019 | 0.9481 |
| A: best epoch 3 | 0.6549 | 0.9164 | 70 | 9 | 7 | 0.8861 | 0.9091 |
| B: best epoch 3 | 0.6752 | 0.9186 | 70 | 8 | 7 | 0.8974 | 0.9091 |

At this reporting threshold A produced fewer unmatched predictions but missed
more annotations; its mAP was lower. These are validation measurements on a
small dataset, not test results or proof of real-world superiority. Some E0
unmatched predictions in crowded `FudanPed00047` visibly correspond to people
without mask annotations, so apparent false positives can include annotation
gaps. In `FudanPed00063`, A missed several small distant annotated people at
the reporting threshold. Similar images may still cross splits despite grouping.
E0 also ranks several object classes before the 300-detection cap, whereas A
predicts only person; that post-processing difference may affect comparison.

Experiment A used all 102 training images, three epochs, batch size 2, a frozen
backbone, SGD, and no augmentation. Tiny-set loss fell from 1.2323 to 0.4906;
this checks that a small batch can be learned, not generalization. The best
checkpoint reproduced its saved validation metrics after offline reload. Full
numeric evidence is in `docs/evidence/experiments/`; dataset visualizations and
checkpoint files stay local. The exact retraining command is in the README.

Experiment B differed from A only by random horizontal training-image flips
with probability 0.5; the associated boxes were flipped jointly. B used the
same recorded split, seed, initialization procedure, optimizer, and training
budget. It outperformed A in this single validation run, while E0 remained higher.
The selected trained checkpoint is B, epoch 3.

### Resolution optimization

On the same B checkpoint and validation split, standard resize 320/640 had mAP
0.6752, median 315.3 ms, and p95 484.0 ms. Reduced resize 256/512 had mAP
0.6191, median 255.6 ms, and p95 421.2 ms. Timings cover model calls on ten
preloaded CPU tensors, 30 predictions after five warm-ups, with four threads on
the i5-1335U laptop. The reduced setting was about 18.9% faster at the median,
but its 0.0561 absolute mAP loss exceeded the 0.02 limit; retain 320/640.
The weights file size does not change. The benchmark is a single local run.

The held-out test and final application remain pending.

## Known limitations and next improvements

The project-trained checkpoint exists locally but is not yet published; it does
not include optimizer state for exact training resumption. The validation set is
small, annotations are incomplete in some scenes, and the
dataset lacks person-free images. Next: build the app, freeze decisions, and
evaluate the held-out test. See the journal for evidence
and the implementation plan for the acceptance criteria.

## Primary references

- [Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
- [Versioned TorchVision model](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)
- [Fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Official compatible CPU builds](https://pytorch.org/get-started/previous-versions/#v2-9-1)
- [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Detection metric](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)
