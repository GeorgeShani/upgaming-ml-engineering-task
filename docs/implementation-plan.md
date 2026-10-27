# 48-hour implementation plan

Agreed budget: 14 focused hours in two calendar days, plus up to 2 hours of
contingency. Break for 10-15 minutes between sessions, take meal breaks, and
sleep a full night. Session durations are budgets, not fabricated work logs.

## Working agreement

AI drafts one small milestone; George runs it, inspects the output, explains
the mechanism, and makes an independent change before the next milestone.
Journal entries explicitly distinguish AI verification from George's work.
Each verified milestone gets a meaningful commit. All learning notes are public.

## Sessions

| Session | Hours | Deliverable | Explain before continuing |
| --- | ---: | --- | --- |
| 1 | 1 | Pinned CPU environment, pretrained prediction, dependency checks | CHW tensors, pixel scaling, pretrained weights |
| 2 | 1.5 | Dataset audit, box overlays, fixed split manifests | Mask instances, boxes, background, leakage |
| 3 | 1.5 | Pretrained E0 validation metrics and predictions | IoU, precision/recall, AP |
| 4 | 2 | Tiny learning check, training loop, experiment A | Losses, gradients, freezing, optimizer |
| 5 | 1 | Error inspection and checkpoint reload verification | Trace one image through the pipeline |
| 6 | 1.5 | Experiment B: add horizontal flips only | Joint image/box transforms, controlled comparison |
| 7 | 1.5 | Resolution optimization and CPU timing | Measured speed/quality trade-off |
| 8 | 1.5 | Streamlit upload, boxes, scores, count, threshold | Cached model, eval mode, empty results |
| 9 | 1.5 | Frozen decisions, test results, reproducible handover | Test-set isolation and limitations |
| 10 | 1 | Demo rehearsal and independent modification | Defend decisions and change code |

## Architecture and constraints

Image/mask -> dataset -> Faster R-CNN with MobileNetV3-Large 320 FPN ->
evaluation/checkpoint -> shared inference -> Streamlit.

Use Penn-Fudan for person detection. Derive one absolute `xyxy` box per mask
instance. Person is label 1; background is 0. Return RGB CHW float tensors in
[0, 1]. Apply geometric transforms jointly to images and boxes.

Split about 60/20/20 with seed 42, balanced by Penn/Fudan source; group confirmed
duplicates or near-duplicates before splitting and record actual counts.
Training updates weights; validation selects settings; test is held out until
all decisions are frozen. All dataset and output locations are configurable.

No video, tracking, segmentation output, authentication, Docker, cloud hosting,
custom architecture, or separate frontend is planned.

## Experiments

- E0: original COCO weights, retain only person detections.
- A: replace the final predictor with person/background outputs, freeze the
  whole backbone including FPN, train remaining detection components.
- B: same initialization, split, order seed, settings, and budget as A; add
  horizontal flip with probability 0.5 as the only planned change.

Defaults: batch size 2, 3 epochs, SGD learning rate 0.005, momentum 0.9,
weight decay 0.0005. Check finite losses, intended parameter updates, and
loss reduction on a fixed tiny subset before full runs. Time a short run.
If training exceeds the budget, use the same fixed, source-balanced 50-image
training subset for A and B; preserve validation/test and disclose the reduction.

Select the best A/B checkpoint using validation mAP at IoU 0.50:0.95; also report
AP50. Use TorchMetrics plus pycocotools rather than custom AP code. Keep the
demo threshold separate from AP evaluation. Report E0 even if it is stronger.
Demo the best trained checkpoint. Record configurations, seeds, dependency
versions, losses, timings, metrics, and representative mistakes.

Save weights with architecture, class mapping, preprocessing settings, and
training configuration. Loading the delivered checkpoint must not redownload
pretrained weights. Provide a download link and SHA256 plus retraining commands.

## Optimization and demonstration

Compare the chosen checkpoint at shorter/max edges 320/640 and 256/512.
Use the same CPU settings and measurement boundary: five warm-ups followed by
30 single-image predictions over ten fixed validation images. Record median
and p95 latency and what is included in timing.

Adopt 256/512 only if median latency improves and validation mAP decreases by
at most 0.02 absolute. Otherwise retain 320/640 and report the unsuccessful
optimization attempt. Resizing does not change checkpoint size.

Streamlit: JPEG/PNG upload, annotated output, confidence slider, person count,
empty results, and useful errors. Share inference with the CLI, cache the model,
use eval mode and disable gradients. Handle grayscale/RGBA and corrupt uploads.

## Completion checks

- Tests: pairing, valid boxes/types, split isolation, forced flip, known metric
  case, checkpoint equivalence, empty predictions, threshold filtering.
- Manual checks: grayscale/RGBA, corrupt upload, missing weights, no-person
  image, and several outside-dataset images (qualitative, not benchmark data).
- Fresh environment reproduces setup and demo; trained weights are accessible.
- Report includes data/model rationale, experiment and optimization tables,
  error analysis, difficulties, strengths, weaknesses, and future improvements.
- Honest AI-use record and journal accompany the source and evidence.
- George explains the full system and makes an unfamiliar small change unaided.

When time is tight, drop visual polish and extra experiments first. Preserve
training, evaluation, the optimization attempt, reproducibility, and learning.
