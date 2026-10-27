# Learning journal

This is a real development record, not a claim about time spent or independent
work. AI-written explanations are labeled; George adds his own words and evidence.

## Milestone 1: environment and a pretrained prediction

Date: 2026-10-27 (Asia/Tbilisi).

Status: AI implementation and automated verification complete; George's review pending.

### What changed (AI draft)

Added a reproducible Python 3.12 CPU environment, a small prediction CLI,
dependency smoke checks, focused tests, and project documentation. The first
model uses existing COCO weights. No training on Penn-Fudan has occurred.

### How it works (AI explanation)

Trace an ordinary image through the program:

1. Pillow opens the file and applies its camera-orientation metadata. Converting
   to RGB gives exactly three color channels even for grayscale or RGBA input.
2. An image normally looks like rows and columns of RGB values from 0 to 255.
   `prepare_image` turns it into a float tensor with shape `[3, height, width]`
   and values from 0 to 1. A tensor is a multidimensional array with a dtype.
3. `load_pretrained_detector` constructs Faster R-CNN with a MobileNet backbone
   and downloads already learned COCO parameters on its first use. Architecture
   describes operations; weights are numbers learned during earlier training.
4. The backbone extracts visual feature maps. A region-proposal network proposes
   candidate regions. Detection heads classify regions and refine their boxes.
   This is a high-level overview; the training milestone will trace the losses.
5. The model performs its own normalization and aspect-preserving resize. We
   scale pixels to [0, 1] but do not independently apply ImageNet normalization.
   TorchVision converts predicted boxes back to the input image coordinates.
6. `model.eval()` selects evaluation behavior. `torch.inference_mode()` separately
   disables gradient recording. Evaluation mode alone does not disable gradients.
7. Predictions contain boxes, labels, and scores. The pretrained model recognizes
   several COCO classes; person has label 1. We keep people above the requested
   threshold and save an annotated image plus machine-readable JSON.

A box `[10, 20, 70, 140]` means left=10, top=20, right=70, bottom=140 pixels;
its width is 60 and height is 120. Coordinates are not normalized to [0, 1].

A confidence of 0.9 is the detector's score, not proof that it is correct and
not necessarily a calibrated 90% probability. Raising the display threshold
can remove boxes; it cannot teach the model new features.

### Why this approach and alternatives (AI explanation)

| Choice | Reason | Alternative and drawback |
| --- | --- | --- |
| Python 3.12 | Installed, with compatible Windows wheels verified | Latest installed Python 3.14 adds compatibility uncertainty for this pinned stack |
| CPU torch/vision pair | Fits hardware and avoids CUDA setup | CUDA builds add unnecessary installation cost on this laptop |
| uv plus lockfile | Recreates direct and transitive dependencies | Unpinned installs may resolve different compiled packages later |
| MobileNet-backed Faster R-CNN | Proven detector, understandable stages, lower compute than heavier backbones | ResNet costs more compute; a custom detector creates unnecessary scope |
| CLI before a UI | Proves image-to-predictions behavior with little integration code | Building UI first makes model and UI failures harder to separate |
| Small inference module | Future CLI and Streamlit can use identical preprocessing/filtering | Duplicate inference code can drift between interfaces |

These are choices for a two-day learning project, not universal best practices.
We have not yet measured model accuracy or compared latency.

### Problems encountered and cause (AI record)

The CPU wheel index includes multiple local builds for the same base version.
Pinning only `torchvision==0.24.1` selected a build for Windows ARM64, while this
laptop is x86-64. Explicit `+cpu` pins resolved the intended compatible build.
Lesson: package names and version numbers alone do not guarantee native binary
compatibility; Python version, platform, architecture, and build flavor matter.

### Evidence from AI-run verification

- `uv sync`: installed Python 3.12.13 environment with torch 2.9.1+cpu and
  torchvision 0.24.1+cpu. Full versions are in the lockfile and environment report.
- `uv run --locked python -m pedestrian.check_environment --output runs/milestone1/environment.json`:
  compiled NMS returned indices `[0, 2]`; the synthetic exact-box example returned
  mAP/AP50 of 1.0. Those values check the metric installation, not model accuracy.
- `uv run --locked pytest -q`: 12 tests passed. The initial pre-implementation
  test run failed because the inference module did not yet exist. The eight
  core inference cases passed after implementation; four additional cases cover
  exact pixel scaling, EXIF orientation, CLI outputs, and corrupt input.
- `uv run --locked python -m pedestrian.predict --image data/person1.jpg --output runs/milestone1`:
  downloaded official weights and predicted one person, score approximately
  0.999205, at display threshold 0.5. Tensor shape was `[3, 640, 480]`, float32,
  range [0, 1]. AI visually inspected the saved box around the person.
- `uv run --locked ruff check .` and `uv run --locked ruff format --check .`:
  passed. `uv lock --check` also passed. The final test run after formatting
  again passed all 12 cases.

Committed evidence: [environment](evidence/milestone-01/environment.json),
[prediction](evidence/milestone-01/predictions.json), and
[annotated image](evidence/milestone-01/annotated.png).
These checks prove a working inference path, not completion of the assignment.

The sample image comes from TorchVision's v0.24.1 release:
https://raw.githubusercontent.com/pytorch/vision/v0.24.1/gallery/assets/person1.jpg

Downloaded sample SHA256:
`a5fff88c6b4a378657cd763c788053ca084f093d9a67d05c93172536881d4de5`.
This external sample is a smoke input, not our future held-out test set.

### George's verification (not completed by AI)

- [ ] Run the README setup/check/test commands yourself and record the outcomes.
- [ ] Run a prediction and inspect both PNG and JSON.
- [ ] Predict what changing `--threshold` to 0.90 will do, then try it in a
  separate output directory. Compare counts and scores. Raising a threshold
  does not necessarily change the count; reason from the actual saved scores.
- [ ] In your own words, explain `[3, H, W]`, [0, 1] pixels, pretrained weights,
  and the difference between evaluation mode and disabling gradients.
- [ ] Independently change the CLI's default threshold from 0.5 to 0.7. Run with
  no threshold argument and verify that JSON records 0.7. Explain why the weights
  did not change. Record the diff; keep README defaults consistent if you retain it.

Write here after doing the exercise:

- Commands I ran: **pending**
- What I expected and what happened: **pending**
- My explanation of the tensor and weights: **pending**
- My independent change and verification: **pending**
- Questions I still have: **pending**

### Next milestone

After George's exercise, implement data inspection and box extraction. First
follow a single image/mask pair and draw its ground-truth boxes; then create
the split manifests. Avoid starting training until the annotations are verified.
