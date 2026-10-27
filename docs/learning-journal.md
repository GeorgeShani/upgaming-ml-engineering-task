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

Follow a single image/mask pair and draw its ground-truth boxes; then create
the split manifests. Avoid starting training until the annotations are verified.

## Milestone 2, first slice: one image and mask

Date: 2026-10-27 (Asia/Tbilisi). AI implementation and verification complete;
George's independent inspection and explanation are pending. The dataset split
and full dataset audit are still to come.

### What changed and why

`pedestrian/annotations.py` extracts person boxes from an instance mask.
`pedestrian/inspect_annotation.py` displays one real image, a colorized mask,
and the boxes derived from that mask. This small slice connects model predictions
to the ground-truth annotations against which predictions will later be measured.

The dataset contains 170 image files and 170 mask files. This is a count, not
yet a complete pairing or duplicate audit. The archive came from the original
Penn-Fudan source and has SHA256
`9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`.
Its readme says copyright is retained and reposting generally requires rights
holder permission. Dataset files and derived visualizations remain local in the
gitignored `data/` and `runs/` directories.

### How mask values become boxes

A mask has one integer at each pixel. Zero means background; each nonzero value
identifies a separate annotated pedestrian in that image. Instance IDs are not
the model's class labels, and they do not identify the same individual across
different images. Every person has the same class label, 1.

For example, consider a mask 5 pixels wide and 3 pixels high:

```text
  x → 0  1  2  3  4
y 0   0  0  0  0  0
  1   0  2  2  0  7
  2   0  2  0  0  7
```

ID 2 occupies columns 1-2 and rows 1-2, so its box is `[1, 1, 3, 3]`.
ID 7 occupies column 4 and rows 1-2, so its box is `[4, 1, 5, 3]`.
Boxes are `[left, top, right, bottom]` with exclusive right and bottom edges.
This makes width `right - left` and makes a one-pixel-wide box have width 1.

The implementation builds one True/False mask per instance ID and calls
TorchVision `masks_to_boxes`. That operation returns the largest occupied pixel
index, so we add 1 to the right and bottom coordinates. An alternative is to
compute min/max pixel indices directly with NumPy; the library operation keeps
the tensor implementation compact but still requires understanding this edge.

The colorized output is for inspection only: its red/blue colors are assigned
for visibility and do not change the original mask or class labels.

### AI-run evidence

On `FudanPed00001`, the image is 559 by 536 pixels, the mask is single-channel,
and nonzero IDs are `[1, 2]`. Derived boxes are `[159, 181, 302, 431]` and
`[419, 170, 535, 486]`; both class labels are 1. I inspected the local box and
color-mask images. They surround the two annotated pedestrians. The numeric
output is recorded in `docs/evidence/milestone-02/annotation.json`.

Three focused annotation tests and the existing 12 tests passed. Ruff lint and
format checks passed. The tests cover nonconsecutive instance IDs, background,
one-pixel regions, box coordinates, and target tensor types.

### George's verification (pending)

- [ ] Run the inspection command from the README and open both output images.
- [ ] Explain why instance IDs 1 and 2 both receive person class label 1.
- [ ] In the toy mask above, calculate ID 7's width and height from its box.
- [ ] Choose another image ID from the dataset, inspect its outputs, and record
  what you saw without publishing the dataset's images.

Commands I ran: **pending**. My explanation: **pending**. My observations and
questions: **pending**.

Next: audit pairing and possible duplicates, then make fixed train/validation/
test manifests before evaluating or training the model.
