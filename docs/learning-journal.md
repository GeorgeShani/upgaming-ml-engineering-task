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

## Milestone 2, second slice: audit and split

Date: 2026-10-27 (Asia/Tbilisi). AI implementation and verification complete;
George's own walkthrough and verification are pending.

### What changed

`pedestrian/dataset_audit.py` matches image IDs to mask IDs, loads every pair,
checks dimensions and mask channels, records per-image instance IDs, screens
image similarity, groups related images, and assigns groups to splits. The
results are in `manifests/audit.json` and `manifests/splits.json`; the manually
reviewed group list is in `manifests/reviewed_groups.json`.

### How the audit works

For example, `PNGImages/FudanPed00001.png` pairs with
`PedMasks/FudanPed00001_mask.png`. The common key is `FudanPed00001`. The
audit checks for missing images or masks, unexpected mask names, unreadable
files, different dimensions, non-single-channel masks, and invalid negative IDs.
It records which nonzero values occur; the number of values is the number of
annotated instances, not the number of classes.

The actual audit found 170 pairs, all with matching image/mask dimensions and
single-channel masks: 74 Fudan images, 96 Penn images, and 423 nonzero mask
instances. The dataset website's historical headline says 345 pedestrians; the
423 is our count of IDs in the downloaded masks. The supplied dataset readme
mentions additional annotations in the release, but we have not independently
established that this completely explains the difference.

### Similarity, groups, and limitations

Images with identical dimensions and identical decoded RGB pixels have the same
SHA256 digest; zero exact duplicate groups were found. A 64-bit difference hash
compares adjacent values in a resized grayscale image. The audit groups image
pairs with a difference-hash Hamming distance at most 3 as a conservative
screening rule. Zero pairs met that particular cutoff.

That zero is not proof of independence: during visual contact-sheet review,
`FudanPed00006` and `FudanPed00008` showed similar views of a scene even though
their difference-hash distance is 14. We recorded 13 visually related groups,
including this pair, in `reviewed_groups.json`. Groups can represent similar
scenes, not necessarily the same person or identical photographs. Any overlapping
groups are combined before splitting. More related people/scenes may remain
undetected; report the resulting leakage risk rather than claiming it is solved.

### Why three splits and why group them?

- **Train:** used to adjust model weights.
- **Validation:** used to compare experiments and select settings.
- **Test:** held back until decisions are frozen for a final assessment.

If related photos appear in both train and evaluation, the model can benefit
from seeing similar content during training, inflating the impression of
generalization. Keeping reviewed and automatically flagged groups together
reduces this risk. It is not a complete guarantee.

We shuffle groups using seed 42, place larger groups first, and choose a split
that keeps source-specific counts near 60/20/20. Keeping entire groups means
the exact target sizes cannot always be guaranteed. This dataset produced:

| Split | Fudan | Penn | Total |
| --- | ---: | ---: | ---: |
| Train | 44 | 58 | 102 |
| Validation | 15 | 19 | 34 |
| Test | 15 | 19 | 34 |

All 170 IDs occur exactly once, and all 13 related groups stay within one split.
For example, the two Fudan images above both ended up in training. Splitting by
image without groups is simpler but ignores related frames; splitting by entire
location would test a harder domain shift but gives only two location groups.

### AI verification and George's exercise

AI ran the audit twice: both audit files matched each other byte-for-byte, as
did both split manifests. The full suite passed 19 tests; Ruff lint and format
checks passed. Tests cover missing pairs, mismatched dimensions, instance counts,
repeatability, source balance, disjoint membership, and linked-group integrity.

George: run the README audit command into `runs/audit-check`; compare the counts
and find `FudanPed00006` and `FudanPed00008` in the split file. Explain in your
own words why seed 42 alone does not prevent leakage, and why a visual group
review cannot prove that no related images remain across splits. Record actual
commands and observations here; independent verification remains **pending**.

Next: use the validation split to evaluate the pretrained E0 detector. The test
split remains untouched for model selection and optimization decisions.

## Steps 2–4: validation baseline, experiment A, and error review

Date: 2026-10-27 (Asia/Tbilisi). AI implemented and ran these steps. George's
independent walkthrough and code verification are pending. The held-out test
split was not used. Experiment B and optimization remain upcoming.

### Step 2: the pretrained baseline

The baseline answers: how does the existing COCO-trained detector perform on
our validation images before training on Penn-Fudan? `pedestrian/data.py` loads
each listed image and converts its instance mask to boxes and person labels.
`pedestrian/evaluate.py` runs the detector on all 34 validation images and keeps
person predictions for evaluation. The dataset loader keeps images as separate
tensors because their sizes differ.

IoU (intersection over union) compares two boxes: intersection area divided by
union area. Identical boxes have IoU 1; disjoint boxes have IoU 0. For the
human-readable counts, a prediction with score at least 0.5 is matched to an
unmatched ground-truth box at IoU at least 0.5. Detections are considered in
descending score order. Each annotation can match at most one prediction.

Our counts give precision = TP / (TP + FP) and recall = TP / (TP + FN).
Unmatched predictions are called FP by this annotation-based calculation;
unmatched annotations are FN. Precision and recall depend on the chosen score
and IoU thresholds.

Average precision instead summarizes the precision/recall curve across scores.
AP50 uses IoU 0.5. COCO-style mAP 0.50:0.95 averages AP across ten IoU thresholds,
from 0.50 to 0.95. The evaluator uses TorchMetrics/pycocotools rather than a
home-grown AP implementation. It sets the detector score floor to 0.001 and
keeps up to 300 detections per image before COCO's own limits; the 0.5 threshold
is for the separate counts, not an extra cutoff for the mAP calculation.

Validation results:

| Model | mAP 0.50:0.95 | AP50 | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 pretrained | 0.6912 | 0.9338 | 73 | 31 | 4 | 0.7019 | 0.9481 |
| A, best epoch 3 | 0.6549 | 0.9164 | 70 | 9 | 7 | 0.8861 | 0.9091 |

These are validation measurements on 77 annotated people, not test results.
The two models are compared under the same validation settings. Their confidence
scores need not be calibrated identically, so the 0.5 operating-point comparison
does not by itself establish which is better at every threshold. E0 ranks its
many object classes together before the 300-detection cap; A predicts only
person. That difference in post-processing can also affect the comparison.

### Step 3: the training loop and experiment A

Why fine-tune? The pretrained model has already learned useful visual patterns.
We adapt it to the person annotations instead of learning every feature from
scratch. We replace the final prediction head with outputs for background and
person, freeze the entire MobileNet/FPN backbone, and train the region proposal
and detection components. A bigger trainable model might adapt more, but costs
more compute and can overfit a small dataset. Freezing is a deliberate trade-off.

For one batch of two images:

1. The dataset yields image tensors plus ground-truth boxes and labels.
2. In training mode, `model(images, targets)` produces several losses: proposal
   objectness and box losses, plus final classification and box losses.
3. Their sum is a single number measuring how poorly the batch fits the training
   objectives. `loss.backward()` computes gradients for trainable parameters.
4. SGD uses those gradients, learning rate 0.005, momentum 0.9, and weight decay
   0.0005 to update the trainable parameters. It does not update frozen backbone
   parameters. `zero_grad` clears old gradients before the next backward pass.
5. After an epoch, switch to evaluation mode and compute validation metrics.
   Save the checkpoint with the highest validation mAP. The next epoch returns
   the model to training mode.

First, the five-step diagnostic repeated two training images. Its loss went
`1.2323 → 0.8825 → 0.5958 → 0.5022 → 0.4906`. It verified finite head gradients,
no backbone parameter gradients, a head update, and an unchanged sampled frozen
backbone parameter. This is a learning check on the same tiny batch, not a
generalization result. We then rebuilt the model from the original pretrained
weights before the real run; the diagnostic updates were not included.

Experiment A used the full 102-image training split, seed 42, batch size 2,
three epochs, CPU with four threads, and no augmentation. Mean summed losses
were approximately 0.4287, 0.3644, and 0.4143; stochastic batch sampling means
loss need not fall every epoch. Validation mAP was 0.6288, 0.6265, and 0.6549,
so epoch 3 was selected. The measured epoch loop took about 135 seconds on this
machine, including validation; this is not a controlled performance benchmark.

Experiment A's mAP was lower than E0's, although its 0.5-threshold precision
was higher. Replacing the pretrained classifier, the small dataset, short run,
and incomplete annotations are plausible factors, not proven causes. Longer
training, unfreezing, or reusing parts of the original head are alternatives
with their own compute and overfitting trade-offs. Do not claim fine-tuning
improved overall mAP in this experiment.

### Step 4: review and checkpoint verification

`pedestrian/review_errors.py` renders ground-truth boxes and score-filtered
predictions side by side in the local, gitignored `runs/` folder. In baseline
`FudanPed00047`, several unmatched predictions visibly correspond to people
without matching mask annotations. In experiment A `FudanPed00063`, several
small distant annotated people are missed at the 0.5 score threshold. These are
specific observations, not a comprehensive error study or proof of causation.

The best checkpoint is `runs/experiment-a/best.pt` (76,005,665 bytes). Its SHA256
is `683225a9fca17f4aac30b00c79a0d66a1042be975e78c188472f1e9c13076fa2`.
It stores learned tensors, architecture/configuration, class mapping, input
preprocessing, seed, training-image IDs, manifest checksum, and dependency
versions. The file is local and gitignored; the README gives exact retraining
commands. A downloadable handover artifact is still a later task. The
checkpoint does not contain optimizer state, so it supports inference and
retraining from the beginning, but not exact resumption of the interrupted run.

Preprocessing and TorchVision metadata were added after the training run; the
SHA256 of the model-state tensors was checked before and after and did not
change. Loading the checkpoint constructs the architecture without downloading
weights. Its re-evaluation reproduced the saved epoch-3 metrics and counts.
An automated test also compares predictions before and after a checkpoint reload.
The final suite passed 24 tests; Ruff lint and formatting checks passed.

### George's verification (pending)

- [ ] Explain why a validation baseline must be measured before fine-tuning.
- [ ] For identical boxes IoU is 1. What is the IoU if a predicted 10×5 box lies
  entirely inside a 10×10 ground-truth box?
- [ ] Calculate A's precision from 70 TP and 9 FP, then explain why AP50 is a
  different quantity from precision at a single confidence threshold.
- [ ] Trace one batch through forward pass → loss → backward → optimizer step.
- [ ] Run the README commands in new output directories, inspect a local error
  image, and record your actual observations and an independent code change.

Commands I ran: **pending**. My explanation: **pending**. My observations and
questions: **pending**. AI-run tests and visual review do not fill these fields.

Next: experiment B changes only the planned horizontal-flip augmentation;
preserve the same split, model initialization, and training budget for a useful
comparison. Optimization and the held-out test are separate later steps.

## Slices 5–6: augmentation and resolution optimization

### Why a horizontal flip changes both image and box

Augmentation creates a modified training example to expose the model to more
variation. For experiment B, each training image is flipped horizontally with
probability 0.5. Validation images are never augmented.

Suppose an image is 10 pixels wide and an annotated box is `[2, 0, 5, 2]`.
Using the project's half-open box coordinates, the flipped box is `[5, 0, 8, 2]`:

```text
new_left  = image_width - old_right = 10 - 5 = 5
new_right = image_width - old_left  = 10 - 2 = 8
```

The top and bottom coordinates and person label remain the same. Flipping only
the pixels would leave the annotation on the wrong side of the image. A targeted
test checks both pixels and boxes and verifies that the input is not mutated.

A separate seeded random generator chooses flips, so those choices do not consume
the global generator used by model training. A and B use the same model seed,
training IDs, shuffle seed, architecture, optimizer, batch size, and epoch budget.
The five-step diagnostic uses unaugmented images in both cases, then training
restarts from the pretrained weights. Separate random generators make the setup
easier to compare; different image content can still change training behavior.

### Why test a smaller inference resolution?

The selected trained checkpoint is evaluated at the standard 320/640 resize and
at 256/512, with the same validation images, model weights, score floor, and CPU
settings. A smaller image may need less computation but can make small pedestrians
harder to detect. This is a hypothesis to measure, not a guaranteed speedup.

Timing uses five warm-ups followed by 30 predictions across ten fixed validation
images, five from each dataset source. It measures the model call on already
loaded CPU tensors, including internal resize and post-processing. File loading
and visualization are excluded. The median is the middle latency; p95 is the
nearest-rank 95th percentile. The weights file is unchanged by resizing.

We use the reduced setting only if its median is faster and its validation mAP
decreases by no more than 0.02 absolute. A single measurement run on one laptop
can vary with operating-system load and does not predict deployment performance
on other hardware. The test split remains held out.

### Experiment B: actual result

AI ran B on the full 102-image training split for three epochs, batch size 2,
seed 42, CPU, four threads. The unaugmented tiny-check losses matched A's
recorded diagnostic exactly; this is a useful check of the common setup, though
we did not save A's actual initial model-state hash for a bytewise comparison.
The optimizer checks the recorded A/B configurations and checkpoint metadata
before selecting a model.

| Validation run | Best epoch | mAP 0.50:0.95 | AP50 | Precision at 0.5 | Recall at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 pretrained | — | 0.6912 | 0.9338 | 0.7019 | 0.9481 |
| A, no augmentation | 3 | 0.6549 | 0.9164 | 0.8861 | 0.9091 |
| B, random horizontal flip | 3 | 0.6752 | 0.9186 | 0.8974 | 0.9091 |

B's epoch mAP values were 0.6120, 0.6283, and 0.6752. Mean summed training
losses were approximately 0.4425, 0.3788, and 0.4531. The epoch loop took about
387 seconds on this laptop. A's earlier run took about 135 seconds, but those
were separate runs under uncontrolled machine conditions; do not treat their
elapsed times as a controlled training-speed comparison.

B was selected over A by validation mAP. E0 remained higher than both trained
runs. This is one run per experiment on a small validation set, so the observed
difference does not establish that flipping will always improve generalization.
The selected B checkpoint is local at `runs/experiment-b/best.pt`; SHA256:
`4c2f1e05ca76cb06294e4f37e2b9bc6ae504bcb02da965b411603d4b9ea44c90`.
Its saved epoch-3 validation metrics were reproduced after reloading.

### Resolution comparison: actual result

The same B checkpoint, 34-image validation split, score floor 0.001, and four
CPU threads were used for both settings. The timing sample contained ten fixed
validation images, each measured three times after five warm-ups. This was one
run on the Intel Core i5-1335U laptop; CPU load and thermal state can affect it.

| Inference resize | mAP 0.50:0.95 | AP50 | AP75 | Median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320/640 | 0.6752 | 0.9186 | 0.7919 | 315.3 ms | 484.0 ms |
| 256/512 | 0.6191 | 0.9292 | 0.7344 | 255.6 ms | 421.2 ms |

The reduced setting's median was about 18.9% faster, but mAP fell by about
0.0561 (5.61 percentage points), exceeding our allowed 0.02 absolute drop. We
therefore kept 320/640. AP50 rose while AP75 and averaged mAP fell; looking at
one IoU threshold alone would have missed the broader localization trade-off.
This is an optimization attempt, not a claim that the selected configuration is
universally fastest. The checkpoint weights and file size did not change.

Recorded summaries: `docs/evidence/experiments/experiment-b.json` and
`docs/evidence/experiments/optimization.json`. The latter includes each timing
sample and image ID; no dataset photographs are committed. The selected
checkpoint and resize are recorded in `manifests/selected-model.json`.

### George's verification (pending)

- [ ] For image width 10, flip box `[1, 2, 4, 7]` and explain each new x value.
- [ ] Explain why changing only the image pixels would corrupt the labels.
- [ ] Explain why B was selected over A even though E0's validation mAP is higher.
- [ ] Explain why 256/512 was rejected although it reduced median latency.
- [ ] Run the README's prediction with the selected checkpoint; record actual
  commands and observations without claiming that AI's checks were your own.

George's observations and independent code change: **pending**. Next: build the
local application, then freeze decisions and evaluate the held-out test split.

## Steps 7–8: local demo and frozen evaluation

### The demo

`app.py` lets a user upload a JPEG or PNG, select a confidence threshold, and
see boxes, scores, and a count. It can also use the local TorchVision example.
The application and CLI share `pedestrian.inference.predict`, so preprocessing
and person filtering are not independently reimplemented for the UI.

`pedestrian/app_support.py` checks the selected checkpoint's SHA256 and its
experiment/epoch metadata before loading it, applies the selected 320/640
resize, and puts the detector in evaluation mode. Streamlit's resource cache
avoids loading the model on every widget interaction; changing the selection or
the checkpoint file's size/modification time causes a reload. We chose a local
Streamlit app to avoid a separate frontend/backend, at the cost of less control
over UI behavior and less flexibility than a dedicated service.

Image uploads are decoded with Pillow, oriented using EXIF, and converted to
RGB; grayscale and RGBA inputs are supported. Invalid files get a readable
message. The server is configured to listen on `127.0.0.1` with a 20 MB upload
limit. The README documents launching it and recovering after retraining, where
the new checkpoint may have a different checksum.

AI browser checks: sample image produced one detection; changing the slider to
1.0 produced zero; grayscale and RGBA blank images produced zero; a corrupt PNG
showed the expected error; uploading a valid image afterwards recovered. An
automated UI test covers slider and missing-checkpoint behavior. A fresh local
virtual environment installed the lockfile and ran inference successfully.

Qualitative checks used three TorchVision images outside Penn-Fudan, without
labels or quantitative claims: `person1.jpg` produced one person detection;
`astronaut.jpg` produced one person detection; and `dog1.jpg` produced one
incorrect person detection at score **0.7585**. AI visually inspected these
outputs. This highlights a real weakness on non-person images, which our
pedestrian-only evaluation dataset does not cover. We did not alter the model
or threshold in response. Source images and generated overlays stay local.

### Freezing decisions

Before evaluating the test split, `pedestrian.final_evaluate freeze` records the
selected checkpoint and checksum, resolution, split manifest, evaluation
thresholds, dependency lock, and relevant source-file hashes. It fingerprints
the held-out files without running the model. Text hashes normalize LF/CRLF so
equivalent Git checkouts do not falsely appear changed; checkpoint and image
bytes remain exact hashes. A test confirms real changes are rejected.

The frozen contract is committed before the test run. The evaluation command
verifies it, uses only the selected B checkpoint and its fixed resize, and
refuses to overwrite an existing result. It reports mAP over scores plus the
separate 0.5 confidence/IoU counts. The held-out results and interpretation
will be added after the actual run, without retuning against them.

The first preflight attempt stopped before loading any test images: AI treated
the selection returned by `load_selected_model` as checkpoint metadata, causing
a `KeyError: config`. No test metrics were produced. The corrected preflight
reads the checkpoint metadata explicitly and has a regression test. The first
freeze remains in Git; a revised freeze records the corrected code before the
test evaluation.
