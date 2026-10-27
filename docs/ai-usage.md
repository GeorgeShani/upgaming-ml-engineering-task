# AI assistance and verification

AI assistance is permitted by the assignment. This record distinguishes code
and explanations drafted by AI from work George has personally verified.

## Milestone 1

**AI assistance:** interpreted the brief, explored the empty repository and
available hardware, discussed the schedule with George, checked primary docs,
drafted setup, inference code, tests, and learning materials. The engineering-
partner skill informed the mechanism explanations and alternative comparisons.

**AI verification:** created the environment, checked compiled NMS and the
evaluation backend against known inputs, ran 12 passing tests, and performed
pretrained inference on the tagged TorchVision sample. Inspected the annotated
image and recorded the actual output in the journal and evidence directory.
AI-run checks do not count as George's independent understanding exercise.

**George's completed decisions:** selected a 12-16-hour focused budget, beginner
ML explanations, pedestrian detection, Streamlit, AI drafts followed by personal
verification, and committed learning notes.

**George's code verification:** pending. Record commands you actually run,
observations, and an independent modification in the journal. Do not claim to
have reviewed or authored code merely because this file exists.

No training outcomes, benchmark numbers, user exercises, or elapsed work hours
are invented. Update this record at each later milestone.

## Milestone 2, first slice

AI drafted the mask-to-box function, focused tests, one-image inspection command,
and journal explanation. AI downloaded the dataset, ran the checks, and visually
inspected the two boxes and the colorized mask for `FudanPed00001`. George's own
inspection and explanation remain pending. Dataset images and visualizations
stay local, consistent with the dataset readme's reposting restrictions.

## Milestone 2, audit and splits

AI wrote and tested the audit and grouped-splitting code, reviewed contact sheets
for obvious related scenes, created the reviewed group list, and generated the
committed manifests. AI verified the reported counts and byte-for-byte repeated
output. The similarity screening and manual review are incomplete safeguards,
not proof that all related images have been identified. George's independent
audit walkthrough and verification remain pending.

## Validation baseline, experiment A, and review

AI drafted the dataset loader, evaluation and matching code, training loop,
checkpoint handling, error-review tool, tests, and explanations. AI ran the
baseline and three training epochs, inspected validation examples, and verified
the saved checkpoint by re-evaluation. Preprocessing and TorchVision version
metadata were added to the saved checkpoints after training; SHA256 of the
model state was checked before and after to verify the weights did not change.
Results and times are measured, not fabricated. George's independent review,
explanation, and code modification remain pending. The held-out test split was
not evaluated for these steps.

## Experiment B and optimization

AI implemented and tested joint image/box flipping, ran the three-epoch B
experiment, checked A/B settings, selected the best trained checkpoint, and
measured the two resize settings. AI recorded the actual samples, decision,
model checksum, and limitations; the test split was not used. George's own
explanations, verification, and independent code change remain pending.

## Local demo and frozen test preparation

AI implemented the local Streamlit interface, checksum-verified selection,
fresh-retraining selection path, and frozen-evaluation checks. AI tested the
browser with a public sample, confidence slider, grayscale/RGBA images, corrupt
upload, and empty output; tested missing checkpoint and checksum errors in
automated tests; and verified setup in a new local virtual environment. AI also
inspected three qualitative images outside Penn-Fudan: a dog image produced a
false person detection. The model was not changed in response. George's own
demo, review, explanation, and independent code change remain pending.
AI's first held-out preflight failed before any test images or metrics were
processed because selection and checkpoint metadata were confused. AI fixed the
check, added a regression test, and recorded a revised freeze before the run.

AI ran the corrected frozen held-out evaluation once, recorded the measured
aggregate metrics and counts, and updated the technical report. AI did not use
the results to retune the checkpoint or display threshold. George's own review,
demo rehearsal, and independent modification are still pending.
