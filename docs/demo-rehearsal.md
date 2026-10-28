# Demo rehearsal: notes to practice aloud

These are AI-prepared prompts, not a record that George has rehearsed them.
Try explaining each answer in your own words and demonstrating it in the code.

1. **From mask to box.** Each nonzero mask value identifies one person. For
   each instance, find the minimum and maximum pixel coordinates and form an
   `xyxy` box. The dataset gives each box class label 1 for person; 0 is
   background. `pedestrian/annotations.py` implements the conversion.
2. **Training.** The pretrained Faster R-CNN uses a MobileNetV3 backbone. We
   froze the whole backbone including its feature pyramid, replaced the final
   predictor for background and person, and trained the remaining detection
   components. Each batch computes losses; backpropagation and SGD update the
   trainable parameters. Freezing reduces compute but limits adaptation.
3. **Selection.** A and B used the same split, initialization procedure and
   budget; B added random horizontal flips and transformed the boxes too. B's
   validation mAP was 0.6752 versus A's 0.6549. E0's pretrained validation
   mAP was higher, 0.6912; we report this honestly and chose B as the best
   trained model. Reduced resolution was faster but lost more validation mAP
   than our predefined 0.02 limit, so we kept 320/640.
4. **Test metrics.** On 34 test images with 87 annotated people, mAP over IoUs
   0.50–0.95 was 0.6659 and AP50 was 0.9442. At confidence 0.5 and IoU 0.5,
   78 matches, 6 unmatched predictions and 9 missed annotations yield
   precision 78/84 = 0.9286 and recall 78/87 = 0.8966. AP summarizes the
   precision–recall curve over confidence scores; the displayed threshold is
   separate. Decisions were frozen before the test and the test was not used
   to choose a model or threshold.
5. **Limitations.** Penn-Fudan is small, has no person-free images, and has
   some incomplete annotations; scene grouping is not guaranteed exhaustive.
   A dog was incorrectly labeled person with score 0.7585 in a qualitative
   check. These results do not prove performance on other environments.
   Independent and more diverse evaluation data would be a useful next step.

During the demo, show the image, count, boxes and scores, then move the slider.
The slider filters displayed predictions; it does not train or change weights.
