# Submission walkthrough

This checklist separates the checks performed by AI from George's own work.
Record actual observations; don't check a box solely because a previous AI run
passed. The original checkpoint is `runs/experiment-b/best.pt`, with SHA-256
`4c2f1e05ca76cb06294e4f37e2b9bc6ae504bcb02da965b411603d4b9ea44c90`.
It is available from the [evaluated checkpoint release](https://github.com/GeorgeShani/upgaming-ml-engineering-task/releases/tag/evaluated-checkpoint-b).

## Run and explain the demo

1. From the project folder, run `uv sync --locked` and then
   `uv run --locked streamlit run app.py`.
2. Open the local URL printed by Streamlit. Upload a JPEG or PNG or select the
   example if available. Record what the boxes, count, and scores show.
3. Move the confidence slider and observe what changes. The slider filters
   displayed predictions; it does not update the weights.
4. Trace the data through `app.py`: image decoding, cached model loading,
   `predict`, filtering, and drawing. Explain why the model is cached and why
   the count can differ when the slider changes.
5. Explain one limitation from the report: the dog example was incorrectly
   labeled as a person, even with a confidence of 0.7585.

## Independent exercise

Make a small presentation-only change yourself. For example, in `app.py` find
the sentence directly below the title and rewrite it in your own words to help
users understand what the confidence slider does. Save the file and use the app
to verify your change. This exercise does not require a model or threshold
change. If you prefer a more challenging exercise, add a download button for
the annotated image using the project's existing drawing function.

Record in `learning-journal.md` what you changed, how you tested it, what you
observed, and any assistance you used. The personal change is pending until you
perform it; do not attribute an AI-generated edit to yourself.

## Rehearsal questions

- How does an instance mask become a box, and what does label 1 mean?
- What was frozen during training, and what did the optimizer update?
- Why did validation select B and the 320/640 resize?
- Why can't the observed test split be reused to select a new model or threshold?
- What do mAP, AP50, precision and recall tell us, and what can they not prove?

The detailed results, limitations, and actual AI contributions are in
`technical-report.md`, `learning-journal.md`, and `ai-usage.md`.
