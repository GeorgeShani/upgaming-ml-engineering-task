# Assignment deliverables

This maps the five submission deliverables in the supplied Computer Vision
End-to-End ML Project brief to the repository. The brief separately describes
a technical interview after submission and permits AI with disclosure.

| Brief requirement | Where to find it |
| --- | --- |
| Source, setup, dependencies, training and use instructions, structure | `README.md`, `pyproject.toml`, `uv.lock`, `pedestrian/`, `app.py` and `tests/` |
| Trained model and working demonstration on new images | README evaluator quick start, `app.py` and the [checkpoint release](https://github.com/GeorgeShani/upgaming-ml-engineering-task/releases/tag/evaluated-checkpoint-b) |
| Experiments, comparisons, metrics and optimization comparison | `docs/technical-report.md` and `docs/evidence/experiments/` |
| Technical report: problem, choices, experiments, difficulties, strengths, weaknesses and future work | `docs/technical-report.md` |
| AI tools: purpose and verification | `docs/ai-usage.md` and `docs/learning-journal.md` |

The system uses open libraries and runs on CPU without a paid cloud service.
The project selects Penn-Fudan, one of the datasets suggested in the brief.
The brief allows pretrained models and does not require a particular accuracy.

The model has known limitations, including errors on non-person images; the
report and demo do not claim production-level reliability. AI use and AI-run
checks are disclosed. The interview requirement to explain and independently
modify the project occurs after submission; AI-authored notes do not establish
that the candidate has personally demonstrated those skills.
