# RMIT Hackathon 2026: multilingual prompt-attack classification

> **Public source-only release.** Competition datasets, submission files, model checkpoints, cached activations and generated experiment artifacts are intentionally excluded. See [validation results](docs/EXPERIMENT_RESULTS.md) for a reproducible-results summary. The recorded Macro F1 scores are offline validation metrics, **not** official Kaggle leaderboard scores.

Approved backbone: **`jhu-clsp/mmBERT-small`**, pinned to revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`.

This project trains one three-class classifier. The first 20 encoder layers stay
frozen; the last two layers, final normalization, and a mean-plus-BOS classification
head are fine-tuned. Frozen activations are cached to make repeated CPU training
practical. The classifier receives prompt text only. IDs, test labels, language
priors, nearest-neighbor labels, and external labeled datasets are not model inputs.
Final inference uses one full-data refit, with no ensemble or test-time voting.

Original competition files remain in `dataset/`. All experiment artifacts are
under `artifacts/`. Only the deliverable belongs in `submission/`.

## Reproduce

From PowerShell at the repository root:

```powershell
python -m venv --system-site-packages .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts/audit_data.py
& .\.venv\Scripts\python.exe scripts/prepare_model.py
& .\.venv\Scripts\python.exe scripts/competition.py cache --max-length 320
& .\.venv\Scripts\python.exe scripts/run_experiments.py
& .\.venv\Scripts\python.exe scripts/verify_forward.py
& .\.venv\Scripts\python.exe scripts/validate_submission.py --experiment final_single_model
```

PyTorch 2.9.1 CPU is supplied by the host. A new environment also needs PyTorch
installed separately. Scripts resolve paths from their own location. Model
preparation downloads only public backbone weights and public competition text;
it uploads nothing and never submits predictions to Kaggle. All subsequent ML
commands run offline. The activation cache resumes completed rows after interruption.
Training writes epoch recovery checkpoints; use `competition.py train --resume`
with the original experiment arguments to resume an interrupted training run.

## Validation and model selection

Seed: `20261022`. Character 4/5-gram TF-IDF cosine similarity of at least 0.92
connects near-duplicate training prompts into groups. All connected members stay
in the same fold. Five fixed folds stratify by the combination of language and
class. Similarity features serve only split construction; they are never classifier
features. The audit found 373 similar pairs, no exact duplicates, and no conflicting
labels within detected groups. Semantic paraphrases and translations can remain.

Fold 0 is used to select regularization, loss weighting, and epoch count, using
folds 2–4 for development training. Fold 1 is reserved entirely from that fitting
and selection, then provides confirmation with the fixed settings and a new
model trained on folds 0, 2, 3, and 4. Each validation model
starts from the pinned pretrained weights. Neither validation checkpoint is used
for final predictions. Final training initializes again and uses all 14,000 labels.
Confirmation duration scales development epochs by the ratio of development to
confirmation fitting rows, rounded to whole epochs. The full refit uses that
confirmed duration; this avoids multiplying the update budget as more labeled
data becomes available.
Pooled features are standardized using statistics fitted only on the current
training partition. Those statistics are fixed buffers inside the classification
head, followed by dropout and the jointly fine-tuned linear output layer.
For a weighted loss, development also compares three strengths of the known
class-weight posterior correction (`0`, `0.5`, `1`). The chosen decision rule is
fixed before confirmation and the full-data refit. It is a single-model class
bias adjustment, with no voting or test-label use.
Metrics include macro F1, per-class precision/recall/F1, confusion matrices, and
per-language macro F1. Development selection has some optimism; the fixed terminal
epoch on the confirmation fold is the stronger independent check.

Tokenization matches the backbone's BOS/EOS postprocessor. Inputs longer than
320 tokens retain approximately 75% from the beginning and 25% from the end.
This truncates 5.81% of all prompts. Frozen activations after layer 16 are stored
in float16; the layer-20 cache is float32. This boundary rounding is also applied
in the independent direct-forward check.

## Artifacts

- `artifacts/audit/report.json`: schemas, hashes, distributions, length shifts,
  duplicates, and split diagnostics.
- `artifacts/provenance/`: official competition pages, model revision, and source
  metadata. Competition pages were obtained from Kaggle's public `PageService`.
- `artifacts/experiments/`: configurations, histories, validation predictions,
  error examples, checkpoints, selected configuration, and final test probabilities.
- `artifacts/submission_validation.json`: independent CSV read-back and provenance checks.
- `artifacts/forward_verification.json`: direct versus cached execution checks.
- `artifacts/final_report.md`: measured results and final configuration, generated after training.

The trained model consists of the pinned approved pretrained checkpoint plus
`artifacts/experiments/final_single_model/final.pt`, which replaces its final two
layers and supplies the trained classification head. Both parts are required.
`competition.py infer --name final_single_model` reproduces the saved predictions
from the verified frozen input cache.

To repeat the selected full-data training independently of the adaptive search's
wall-clock budget, use the completed cache and the recorded configuration:

```powershell
& .\.venv\Scripts\python.exe scripts/competition.py train --name final_single_model_reproduced --final --epochs 2 --lr 0.0001 --weight-power 0.35 --init-c 0.1 --decision-power 1
& .\.venv\Scripts\python.exe scripts/competition.py infer --name final_single_model_reproduced
```

No Kaggle notebook is created because university, campus, and team identifiers
were not provided. No submission attempts are consumed by this project.
