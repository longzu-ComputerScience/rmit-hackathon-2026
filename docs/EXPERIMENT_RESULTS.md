# Experimental results (offline validation)

This summary is transcribed from the locally generated `artifacts/final_report.md`. The full report and artifacts remain on the original machine; no Kaggle test labels or hidden labels were used to estimate these metrics.

## Task
Three-way multilingual text classification: `BENIGN`, `JAILBREAK`, and `PROMPT_INJECTION`.

- Backbone: pinned `jhu-clsp/mmBERT-small`.
- Last two Transformer layers, final normalization, and a mean-plus-BOS classification head are fine-tuned; the first 20 layers are frozen.
- Trained on a CPU-only 16 GB host with frozen-activation caching.
- One final full-data model; no ensemble or test-time voting.

## Validation design

Five fixed, group-aware folds, stratified by language and class. Near-duplicate prompts (character 4/5-gram TF-IDF cosine >= 0.92) stay in the same fold. Fold 0 guided development choices; fold 1 was reserved for independent confirmation, using a freshly trained model with frozen hyperparameters. The final model was refit with all 14,000 labeled training examples.

| Measurement | Macro F1 |
| --- | ---: |
| Frozen-backbone linear probe, best development | 0.868527 |
| Fine-tuned model, best development | 0.902077 |
| **Independent confirmation (terminal epoch)** | **0.895587** |

Confirmation group-bootstrap 95% interval: **0.878566–0.911356**.

| Class | Confirmation precision | Confirmation recall | Confirmation F1 |
| --- | ---: | ---: | ---: |
| BENIGN | 0.958663 | 0.903725 | 0.930383 |
| JAILBREAK | 0.836901 | 0.951333 | 0.890456 |
| PROMPT_INJECTION | 0.911765 | 0.824468 | 0.865922 |

## Output validation

The local submission validator reported a CSV with 6,000 rows, matching the sample-submission ordering, unique IDs and valid label strings. Inference verification compared cached and direct model execution and reported identical sampled predictions. The submission CSV is **not included** in this repository.

## Limits

- All F1 values are offline estimates, **not official Kaggle leaderboard results**.
- Near-duplicate grouping does not guarantee semantic paraphrases or translations are grouped perfectly.
- The development score is optimistic because it was used for model selection; independent confirmation provides a stronger check.
- Reproduction needs competition data and model weights obtained independently under their respective access terms.
- Runtime and results depend on hardware and dependency versions; no trained checkpoint is distributed.

See [README](../README.md) for commands, selected hyperparameters and model setup.
