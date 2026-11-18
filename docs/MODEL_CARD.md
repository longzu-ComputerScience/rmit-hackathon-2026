# Model card — Multilingual Prompt-Attack Classification

## Intended use

Research and educational classification of multilingual prompts into `BENIGN`, `JAILBREAK` and `PROMPT_INJECTION`. The model is intended for evaluating NLP training and validation methods, not for standalone high-stakes moderation or security enforcement.

## Model details

- **Base:** `jhu-clsp/mmBERT-small` at `abc32620dd4f6ab06f5fbe905dc25f310618e09f`
- **Adaptation:** Freeze the first 20 encoder layers; fine-tune the final two, final normalization and a mean-plus-BOS classification head.
- **Input:** Text only; maximum 320 tokens, preserving approximately 75% from the head and 25% from the tail.
- **Output:** Three label probabilities and a single predicted label.
- **Training:** 14,000 labeled examples; one final full-data refit after development selection and confirmation.
- **Compute:** Optimized for CPU-only experiments using cached frozen features. Checkpoints and caches are not publicly distributed.

## Evaluation evidence

| Split | Metric | Value |
| --- | --- | ---: |
| Development | Frozen-backbone Macro F1 | 0.868527 |
| Development | Fine-tuned Macro F1 | 0.902077 |
| Independent confirmation | Macro F1 | **0.895587** |
| Independent confirmation | Group-bootstrap 95% CI | **0.878566–0.911356** |

Confirmation per-class F1: BENIGN 0.930383; JAILBREAK 0.890456; PROMPT_INJECTION 0.865922. English confirmation Macro F1 0.835171; French 0.906010; Italian 0.932136; Portuguese 0.885420. Reported measurements derive from local run artifacts summarized in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), not a public competition leaderboard.

## Validation protocol

Five stratified group-aware folds use character 4/5-gram similarity with threshold 0.92 to connect near-duplicates. Fold 0 guides development, fold 1 serves as independent confirmation with the selected configuration. Final model is refit with all labeled examples. The grouping cannot guarantee removal of semantic or translated near-duplicates.

## Risks and limitations

- Distribution shift, adversarial rephrasing and unseen languages may reduce performance substantially.
- False negatives can miss attacks, and false positives can flag benign prompts.
- Prompts longer than 320 tokens lose middle context.
- Classes and labels reflect the competition definition, not a universal taxonomy of attacks.
- No prospective production evaluation, runtime SLA, or deployed service is claimed.
- Local confirmation is a stronger estimate than development, but does not guarantee generalization.

## Data and weights

Competition files, trained weights, raw prompts and submission outputs are excluded from public Git. The backbone must be downloaded under its respective terms. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
