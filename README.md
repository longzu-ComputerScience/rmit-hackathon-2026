# Multilingual Prompt-Attack Classification

[![Source quality](https://github.com/longzu-ComputerScience/rmit-hackathon-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/longzu-ComputerScience/rmit-hackathon-2026/actions/workflows/ci.yml)

**A leakage-aware NLP experiment for detecting benign prompts, jailbreak attempts, and prompt injections across languages.** Developed for RMIT Hackathon 2026, using partial fine-tuning of an open pretrained Transformer on a CPU-only machine.

**Scope:** Research pipeline and offline evaluation, **not** a deployed detection service. This public repository contains source and summarized findings only; competition data, trained checkpoints, cached features and submissions are excluded.

## Key results

| Experiment | Macro F1 | Evaluation split |
| --- | ---: | --- |
| Frozen-backbone linear probe | 0.8685 | Development |
| Fine-tuned encoder + classification head | **0.9021** | Development (model selection) |
| Selected configuration | **0.8956** | Independent confirmation fold |

The confirmation score is the stronger estimate; development was used for model selection. The group-bootstrap 95% confidence interval for the confirmation fold is **0.8786–0.9114**. These are **offline validation results, not official Kaggle leaderboard scores**. See [detailed experiment results](docs/EXPERIMENT_RESULTS.md).

## Problem and approach

Given a prompt, predict one of three labels:

- `BENIGN`: normal, non-attack input
- `JAILBREAK`: attempt to bypass model instructions or safety constraints
- `PROMPT_INJECTION`: attempt to redirect an assistant through malicious instructions in its input or context

```text
Competition train.csv (14,000 labeled rows)
          |
          v
Data audit + near-duplicate grouping (character n-gram similarity)
          |
          v
5 stratified group-aware folds
          |
          +--> Development: fit on folds 2–4 / select on fold 0
          |         |
          |         v
          |    Frozen linear probe vs. last-2-layer fine-tuning
          |         |
          +--> Independent confirmation: train on 0,2,3,4 / evaluate on 1
          |         |
          v         v
Final refit on all 14,000 labels with selected configuration
          |
          v
Single-model prediction + schema/provenance checks
          |
          v
submission.csv (6,000 test rows, kept outside Git)
```

**Model:** [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), pinned to commit `abc32620dd4f6ab06f5fbe905dc25f310618e09f`. The first 20 Transformer layers are frozen; the final two layers, final normalization, and classification head are fine-tuned. Frozen activations are cached to reduce CPU cost. The output uses one model, not an ensemble.

**Leakage controls:** Duplicate-linked examples stay in the same fold. Training-only feature statistics are used for normalization. IDs, test labels, nearest-neighbor labels, and language priors are not classifier inputs. Hyperparameters are selected on development data; the confirmation fold remains unseen during selection.

## Repository layout

```text
.
├── .github/workflows/ci.yml     # Public offline checks; no private data
├── docs/
│   ├── EXPERIMENT_RESULTS.md    # Reported metrics and limitations
│   ├── MODEL_CARD.md            # Model scope, risks, evaluation
│   └── REPRODUCIBILITY.md       # Requirements, data contract, commands
├── scripts/
│   ├── project_contract.py      # Version pin and dependency-free invariants
│   ├── audit_data.py            # Data/schema audit and grouped folds
│   ├── prepare_model.py         # Pinned backbone download + SHA verification
│   ├── competition.py           # Tokenization, cached features, train, infer
│   ├── run_experiments.py       # Fixed selection / confirmation / final refit
│   ├── verify_forward.py        # Cached vs direct model equivalence
│   ├── validate_submission.py   # Output schema and ID alignment
│   └── report_results.py        # Evidence-backed technical report
├── tests/                       # Dependency-free unit and integrity tests
├── requirements.txt
└── README.md
```

Local-only folders `dataset/`, `artifacts/`, `submission/`, `.cache/`, and `.venv/` are ignored intentionally.

## Quick verification (no dataset or model required)

Python **3.12**:

```shell
python -m unittest discover -s tests -v
```

These checks exercise token truncation, split-leakage constraints, submission contracts, source parsing and privacy boundaries. GitHub Actions runs them on pushes and pull requests. **They do not certify training, dataset availability or model accuracy.**

## Reproduce the full experiment

The full run requires access to the original competition CSVs and a compatible PyTorch CPU installation. See [Reproducibility Guide](docs/REPRODUCIBILITY.md) for both Windows and Linux/macOS setup, approved model revision, hardware expectations and validation steps.

From PowerShell, once the environment and data are prepared:

```powershell
& .\.venv\Scripts\python.exe scripts/audit_data.py
& .\.venv\Scripts\python.exe scripts/prepare_model.py
& .\.venv\Scripts\python.exe scripts/competition.py cache --max-length 320
& .\.venv\Scripts\python.exe scripts/run_experiments.py
& .\.venv\Scripts\python.exe scripts/verify_forward.py
& .\.venv\Scripts\python.exe scripts/validate_submission.py --experiment final_single_model
& .\.venv\Scripts\python.exe scripts/report_results.py
```

The search plan is fixed for reproducibility rather than conditioned on the historical competition deadline. **Do not expect identical floating-point scores across hardware and dependency changes.** Reproduction can be CPU- and disk-intensive.

## Findings and limitations

- The independently confirmed **Macro F1 = 0.8956** demonstrates a useful baseline, not resistance to every possible attack.
- Confirmation F1 by class: BENIGN **0.9304**, JAILBREAK **0.8905**, PROMPT_INJECTION **0.8659**. See the full [results table](docs/EXPERIMENT_RESULTS.md).
- The similarity-based split reduces obvious near-duplicate leakage but may miss semantic paraphrases and translations.
- The 320-token head/tail strategy truncates **5.81%** of the available prompts, potentially losing middle-context attacks.
- The final checkpoint and private competition files are **not distributed**, so an independent reader must obtain permitted sources and perform training.

See the [model card](docs/MODEL_CARD.md) for scope and responsible-use caveats.

## Provenance and use

The backbone is third-party and governed by its own license; competition datasets and submissions are governed by their source terms. They are not redistributed here. This repository is an educational research demonstration, not an autonomous safety filter or a production security guarantee.

**Project status:** Completed recorded offline experiment; public code and lightweight CI are maintained separately from local experiment artifacts.
