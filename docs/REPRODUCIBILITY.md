# Reproducibility guide

## 1. Scope

The repository is a **source-only public release**. Full replay requires approved access to the original RMIT Hackathon 2026 competition files; the source code cannot regenerate those private CSVs. The reported numbers come from the recorded original experiment, not from a new public CI run.

The code was developed/tested with **Python 3.12.9** and **PyTorch 2.9.1+cpu** on Windows; the remaining package versions are pinned in [requirements.txt](../requirements.txt). CPU training can take substantial time and create gigabytes of cache/checkpoint data.

## 2. Environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Select the CPU-compatible PyTorch wheel for your OS using official instructions.
.venv/bin/python -m pip install torch==2.9.1
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

PyTorch wheel availability varies by OS/architecture. Use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) when a pinned wheel is not available. The pipeline uses ModernBERT internals, so changing `transformers` requires compatibility retesting.

## 3. Input contract

Place competition-authorized files at:

```text
dataset/
├── train.csv               # id,text,language,label; 14,000 rows
├── test.csv                # id,text,language; 6,000 rows
└── sample_submission.csv   # id,label; 6,000 rows
```

Do **not** copy these inputs into public commits. The schema is validated by `scripts/audit_data.py`. The code checks IDs, duplicate groups, class distributions and fold boundaries, and records hashes in `artifacts/audit/`.

## 4. Full experiment

Run from the repository root after installing dependencies:

```shell
python scripts/audit_data.py
python scripts/prepare_model.py
python scripts/competition.py cache --max-length 320
python scripts/run_experiments.py
python scripts/verify_forward.py
python scripts/validate_submission.py --experiment final_single_model
python scripts/report_results.py
```

`prepare_model.py` downloads only the pinned public backbone commit `abc32620dd4f6ab06f5fbe905dc25f310618e09f`; it explicitly requests `pytorch_model.bin`, the format consumed by the training pipeline. Previously captured competition metadata is preserved in local ignored artifacts; the public download script no longer depends on Kaggle endpoints.

The training script runs the declared development comparisons, then the independent confirmation and one final full-data refit. It no longer changes the model-search plan based on the current date or original competition deadline.

## 5. Evidence and acceptance

- `python -m unittest discover -s tests -v`: dependency-free tests for token trimming, group/fold safety, submission contracts and public file layout.
- `scripts/verify_forward.py`: checks cached-versus-direct outputs for selected samples; this is **not** full test-set model evaluation.
- `scripts/validate_submission.py`: asserts exact schema/order, checkpoint/file hashes, label alignment, and 6,000-row completion; it does **not** assert prediction accuracy.
- `scripts/report_results.py`: regenerates the detailed local technical report from measured artifacts.
- `docs/EXPERIMENT_RESULTS.md`: compact public summary (not a substitute for independent execution).

The original validation checkpoints, raw data, probability arrays and final submission are **not** committed. Their omission is intentional and limits exact external replay until the reader obtains the source assets.

## 6. Known limits

This is a CPU-oriented, non-distributed research runner. Expect memory/disk requirements beyond what public CI can reasonably provide. Floating-point differences, changes in pretrained APIs, and unhandled semantic near-duplicates can affect measurements; do not present the offline Macro F1 as an official competition rank.
