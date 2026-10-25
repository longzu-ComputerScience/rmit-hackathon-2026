"""Build the final technical report from measured experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    artifacts = ROOT / "artifacts"
    exps = artifacts / "experiments"
    selection = read(exps / "selection.json")
    selected = selection["development"]
    dev = read(exps / selected["name"] / "calibrated_metrics.json")
    confirmation = selection["confirmation"]
    final = read(exps / "final_single_model/config.json")
    validation = read(artifacts / "submission_validation.json")
    forward = read(artifacts / "forward_verification.json")
    baseline = read(exps / "frozen_baseline.json")
    raw_baseline_path = exps / "raw_frozen_baseline.json"
    raw_baseline = read(raw_baseline_path) if raw_baseline_path.exists() else []
    comparison = read(exps / "comparison.json")
    train = pd.read_csv(ROOT / "dataset/train.csv")
    fold = pd.read_csv(artifacts / "audit/folds.csv")
    confirm_dir = exps / "tail2_confirmation_fold1"
    ix = np.load(confirm_dir / "last_valid_indices.npy")
    probabilities = np.load(confirm_dir / "last_valid_probabilities.npy")
    y = np.array([LABELS.index(label) for label in train.label.iloc[ix]])
    pred = probabilities.argmax(1)
    assert abs(f1_score(y,pred,labels=[0,1,2],average="macro")-confirmation["macro_f1"]) < 1e-12
    groups = fold.group.iloc[ix].to_numpy()
    group_indices = {int(group):np.where(groups==group)[0] for group in np.unique(groups)}
    rng = np.random.default_rng(20261022)
    unique = np.array(list(group_indices))
    scores = []
    for _ in range(400):
        sample = np.concatenate([group_indices[int(g)] for g in rng.choice(unique,size=len(unique),replace=True)])
        scores.append(f1_score(y[sample],pred[sample],average="macro",labels=[0,1,2],zero_division=0))
    ci = np.quantile(scores,[.025,.975]).tolist()
    diagnostics = train.iloc[ix].copy()
    diagnostics["prediction"] = np.array(LABELS)[pred]
    diagnostics["confidence"] = probabilities.max(1)
    errors = diagnostics[diagnostics.label!=diagnostics.prediction]
    errors.to_csv(confirm_dir / "terminal_errors.csv",index=False)
    summary = {"development_macro_f1":dev["macro_f1"],"confirmation_macro_f1":confirmation["macro_f1"],
               "confirmation_group_bootstrap_95pct_ci":ci,
               "confirmation_errors_by_language":errors.language.value_counts().to_dict(),
               "confirmation_confusions":errors.groupby(["label","prediction"]).size().rename("count").reset_index().to_dict(orient="records"),
               "confirmation_error_char_length_median":float(errors.text.str.len().median()) if len(errors) else 0,
               "confirmation_correct_char_length_median":float(diagnostics.loc[diagnostics.label==diagnostics.prediction,"text"].str.len().median())}
    (artifacts / "results_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines = ["# Measured competition results", "",
             "Approved model: **jhu-clsp/mmBERT-small**; one backbone and one final full-data refit.",
             "Pinned revision: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`.","",
             "Selected for multilingual coverage and 384-dimensional hidden states, which permit real CPU fine-tuning on this 16 GB host. Larger encoders and decoder LLMs were not practical without CUDA. Safety-specific English encoders were not selected because most examples are non-English.","",
             "## Validation", "",
             "Fixed seed 20261022; five group-aware folds stratified by language and label. 373 high-similarity pairs are kept within folds. No detected duplicate-group label conflicts. Development trains on folds 2-4 and selects settings on fold 0. Fold 1 is fully reserved from development fitting and selection; a fresh model trains on the other four folds. Before confirmation, its epoch count is scaled by development/confirmation fitting rows to approximately preserve the selected update budget; the full refit uses that confirmed duration. No test labels, test-set class balancing, or leaderboard feedback are used.", "",
             f"- Best frozen-backbone linear-probe development Macro F1: **{max(row['macro_f1'] for row in baseline):.6f}**.",
             f"- Best fine-tuned development Macro F1: **{dev['macro_f1']:.6f}**.",
             f"- Independent confirmation terminal-epoch Macro F1: **{confirmation['macro_f1']:.6f}**.",
             f"- Confirmation group-bootstrap 95% interval: **{ci[0]:.6f}–{ci[1]:.6f}**.",
             f"- Descriptive two-split mean ± population SD: **{selection['two_fold_macro_f1_mean']:.6f} ± {selection['two_fold_macro_f1_std']:.6f}**. Training-set sizes differ; development selection makes this mean optimistic.","",
             "| Class | Development F1 | Confirmation precision | Confirmation recall | Confirmation F1 |",
             "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        row = confirmation["report"][label]
        lines.append(f"| {label} | {dev['report'][label]['f1-score']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} |")
    lines += ["","| Language | Development Macro F1 | Confirmation Macro F1 |","|---|---:|---:|"]
    for lang in sorted(dev["by_language"]):
        lines.append(f"| {lang} | {dev['by_language'][lang]:.6f} | {confirmation['by_language'][lang]:.6f} |")
    lines += ["","## Experiments", "", "| Configuration | Best development Macro F1 | Selected epoch |", "|---|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['name']} | {row['macro_f1']:.6f} | {row['epoch']} |")
    if raw_baseline:
        lines += ["",f"Initial unstandardized linear fits reached only {max(row['macro_f1'] for row in raw_baseline):.6f} Macro F1 and hit the 500-iteration limit. Training-partition standardization resolved convergence; the selected standardized C=0.1 fit converged in {next(row['iterations'] for row in baseline if row['C']==.1)} iterations."]
    lines += ["",f"Selected loss-weight exponent: **{final['weight_power']}**. Weights are computed only from the relevant training partition. The selected head is initialized by a regularized linear fit on that same partition and then jointly fine-tuned with the last two encoder layers.","",
              "## Final configuration", "",
              f"- All {final['train_rows']:,} labeled examples; {final['epochs']} epochs; new pretrained initialization.",
              f"- Last 2 transformer layers, final normalization, and mean-plus-BOS head: {final['trainable_parameters']:,} trainable parameters; first 20 layers frozen.",
              f"- AdamW: encoder LR {final['lr']}, head LR {final['lr']*final['head_lr_multiplier']}; weight decay 0.01; 6% linear warmup and decay; gradient clipping 1.0.",
              f"- Bucketed microbatches up to {final['micro_batch_size']}; gradient accumulation targets 32 examples; training-partition feature standardization inside the head; dropout 0.1.",
              f"- Class-weight posterior correction power {final.get('decision_weight_power',0.)}, selected from 0/0.5/1 on development only and fixed before confirmation.",
              f"- Maximum 320 tokens; head/tail truncation affects 5.81% of train and test combined; cached frozen computation, float32 optimization, 8 CPU threads.",
              "- The approved pretrained checkpoint plus `artifacts/experiments/final_single_model/final.pt` form the trained model. No ensemble, bagging, or prediction override.","",
              "## Deliverable verification", "",
              f"`submission/submission.csv`: {validation['rows']:,} rows, exactly `id,label`, unique and complete test IDs, sample-submission ordering, three valid label strings, no missing values or index column. Independently read back with the standard CSV parser. The submission directory contains exactly this one file.",
              f"Submission SHA-256: `{validation['submission_sha256']}`.",
              f"Cached versus direct trained-model inference passed; maximum probability discrepancy {forward['max_probability_error']:.8f}; identical checked predictions. Original dataset hashes are unchanged.",
              "No automatic Kaggle submission was made.","",
              "## Limitations", "",
              "CPU and available-time constraints limited the search to one approved backbone, two loss settings, and two evaluation folds. Upper-layer fine-tuning freezes most parameters. Some semantic paraphrases or translations may cross folds despite character-level grouping. Long prompts lose middle content. The confidence interval covers validation sampling variation, not all model-selection or distribution-shift uncertainty. Local F1 does not guarantee a leaderboard rank.", "",
              "## Confirmation error analysis", "", "```json",json.dumps(summary,indent=2),"```", "",
              "Rows and full confusion matrices are retained in experiment error CSVs and histories for inspection."]
    (artifacts / "final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
