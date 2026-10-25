"""Run measured model selection, independent fold confirmation, and full refit."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

import competition as c


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_decision_rule(result):
    directory = c.ROOT / "artifacts/experiments" / result["name"]
    config = read(directory / "config.json")
    probabilities = np.load(directory / "valid_probabilities.npy")
    indices = np.load(directory / "valid_indices.npy")
    train,_,_ = c.frames()
    y = np.array([c.LABELS.index(label) for label in train.label.iloc[indices]])
    candidates = []
    for power in (0.,.5,1.):
        corrected = c.decision_correct(probabilities,config["weights"],power)
        metrics = c.evaluate(y,corrected,train,indices)
        candidates.append((metrics["macro_f1"],power,metrics,corrected))
    score,power,metrics,corrected = max(candidates,key=lambda row:row[0])
    result.update({"raw_macro_f1":result["macro_f1"],"macro_f1":score,"decision_weight_power":power})
    c.dump(directory / "calibrated_metrics.json",metrics)
    c.dump(directory / "decision_comparison.json",[{"power":p,"macro_f1":s} for s,p,_,_ in candidates])
    np.save(directory / "calibrated_valid_probabilities.npy",corrected)
    print("Development decision rule",result,flush=True)
    return result


def main():
    c.seed_all()
    started = time.monotonic()
    while True:
        try:
            if read(c.CACHE / "metadata.json").get("complete", False):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(10)
    print("Frozen cache complete; beginning measured experiments", flush=True)
    c.refine_cache()
    c.baseline()
    base = read(c.ROOT / "artifacts/experiments/frozen_baseline.json")
    init_c = max(base, key=lambda row:row["macro_f1"])["C"]
    # A short, predeclared search over imbalance treatment; all other settings fixed.
    runs = [("tail2_scaled_weighted_fold0", .35), ("tail2_scaled_unweighted_fold0", 0.)]
    summary = []
    for name, weight in runs:
        directory = c.ROOT / "artifacts/experiments" / name
        complete = directory.joinpath("history.json").exists() and len(read(directory / "history.json")) == 2
        if not complete:
            c.train_model(name, fold=0, epochs=2, lr=5e-5, weight_power=weight, init_c=init_c)
        history = read(c.ROOT / "artifacts/experiments" / name / "history.json")
        best = max(history, key=lambda row:row["macro_f1"])
        summary.append(select_decision_rule({"name":name,"weight_power":weight,"macro_f1":best["macro_f1"],"epoch":best["epoch"],"learning_rate":5e-5}))
        c.dump(c.ROOT / "artifacts/experiments/comparison.json", summary)
    selected = max(summary, key=lambda row:row["macro_f1"])
    selected_history = read(c.ROOT / "artifacts/experiments" / selected["name"] / "history.json")
    epoch_seconds = selected_history[-1]["elapsed_seconds"]/len(selected_history)
    public = read(c.ROOT / "artifacts/provenance/kaggle_GetCompetition.txt")
    deadline = datetime.fromisoformat(public["deadline"].replace("Z","+00:00")).timestamp()
    remaining = deadline-datetime.now(timezone.utc).timestamp()
    # Only spend more compute when the learning curve and measured runtime justify it.
    # Reserve confirmation, a full-data refit, verification, and 15 minutes for recovery.
    still_improving = len(selected_history)>1 and selected_history[-1]["macro_f1"] > selected_history[-2]["macro_f1"]+.001
    extra_budget = epoch_seconds*(3+3*1.35+3*1.7)+15*60
    name = "tail2_higher_lr_fold0"
    candidate_dir = c.ROOT / "artifacts/experiments" / name
    existing_candidate = candidate_dir.joinpath("history.json").exists()
    if existing_candidate or (still_improving and remaining > extra_budget and selected["macro_f1"] < .985):
        if not existing_candidate:
            print("Learning curve and measured runtime justify one higher-LR experiment",flush=True)
            c.train_model(name,fold=0,epochs=3,lr=1e-4,weight_power=selected["weight_power"],init_c=init_c)
        history = read(c.ROOT / "artifacts/experiments" / name / "history.json")
        best = max(history,key=lambda row:row["macro_f1"])
        summary.append(select_decision_rule({"name":name,"weight_power":selected["weight_power"],"macro_f1":best["macro_f1"],
                        "epoch":best["epoch"],"learning_rate":1e-4}))
        c.dump(c.ROOT / "artifacts/experiments/comparison.json",summary)
        selected = max(summary,key=lambda row:row["macro_f1"])
    train,_,folds = c.frames()
    dev_config = read(c.ROOT / "artifacts/experiments" / selected["name"] / "config.json")
    confirmation_rows = int((folds.fold != 1).sum())
    # Larger fitting partitions have more updates per epoch. Preserve roughly
    # the selected development update budget, then use that validated duration
    # for the full-data refit rather than multiplying epochs with dataset size.
    epochs = max(1,min(selected["epoch"],round(selected["epoch"]*dev_config["train_rows"]/confirmation_rows)))
    print("Selected on development fold only:", selected, flush=True)
    confirmation = "tail2_confirmation_fold1"
    c.train_model(confirmation, fold=1, epochs=epochs, lr=selected["learning_rate"], weight_power=selected["weight_power"], init_c=init_c,decision_power=selected["decision_weight_power"])
    confirm_history = read(c.ROOT / "artifacts/experiments" / confirmation / "history.json")
    # Report the predetermined terminal epoch, even if a different confirmation epoch scored better.
    confirmation_metric = confirm_history[-1]
    selection = {"development":selected,"confirmation":confirmation_metric,"epochs":epochs,
                 "development_epochs":selected["epoch"],"confirmation_epochs":epochs,
                 "duration_policy":"Round development epochs * development fitting rows / confirmation fitting rows; use the confirmed duration for full refit.",
                 "learning_rate":selected["learning_rate"],"init_c":init_c,"weight_power":selected["weight_power"],
                 "decision_weight_power":selected["decision_weight_power"],
                 "two_fold_macro_f1_mean":float(np.mean([selected["macro_f1"],confirmation_metric["macro_f1"]])),
                 "two_fold_macro_f1_std":float(np.std([selected["macro_f1"],confirmation_metric["macro_f1"]])),
                 "selection_seconds":time.monotonic()-started,
                 "note":"Development uses folds 2-4 for fitting and fold 0 for selection. Fold 1 is fully reserved until confirmation with fixed terminal epoch; confirmation trains on all other folds. Development and confirmation use different training-set sizes, so their mean/SD is descriptive. No fold predictions are ensembled."}
    c.dump(c.ROOT / "artifacts/experiments/selection.json", selection)
    c.train_model("final_single_model", epochs=epochs, lr=selected["learning_rate"], weight_power=selected["weight_power"], final=True, init_c=init_c,decision_power=selected["decision_weight_power"])
    c.infer("final_single_model")
    subprocess.run([sys.executable, str(c.ROOT / "scripts/validate_submission.py"), "--experiment", "final_single_model"], check=True)
    print("End-to-end workflow completed and independently validated", flush=True)


if __name__ == "__main__":
    main()
