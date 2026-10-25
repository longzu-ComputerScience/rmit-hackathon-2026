"""Independent, standard-library read-back validation of the final deliverable."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"BENIGN", "JAILBREAK", "PROMPT_INJECTION"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    return columns, rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    path = ROOT / "submission/submission.csv"
    assert path.is_file(), f"Final submission missing: {path}"
    assert [p.name for p in path.parent.iterdir()] == ["submission.csv"], "Submission folder must contain exactly one file"
    columns, submission = read_csv(path)
    _, test = read_csv(ROOT / "dataset/test.csv")
    sample_columns, sample = read_csv(ROOT / "dataset/sample_submission.csv")
    assert columns == sample_columns == ["id", "label"]
    assert len(submission) == len(test) == len(sample) == 6000
    ids = [row["id"] for row in submission]
    assert all(ids) and len(set(ids)) == len(ids)
    assert set(ids) == {row["id"] for row in test} == {row["id"] for row in sample}
    assert ids == [row["id"] for row in sample], "Submission must follow the sample order"
    assert all(row["label"] in LABELS and set(row) == {"id", "label"} for row in submission)
    audit = json.loads((ROOT / "artifacts/audit/report.json").read_text(encoding="utf-8"))
    for filename, metadata in audit["files"].items():
        assert sha256(ROOT / "dataset" / filename) == metadata["sha256"], f"Original dataset was modified: {filename}"
    experiment = ROOT / "artifacts/experiments" / args.experiment
    cfg = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "artifacts/experiments/selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "inference_manifest.json").read_text(encoding="utf-8"))
    history = json.loads((experiment / "history.json").read_text(encoding="utf-8"))
    assert cfg["final"] and cfg["train_rows"] == 14000
    assert cfg["epochs"] == selection["epochs"] and cfg["lr"] == selection["learning_rate"]
    assert cfg["weight_power"] == selection["weight_power"]
    assert cfg.get("decision_weight_power",0.) == selection.get("decision_weight_power",0.)
    assert len(history) == cfg["epochs"] and history[-1]["epoch"] == cfg["epochs"]
    assert history[-1]["optimizer_steps"] > 0 and all(math.isfinite(row["train_loss"]) for row in history)
    assert manifest["ensemble"] is False
    assert manifest["backbone"]["model_id"] == "jhu-clsp/mmBERT-small"
    assert sha256(ROOT / manifest["trained_checkpoint"]) == manifest["checkpoint_sha256"]
    assert sha256(path) == manifest["submission_sha256"]
    import numpy as np
    probabilities = np.load(experiment / "test_probabilities.npy")
    assert probabilities.shape == (len(test),3) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-5)
    expected = {row["id"]:manifest["label_order"][int(index)] for row,index in zip(test,probabilities.argmax(1))}
    assert all(row["label"] == expected[row["id"]] for row in submission), "Probability-to-ID alignment differs"
    report = {"passed":True,"rows":len(submission),"columns":columns,"unique_ids":len(set(ids)),
              "label_counts":dict(Counter(row["label"] for row in submission)),
              "submission_sha256":sha256(path),"original_datasets_unchanged":True,
              "trained_model_manifest_verified":True,"probability_to_id_alignment_verified":True,"experiment":args.experiment}
    destination = ROOT / "artifacts/submission_validation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
