"""Read-only data audit and duplicate-aware validation fold construction."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "audit"
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
SEED = 20261022


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "dataset/train.csv", dtype=str)
    test = pd.read_csv(ROOT / "dataset/test.csv", dtype=str)
    sample = pd.read_csv(ROOT / "dataset/sample_submission.csv", dtype=str)
    assert list(train) == ["id", "text", "language", "label"]
    assert list(test) == ["id", "text", "language"]
    assert list(sample) == ["id", "label"]
    assert len(train) == 14000 and len(test) == len(sample) == 6000
    assert set(train.label) == set(LABELS)
    for df in (train, test, sample):
        assert not df.isna().any().any() and df.id.is_unique
    assert set(train.id).isdisjoint(test.id)
    assert set(test.id) == set(sample.id)

    report = {
        "seed": SEED,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted((ROOT / "dataset").glob("*.csv"))
        },
        "shapes": {"train": list(train.shape), "test": list(test.shape), "sample": list(sample.shape)},
        "nulls": {"train": train.isna().sum().to_dict(), "test": test.isna().sum().to_dict()},
        "class_counts": train.label.value_counts().to_dict(),
        "language_class_counts": pd.crosstab(train.language, train.label).to_dict(),
        "test_languages": test.language.value_counts().to_dict(),
        "sample_order_matches_test": bool(sample.id.equals(test.id)),
        "exact_train_duplicates": int(train.text.duplicated().sum()),
        "exact_train_test_overlap": len(set(train.text) & set(test.text)),
    }
    for split, df in (("train", train), ("test", test)):
        for unit, lengths in (("characters", df.text.str.len()), ("words", df.text.str.split().str.len())):
            report[f"{split}_{unit}_quantiles"] = lengths.quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_dict()
        report[f"{split}_lengths_by_language"] = df.assign(n=df.text.str.len()).groupby("language").n.describe().to_dict()
    stat = ks_2samp(train.text.str.len(), test.text.str.len())
    report["character_length_ks"] = {"statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}
    report["id_numeric_range_by_label_diagnostic_only"] = train.assign(
        number=train.id.str.extract(r"(\d+)", expand=False).astype(int)
    ).groupby("label").number.agg(["min", "max", "median"]).to_dict()

    normalized = train.text.map(normalize)
    parent = np.arange(len(train))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first = {}
    for i, text in enumerate(normalized):
        if text in first:
            union(i, first[text])
        else:
            first[text] = i
    report["normalized_train_duplicates"] = int(normalized.duplicated().sum())
    report["normalized_train_test_overlap"] = len(set(normalized) & set(test.text.map(normalize)))

    # These features are used ONLY to prevent validation leakage, never as classifier inputs.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2,
                                 max_features=100000, dtype=np.float32, sublinear_tf=True)
    x = vectorizer.fit_transform(normalized)
    pairs = []
    for start in range(0, len(train), 96):
        similarity = (x[start:start + 96] @ x.T).tocsr()
        similarity.data[similarity.data < .92] = 0
        similarity.eliminate_zeros()
        coo = similarity.tocoo()
        for row, col, score in zip(coo.row, coo.col, coo.data):
            i, j = start + int(row), int(col)
            if i < j:
                union(i, j)
                pairs.append((i, j, float(score)))
        if start % 1920 == 0:
            print(f"Duplicate scan {start}/{len(train)}, pairs={len(pairs)}", flush=True)
    groups = np.array([find(i) for i in range(len(train))])
    grouped = train.assign(group=groups)
    conflicts = grouped.groupby("group").label.nunique()
    report["duplicate_groups"] = int(len(np.unique(groups)))
    report["near_duplicate_pairs_cosine_ge_0.92"] = len(pairs)
    report["groups_with_conflicting_labels"] = int((conflicts > 1).sum())
    report["largest_group_sizes"] = grouped.group.value_counts().head(20).tolist()
    pair_frame = pd.DataFrame(pairs, columns=["row_a", "row_b", "cosine"])
    if len(pair_frame):
        pair_frame["id_a"] = train.id.iloc[pair_frame.row_a].to_numpy()
        pair_frame["id_b"] = train.id.iloc[pair_frame.row_b].to_numpy()
        pair_frame["label_a"] = train.label.iloc[pair_frame.row_a].to_numpy()
        pair_frame["label_b"] = train.label.iloc[pair_frame.row_b].to_numpy()
    pair_frame.to_csv(OUT / "duplicate_pairs.csv", index=False)
    grouped[grouped.group.isin(conflicts[conflicts > 1].index)].to_csv(OUT / "label_conflicts.csv", index=False)

    folds = np.full(len(train), -1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(train, train.language + "_" + train.label, groups)):
        folds[valid] = fold
    assignments = pd.DataFrame({"id": train.id, "group": groups, "fold": folds})
    assert assignments.groupby("group").fold.nunique().max() == 1
    assert (folds >= 0).all()
    assignments.to_csv(OUT / "folds.csv", index=False)
    report["fold_counts"] = pd.crosstab(folds, train.label).to_dict()
    report["fold_language_class_counts"] = grouped.assign(fold=folds).groupby(["fold", "language", "label"]).size().to_dict()
    # Convert tuple keys for portable JSON.
    report["fold_language_class_counts"] = {str(k): int(v) for k, v in report["fold_language_class_counts"].items()}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if "language_class" not in k and "lengths_by" not in k}, indent=2), flush=True)


if __name__ == "__main__":
    main()
