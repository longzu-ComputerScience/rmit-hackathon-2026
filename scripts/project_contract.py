"""Small dependency-free invariants shared by the research pipeline and CI tests.

These checks intentionally avoid importing torch or loading competition data, so a
public clone can validate core assumptions without downloading private artifacts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

MODEL_ID = "jhu-clsp/mmBERT-small"
MODEL_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"
MODEL_WEIGHTS = "pytorch_model.bin"
LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")


def trim_head_tail(tokens: Sequence[int], max_length: int) -> list[int]:
    """Reserve two special-token slots, keeping 75% head and 25% tail."""
    if max_length < 4:
        raise ValueError("max_length must reserve room for special tokens")
    budget = max_length - 2
    items = list(tokens)
    if len(items) <= budget:
        return items
    tail = budget // 4
    if not tail:
        return items[:budget]
    return items[: budget - tail] + items[-tail:]


def validate_group_folds(groups: Sequence[int], folds: Sequence[int]) -> None:
    """Disallow the same near-duplicate group in multiple folds."""
    if len(groups) != len(folds) or not groups:
        raise ValueError("Group and fold assignments must be nonempty and aligned")
    group_to_fold: dict[int, int] = {}
    for group, fold in zip(groups, folds):
        if fold < 0:
            raise ValueError("Unassigned fold")
        if group in group_to_fold and group_to_fold[group] != fold:
            raise ValueError("Duplicate group crosses validation folds")
        group_to_fold[group] = fold


def validate_submission_records(
    rows: Sequence[Mapping[str, str]],
    expected_ids: Sequence[str],
    labels: set[str] | frozenset[str],
) -> None:
    """Check ordering, IDs, label values and column contract (not ML accuracy)."""
    if not rows or len(rows) != len(expected_ids):
        raise ValueError("Submission row count does not match test set")
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected test IDs are empty or duplicated")
    for row, expected_id in zip(rows, expected_ids):
        if set(row) != {"id", "label"}:
            raise ValueError("Submission must have exactly id,label columns")
        if row["id"] != expected_id:
            raise ValueError("Submission IDs are misordered or misaligned")
        if row["label"] not in labels:
            raise ValueError("Unknown submission label")
