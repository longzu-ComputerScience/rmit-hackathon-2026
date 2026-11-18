"""Offline tests for the pipeline invariants; no model or contest data needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from project_contract import (
    LABELS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS,
    trim_head_tail,
    validate_group_folds,
    validate_submission_records,
)


class TokenTruncationTests(unittest.TestCase):
    def test_short_sequence_is_unchanged(self):
        self.assertEqual(trim_head_tail([1, 2, 3], 8), [1, 2, 3])

    def test_head_and_tail_preserved(self):
        # A max length of 10 leaves 8 content tokens: 6 head, 2 tail.
        self.assertEqual(
            trim_head_tail(list(range(20)), 10),
            [0, 1, 2, 3, 4, 5, 18, 19],
        )

    def test_exact_budget(self):
        self.assertEqual(trim_head_tail(list(range(8)), 10), list(range(8)))

    def test_minimum_length(self):
        with self.assertRaises(ValueError):
            trim_head_tail([1, 2, 3], 3)

    def test_no_tail_for_tiny_budget(self):
        self.assertEqual(trim_head_tail(list(range(9)), 5), [0, 1, 2])


class FoldIntegrityTests(unittest.TestCase):
    def test_same_group_stays_in_one_fold(self):
        validate_group_folds([2, 2, 3, 4, 4], [0, 0, 1, 2, 2])

    def test_group_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_group_folds([2, 2, 3], [0, 1, 2])

    def test_missing_assignment_rejected(self):
        with self.assertRaises(ValueError):
            validate_group_folds([1, 2], [0, -1])

    def test_unaligned_assignments_rejected(self):
        with self.assertRaises(ValueError):
            validate_group_folds([1, 2], [0])


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["q-1", "q-2"]
        self.rows = [
            {"id": "q-1", "label": "BENIGN"},
            {"id": "q-2", "label": "PROMPT_INJECTION"},
        ]

    def test_valid_submission(self):
        validate_submission_records(self.rows, self.ids, set(LABELS))

    def test_swapped_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission_records(list(reversed(self.rows)), self.ids, set(LABELS))

    def test_wrong_label_rejected(self):
        bad = [dict(self.rows[0]), dict(self.rows[1])]
        bad[1]["label"] = "OTHER"
        with self.assertRaises(ValueError):
            validate_submission_records(bad, self.ids, set(LABELS))

    def test_extra_column_rejected(self):
        bad = [dict(row) for row in self.rows]
        bad[0]["confidence"] = "0.9"
        with self.assertRaises(ValueError):
            validate_submission_records(bad, self.ids, set(LABELS))

    def test_duplicate_test_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission_records(self.rows, ["q-1", "q-1"], set(LABELS))


class BackboneContractTests(unittest.TestCase):
    def test_approved_revision_is_full_sha(self):
        self.assertEqual(MODEL_ID, "jhu-clsp/mmBERT-small")
        self.assertEqual(len(MODEL_REVISION), 40)
        self.assertTrue(all(char in "0123456789abcdef" for char in MODEL_REVISION))

    def test_checkpoint_format_is_explicit(self):
        self.assertEqual(MODEL_WEIGHTS, "pytorch_model.bin")


if __name__ == "__main__":
    unittest.main()
