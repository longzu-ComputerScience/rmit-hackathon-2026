"""Source and release hygiene tests that work on a fresh public clone."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceLayoutTests(unittest.TestCase):
    def test_all_python_scripts_parse(self):
        scripts = sorted((ROOT / "scripts").glob("*.py"))
        self.assertGreaterEqual(len(scripts), 9)
        for script in scripts:
            with self.subTest(script=script.name):
                ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    def test_ignore_rules_cover_non_public_assets(self):
        patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        for required in ("dataset/", "submission/", "artifacts/", ".venv/", ".cache/", ".env"):
            with self.subTest(pattern=required):
                self.assertIn(required, patterns)

    def test_public_documentation_exists(self):
        for relative in (
            "README.md",
            "docs/EXPERIMENT_RESULTS.md",
            "docs/REPRODUCIBILITY.md",
            "docs/MODEL_CARD.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
