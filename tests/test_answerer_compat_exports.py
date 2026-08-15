"""P1 compatibility gate: src.generation.answerer must keep re-exporting every
symbol it defined or imported before the generation split, so existing imports
(scripts, provider, tests) keep working unchanged.

The name lists are machine-generated from the pre-split module namespace
(see tests/fixtures/p1_answerer_namespace.json).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import src.generation.answerer as answerer_module

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "p1_answerer_namespace.json"


class AnswererCompatExportTests(unittest.TestCase):
    def test_previously_defined_names_are_still_importable(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        missing = [name for name in payload["defined"] if not hasattr(answerer_module, name)]
        self.assertEqual(
            missing,
            [],
            f"answerer compatibility shim lost {len(missing)} previously defined names: {missing}",
        )

    def test_previously_imported_names_are_still_importable(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        missing = [name for name in payload["previously_imported"] if not hasattr(answerer_module, name)]
        self.assertEqual(
            missing,
            [],
            f"answerer compatibility shim lost {len(missing)} previously imported names: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
