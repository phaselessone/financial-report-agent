"""Evidence search tool tests: checklist v3.0 §P4 (local deterministic lookup)."""

from __future__ import annotations

import unittest

from src.tools.evidence_search import (
    evidence_rows_for_doc,
    lookup_evidence,
    lookup_evidence_in_pool,
)


def make_row(*, chunk_id: str, doc_id: str, text: str = "", evidence_id: str = "") -> dict:
    row = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "child_text": text,
        "support_span": text,
    }
    if evidence_id:
        row["evidence_id"] = evidence_id
    return row


class LookupEvidenceTests(unittest.TestCase):
    def test_hit_by_chunk_id(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1"),
            make_row(chunk_id="c2", doc_id="d1", evidence_id="e2"),
        ]
        result = lookup_evidence("c2", rows=rows)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c2")

    def test_hit_by_evidence_id(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1"),
            make_row(chunk_id="c2", doc_id="d1", evidence_id="e2"),
        ]
        result = lookup_evidence("e1", rows=rows)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c1")

    def test_miss_returns_found_false(self) -> None:
        rows = [make_row(chunk_id="c1", doc_id="d1", evidence_id="e1")]
        result = lookup_evidence("missing", rows=rows)
        self.assertFalse(result["found"])
        self.assertIsNone(result["row"])

    def test_missing_evidence_id_key_is_tolerated(self) -> None:
        rows = [make_row(chunk_id="c1", doc_id="d1")]
        result = lookup_evidence("c1", rows=rows)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c1")

    def test_chunk_id_match_preferred_over_evidence_id(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1"),
            # a second row whose evidence_id equals the first row's chunk_id
            make_row(chunk_id="c2", doc_id="d2", evidence_id="c1"),
        ]
        result = lookup_evidence("c1", rows=rows)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c1")

    def test_duplicate_chunk_ids_first_wins(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", text="first"),
            make_row(chunk_id="c1", doc_id="d2", text="second"),
        ]
        result = lookup_evidence("c1", rows=rows)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["text"], "first")

    def test_empty_rows_safe(self) -> None:
        result = lookup_evidence("c1", rows=[])
        self.assertFalse(result["found"])
        self.assertIsNone(result["row"])


class LookupEvidenceInPoolTests(unittest.TestCase):
    def test_pool_hit_by_chunk_id_key(self) -> None:
        pool = {
            "c1": make_row(chunk_id="c1", doc_id="d1", evidence_id="e1"),
            "c2": make_row(chunk_id="c2", doc_id="d1", evidence_id="e2"),
        }
        result = lookup_evidence_in_pool("c2", pool=pool)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c2")

    def test_pool_hit_by_evidence_id_alias(self) -> None:
        pool = {
            "c1": make_row(chunk_id="c1", doc_id="d1", evidence_id="e1"),
            "c2": make_row(chunk_id="c2", doc_id="d1", evidence_id="e2"),
        }
        result = lookup_evidence_in_pool("e1", pool=pool)
        self.assertTrue(result["found"])
        self.assertEqual(result["row"]["chunk_id"], "c1")

    def test_pool_miss(self) -> None:
        pool = {"c1": make_row(chunk_id="c1", doc_id="d1", evidence_id="e1")}
        result = lookup_evidence_in_pool("missing", pool=pool)
        self.assertFalse(result["found"])
        self.assertIsNone(result["row"])


class EvidenceRowsForDocTests(unittest.TestCase):
    def test_filters_and_preserves_order(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1"),
            make_row(chunk_id="c2", doc_id="d2"),
            make_row(chunk_id="c3", doc_id="d1"),
            make_row(chunk_id="c4", doc_id="d3"),
        ]
        result = evidence_rows_for_doc("d1", rows=rows)
        self.assertEqual([row["chunk_id"] for row in result], ["c1", "c3"])

    def test_rows_missing_doc_id_skipped(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1"),
            {"chunk_id": "c2", "text": "no doc_id"},
            make_row(chunk_id="c3", doc_id="d1"),
        ]
        result = evidence_rows_for_doc("d1", rows=rows)
        self.assertEqual([row["chunk_id"] for row in result], ["c1", "c3"])

    def test_empty_rows_safe(self) -> None:
        self.assertEqual(evidence_rows_for_doc("d1", rows=[]), [])


if __name__ == "__main__":
    unittest.main()
