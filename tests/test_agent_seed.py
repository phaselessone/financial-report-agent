"""Agent benchmark seed tests (checklist v3.0 §P3 benchmark categories)."""

from __future__ import annotations

import shutil
import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.answer_eval import _materialize_answer_eval_seed
from src.evaluation.benchmark_assets import build_agent_seed_draft, write_agent_seed_draft

INDUSTRIES = ("semiconductor", "new_energy", "consumer", "liquor")
INDUSTRY_NAMES = {
    "semiconductor": "半导体",
    "new_energy": "新能源",
    "consumer": "消费",
    "liquor": "白酒",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "outputs" / "test_scratch_agent_seed"


def make_chunk(*, chunk_id: str, doc_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "child_text": text,
        "support_span": text,
        "section_title": "",
        "section_path": "",
        "chunk_type": "text",
        "element_type": "paragraph",
    }


def make_doc(*, doc_id: str, industry: str, short_title: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc_key": f"{industry}-{doc_id}",
        "file_name": f"{industry}_{doc_id}.pdf",
        "short_title": short_title,
        "title_topic": short_title,
    }


def build_test_corpus() -> tuple[list[dict], list[dict]]:
    """Two docs per industry; each doc carries one chunk."""
    chunks: list[dict] = []
    manifest: list[dict] = []
    for industry in INDUSTRIES:
        for ordinal in (1, 2):
            doc_id = f"{industry}-doc{ordinal}"
            title = f"{INDUSTRY_NAMES[industry]}行业周报{ordinal}"
            manifest.append(make_doc(doc_id=doc_id, industry=industry, short_title=title))
            chunks.append(
                make_chunk(
                    chunk_id=f"{doc_id}-c1",
                    doc_id=doc_id,
                    text=f"{title}显示，{INDUSTRY_NAMES[industry]}景气度上行{ordinal}0%。",
                )
            )
    return chunks, manifest


def make_retrieval_row(*, industry: str, intent: str, ordinal: int, target_doc_keys: list[str], query: str) -> dict:
    return {
        "question_id": f"{industry}_{intent}_{ordinal:02d}",
        "query": query,
        "question_type": "comparison" if intent == "comparison" else "fact",
        "intent": intent,
        "industry": industry,
        "target_doc_keys": target_doc_keys,
    }


def build_test_retrieval_seed() -> list[dict]:
    rows: list[dict] = []
    for industry in INDUSTRIES:
        doc1, doc2 = f"{industry}-doc1", f"{industry}-doc2"
        title1, title2 = f"{INDUSTRY_NAMES[industry]}行业周报1", f"{INDUSTRY_NAMES[industry]}行业周报2"
        rows.append(
            make_retrieval_row(
                industry=industry,
                intent="numeric_fact",
                ordinal=1,
                target_doc_keys=[f"{industry}-{doc1}"],
                query=f"{INDUSTRY_NAMES[industry]}行业周报中的关键数据是怎样的？",
            )
        )
        rows.append(
            make_retrieval_row(
                industry=industry,
                intent="comparison",
                ordinal=1,
                target_doc_keys=[f"{industry}-{doc1}", f"{industry}-{doc2}"],
                query=f"《{title1}》与《{title2}》分别关注哪些重点？",
            )
        )
        rows.append(
            make_retrieval_row(
                industry=industry,
                intent="comparison",
                ordinal=2,
                target_doc_keys=[f"{industry}-{doc1}", f"{industry}-{doc2}"],
                query=f"《{title1}》与《{title2}》的行业观点有何差异？",
            )
        )
        rows.append(
            make_retrieval_row(
                industry=industry,
                intent="inductive",
                ordinal=1,
                target_doc_keys=[f"{industry}-{doc1}", f"{industry}-{doc2}"],
                query=f"近期{INDUSTRY_NAMES[industry]}行业报告共同强调了哪些主题？",
            )
        )
    return rows


class AgentSeedBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks, self.manifest = build_test_corpus()
        self.seed_rows = build_test_retrieval_seed()

    def test_build_agent_seed_draft_produces_four_categories(self) -> None:
        rows = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        self.assertEqual(len(rows), 20)
        counter = Counter(row["category"] for row in rows)
        self.assertEqual(
            dict(counter),
            {"agent_recovery": 4, "agent_multi_source": 8, "agent_numeric_missing": 4, "agent_abstain": 4},
        )

    def test_answerable_rows_resolve_gold_chunks_abstain_rows_do_not(self) -> None:
        rows = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        chunk_ids = {chunk["chunk_id"] for chunk in self.chunks}
        for row in rows:
            if row["must_abstain"]:
                self.assertEqual(row["target_doc_keys"], [])
                self.assertEqual(row["gold_chunk_ids"], [])
            else:
                self.assertTrue(row["target_doc_keys"])
                self.assertTrue(row["gold_chunk_ids"])
                self.assertTrue(set(row["gold_chunk_ids"]) <= chunk_ids)
                self.assertFalse(row["must_abstain"])

    def test_recovery_rows_are_weakened_and_flagged(self) -> None:
        rows = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        recovery_rows = [row for row in rows if row["category"] == "agent_recovery"]
        self.assertEqual(len(recovery_rows), 4)
        for row in recovery_rows:
            self.assertTrue(row["must_recover"])
            self.assertEqual(row["expected_first_failure"], "source_diversity_missing")
            self.assertNotIn("《", row["query"])
            self.assertEqual(row["domain_hint"], "")
        numeric_rows = [row for row in rows if row["category"] == "agent_numeric_missing"]
        for row in numeric_rows:
            self.assertTrue(row["must_recover"])
            self.assertEqual(row["expected_first_failure"], "numeric_missing")
            self.assertEqual(row["domain_hint"], row["industry"])
        for row in rows:
            if row["category"] not in ("agent_recovery", "agent_numeric_missing"):
                self.assertEqual(row["domain_hint"], row["industry"])

    def test_question_ids_are_unique(self) -> None:
        rows = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        ids = [row["question_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_builder_is_deterministic(self) -> None:
        first = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        second = build_agent_seed_draft(retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest)
        self.assertEqual(first, second)


class AgentSeedMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        self.chunks, self.manifest = build_test_corpus()
        self.seed_rows = build_test_retrieval_seed()
        self.agent_rows = build_agent_seed_draft(
            retrieval_seed_rows=self.seed_rows, chunks=self.chunks, manifest=self.manifest
        )

    def tearDown(self) -> None:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    def _materialize(self, seed_path: Path, preserve_fields: tuple[str, ...]):
        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        docs_by_key = {row["doc_key"]: row for row in self.manifest}
        return _materialize_answer_eval_seed(
            seed_path=seed_path,
            chunks=self.chunks,
            chunk_lookup=chunk_lookup,
            docs_by_key=docs_by_key,
            preserve_fields=preserve_fields,
        )

    def test_preserve_fields_carries_category_metadata_into_materialized_rows(self) -> None:
        seed_path = SCRATCH_DIR / "agent_seed.jsonl"
        write_agent_seed_draft(self.agent_rows, seed_path)
        resolved, missing = self._materialize(
            seed_path, preserve_fields=("category", "must_recover", "expected_first_failure", "domain_hint")
        )
        self.assertEqual(len(resolved), 20)
        self.assertEqual(missing, [])
        for row in resolved:
            self.assertIn("category", row)
            self.assertIn("must_recover", row)
            self.assertIn("expected_first_failure", row)
            self.assertIn("domain_hint", row)
        self.assertEqual(
            Counter(row["category"] for row in resolved),
            {"agent_recovery": 4, "agent_multi_source": 8, "agent_numeric_missing": 4, "agent_abstain": 4},
        )

    def test_default_materialization_drops_extra_fields(self) -> None:
        seed_path = SCRATCH_DIR / "agent_seed.jsonl"
        write_agent_seed_draft(self.agent_rows[:4], seed_path)
        resolved, _ = self._materialize(seed_path, preserve_fields=())
        for row in resolved:
            self.assertNotIn("category", row)
            self.assertNotIn("must_recover", row)


if __name__ == "__main__":
    unittest.main()
